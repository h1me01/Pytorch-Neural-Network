import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
import fixed_seed
import sparse

class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, max_samples=None):
        if max_samples is not None:
            self.df = pd.read_csv(file_path, nrows=max_samples)
        else:
            self.df = pd.read_csv(file_path)
        self.fens = self.df.iloc[:, 0]
        self.evals = self.df.iloc[:, 1]    

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        fen_data = self.fens.iloc[idx]
        eval_data = self.evals.iloc[idx]
        inputs = sparse.get(fen_data)
        return torch.as_tensor(inputs, dtype=torch.float32), torch.as_tensor(eval_data, dtype=torch.float32)

class MPELoss(nn.Module):
    def __init__(self, power=2.5):
        super(MPELoss, self).__init__()
        self.power = power

    def forward(self, outputs, targets):
        diff = outputs - targets
        abs_diff = torch.abs(diff)
        return torch.mean(torch.pow(abs_diff, self.power))

class CustomSigmoid(nn.Module):
    def __init__(self, scalar=0.0015):
        super(CustomSigmoid, self).__init__()
        self.scalar = torch.tensor(scalar).cuda(0)

    def forward(self, x):
        return 1 / (1 + torch.exp(-x * self.scalar))

class ChessNN(nn.Module):
    def __init__(self):
        super(ChessNN, self).__init__()
        self.fc1 = nn.Linear(768, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.custom_sigmoid = CustomSigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.custom_sigmoid(self.fc3(x))
        return x

def he_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def save_checkpoint(epoch, model, optimizer, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def train(net, criterion, optimizer, train_loader, val_loader, epochs=50, resume=False):
    print('Model Architecture:\n')
    print(net)

    print(f'\nTraining dataset size: {len(train_loader.dataset)} samples')
    print(f'Validation dataset size: {len(val_loader.dataset)} samples\n')

    net.cuda(0)

    start_epoch = 0
    if resume:
        checkpoint = torch.load('checkpoint/net_checkpoint.tar')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resuming training from epoch {start_epoch}...\n')

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        
        # Training phase
        net.train()
        running_train_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.cuda(0, non_blocking=True)
            targets = targets.cuda(0, non_blocking=True)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            
            # Gradient clipping
            #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        
        # Validation phase
        net.eval()
        running_val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.cuda(0, non_blocking=True)
                targets = targets.cuda(0, non_blocking=True)
                outputs = net(inputs)
                loss = criterion(outputs.squeeze(), targets)
                running_val_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_time = time.time() - start_time

        print(f'Epoch {epoch+1:02d}/{epochs} | '
              f'Training Loss: {epoch_train_loss:.7f} | '
              f'Validation Loss: {epoch_val_loss:.7f} | '
              f'Time per Epoch: {epoch_time:.2f} seconds')
        
        save_checkpoint(epoch, net, optimizer, f'checkpoint/net_epoch_{epoch+1}_checkpoint.tar')
        torch.save(net.state_dict(), f'weights/weights_epoch-{epoch+1}_768-64-64-1_b-128.nnue')

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    #fixed_seed.set_seed(42)

    train_data_path = 'data/data.csv'
    val_data_path = 'data/val_data.csv'

    train_dataset = ChessDataset(train_data_path, max_samples=None)
    val_dataset = ChessDataset(val_data_path, max_samples=None)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    net = ChessNN() 
    net.apply(he_init)
    
    criterion = MPELoss(power=2.5)
    optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.95, 0.999))
    
    train(net, criterion, optimizer, train_loader, val_loader, epochs=50, resume=False)
