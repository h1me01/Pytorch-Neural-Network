import torch
import torch.nn as nn
import torch.optim as optim
import time

import fixed_seed
import model
import loss
import dataloader as dl

class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, path, max_samples=None):
        self.data_loader = dl.DataLoader(path)
        self.size = min(self.data_loader.get_size(), max_samples) if max_samples is not None else self.data_loader.get_size()
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input1_slice, input2_slice, score = self.data_loader.get_data(idx)

        return (
            torch.as_tensor(input1_slice, dtype=torch.float32),
            torch.as_tensor(input2_slice, dtype=torch.float32),
            torch.as_tensor(score, dtype=torch.float32),
        )
    
def save_checkpoint(epoch, model, optimizer, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def train(model, criterion, optimizer, train_loader, val_loader, device, epochs=500, resume=False):
    print('Model Architecture:\n')
    print(model)

    print(f'\nTraining dataset size: {len(train_loader.dataset)} samples')
    print(f'Validation dataset size: {len(val_loader.dataset)} samples\n')

    model.to(device)

    start_epoch = 0
    if resume:
        try:
            checkpoint = torch.load('checkpoint/nn-e32-checkpoint.tar', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f'Resuming training from epoch {start_epoch}...\n')
        except FileNotFoundError:
            print("Checkpoint not found, starting from scratch.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        
        # training phase
        model.train()
        running_train_loss = 0
        for inputs1, inputs2, targets in train_loader: 
            inputs1 = inputs1.to(device, non_blocking=True)
            inputs2 = inputs2.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs1, inputs2) 
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            
            optimizer.step()
            running_train_loss += loss.item() * inputs1.size(0)
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        
        # validation phase
        model.eval()
        running_val_loss = 0
        with torch.no_grad():
            for inputs1, inputs2, targets in val_loader: 
                inputs1 = inputs1.to(device, non_blocking=True)
                inputs2 = inputs2.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(inputs1, inputs2) 
                loss = criterion(outputs.squeeze(), targets)
                running_val_loss += loss.item() * inputs1.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_time = time.time() - start_time

        print(f'Epoch {epoch+1:02d}/{epochs} | '
              f'Training Loss: {epoch_train_loss:.7f} | '
              f'Validation Loss: {epoch_val_loss:.7f} | '
              f'Time per Epoch: {epoch_time:.2f} seconds')
        
        save_checkpoint(epoch, model, optimizer, f'checkpoint/nn-e{epoch+1}-checkpoint.tar')
        torch.save(model.state_dict(), f'weights/nn-e{epoch+1}b512-768-2x256-1.nnue')

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    #fixed_seed.set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data_path = r"C:\Users\semio\Downloads\chess_data.bin"
    val_data_path = r"C:\Users\semio\Downloads\chess_val_data.bin"

    train_dataset = ChessDataset(train_data_path, max_samples=None)
    val_dataset = ChessDataset(val_data_path, max_samples=None)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False)
    
    model = model.NN() 
    
    criterion = loss.MPE(power=2.5)
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.95, 0.999))
    
    train(model, criterion, optimizer, train_loader, val_loader, device, epochs=50, resume=False)
    