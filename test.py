import torch
import nn

def test_net(net, criterion, val_loader, weights_path):
    net.cuda(0)
    net.load_state_dict(torch.load(weights_path, weights_only=True))
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
    print(f'Validation Loss with weights from {weights_path}: {epoch_val_loss:.7f}')

if __name__ == "__main__":
    val_data_path = 'data/val_data.csv'
    val_dataset = nn.ChessDataset(val_data_path, max_samples=None)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

    net = nn.ChessNN()
    criterion = nn.MPELoss(power=2.5)

    test_net(net, criterion, val_loader, 'weights/nn-e10b256-768-512-1.nnue')
