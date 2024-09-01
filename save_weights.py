import torch
import torch.nn as nn
import sparse

class ChessNN(nn.Module):
    def __init__(self):
        super(ChessNN, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def save_weights_and_biases(model, filename='nn-768-512-1.txt'):
    with open(filename, 'w') as f:
        for name, param in model.named_parameters():
            if "weight" in name or "bias" in name:
                param_array = param.detach().cpu().numpy()
                param_array = param_array.flatten()
                
                f.write(f"{name}\n")
                f.write(' '.join(map(str, param_array)) + '\n')
                
def predict(net, fen):
    input_fen = sparse.get(fen)
    input_fen = torch.tensor(input_fen, dtype=torch.float32).unsqueeze(0)
    net.eval()
    
    with torch.no_grad():
        output = net(input_fen)
    output = output.squeeze() 

    return output.item()

if __name__ == '__main__':
    weights_path = 'main_weights/nn-e15b128-768-512-1.nnue'

    net = ChessNN()
    net.load_state_dict(torch.load(weights_path, weights_only=True))

    fen = 'rnbqkb1r/pppppppp/5n2/8/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 0 3'
    print(predict(net, fen))

    save_weights_and_biases(net)
