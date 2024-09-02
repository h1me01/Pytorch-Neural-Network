import torch
import torch.nn as nn
import sparse

class ChessNN(nn.Module):
    def __init__(self):
        super(ChessNN, self).__init__()
        self.fc1 = nn.Linear(2 * 768, 2 * 512) 
        self.fc2 = nn.Linear(2 * 512, 1)

    def forward(self, x1, x2):
        merged_input = torch.cat((x1, x2), dim=1)
        x = torch.relu(self.fc1(merged_input)) 
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
    input1, input2 = sparse.get(fen)
    input1 = torch.tensor(input1, dtype=torch.float32).unsqueeze(0)
    input2 = torch.tensor(input2, dtype=torch.float32).unsqueeze(0)
    net.eval()
    
    with torch.no_grad():
        output = net(input1, input2)
    output = output.squeeze() 

    return output.item()

if __name__ == '__main__':
    weights_path = 'weights/nn-e1b256-768-512-1.nnue'

    net = ChessNN()
    net.load_state_dict(torch.load(weights_path, weights_only=True))

    fen = '2b1rrk1/ppp2pp1/7p/3Pp3/4P1nP/P1N2Nq1/1P2R1P1/3B1RK1 b - - 14 31'
    print(predict(net, fen))

    save_weights_and_biases(net)
