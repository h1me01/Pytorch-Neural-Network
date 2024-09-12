import torch
import torch.nn as nn
import numpy as np
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

def save_quantisized(net, filename='nn-768-512-1-quantized.txt'):
    scalar_1 = 16
    scalar_2 = 512
    
    with open(filename, 'w') as f:
        for name, param in net.named_parameters():
            param_array = param.detach().cpu().numpy()

            if "fc1" in name:
                if "bias" in name:
                    param_quantized = np.round(param_array * scalar_1).astype(np.int16)
                else:
                    param_quantized = np.round(param_array * scalar_1).astype(np.int16)
            elif "fc2" in name:
                if "bias" in name:
                    param_quantized = np.round(param_array * (scalar_1 * scalar_2)).astype(np.int32)
                else:
                    param_quantized = np.round(param_array * scalar_2).astype(np.int16)

            f.write(f"{name}\n")
            f.write(' '.join(map(str, param_quantized.flatten())) + '\n')

def save(net, filename='nn-768-512-1.txt'):
    with open(filename, 'w') as f:
        for name, param in net.named_parameters():
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
    weights_path = 'main_weights/nn-e20b256-768-512-1.nnue'

    net = ChessNN()
    net.load_state_dict(torch.load(weights_path, weights_only=True))

    fen = '8/5p2/5k2/p1R2p1p/7P/4K1P1/8/5r2 b - - 9 46'
    print(predict(net, fen))

    save(net)
   # save_quantisized(net)
