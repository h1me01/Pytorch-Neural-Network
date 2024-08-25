import nn
import torch
import fixed_seed
import sparse

def save_weights_and_biases(model):
    for name, param in model.named_parameters():
        if "weight" in name or "bias" in name:
            param_array = param.detach().cpu().numpy()
            param_array = param_array.flatten()
            
            file_name = f"{name}.txt"    
            with open(file_name, 'w') as f:
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
    #fixed_seed.set_seed(42)

    weights_path = 'main_weights/net_weights_epoch_1.nnue'

    net = nn.ChessNN()
    net.load_state_dict(torch.load(weights_path, weights_only=True))

    #fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    #print(predict(net, fen))

    save_weights_and_biases(net)
