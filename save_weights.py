import torch
import model

def save(net, filename='nn-768-2x256-1.txt'):
    with open(filename, 'w') as f:
        for name, param in net.named_parameters():
            if "weight" in name or "bias" in name:
                param_array = param.detach().cpu().numpy()
                param_array = param_array.flatten()
                
                f.write(f"{name}\n")
                f.write(' '.join(map(str, param_array)) + '\n')

if __name__ == '__main__':
    weights_path = 'main_weights/nn-e32b256-768-2x256-1.nnue'

    model = model.NN()
    model.load_state_dict(torch.load(weights_path, weights_only=True))

    save(model)
