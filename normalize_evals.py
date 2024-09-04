import pandas as pd
import torch

max_position = None

file_path = r'C:\Users\semio\Documents\Coding\Projects\Astra-Data\TrainingData\chess_data_d9.csv'
df = pd.read_csv(file_path, header=None, nrows=max_position)
df[1] = df[1].astype(float)

def process_eval(fen, eval_val):
    stm = 0 if 'w' in fen else 1
    if stm == 1:
        eval_val = -eval_val
    
    eval_val = torch.tensor(eval_val, dtype=torch.float32)
    sigmoid_scalar = 0.0015
    eval_val = 1 / (1 + torch.exp(-eval_val * sigmoid_scalar))
    
    return eval_val.item()

for index, row in df.iterrows():
    fen = row[0]  
    eval_val = row[1] 
    processed_eval = process_eval(fen, eval_val)
    df.at[index, 1] = processed_eval 
    if (index + 1) % 100000 == 0:
        print(f'Processed {index + 1} rows')

output_file_path = 'data/data.csv'
df.to_csv(output_file_path, index=False, header=False)
print('Processing complete. Data saved to', output_file_path)
