import pandas as pd

def shuffle_csv(input_file, output_file):
    df = pd.read_csv(input_file, header=None)
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    shuffled_df.to_csv(output_file, header=False, index=False)
    print(f"Shuffled data saved to {output_file}")

input_file = r'C:\Users\semio\Documents\Coding\Projects\Astra-Data\TrainingData\chess_data_d9.csv'
output_file = "shuffled_output.csv" 
shuffle_csv(input_file, output_file)
