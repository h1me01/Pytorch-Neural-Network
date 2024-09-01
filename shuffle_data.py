import pandas as pd

def shuffle_csv(input_file, output_file):
    df = pd.read_csv(input_file, header=None)
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    shuffled_df.to_csv(output_file, header=False, index=False)
    print(f"Shuffled data saved to {output_file}")

input_file = 'data/data.csv'
output_file = "shuffled_output.csv" 
shuffle_csv(input_file, output_file)
