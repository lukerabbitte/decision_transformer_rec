import pandas as pd


def modify_data(filepath):
    data = pd.read_csv(filepath, delimiter="\t")

    # Change values in the 'returns' column
    data.iloc[:, 2] = data.iloc[:, 2].apply(lambda x: 5 if x in [4, 5] else 1)

    # Save modified DataFrame to a new TSV file
    new_filepath = filepath.replace('.tsv', '_modified.tsv')
    data.to_csv(new_filepath, sep='\t', index=False)

    return new_filepath


# Example usage
filepath = '/home/luke/code/decision_transformer_rec/goodreads/goodreads_eval_data.tsv'
new_file = modify_data(filepath)
print(f"Modified data saved to: {new_file}")
