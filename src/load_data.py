import pandas as pd
import os
# imports for file handling and dataset loading
from datasets import load_dataset

# main entry point for data download
def main():
    print("--- Loading Dataset from Hugging Face (Raw) ---")
    
    # downloads raw dataset directly from huggingface hub
    # attempts to load qevasion dataset
    try:
        ds = load_dataset("ailsntua/QEvasion")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # transforms dataset to pandas for easier manipulation
    train_df = ds['train'].to_pandas()
    test_df = ds['test'].to_pandas()
    
    print(f"Train Shape: {train_df.shape}")
    print(f"Test Shape: {test_df.shape}")
    print(f"Columns: {train_df.columns.tolist()}")
    
    # writes validated dataframes to raw csv files
    # creates directory if it does not exist
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"Saved raw CSVs to {output_dir}")

if __name__ == "__main__":
    main()