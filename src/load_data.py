import pandas as pd
import os
from datasets import load_dataset

def main():
    print("--- Loading Dataset from Hugging Face (Raw) ---")
    
    # load dataset
    try:
        ds = load_dataset("ailsntua/QEvasion")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # convert to pandas
    train_df = ds['train'].to_pandas()
    test_df = ds['test'].to_pandas()
    
    print(f"Train Shape: {train_df.shape}")
    print(f"Test Shape: {test_df.shape}")
    print(f"Columns: {train_df.columns.tolist()}")
    
    # save to raw
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"Saved raw CSVs to {output_dir}")

if __name__ == "__main__":
    main()