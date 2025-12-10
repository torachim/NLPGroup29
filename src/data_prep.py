import pandas as pd
import contractions
import nltk
import re
from datasets import load_dataset
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# download nltk resources
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# init lemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # safety check for empty strings
    if not isinstance(text, str): return ""
    
    # lowercase
    text = text.lower()
    
    # expand contractions
    # e.g. don't -> do not
    text = contractions.fix(text)
    
    # remove special chars (optional but good for baseline)
    # keep only alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # lemmatization
    # split into words, lemmatize, join back
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    text = " ".join(words)
    
    return text

def main():
    # load dataset from huggingface
    print("loading dataset ailsntua/QEvasion...")
    dataset = load_dataset("ailsntua/QEvasion")
    
    # convert to pandas for easier processing
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()
    
    # sanity check raw shapes
    print(f"raw train shape: {train_df.shape}")
    print(f"raw test shape: {test_df.shape}")
    
    # preprocessing columns
    # applying to both question and answer
    print("applying preprocessing...")
    
    # process train
    train_df['clean_question'] = train_df['question'].apply(clean_text)
    train_df['clean_answer'] = train_df['interview_answer'].apply(clean_text)
    
    # process test
    test_df['clean_question'] = test_df['question'].apply(clean_text)
    test_df['clean_answer'] = test_df['interview_answer'].apply(clean_text)
    
    # sanity check processed sample
    print("\nsample cleaned data:")
    print(train_df[['question', 'clean_question']].iloc[0].to_dict())
    
    # save to disk
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nsaved processed files to {train_path} and {test_path}")

if __name__ == "__main__":
    main()