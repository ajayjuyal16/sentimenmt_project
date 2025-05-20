import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources (only once)
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stopwords and spaCy model
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    if not isinstance(text, str):
        return ""  # Return empty string for non-string input
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)    # Remove punctuation and numbers
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    doc = nlp(" ".join(tokens))
    lemmatized = [token.lemma_ for token in doc]
    return " ".join(lemmatized)

def preprocess_generic_file(input_file, output_file, text_column="text"):
    try:
        df = pd.read_csv(input_file)
        df.columns = [col.lower().strip() for col in df.columns]  # Normalize columns

        if text_column not in df.columns:
            print(f"❌ Text column '{text_column}' not found in {input_file}. Skipping.")
            return

        df[text_column] = df[text_column].fillna("")
        df['clean_text'] = df[text_column].apply(clean_text)

        df.to_csv(output_file, index=False)
        print(f"✅ Preprocessed and saved: {output_file}")
    except Exception as e:
        print(f"❌ Error processing {input_file}: {e}")

def preprocess_reddit_file(input_file, output_file):
    try:
        df = pd.read_csv(input_file)
        df.columns = [col.lower().strip() for col in df.columns]

        # Try to detect best text column
        for col_name in ['text', 'body', 'selftext', 'title']:
            if col_name in df.columns:
                text_col = col_name
                break
        else:
            print("❌ No suitable text column found in Reddit data. Skipping.")
            return

        df[text_col] = df[text_col].fillna("")
        df['clean_text'] = df[text_col].apply(clean_text)

        df.to_csv(output_file, index=False)
        print(f"✅ Preprocessed and saved: {output_file} (using '{text_col}' as text column)")
    except Exception as e:
        print(f"❌ Error processing {input_file}: {e}")

# Run preprocessing on all social media datasets
preprocess_reddit_file("reddit_data.csv", "reddit_data_clean_final.csv")
preprocess_generic_file("twitter_data.csv", "twitter_data_clean_final.csv", text_column="text")
preprocess_generic_file("tumblr_data.csv", "tumblr_data_clean_final.csv", text_column="text")
preprocess_generic_file("youtube_data.csv", "youtube_data_clean_final.csv", text_column="text")
