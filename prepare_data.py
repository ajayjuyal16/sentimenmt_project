import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def load_and_prepare_data(files):
    print("📥 Loading and combining data...")
    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file)
            # Normalize columns to lowercase and strip spaces
            df.columns = [col.lower().strip() for col in df.columns]

            # Check required columns
            if "clean_text" not in df.columns or "sentiment" not in df.columns:
                print(f"⚠️ Skipping {file}: missing 'clean_text' or 'sentiment' columns.")
                continue

            # Drop rows with missing values in those columns
            df = df.dropna(subset=["clean_text", "sentiment"])

            # Append if not empty
            if not df.empty:
                dfs.append(df)
            else:
                print(f"⚠️ Skipping {file}: no valid data after dropping NA.")
        except Exception as e:
            print(f"❌ Failed to load {file}: {e}")

    if not dfs:
        raise ValueError("No valid data loaded from files!")

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"🧮 Total combined rows: {len(combined_df)}")
    return combined_df

def map_sentiments(df):
    print("🔄 Mapping sentiment labels to numeric values...")
    label_map = {"positive": 1, "neutral": 0, "negative": -1}
    # Lowercase sentiment strings and map
    df["sentiment"] = df["sentiment"].str.lower().map(label_map)
    # Drop rows with unmapped or missing sentiment values
    df = df.dropna(subset=["sentiment"])
    df["sentiment"] = df["sentiment"].astype(int)
    print(f"✅ Sentiment mapping complete. Rows after mapping: {len(df)}")
    return df

def vectorize_and_split(df):
    print("🔢 Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["sentiment"]

    print("📊 Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("💾 Saving vectorized data and vectorizer to disk...")
    joblib.dump((X_train, X_test, y_train, y_test, vectorizer), "vectorized_data.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

    print("✅ Data vectorization and splitting done.")

def main():
    files = [
        "reddit_clean_final.csv",
        "youtube_clean_final.csv",
        "tumblr_clean_final.csv",
        "twitter_clean_final.csv"
    ]

    combined_df = load_and_prepare_data(files)
    combined_df = map_sentiments(combined_df)
    vectorize_and_split(combined_df)

if __name__ == "__main__":
    main()
