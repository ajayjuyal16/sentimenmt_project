import pandas as pd

files = [
    "reddit_clean_final.csv",
    "youtube_clean_final.csv",
    "tumblr_clean_final.csv",
    "twitter_clean_final.csv"
]

dataframes = [pd.read_csv(file) for file in files]
all_data = pd.concat(dataframes, ignore_index=True)
all_data.columns = [col.lower().strip() for col in all_data.columns]

# Map sentiments to numeric as before
label_map = {"positive": 1, "neutral": 0, "negative": -1}
all_data["sentiment"] = all_data["sentiment"].str.lower().map(label_map)

print("Class distribution:")
print(all_data["sentiment"].value_counts())
