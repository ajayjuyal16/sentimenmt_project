import streamlit as st
import joblib
import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv

# Import fetch functions
from twitter_fetch import fetch_own_tweets
from reddit_fetch import fetch_reddit_posts
from tumblr_fetch import fetch_tumblr_posts
from youtube_fetch import fetch_youtube_comments

nltk.download('punkt')
nltk.download('stopwords')

load_dotenv()

stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

with open("sentiment_model.pkl", "rb") as f:
    model = joblib.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = joblib.load(f)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    doc = nlp(" ".join(tokens))
    lemmatized = [token.lemma_ for token in doc]
    return " ".join(lemmatized)

sentiment_map = {
    1: "Positive 😊",
    0: "Neutral 😐",
    -1: "Negative 😠"
}

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="centered")
st.title("🚀 Cross-Platform Sentiment Analysis")

platform = st.selectbox("Select Input Source", [
    "Manual Text",
    "Twitter (by username)",
    "YouTube Comments (by video ID)",
    "Reddit Posts (by subreddit)",
    "Tumblr Posts (by blog name)"
])

user_input = None
analyze_manual = False  # Flag for manual text analyze button

if platform == "Manual Text":
    user_input = st.text_area("Enter your text here:")
    analyze_manual = st.button("Analyze Text")

elif platform == "Twitter (by username)":
    username = st.text_input("Enter Twitter (X) username:")
    if st.button("Fetch Tweets") and username:
        tweets = fetch_own_tweets(username)
        if tweets:
            st.write(f"Showing last {len(tweets)} tweets by @{username}:")
            user_input = " ".join(tweets)
            for i, tweet in enumerate(tweets, start=1):
                st.write(f"{i}. {tweet}")
        else:
            st.warning("No tweets found or error fetching tweets.")

elif platform == "YouTube Comments (by video ID)":
    video_id = st.text_input("Enter YouTube video ID:")
    if st.button("Fetch Comments") and video_id:
        comments = fetch_youtube_comments(video_id)
        if comments:
            st.write(f"Showing {len(comments)} comments from video {video_id}:")
            user_input = " ".join(comments)
            for i, comment in enumerate(comments, start=1):
                st.write(f"{i}. {comment}")
        else:
            st.warning("No comments found or error fetching comments.")

elif platform == "Reddit Posts (by subreddit)":
    subreddit = st.text_input("Enter subreddit name:")
    if st.button("Fetch Reddit Posts") and subreddit:
        posts = fetch_reddit_posts(subreddit)
        if posts:
            st.write(f"Showing {len(posts)} posts from r/{subreddit}:")
            user_input = " ".join(posts)
            for i, post in enumerate(posts, start=1):
                st.write(f"{i}. {post}")
        else:
            st.warning("No posts found or error fetching Reddit posts.")

elif platform == "Tumblr Posts (by blog name)":
    blog_name = st.text_input("Enter Tumblr blog name:")
    if st.button("Fetch Tumblr Posts") and blog_name:
        posts = fetch_tumblr_posts(blog_name)
        if posts:
            st.write(f"Showing {len(posts)} posts from {blog_name}.tumblr.com:")
            user_input = " ".join(posts)
            for i, post in enumerate(posts, start=1):
                st.write(f"{i}. {post}")
        else:
            st.warning("No posts found or error fetching Tumblr posts.")

# Perform sentiment prediction:
# For Manual Text, only analyze if button clicked
# For other platforms, analyze immediately after fetching data

if (platform == "Manual Text" and analyze_manual and user_input) or (platform != "Manual Text" and user_input):
    cleaned_text = clean_text(user_input)
    vect = vectorizer.transform([cleaned_text])
    pred = model.predict(vect)[0]
    st.subheader("Sentiment Prediction:")
    st.write(sentiment_map.get(pred, "Unknown"))
