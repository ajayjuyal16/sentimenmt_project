import praw
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

def fetch_reddit_posts(subreddit="technology", limit=50):
    posts = reddit.subreddit(subreddit).hot(limit=limit)
    data = []
    for post in posts:
        data.append([post.created_utc, post.title, post.selftext])
    df = pd.DataFrame(data, columns=["Date", "Title", "Text"])
    df.to_csv("reddit_data.csv", index=False)
    print("✅ Saved Reddit data to reddit_data.csv")

fetch_reddit_posts()

