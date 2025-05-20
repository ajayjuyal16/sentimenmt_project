import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def fetch_tumblr_posts(tag="AI", limit=20):
    key = os.getenv("TUMBLR_API_KEY")
    url = f"https://api.tumblr.com/v2/tagged?tag={tag}&api_key={key}"
    response = requests.get(url).json()
    data = []
    for post in response.get("response", [])[:limit]:
        text = post.get("summary") or post.get("caption", "")
        data.append([post.get("timestamp"), post.get("blog_name"), text])
    df = pd.DataFrame(data, columns=["Date", "User", "Text"])
    df.to_csv("tumblr_data.csv", index=False)
    print("✅ Saved Tumblr data to tumblr_data.csv")

fetch_tumblr_posts()

