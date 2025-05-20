import requests
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

def fetch_youtube_comments(video_id="Jt7DqX2AlNo", max_results=20):
    key = os.getenv("YOUTUBE_API_KEY")
    url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&key={key}&maxResults={max_results}"
    response = requests.get(url).json()
    data = []
    for item in response.get("items", []):
        snippet = item["snippet"]["topLevelComment"]["snippet"]
        data.append([snippet["publishedAt"], snippet["authorDisplayName"], snippet["textDisplay"]])
    df = pd.DataFrame(data, columns=["Date", "User", "Text"])
    df.to_csv("youtube_data.csv", index=False)
    print("✅ Saved YouTube data to youtube_data.csv")

fetch_youtube_comments("Jt7DqX2AlNo")  # Replace with any video ID

