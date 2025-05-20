import tweepy
import pandas as pd
import os
import time
from dotenv import load_dotenv
from tweepy.errors import Unauthorized, TooManyRequests, NotFound, TweepyException

load_dotenv()

# Initialize Tweepy Client for v2 API with rate-limit handling
try:
    client = tweepy.Client(
        bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
        consumer_key=os.getenv("TWITTER_API_KEY"),
        consumer_secret=os.getenv("TWITTER_API_SECRET"),
        access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
        access_token_secret=os.getenv("TWITTER_ACCESS_SECRET"),
        wait_on_rate_limit=True
    )
except Exception as e:
    print("❌ Error initializing Twitter client:", e)
    exit()

def fetch_own_tweets(username="elonmusk", max_results=30, retries=5):
    attempt = 0
    while attempt < retries:
        try:
            user_response = client.get_user(username=username)
            if user_response.data is None:
                print(f"❌ User '{username}' not found.")
                return None

            user_id = user_response.data.id

            tweets_response = client.get_users_tweets(
                id=user_id,
                max_results=max_results,
                tweet_fields=["created_at", "text"]
            )

            if tweets_response.data is None:
                print(f"⚠️ No tweets found for user '{username}'.")
                return None

            data = []
            for tweet in tweets_response.data:
                data.append([tweet.created_at, username, tweet.text])

            df = pd.DataFrame(data, columns=["Date", "User", "Text"])
            df.to_csv("twitter_data.csv", index=False)
            print("✅ Saved Twitter data to twitter_data.csv")
            return df

        except TooManyRequests:
            wait_time = 60 * (2 ** attempt)  # exponential backoff: 1min, 2min, 4min...
            print(f"⏳ Rate limit exceeded. Sleeping for {wait_time} seconds before retrying...")
            time.sleep(wait_time)
            attempt += 1

        except Unauthorized:
            print("❌ Unauthorized: Check your API credentials (.env file).")
            return None

        except NotFound:
            print(f"❌ User '{username}' not found (404).")
            return None

        except TweepyException as te:
            print("❌ Tweepy error:", te)
            return None

        except Exception as e:
            print("❌ General error:", e)
            return None

    print("❌ Max retries reached. Failed to fetch tweets due to rate limits.")
    return None

# Example usage:
if __name__ == "__main__":
    fetch_own_tweets("elonmusk", max_results=30)
