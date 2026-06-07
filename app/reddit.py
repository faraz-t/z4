from __future__ import annotations

import praw

def main() -> None:
    reddit = praw.Reddit(
        client_id="YOUR_CLIENT_ID",
        client_secret="YOUR_CLIENT_SECRET",
        user_agent="my_reddit_app",
    )

    subreddit = reddit.subreddit("python")

    for post in subreddit.hot(limit=10):
        print(post.title)
        print(post.score)

if __name__ == "__main__":
    main()