from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "reddit"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

DEFAULT_CSV_OUT = PROCESSED_DIR / "reddit_comments.csv"

REDDIT_BASE = "https://www.reddit.com"
USER_AGENT = "z4-reddit-json-scraper/0.1"

CSV_COLUMNS = ["subreddit", "score", "upvote_ratio", "type", "comment"]
SKIPPED_COMMENT_BODIES = {"", "[deleted]", "[removed]"}


def normalize_reddit_input(reddit_input: str, sort: str = "hot") -> tuple[str, str]:
    """Return (kind, value), where kind is either 'subreddit' or 'post'."""
    raw = reddit_input.strip()
    if not raw:
        raise ValueError("Reddit input cannot be empty.")

    if raw.startswith("http://") or raw.startswith("https://"):
        parsed = urlparse(raw)
        path = parsed.path.strip("/")
    else:
        path = raw.strip("/")

    if path.endswith(".json"):
        path = path[:-5]

    parts = [p for p in path.split("/") if p]
    if parts and parts[0] == "r":
        parts = parts[1:]

    if len(parts) >= 3 and parts[1] == "comments":
        subreddit = parts[0]
        post_id = parts[2]
        return "post", f"{REDDIT_BASE}/r/{subreddit}/comments/{post_id}.json"

    if len(parts) == 1:
        subreddit = parts[0]
        return "subreddit", f"{REDDIT_BASE}/r/{subreddit}/{sort}.json"

    raise ValueError(
        "Expected a subreddit like 'webscraping' or a post path like "
        "'webscraping/comments/1ga5pg6'."
    )


def fetch_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
) -> Any:
    response = session.get(
        url,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )

    if response.status_code == 429:
        raise RuntimeError("Reddit rate-limited the request. Try again later.")

    response.raise_for_status()
    return response.json()


def iter_comment_nodes(comment_listing: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten visible Reddit comments. Reddit 'more' placeholders are skipped."""
    comments: list[dict[str, Any]] = []
    children = comment_listing.get("data", {}).get("children", [])

    for child in children:
        kind = child.get("kind")
        if kind == "more":
            continue

        if kind != "t1":
            continue

        data = child.get("data", {})
        comments.append(data)

        replies = data.get("replies")
        if isinstance(replies, dict):
            comments.extend(iter_comment_nodes(replies))

    return comments


def rows_from_post_json(
    post_json: list[Any],
    keep_deleted: bool = False,
) -> list[dict[str, Any]]:
    """Convert one Reddit post .json response into comment rows."""
    if not isinstance(post_json, list) or len(post_json) < 2:
        raise ValueError("Unexpected Reddit post JSON shape.")

    post_listing = post_json[0]
    comment_listing = post_json[1]

    post_children = post_listing.get("data", {}).get("children", [])
    if not post_children:
        return []

    post_data = post_children[0].get("data", {})
    subreddit = post_data.get("subreddit", "")
    upvote_ratio = post_data.get("upvote_ratio", None)

    rows: list[dict[str, Any]] = []

    for comment in iter_comment_nodes(comment_listing):
        body = (comment.get("body") or "").strip()

        if not keep_deleted and body in SKIPPED_COMMENT_BODIES:
            continue

        rows.append(
            {
                "subreddit": subreddit,
                "score": comment.get("score", None),
                "upvote_ratio": upvote_ratio,
                "type": "comment",
                "comment": body,
                # Kept for compatibility with app.llm.run_llm.
                "date": int(comment.get("created_utc", 0) or 0),
            }
        )

    return rows


def post_urls_from_subreddit_json(subreddit_json: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    children = subreddit_json.get("data", {}).get("children", [])

    for child in children:
        if child.get("kind") != "t3":
            continue

        permalink = child.get("data", {}).get("permalink")
        if permalink:
            urls.append(f"{REDDIT_BASE}{permalink.rstrip('/')}.json")

    return urls


def scrape_post(
    session: requests.Session,
    post_url: str,
    comment_limit: int,
    keep_deleted: bool,
) -> list[dict[str, Any]]:
    post_json = fetch_json(
        session,
        post_url,
        params={"limit": comment_limit, "raw_json": 1},
    )

    return rows_from_post_json(post_json, keep_deleted=keep_deleted)


def scrape_subreddit(
    session: requests.Session,
    subreddit_url: str,
    post_limit: int,
    comment_limit: int,
    sleep_seconds: float,
    keep_deleted: bool,
) -> list[dict[str, Any]]:
    subreddit_json = fetch_json(
        session,
        subreddit_url,
        params={"limit": post_limit, "raw_json": 1},
    )

    post_urls = post_urls_from_subreddit_json(subreddit_json)
    rows: list[dict[str, Any]] = []

    for i, post_url in enumerate(post_urls, start=1):
        print(f"Fetching post {i}/{len(post_urls)}: {post_url}")

        try:
            rows.extend(scrape_post(session, post_url, comment_limit, keep_deleted))
        except Exception as e:
            print(f"Warning: skipped {post_url}: {e}")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return rows


def write_outputs(
    rows: list[dict[str, Any]],
    csv_output: Path,
    json_output: Path | None,
) -> None:
    csv_output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(csv_output, index=False)
    print(f"Saved {len(rows)} comment row(s) to {csv_output}")

    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)

        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

        print(f"Saved raw JSON rows to {json_output}")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert Reddit .json comments into CSV rows.")

    p.add_argument(
        "reddit_input",
        help="Subreddit or post path/link, e.g. 'webscraping' or 'webscraping/comments/1ga5pg6'.",
    )

    p.add_argument("--output", type=Path, default=DEFAULT_CSV_OUT)
    p.add_argument("--json-output", type=Path, default=None)
    p.add_argument("--sort", choices=["hot", "new", "top", "rising"], default="hot")
    p.add_argument("--post-limit", type=int, default=25)
    p.add_argument("--comment-limit", type=int, default=500)
    p.add_argument("--sleep", type=float, default=1.0)
    p.add_argument("--keep-deleted", action="store_true")

    args = p.parse_args()
    kind, url = normalize_reddit_input(args.reddit_input, sort=args.sort)

    with requests.Session() as session:
        if kind == "post":
            rows = scrape_post(session, url, args.comment_limit, args.keep_deleted)
        else:
            rows = scrape_subreddit(
                session,
                url,
                args.post_limit,
                args.comment_limit,
                args.sleep,
                args.keep_deleted,
            )

    write_outputs(rows, args.output, args.json_output)


if __name__ == "__main__":
    main()