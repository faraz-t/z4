from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

APP_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_DIR = APP_DIR.parent.parent / "data" / "reddit" / "raw"
DEFAULT_OUTPUT = APP_DIR.parent.parent / "data" / "reddit" / "processed" / "reddit_comments.csv"

CSV_COLUMNS = [
    "id",
    "parent",
    "post_url",
    "subreddit",
    "score",
    "upvote_ratio",
    "type",
    "comment",
]

SKIPPED_COMMENT_BODIES = {"", "[deleted]", "[removed]"}


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def full_id(data: dict[str, Any], fallback_prefix: str) -> str | None:
    """
    Prefer Reddit's full object name, e.g.:
      - t3_1ug6xp5 for posts
      - t1_otxgf62 for comments

    Fall back to prefix + raw id when name is missing.
    """
    name = data.get("name")
    if name:
        return name

    raw_id = data.get("id")
    if raw_id:
        return f"{fallback_prefix}_{raw_id}"

    return None


def get_post_data(reddit_json: Any) -> dict[str, Any]:
    """
    Reddit post .json files usually look like:

        [
            post_listing,
            comment_listing
        ]

    The post data is stored at:

        reddit_json[0]["data"]["children"][0]["data"]
    """
    if not isinstance(reddit_json, list) or len(reddit_json) < 1:
        return {}

    post_listing = reddit_json[0]
    if not isinstance(post_listing, dict):
        return {}

    children = post_listing.get("data", {}).get("children", [])
    if not children:
        return {}

    first_child = children[0]
    if first_child.get("kind") != "t3":
        return {}

    return first_child.get("data", {})


def get_comment_listing(reddit_json: Any) -> dict[str, Any]:
    """
    For a Reddit post .json file, comments are usually in reddit_json[1].
    """
    if not isinstance(reddit_json, list) or len(reddit_json) < 2:
        return {}

    comment_listing = reddit_json[1]
    if not isinstance(comment_listing, dict):
        return {}

    return comment_listing


def post_url_from_post_data(post_data: dict[str, Any]) -> str | None:
    url = post_data.get("url")
    if url:
        return url

    permalink = post_data.get("permalink")
    if permalink:
        return f"https://www.reddit.com{permalink}"

    return None


def post_row_from_post_data(post_data: dict[str, Any]) -> dict[str, Any]:
    post_id = full_id(post_data, "t3")
    post_url = post_url_from_post_data(post_data)

    title = (post_data.get("title") or "").strip()
    selftext = (post_data.get("selftext") or "").strip()

    if title and selftext:
        body = f"{title}\n\n{selftext}"
    else:
        body = title or selftext

    return {
        "id": post_id,
        "parent": None,
        "post_url": post_url,
        "subreddit": post_data.get("subreddit", ""),
        "score": post_data.get("score", None),
        "upvote_ratio": post_data.get("upvote_ratio", None),
        "type": "post",
        "comment": body,
    }


def iter_comments(comment_listing: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Recursively flatten visible comments from a Reddit comment listing.

    Reddit sometimes includes "more" placeholders for comments that were not
    expanded in the saved .json file. Since this is an offline parser, those
    are skipped because the actual comment bodies are not present.
    """
    comments: list[dict[str, Any]] = []
    children = comment_listing.get("data", {}).get("children", [])

    for child in children:
        kind = child.get("kind")

        if kind == "more":
            continue

        if kind != "t1":
            continue

        comment_data = child.get("data", {})
        comments.append(comment_data)

        replies = comment_data.get("replies")
        if isinstance(replies, dict):
            comments.extend(iter_comments(replies))

    return comments


def comment_row_from_comment_data(
    comment_data: dict[str, Any],
    post_data: dict[str, Any],
) -> dict[str, Any]:
    post_url = post_url_from_post_data(post_data)

    return {
        "id": full_id(comment_data, "t1"),
        "parent": comment_data.get("parent_id", None),
        "post_url": post_url,
        "subreddit": comment_data.get("subreddit", post_data.get("subreddit", "")),
        "score": comment_data.get("score", None),
        "upvote_ratio": post_data.get("upvote_ratio", None),
        "type": "comment",
        "comment": (comment_data.get("body") or "").strip(),
    }


def rows_from_reddit_json(
    reddit_json: Any,
    source_file: Path,
    keep_deleted: bool,
    include_posts: bool,
) -> list[dict[str, Any]]:
    post_data = get_post_data(reddit_json)
    comment_listing = get_comment_listing(reddit_json)

    if not post_data:
        print(f"Warning: skipped {source_file}; could not find post data.")
        return []

    rows: list[dict[str, Any]] = []

    if include_posts:
        rows.append(post_row_from_post_data(post_data))

    if not comment_listing:
        return rows

    for comment_data in iter_comments(comment_listing):
        row = comment_row_from_comment_data(
            comment_data=comment_data,
            post_data=post_data,
        )

        if not keep_deleted and row["comment"] in SKIPPED_COMMENT_BODIES:
            continue

        rows.append(row)

    return rows


def extract_folder(
    input_dir: Path,
    keep_deleted: bool,
    include_posts: bool,
) -> list[dict[str, Any]]:
    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No .json files found in {input_dir}")

    rows: list[dict[str, Any]] = []

    for path in json_files:
        try:
            reddit_json = load_json(path)
            file_rows = rows_from_reddit_json(
                reddit_json=reddit_json,
                source_file=path,
                keep_deleted=keep_deleted,
                include_posts=include_posts,
            )
            rows.extend(file_rows)
            print(f"Extracted {len(file_rows)} row(s) from {path.name}")
        except Exception as e:
            print(f"Warning: failed to process {path}: {e}")

    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(rows)} total row(s) to {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract Reddit posts/comments from locally saved Reddit .json files."
    )

    p.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Folder containing Reddit .json files. Default: {DEFAULT_INPUT_DIR}",
    )

    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT}",
    )

    p.add_argument(
        "--keep-deleted",
        action="store_true",
        help="Keep [deleted] and [removed] comments instead of skipping them.",
    )

    p.add_argument(
        "--comments-only",
        action="store_true",
        help="Only output comments. By default, the original post is included too.",
    )

    args = p.parse_args()

    rows = extract_folder(
        input_dir=args.input_dir,
        keep_deleted=args.keep_deleted,
        include_posts=not args.comments_only,
    )

    write_csv(rows, args.output)


if __name__ == "__main__":
    main()
