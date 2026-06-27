from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent.parent

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "reddit" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "reddit" / "processed" / "reddit_comments.csv"

CSV_COLUMNS = ["subreddit", "score", "upvote_ratio", "type", "comment"]
SKIPPED_COMMENT_BODIES = {"", "[deleted]", "[removed]"}


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def iter_comments(comment_listing: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Recursively flatten visible comments from a Reddit comment listing.

    Reddit sometimes includes "more" placeholders for comments that were not
    expanded in the .json file. Since this is an offline parser, those are
    skipped because the actual comment bodies are not present in the file.
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


def rows_from_reddit_json(
    reddit_json: Any,
    source_file: Path,
    keep_deleted: bool,
) -> list[dict[str, Any]]:
    post_data = get_post_data(reddit_json)
    comment_listing = get_comment_listing(reddit_json)

    if not post_data or not comment_listing:
        print(f"Warning: skipped {source_file}; not a Reddit post .json file.")
        return []

    subreddit = post_data.get("subreddit", "")
    upvote_ratio = post_data.get("upvote_ratio", None)

    rows: list[dict[str, Any]] = []

    for comment in iter_comments(comment_listing):
        body = (comment.get("body") or "").strip()

        if not keep_deleted and body in SKIPPED_COMMENT_BODIES:
            continue

        rows.append(
            {
                "subreddit": comment.get("subreddit", subreddit),
                "score": comment.get("score", None),
                "upvote_ratio": upvote_ratio,
                "type": "comment",
                "comment": body,
            }
        )

    return rows


def extract_folder(input_dir: Path, keep_deleted: bool) -> list[dict[str, Any]]:
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
            )
            rows.extend(file_rows)
            print(f"Extracted {len(file_rows)} comment(s) from {path.name}")
        except Exception as e:
            print(f"Warning: failed to process {path}: {e}")

    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(rows)} total comment row(s) to {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract Reddit comments from locally saved Reddit .json files."
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

    args = p.parse_args()

    rows = extract_folder(
        input_dir=args.input_dir,
        keep_deleted=args.keep_deleted,
    )

    write_csv(rows, args.output)


if __name__ == "__main__":
    main()
