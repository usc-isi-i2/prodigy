"""End-to-end checks for the Facebook page-reference table builder."""

from __future__ import annotations

import pickle
from pathlib import Path
import subprocess
import sys
import tempfile

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "build_facebook_page_reference_tables.py"
PREPARE_EMBEDDING_INPUT = (
    Path(__file__).resolve().parents[1] / "prepare_facebook_page_embedding_input.py"
)


def page_row(page_id: str, post_id: str, target_url: str | None = None) -> dict:
    links = [] if target_url is None else [{"expanded": target_url}]
    return {
        "platform": "Facebook",
        "platformId": post_id,
        "id": post_id,
        "date": "2022-02-24T12:00:00Z",
        "updated": "2022-02-24T13:00:00Z",
        "postUrl": f"https://www.facebook.com/{page_id}/posts/{post_id}",
        "expandedLinks": links,
        "message": "must not be retained",
        "history": [{"actual": 99}],
        "account": {
            "accountType": "facebook_page",
            "platformId": page_id,
            "id": page_id,
            "name": f"Page {page_id}",
            "url": f"https://www.facebook.com/{page_id}",
            "pageDescription": f"Description {page_id}",
            "pageCategory": "NEWS_SITE",
            "pageAdminTopCountry": "US",
            "pageCreatedDate": "2012-03-04T00:00:00Z",
            "verified": page_id == "a",
            "subscriberCount": 100,
        },
    }


def test_recursive_build_and_growth() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw = root / "raw"
        day1 = raw / "2022-02-24"
        day2 = raw / "2022-02-25"
        day1.mkdir(parents=True)
        day2.mkdir(parents=True)
        with (day1 / "part1.pkl").open("wb") as handle:
            pickle.dump([
                page_row("a", "a1", "https://www.facebook.com/b/posts/b1"),
                page_row("b", "b1"),
            ], handle)
        c_row = page_row("c", "c1", "https://www.facebook.com/a/posts/a1")
        c_row["date"] = c_row["updated"] = "2022-02-25T12:00:00Z"
        with (day2 / "part1.pkl").open("wb") as handle:
            pickle.dump([c_row], handle)

        output = root / "derived"
        subprocess.run(
            [sys.executable, str(SCRIPT), "--input-root", str(raw), "--output-root", str(output)],
            check=True,
        )

        inputs = pd.read_parquet(output / "input_files.parquet")
        assert inputs["relative_path"].tolist() == [
            "2022-02-24/part1.pkl", "2022-02-25/part1.pkl"
        ]
        events = pd.read_parquet(output / "page_reference_events.parquet")
        assert len(events) == 2
        assert set(events["source_file"]) == {
            "2022-02-24/part1.pkl", "2022-02-25/part1.pkl"
        }
        forbidden = {"message", "caption", "imageText", "media", "history", "statistics"}
        assert forbidden.isdisjoint(events.columns)

        profiles = pd.read_parquet(output / "page_profiles.parquet")
        assert len(profiles) == 3
        assert set(profiles["page_created_date"]) == {"2012-03-04T00:00:00Z"}
        growth = pd.read_parquet(output / "growth_by_cutoff.parquet")
        assert growth["primary_nodes"].tolist() == [2, 3]
        assert growth["primary_directed_edges"].tolist() == [1, 2]

        embedding_input = root / "embedding_input"
        subprocess.run(
            [
                sys.executable,
                str(PREPARE_EMBEDDING_INPUT),
                "--tables-root",
                str(output),
                "--output-root",
                str(embedding_input),
            ],
            check=True,
        )
        selected_profiles = pd.read_parquet(embedding_input / "page_profiles.parquet")
        assert selected_profiles["account_id"].tolist() == ["a", "b", "c"]
        assert "message" not in selected_profiles.columns


if __name__ == "__main__":
    test_recursive_build_and_growth()
    print("ok: recursive build, privacy fields, targets, and growth")
