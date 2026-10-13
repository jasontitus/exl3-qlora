"""Report unique discovered commits; never equate records with qualified examples."""
import argparse
import json
from pathlib import Path


def status(store):
    rows = []
    for repository in sorted((store / "commits").glob("*")):
        commits = list(repository.glob("*/summary.json"))
        candidates = list((store / "candidates" / repository.name).glob("*"))
        rows.append({"repository_key": repository.name,
                     "unique_commits": len(commits),
                     "unique_candidate_commits": sum(p.is_dir() for p in candidates),
                     "details_fetched": len(list(repository.glob("*/detail.json")))})
    runs = sorted((store / "runs").glob("*/run.json"))
    return {"repositories": rows,
            "unique_commits": sum(r["unique_commits"] for r in rows),
            "unique_candidate_commits": sum(r["unique_candidate_commits"] for r in rows),
            "completed_collection_runs": len(runs),
            "latest_receipt": str(runs[-1].resolve()) if runs else None,
            "training_admission": "Discovery records are unreviewed and ineligible; no admission performed by this collector."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path(__file__).resolve().parents[1] / "store")
    args = parser.parse_args()
    print(json.dumps(status(args.store), indent=2))
