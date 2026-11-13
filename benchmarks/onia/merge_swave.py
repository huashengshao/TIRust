#!/usr/bin/env python3
"""Concatenate onia_swave numerator files of the same state, process and
diagrams over disjoint point ranges (onia_swave's FIRST_POINT argument).

usage: merge_swave.py OUTPUT_JSON INPUT_JSON [INPUT_JSON...]
"""
import json
import sys
from pathlib import Path


def main():
    output, inputs = Path(sys.argv[1]), [json.loads(Path(f).read_text()) for f in sys.argv[2:]]
    merged = inputs[0]
    for other in inputs[1:]:
        assert (other["state"], other.get("process")) == (merged["state"], merged.get("process"))
        assert [d["name"] for d in other["diagrams"]] == [d["name"] for d in merged["diagrams"]]
        for mine, theirs in zip(merged["diagrams"], other["diagrams"]):
            mine["points"].extend(theirs["points"])
            mine["rank"] = max(mine["rank"], theirs["rank"])
    indices = [p["index"] for p in merged["diagrams"][0]["points"]]
    assert len(indices) == len(set(indices)), "overlapping point ranges"
    output.write_text(json.dumps(merged) + "\n")
    print(output, len(indices), "points")


if __name__ == "__main__":
    main()
