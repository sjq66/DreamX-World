#!/usr/bin/env python3
"""Create AR revisitation prompts for long-memory KV cache evaluation."""

import argparse
import json
from pathlib import Path


PATTERNS = {
    # Pure view loop: long left yaw, return, long up pitch, return.
    # The final view should revisit the initial camera orientation.
    "yaw_pitch_loop": (["j", "l", "i", "k"], [1, 1, 1, 1]),
    # Forward motion with reciprocal yaw segments. This is harder because
    # translation does not exactly close, but it creates stronger parallax.
    "parallax_loop": (["wj", "sl", "wl", "sj"], [1, 1, 1, 1]),
    # Two long sideways sweeps, then reverse them.
    "lateral_loop": (["aj", "dl", "al", "dj"], [1, 1, 1, 1]),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="configs/dreamx/eval.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--pattern", choices=sorted(PATTERNS), default="yaw_pitch_loop")
    parser.add_argument("--task_prefix", default="revisit")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.input, "r", encoding="utf-8") as f:
        items = json.load(f)

    action_seq, action_speed_list = PATTERNS[args.pattern]
    selected = items[: args.limit]
    output_items = []
    for idx, item in enumerate(selected):
        out = dict(item)
        out["task_id"] = f"{args.task_prefix}_{args.pattern}_{idx:03d}"
        out["action_seq"] = list(action_seq)
        out["action_speed_list"] = list(action_speed_list)
        out["revisitation"] = {
            "pattern": args.pattern,
            "anchor_frame": 0,
            "revisit_frame": -1,
            "description": "The last frame should revisit the first camera view.",
        }
        output_items.append(out)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_items, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(output_items)} revisitation items to {output_path}")
    print(f"Pattern: {args.pattern} action_seq={action_seq} weights={action_speed_list}")


if __name__ == "__main__":
    main()
