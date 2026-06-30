#!/usr/bin/env python3
"""Evaluate start/end revisitation consistency across AR output directories."""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def parse_run(spec):
    if "=" not in spec:
        raise argparse.ArgumentTypeError("--run must be label=directory")
    label, root = spec.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("run label cannot be empty")
    return label, Path(root)


def list_videos(root):
    return {p.name: p for p in sorted(root.glob("*.mp4"))}


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Empty video: {path}")
    return frames


def frame_at(frames, index):
    if index < 0:
        index = len(frames) + index
    index = min(max(index, 0), len(frames) - 1)
    return frames[index], index


def luma(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)


def lab_mean(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab.reshape(-1, 3).mean(axis=0)


def laplacian_var(frame):
    return float(cv2.Laplacian(luma(frame).astype(np.uint8), cv2.CV_64F).var())


def edge_map(frame):
    gray = luma(frame).astype(np.uint8)
    return cv2.Canny(gray, 80, 160).astype(np.float32) / 255.0


def corrcoef(a, b):
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return float("nan")
    return float(np.dot(a, b) / denom)


def temporal_luma_delta(frames):
    vals = []
    prev = None
    for frame in frames:
        cur = luma(frame)
        if prev is not None:
            vals.append(float(np.mean(np.abs(cur - prev))))
        prev = cur
    return float(np.mean(vals)) if vals else float("nan")


def add_label(frame, text):
    bar_h = max(32, frame.shape[0] // 28)
    out = np.zeros((frame.shape[0] + bar_h, frame.shape[1], 3), dtype=np.uint8)
    out[bar_h:] = frame
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (18, 18, 18), -1)
    cv2.putText(
        out,
        text,
        (12, int(bar_h * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.45, bar_h / 52.0),
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return out


def resize(frame, scale):
    if scale == 1.0:
        return frame
    return cv2.resize(
        frame,
        (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )


def evaluate_video(path, anchor_index, revisit_index):
    frames = read_video(path)
    anchor, anchor_i = frame_at(frames, anchor_index)
    revisit, revisit_i = frame_at(frames, revisit_index)
    if anchor.shape[:2] != revisit.shape[:2]:
        revisit = cv2.resize(revisit, (anchor.shape[1], anchor.shape[0]), interpolation=cv2.INTER_AREA)

    anchor_luma = luma(anchor)
    revisit_luma = luma(revisit)
    anchor_edges = edge_map(anchor)
    revisit_edges = edge_map(revisit)
    anchor_sharpness = laplacian_var(anchor)
    revisit_sharpness = laplacian_var(revisit)

    return {
        "frames": len(frames),
        "anchor_frame": anchor_i,
        "revisit_frame": revisit_i,
        "revisit_luma_mae": float(np.mean(np.abs(anchor_luma - revisit_luma))),
        "revisit_lab_delta": float(np.linalg.norm(lab_mean(anchor) - lab_mean(revisit))),
        "revisit_edge_mae": float(np.mean(np.abs(anchor_edges - revisit_edges))),
        "revisit_edge_corr": corrcoef(anchor_edges, revisit_edges),
        "temporal_luma_delta_mean": temporal_luma_delta(frames),
        "anchor_sharpness": anchor_sharpness,
        "revisit_sharpness": revisit_sharpness,
        "revisit_sharpness_ratio": revisit_sharpness / anchor_sharpness if anchor_sharpness > 1e-8 else float("nan"),
        "_anchor_image": anchor,
        "_revisit_image": revisit,
    }


def mean_metric(rows, label, metric):
    vals = [float(r[metric]) for r in rows if r["run"] == label and not math.isnan(float(r[metric]))]
    return float(np.mean(vals)) if vals else float("nan")


def closure(rows, small_label, candidate_label, large_label, metric):
    small = mean_metric(rows, small_label, metric)
    candidate = mean_metric(rows, candidate_label, metric)
    large = mean_metric(rows, large_label, metric)
    denom = small - large
    if math.isnan(denom) or abs(denom) < 1e-8:
        return float("nan")
    return (small - candidate) / denom


def write_strip(video_name, per_run_results, output_dir, scale):
    cells = []
    for label, result in per_run_results:
        anchor = add_label(resize(result["_anchor_image"], scale), f"{label} anchor")
        revisit = add_label(resize(result["_revisit_image"], scale), f"{label} revisit")
        cells.append(np.concatenate([anchor, revisit], axis=0))
    strip = np.concatenate(cells, axis=1)
    out = output_dir / "revisit_strips" / f"{Path(video_name).stem}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), strip)


def write_summary(rows, labels, output_dir, small_label, candidate_label, large_label):
    lines = [
        "# Revisitation Memory Summary",
        "",
        "Lower revisit luma/LAB/edge MAE means the final revisited view is closer to the initial view.",
        "Higher edge correlation and healthy sharpness/motion are used as anti-collapse probes.",
        "",
        "## Aggregate Metrics",
        "",
        "| Run | revisit luma MAE | revisit LAB delta | edge MAE | edge corr | temporal delta | sharpness ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in labels:
        lines.append(
            f"| {label} | "
            f"{mean_metric(rows, label, 'revisit_luma_mae'):.4f} | "
            f"{mean_metric(rows, label, 'revisit_lab_delta'):.4f} | "
            f"{mean_metric(rows, label, 'revisit_edge_mae'):.4f} | "
            f"{mean_metric(rows, label, 'revisit_edge_corr'):.4f} | "
            f"{mean_metric(rows, label, 'temporal_luma_delta_mean'):.4f} | "
            f"{mean_metric(rows, label, 'revisit_sharpness_ratio'):.4f} |"
        )

    if small_label and candidate_label and large_label:
        lines.extend([
            "",
            "## Gap Closure",
            "",
            f"Small FIFO: `{small_label}`",
            f"Candidate: `{candidate_label}`",
            f"Large FIFO: `{large_label}`",
            "",
            "| Metric | Gap closure |",
            "| --- | ---: |",
        ])
        for metric in ("revisit_luma_mae", "revisit_lab_delta", "revisit_edge_mae"):
            lines.append(f"| {metric} | {closure(rows, small_label, candidate_label, large_label, metric):.4f} |")

    lines.extend([
        "",
        "## Artifacts",
        "",
        "- `metrics.csv`: per-video, per-run revisitation metrics",
        "- `revisit_strips/*.jpg`: anchor/revisit frame strips for visual inspection",
        "",
    ])
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True,
                        help="Run specification label=directory. May be repeated.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--anchor_frame", type=int, default=0)
    parser.add_argument("--revisit_frame", type=int, default=-1)
    parser.add_argument("--strip_scale", type=float, default=0.35)
    parser.add_argument("--small_fifo_label")
    parser.add_argument("--candidate_label")
    parser.add_argument("--large_fifo_label")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = args.run
    labels = [label for label, _ in runs]
    video_sets = [set(list_videos(root)) for _, root in runs]
    matched = sorted(set.intersection(*video_sets)) if video_sets else []
    rows = []

    for video_name in matched:
        per_run_results = []
        for label, root in runs:
            result = evaluate_video(root / video_name, args.anchor_frame, args.revisit_frame)
            row = {k: v for k, v in result.items() if not k.startswith("_")}
            row["run"] = label
            row["video"] = video_name
            rows.append(row)
            per_run_results.append((label, result))
        write_strip(video_name, per_run_results, output_dir, args.strip_scale)

    metrics_path = output_dir / "metrics.csv"
    if rows:
        fieldnames = ["run", "video"] + [k for k in rows[0] if k not in ("run", "video")]
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        metrics_path.write_text("", encoding="utf-8")

    write_summary(
        rows,
        labels,
        output_dir,
        args.small_fifo_label,
        args.candidate_label,
        args.large_fifo_label,
    )
    print(f"Matched {len(matched)} videos")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
