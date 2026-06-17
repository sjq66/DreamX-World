#!/usr/bin/env python3
"""Build visual and metric comparisons for paired AR inference outputs."""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def list_videos(root):
    return {p.name: p for p in sorted(Path(root).glob("*.mp4"))}


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def frame_luma(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)


def lab_mean(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab.reshape(-1, 3).mean(axis=0)


def laplacian_var(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


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


def resize_for_report(frame, scale):
    if scale == 1.0:
        return frame
    width = max(1, int(frame.shape[1] * scale))
    height = max(1, int(frame.shape[0] * scale))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def safe_mean(values):
    return float(np.mean(values)) if values else float("nan")


def safe_std(values):
    return float(np.std(values)) if values else float("nan")


def write_side_by_side(
    baseline_path,
    candidate_path,
    output_path,
    baseline_label,
    candidate_label,
    scale,
):
    cap_a = open_video(baseline_path)
    cap_b = open_video(candidate_path)
    fps = cap_a.get(cv2.CAP_PROP_FPS) or 16

    ret_a, frame_a = cap_a.read()
    ret_b, frame_b = cap_b.read()
    if not ret_a or not ret_b:
        raise RuntimeError(f"Empty video pair: {baseline_path}, {candidate_path}")

    frame_a = resize_for_report(frame_a, scale)
    frame_b = resize_for_report(frame_b, scale)
    if frame_a.shape[:2] != frame_b.shape[:2]:
        frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]), interpolation=cv2.INTER_AREA)

    sample = np.concatenate([
        add_label(frame_a, baseline_label),
        add_label(frame_b, candidate_label),
    ], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (sample.shape[1], sample.shape[0]),
    )
    writer.write(sample)

    while True:
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()
        if not ret_a or not ret_b:
            break
        frame_a = resize_for_report(frame_a, scale)
        frame_b = resize_for_report(frame_b, scale)
        if frame_a.shape[:2] != frame_b.shape[:2]:
            frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]), interpolation=cv2.INTER_AREA)
        writer.write(np.concatenate([
            add_label(frame_a, baseline_label),
            add_label(frame_b, candidate_label),
        ], axis=1))

    writer.release()
    cap_a.release()
    cap_b.release()


def make_frame_strip(frame_ids, frames_a, frames_b, labels, output_path):
    cells = []
    for frame_id, a, b in zip(frame_ids, frames_a, frames_b):
        a = add_label(a, f"{labels[0]} f={frame_id}")
        b = add_label(b, f"{labels[1]} f={frame_id}")
        cells.append(np.concatenate([a, b], axis=0))
    strip = np.concatenate(cells, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), strip)


def compare_pair(
    baseline_path,
    candidate_path,
    output_dir,
    baseline_label,
    candidate_label,
    scale,
    strip_frames,
):
    cap_a = open_video(baseline_path)
    cap_b = open_video(candidate_path)

    fps = cap_a.get(cv2.CAP_PROP_FPS) or 16
    width = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
    total_b = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(total_a, total_b)

    prev_a = prev_b = None
    first_lab_a = first_lab_b = None
    last_lab_a = last_lab_b = None
    temporal_a = []
    temporal_b = []
    temporal_a_tail = []
    temporal_b_tail = []
    color_a = []
    color_b = []
    brightness_a = []
    brightness_b = []
    sharpness_a = []
    sharpness_b = []
    paired_diff = []
    sample_ids = set()
    if total > 0:
        sample_ids = set(np.linspace(0, total - 1, strip_frames, dtype=int).tolist())
    samples_a = []
    samples_b = []
    sample_frame_ids = []

    i = 0
    while True:
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()
        if not ret_a or not ret_b:
            break
        if frame_a.shape[:2] != frame_b.shape[:2]:
            frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]), interpolation=cv2.INTER_AREA)

        luma_a = frame_luma(frame_a)
        luma_b = frame_luma(frame_b)
        lab_a = lab_mean(frame_a)
        lab_b = lab_mean(frame_b)
        if first_lab_a is None:
            first_lab_a = lab_a
            first_lab_b = lab_b
        last_lab_a = lab_a
        last_lab_b = lab_b

        if prev_a is not None:
            da = float(np.mean(np.abs(luma_a - prev_a)))
            db = float(np.mean(np.abs(luma_b - prev_b)))
            temporal_a.append(da)
            temporal_b.append(db)
            if total and i >= int(total * 0.75):
                temporal_a_tail.append(da)
                temporal_b_tail.append(db)
        prev_a = luma_a
        prev_b = luma_b

        color_a.append(float(np.linalg.norm(lab_a - first_lab_a)))
        color_b.append(float(np.linalg.norm(lab_b - first_lab_b)))
        brightness_a.append(float(np.mean(luma_a)))
        brightness_b.append(float(np.mean(luma_b)))
        sharpness_a.append(laplacian_var(frame_a))
        sharpness_b.append(laplacian_var(frame_b))
        paired_diff.append(float(np.mean(np.abs(luma_a - luma_b))))

        if i in sample_ids:
            sample_frame_ids.append(i)
            samples_a.append(resize_for_report(frame_a, scale))
            samples_b.append(resize_for_report(frame_b, scale))
        i += 1

    cap_a.release()
    cap_b.release()

    stem = baseline_path.stem
    write_side_by_side(
        baseline_path,
        candidate_path,
        output_dir / "side_by_side" / f"{stem}.mp4",
        baseline_label,
        candidate_label,
        scale,
    )
    if samples_a and samples_b:
        make_frame_strip(
            sample_frame_ids,
            samples_a,
            samples_b,
            (baseline_label, candidate_label),
            output_dir / "frame_strips" / f"{stem}.jpg",
        )

    return {
        "video": baseline_path.name,
        "fps": fps,
        "width": width,
        "height": height,
        "baseline_frames": total_a,
        "candidate_frames": total_b,
        "paired_frames": i,
        "baseline_temporal_luma_delta_mean": safe_mean(temporal_a),
        "candidate_temporal_luma_delta_mean": safe_mean(temporal_b),
        "delta_temporal_luma_delta_mean": safe_mean(temporal_b) - safe_mean(temporal_a),
        "baseline_tail_temporal_luma_delta_mean": safe_mean(temporal_a_tail),
        "candidate_tail_temporal_luma_delta_mean": safe_mean(temporal_b_tail),
        "delta_tail_temporal_luma_delta_mean": safe_mean(temporal_b_tail) - safe_mean(temporal_a_tail),
        "baseline_color_drift_lab_mean": safe_mean(color_a),
        "candidate_color_drift_lab_mean": safe_mean(color_b),
        "delta_color_drift_lab_mean": safe_mean(color_b) - safe_mean(color_a),
        "baseline_end_color_drift_lab": float(np.linalg.norm(last_lab_a - first_lab_a)) if last_lab_a is not None else math.nan,
        "candidate_end_color_drift_lab": float(np.linalg.norm(last_lab_b - first_lab_b)) if last_lab_b is not None else math.nan,
        "baseline_brightness_std": safe_std(brightness_a),
        "candidate_brightness_std": safe_std(brightness_b),
        "baseline_sharpness_laplacian_mean": safe_mean(sharpness_a),
        "candidate_sharpness_laplacian_mean": safe_mean(sharpness_b),
        "paired_luma_absdiff_mean": safe_mean(paired_diff),
    }


def write_summary(rows, output_dir, baseline_label, candidate_label):
    output = output_dir / "summary.md"
    if not rows:
        output.write_text("# AR Comparison Summary\n\nNo matched videos found.\n", encoding="utf-8")
        return

    def avg(key):
        vals = [r[key] for r in rows if not math.isnan(float(r[key]))]
        return float(np.mean(vals)) if vals else math.nan

    lines = [
        "# AR Comparison Summary",
        "",
        f"Baseline: `{baseline_label}`",
        f"Candidate: `{candidate_label}`",
        "",
        "Lower temporal/color drift is usually better, but these are probe metrics, not ground-truth quality scores.",
        "Use the generated side-by-side videos and frame strips for the main visual judgment.",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Baseline | Candidate | Delta (candidate - baseline) |",
        "| --- | ---: | ---: | ---: |",
        f"| temporal luma delta mean | {avg('baseline_temporal_luma_delta_mean'):.4f} | {avg('candidate_temporal_luma_delta_mean'):.4f} | {avg('delta_temporal_luma_delta_mean'):.4f} |",
        f"| tail temporal luma delta mean | {avg('baseline_tail_temporal_luma_delta_mean'):.4f} | {avg('candidate_tail_temporal_luma_delta_mean'):.4f} | {avg('delta_tail_temporal_luma_delta_mean'):.4f} |",
        f"| color drift LAB mean | {avg('baseline_color_drift_lab_mean'):.4f} | {avg('candidate_color_drift_lab_mean'):.4f} | {avg('delta_color_drift_lab_mean'):.4f} |",
        f"| brightness std | {avg('baseline_brightness_std'):.4f} | {avg('candidate_brightness_std'):.4f} | {(avg('candidate_brightness_std') - avg('baseline_brightness_std')):.4f} |",
        f"| sharpness laplacian mean | {avg('baseline_sharpness_laplacian_mean'):.4f} | {avg('candidate_sharpness_laplacian_mean'):.4f} | {(avg('candidate_sharpness_laplacian_mean') - avg('baseline_sharpness_laplacian_mean')):.4f} |",
        "",
        "## Artifacts",
        "",
        "- `metrics.csv`: per-video metric table",
        "- `side_by_side/*.mp4`: baseline and candidate videos placed next to each other",
        "- `frame_strips/*.jpg`: sampled frame strips for fast inspection",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_label", default="baseline_fifo")
    parser.add_argument("--candidate_label", default="kv_similarity")
    parser.add_argument("--side_by_side_scale", type=float, default=0.5)
    parser.add_argument("--strip_frames", type=int, default=8)
    args = parser.parse_args()

    baseline = list_videos(args.baseline_dir)
    candidate = list_videos(args.candidate_dir)
    matched = sorted(set(baseline) & set(candidate))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in matched:
        print(f"Comparing {name}")
        rows.append(compare_pair(
            baseline[name],
            candidate[name],
            output_dir,
            args.baseline_label,
            args.candidate_label,
            args.side_by_side_scale,
            args.strip_frames,
        ))

    metrics_path = output_dir / "metrics.csv"
    if rows:
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        metrics_path.write_text("", encoding="utf-8")

    write_summary(rows, output_dir, args.baseline_label, args.candidate_label)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
