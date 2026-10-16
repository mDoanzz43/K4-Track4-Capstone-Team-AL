"""Reproducible evaluation for the vision -> trajectory -> robot pipeline.

Run from the repository root:
    python eval/run_eval.py

Generated files are intentionally committed as demo evidence.  The dataset is
synthetic and contains no personal, patient, or student data.
"""
from __future__ import annotations

import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.robot_sim import DrawingSim
from backend.trajectory import TrajectoryConfig, build_trajectory
from backend.vision import NoCharacterFound, extract_strokes, render_strokes

OUT = ROOT / "eval" / "results"
PASS_DIR = OUT / "vision_pass"
FAIL_DIR = OUT / "failed_cases"


def text_image(ch: str, thickness: int = 26, size: int = 400) -> np.ndarray:
    image = np.full((size, size, 3), 255, np.uint8)
    cv2.putText(
        image, ch, (int(size * 0.22), int(size * 0.8)),
        cv2.FONT_HERSHEY_SIMPLEX, size / 44, (0, 0, 0), thickness, cv2.LINE_AA,
    )
    return image


def coverage(result) -> float:
    """Fraction of ink pixels reached by the extracted centre-line strokes."""
    binary = result.binary
    half_width = cv2.distanceTransform(binary, cv2.DIST_L2, 5).max()
    canvas = np.full(binary.shape, 255, np.uint8)
    for stroke in result.strokes_px:
        cv2.polylines(canvas, [np.round(stroke).astype(np.int32)], False, 0, 1)
    distance = cv2.distanceTransform(canvas, cv2.DIST_L2, 5)
    return float((distance[binary > 0] <= half_width + 3).mean())


def save_triptych(path: Path, source: np.ndarray, result) -> None:
    size = 400
    source = cv2.resize(source, (size, size))
    binary = cv2.cvtColor(cv2.resize(result.binary, (size, size)), cv2.COLOR_GRAY2BGR)
    traced = cv2.resize(render_strokes(result.strokes, size=size), (size, size))
    panel = np.hstack([source, binary, traced])
    for x, label in zip((8, size + 8, 2 * size + 8), ("INPUT", "BINARY", "EXTRACTED")):
        cv2.putText(panel, label, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 220), 2)
    cv2.imwrite(str(path), panel)


def evaluate_vision() -> list[dict]:
    rows = []
    for char in ("I", "L", "O", "T", "A", "X", "8", "B", "K"):
        source = text_image(char)
        started = time.perf_counter()
        result = extract_strokes(source, "canvas")
        elapsed_ms = (time.perf_counter() - started) * 1000
        cov = coverage(result)
        row = {
            "case_id": f"canvas_{char}",
            "status": "pass" if cov >= 0.98 else "fail",
            "coverage": round(cov, 5),
            "latency_ms": round(elapsed_ms, 2),
            "num_strokes": len(result.strokes),
            "num_points": sum(len(s) for s in result.strokes),
        }
        rows.append(row)
        save_triptych(PASS_DIR / f"{row['case_id']}.png", source, result)
    return rows


def evaluate_known_failures() -> list[dict]:
    rows = []

    # A real limitation: the character is present, but its contrast is below
    # the pipeline's deliberate noise rejection threshold.
    faded = np.full((400, 400, 3), 235, np.uint8)
    cv2.putText(faded, "A", (80, 320), cv2.FONT_HERSHEY_SIMPLEX, 8, (220, 220, 220), 18, cv2.LINE_AA)
    try:
        extract_strokes(faded, "photo")
        actual = "unexpectedly accepted"
        reproduced = False
    except NoCharacterFound as exc:
        actual = str(exc)
        reproduced = True
    evidence = faded.copy()
    cv2.putText(evidence, "KNOWN FAILURE: LOW CONTRAST", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 220), 2)
    cv2.imwrite(str(FAIL_DIR / "F01_low_contrast_input.png"), evidence)
    rows.append({
        "case_id": "F01_low_contrast_photo",
        "reproduced": reproduced,
        "expected": "extract the visible letter A",
        "actual": actual,
        "cause": "ink is less than about 12% darker than the local paper and is rejected as noise",
        "mitigation": "increase lighting/contrast, use darker ink, or draw with canvas mode",
        "evidence": "results/failed_cases/F01_low_contrast_input.png",
    })

    # An expected rejection, kept separately so the team does not present it as
    # a model failure.
    blank = np.full((400, 400, 3), 255, np.uint8)
    try:
        extract_strokes(blank, "canvas")
        rejected = False
        actual = "accepted"
    except NoCharacterFound as exc:
        rejected = True
        actual = str(exc)
    cv2.imwrite(str(FAIL_DIR / "V01_blank_expected_rejection.png"), blank)
    rows.append({
        "case_id": "V01_blank_validation",
        "reproduced": rejected,
        "expected": "reject blank input",
        "actual": actual,
        "cause": "input validation; this is not a system failure",
        "mitigation": "ask the user to provide at least one visible stroke",
        "evidence": "results/failed_cases/V01_blank_expected_rejection.png",
    })
    return rows


def evaluate_robot() -> list[dict]:
    cases = {
        "line": [[(0.15, 0.5), (0.85, 0.5)]],
        "corner": [[(0.2, 0.8), (0.2, 0.2), (0.8, 0.2)]],
        "two_strokes": [[(0.2, 0.2), (0.8, 0.8)], [(0.8, 0.2), (0.2, 0.8)]],
    }
    sim = DrawingSim(TrajectoryConfig())
    rows = []
    try:
        for case_id, strokes in cases.items():
            sim.clear_paper()
            trajectory = build_trajectory(strokes, sim.cfg)
            stats = sim.run(trajectory, lambda *_: None, speed=8, realtime=False)
            frame = sim.render()
            cv2.imwrite(str(OUT / f"robot_{case_id}.jpg"), frame)
            rows.append({"case_id": case_id, **stats, "planned_duration_s": round(trajectory.duration, 3)})
    finally:
        sim.close()
    return rows


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    PASS_DIR.mkdir(parents=True, exist_ok=True)
    FAIL_DIR.mkdir(parents=True, exist_ok=True)
    vision = evaluate_vision()
    failures = evaluate_known_failures()
    robot = evaluate_robot()
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "dataset": {"type": "synthetic", "contains_personal_data": False},
        "vision_cases": vision,
        "robot_cases": robot,
        "known_failures": failures,
        "summary": {
            "vision_passed": sum(r["status"] == "pass" for r in vision),
            "vision_total": len(vision),
            "vision_mean_coverage": round(statistics.mean(r["coverage"] for r in vision), 5),
            "vision_min_coverage": min(r["coverage"] for r in vision),
            "vision_mean_latency_ms": round(statistics.mean(r["latency_ms"] for r in vision), 2),
            "robot_mean_rms_mm": round(statistics.mean(r["tracking_rms_mm"] for r in robot), 3),
            "robot_worst_max_mm": max(r["tracking_max_mm"] for r in robot),
            "known_failures_reproduced": sum(r["reproduced"] for r in failures if r["case_id"].startswith("F")),
            "validation_cases_passed": sum(r["reproduced"] for r in failures if r["case_id"].startswith("V")),
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    s = payload["summary"]
    report = f"""# Evaluation report

Generated: `{payload['generated_at_utc']}`  
Commit: `{payload['commit']}`  
Dataset: synthetic, no personal/patient/student data.

| Metric | Result |
|---|---:|
| Vision cases passed | {s['vision_passed']}/{s['vision_total']} |
| Mean / minimum ink coverage | {s['vision_mean_coverage']:.3%} / {s['vision_min_coverage']:.3%} |
| Mean vision latency | {s['vision_mean_latency_ms']:.2f} ms/image |
| Mean robot tracking RMS | {s['robot_mean_rms_mm']:.3f} mm |
| Worst robot maximum error | {s['robot_worst_max_mm']:.3f} mm |
| Known failures reproduced | {s['known_failures_reproduced']}/1 |
| Validation cases passed | {s['validation_cases_passed']}/1 |

These results describe this synthetic dataset and PyBullet simulation only. They
are not claims about OCR accuracy, real robot accuracy, or webcam gesture accuracy.
See `metrics.json` for per-case measurements and `failed_cases.md` for failure analysis.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(s, indent=2))
    return 0 if s["vision_passed"] == s["vision_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
