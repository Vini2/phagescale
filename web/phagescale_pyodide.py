from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from pyodide.ffi import to_js


def _runs_for_mask(mask: np.ndarray, y_offset: int) -> list[dict]:
    candidates: list[dict] = []
    height, width = mask.shape
    min_len = max(16, int(width * 0.035))

    for y in range(height):
        row = mask[y]
        padded = np.concatenate(([False], row, [False]))
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        for start, end in zip(changes[::2], changes[1::2]):
            length = end - start
            if length >= min_len:
                candidates.append(
                    {
                        "x1": int(start),
                        "x2": int(end - 1),
                        "y1": int(y + y_offset),
                        "y2": int(y + y_offset),
                        "runs": 1,
                    }
                )
    return candidates


def _merge_horizontal_runs(runs: list[dict]) -> list[dict]:
    runs.sort(key=lambda item: (item["y1"], item["x1"]))
    merged: list[dict] = []
    active: list[dict] = []

    for run in runs:
        next_active: list[dict] = []
        target = None
        for component in active:
            vertical_touch = run["y1"] <= component["y2"] + 2
            overlap = min(run["x2"], component["x2"]) - max(run["x1"], component["x1"])
            if vertical_touch and overlap >= -4:
                target = component
                break
            next_active.append(component)

        if target is None:
            target = run.copy()
            active.append(target)
        else:
            target["x1"] = min(target["x1"], run["x1"])
            target["x2"] = max(target["x2"], run["x2"])
            target["y1"] = min(target["y1"], run["y1"])
            target["y2"] = max(target["y2"], run["y2"])
            target["runs"] += 1

        for component in active:
            if component["y2"] < run["y1"] - 3 and component not in merged:
                merged.append(component)
            elif component not in next_active:
                next_active.append(component)
        active = next_active

    for component in active:
        if component not in merged:
            merged.append(component)
    return merged


def _score_candidate(candidate: dict, image_width: int, image_height: int) -> float:
    width = candidate["x2"] - candidate["x1"] + 1
    height = candidate["y2"] - candidate["y1"] + 1
    if width <= 0 or height <= 0:
        return -1.0

    aspect = width / max(1, height)
    if aspect < 5:
        return -1.0
    if width > image_width * 0.65:
        return -1.0
    if height > image_height * 0.08:
        return -1.0

    lower_bonus = candidate["y2"] / max(1, image_height)
    compactness = min(aspect / 18.0, 1.0)
    return width * (1.0 + compactness + lower_bonus * 0.35)


def detect_scale_bar(rgba_data, width: int, height: int):
    """Detect a horizontal TEM scale bar in browser canvas RGBA pixels."""
    pixels = np.asarray(rgba_data.to_py(), dtype=np.uint8).reshape((height, width, 4))
    rgb = pixels[:, :, :3].astype(np.float32)
    gray = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

    y0 = int(height * 0.55)
    crop = gray[y0:, :]
    dark_cutoff = float(np.percentile(crop, 12))
    bright_cutoff = float(np.percentile(crop, 88))

    candidates: list[dict] = []
    for polarity, mask in (
        ("dark", crop <= dark_cutoff),
        ("light", crop >= bright_cutoff),
    ):
        runs = _runs_for_mask(mask, y0)
        for component in _merge_horizontal_runs(runs):
            score = _score_candidate(component, width, height)
            if score > 0:
                component["polarity"] = polarity
                component["score"] = score
                candidates.append(component)

    if not candidates:
        return to_js({"found": False, "message": "No horizontal scale bar candidate found."})

    best = max(candidates, key=lambda item: item["score"])
    bar_width = best["x2"] - best["x1"] + 1
    bbox = {
        "x": best["x1"],
        "y": best["y1"],
        "width": bar_width,
        "height": best["y2"] - best["y1"] + 1,
    }
    return to_js(
        {
            "found": True,
            "lengthPx": float(bar_width),
            "bbox": bbox,
            "polarity": best["polarity"],
            "message": f"Detected {best['polarity']} scale bar.",
        }
    )


def _point_xy(point) -> tuple[float, float]:
    try:
        return float(point["x"]), float(point["y"])
    except Exception:
        return float(point.x), float(point.y)


def measure_path(points: Iterable, nm_per_px: float):
    py_points = points.to_py() if hasattr(points, "to_py") else points
    coords = [_point_xy(point) for point in py_points]
    px = 0.0
    for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
        px += math.hypot(x2 - x1, y2 - y1)
    return to_js({"lengthPx": px, "lengthNm": px * float(nm_per_px)})
