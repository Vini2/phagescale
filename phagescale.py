"""
TEM phage tail length (nm) from a single image.

This tuned version uses:
- robust scale bar detection in the bottom region
- capsid (head) detection with Hough circles + contrast scoring
- direction-guided centerline tracing for the tail
"""

from __future__ import annotations

import heapq
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import click
import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize


CLICK_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}
VERSION = "0.1.0"


class OrderedGroup(click.Group):
    def list_commands(self, ctx: click.Context) -> list[str]:
        return list(self.commands)


@dataclass
class Config:
    # Scale bar detection.
    bottom_crop_frac: float = 0.24
    bar_min_len_frac: float = 0.04
    bar_max_len_frac: float = 0.45
    bar_max_height_frac: float = 0.18
    bar_hough_thresh: int = 30
    bar_hough_min_line_gap: int = 12
    bar_hough_min_line_len: int = 25
    blackhat_kernel_w_frac: float = 0.25
    blackhat_kernel_h: int = 9

    # Head circle detection.
    head_min_radius_frac: float = 0.025
    head_max_radius_frac: float = 0.11
    head_hough_dp: float = 1.2
    head_hough_param1: int = 120
    head_hough_param2: int = 18
    head_min_dist_frac: float = 0.16

    # Tail response and tracing.
    tail_line_kernel_scale: float = 2.2
    tail_theta_ray_start_scale: float = 1.2
    tail_theta_ray_end_scale: float = 6.0
    tail_theta_step_deg: int = 2
    tail_trace_step_px: float = 2.0
    tail_trace_local_angle_span_deg: int = 35
    tail_trace_global_angle_span_deg: int = 55
    tail_trace_max_dist_scale: float = 6.2
    tail_trace_border_margin_frac: float = 0.11
    tail_trace_fail_limit: int = 18
    tail_trace_max_steps: int = 500
    tail_trace_loop_back_steps: int = 16


@dataclass
class TailMeasurement:
    tail_nm: float
    tail_px: float
    bar_px: int
    px_per_nm: float
    head_yx: Tuple[int, int]
    head_r: float
    theta_deg: float
    tail_points: list[Tuple[float, float]]
    scale_bbox_xywh: tuple[int, int, int, int] | None = None
    scale_polarity: str | None = None


@dataclass(frozen=True)
class AnnotatedScaleBarDetection:
    length_px: float
    bbox_xywh: tuple[int, int, int, int]
    polarity: str


@dataclass(frozen=True)
class AnnotatedTailMeasurement:
    image_path: Path
    tail_px: float
    bar_px: float
    scale_nm: float
    tail_nm: float
    tail_path_yx: list[tuple[int, int]]
    scale_bbox_xywh: tuple[int, int, int, int]
    scale_polarity: str


def _odd(n: int, minimum: int = 3) -> int:
    n = max(minimum, int(n))
    return n if n % 2 == 1 else n + 1


def _normalize01(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32, copy=False)
    mn = float(np.min(a))
    mx = float(np.max(a))
    if mx <= mn:
        return np.zeros_like(a, dtype=np.float32)
    return (a - mn) / (mx - mn + 1e-6)


def _angle_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _find_scale_bar_px(gray: np.ndarray, cfg: Config, debug: bool = False) -> int:
    del cfg  # Scale bar detection now shares the annotated-image implementation.
    detection = _find_bottom_scale_bar(gray, debug=debug)
    return int(round(detection.length_px))


def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
    length = max(5, int(length))
    if length % 2 == 0:
        length += 1
    k = np.zeros((length, length), dtype=np.uint8)
    c = length // 2
    rad = np.deg2rad(angle_deg)
    dx = int(round((length // 2) * np.cos(rad)))
    dy = int(round((length // 2) * np.sin(rad)))
    cv2.line(k, (c - dx, c - dy), (c + dx, c + dy), 1, 1)
    return k


def _detect_head_circle(gray: np.ndarray, cfg: Config) -> Tuple[Tuple[int, int], float]:
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)

    yy, xx = np.indices(gray.shape)

    def score_candidate(x: int, y: int, r: int) -> Optional[Tuple[float, float]]:
        if x - r < 1 or y - r < 1 or x + r >= w - 1 or y + r >= h - 1:
            return None
        if y > int(0.78 * h):
            return None
        if x < int(0.08 * w) or x > int(0.92 * w):
            return None

        d = np.sqrt((yy - y) ** 2 + (xx - x) ** 2)
        inner = gray[d <= 0.72 * r]
        ring = gray[(d >= 0.92 * r) & (d <= 1.25 * r)]
        outer = gray[(d >= 1.45 * r) & (d <= 1.90 * r)]
        if inner.size < 80 or ring.size < 80 or outer.size < 100:
            return None

        c1 = float(np.mean(inner) - np.mean(ring))
        c2 = float(np.mean(outer) - np.mean(ring))
        score = 1.2 * c1 + 0.55 * c2 - 0.006 * abs(x - 0.5 * w) - 0.008 * abs(y - 0.35 * h)
        return score, c1

    min_r = max(10, int(min(h, w) * cfg.head_min_radius_frac))
    max_r = max(min_r + 3, int(min(h, w) * cfg.head_max_radius_frac))
    min_dist = max(24, int(min(h, w) * cfg.head_min_dist_frac))

    best = None  # (score, y, x, r, c1)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=float(cfg.head_hough_dp),
        minDist=min_dist,
        param1=float(cfg.head_hough_param1),
        param2=float(cfg.head_hough_param2),
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is not None:
        for x, y, r in np.round(circles[0, :]).astype(int):
            s = score_candidate(int(x), int(y), int(r))
            if s is None:
                continue
            score, c1 = s
            if best is None or score > best[0]:
                best = (score, int(y), int(x), int(r), c1)

    if best is not None:
        return (best[1], best[2]), float(best[3])

    # Fallback: relaxed Hough for challenging images.
    fb_min_r = max(12, int(min(h, w) * 0.03))
    fb_max_r = max(fb_min_r + 4, int(min(h, w) * 0.22))
    fb_min_dist = max(20, int(min(h, w) * 0.10))
    for p2 in (14, 12, 10, 8):
        circles_fb = cv2.HoughCircles(
            cv2.GaussianBlur(gray, (7, 7), 1.2),
            cv2.HOUGH_GRADIENT,
            dp=1.1,
            minDist=fb_min_dist,
            param1=70.0,
            param2=float(p2),
            minRadius=fb_min_r,
            maxRadius=fb_max_r,
        )
        if circles_fb is None:
            continue

        for x, y, r in np.round(circles_fb[0, :]).astype(int):
            s = score_candidate(int(x), int(y), int(r))
            if s is None:
                continue
            score, c1 = s
            if best is None or score > best[0]:
                best = (score, int(y), int(x), int(r), c1)
    if best is not None and best[0] > 2.0:
        return (best[1], best[2]), float(best[3])

    raise RuntimeError("Could not detect phage head circle.")


def _estimate_tail_direction_deg(dog_norm: np.ndarray, head_yx: Tuple[int, int], head_r: float, cfg: Config) -> float:
    h, w = dog_norm.shape
    hy, hx = head_yx
    ray_thr = float(np.quantile(dog_norm, 0.80))

    d0 = int(max(4, cfg.tail_theta_ray_start_scale * head_r))
    d1 = int(max(d0 + 10, cfg.tail_theta_ray_end_scale * head_r))

    best = None  # (score, theta)
    for theta in range(0, 360, max(1, int(cfg.tail_theta_step_deg))):
        rad = np.deg2rad(theta)
        vals = []
        for d in range(d0, d1, 2):
            y = int(round(hy + d * np.sin(rad)))
            x = int(round(hx + d * np.cos(rad)))
            if y < 2 or y >= h - 2 or x < 2 or x >= w - 2:
                break
            vals.append(float(dog_norm[y, x]))
        if len(vals) < 15:
            continue

        arr = np.array(vals, dtype=np.float32)
        score = float(np.mean(arr) + 0.9 * np.quantile(arr, 0.9) + 0.6 * np.mean(arr > ray_thr))
        if best is None or score > best[0]:
            best = (score, float(theta))

    if best is None:
        return 90.0
    return best[1]


def _build_tail_response(gray: np.ndarray, head_r: float, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    g = gray.astype(np.float32)
    dog = np.abs(cv2.GaussianBlur(g, (0, 0), sigmaX=1.2, sigmaY=1.2) - cv2.GaussianBlur(g, (0, 0), sigmaX=4.0, sigmaY=4.0))

    line_len = max(9, int(cfg.tail_line_kernel_scale * head_r))
    tophat = np.zeros_like(g, dtype=np.float32)
    blackhat = np.zeros_like(g, dtype=np.float32)
    for ang in range(0, 180, 15):
        kern = _line_kernel(line_len, ang)
        tophat = np.maximum(tophat, cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kern).astype(np.float32))
        blackhat = np.maximum(blackhat, cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kern).astype(np.float32))

    dog_n = _normalize01(dog)
    tophat_n = _normalize01(tophat)
    blackhat_n = _normalize01(blackhat)
    resp = _normalize01(0.35 * dog_n + 0.30 * tophat_n + 0.35 * blackhat_n)
    return dog_n, resp


def _trace_tail_centerline(
    resp_norm: np.ndarray,
    head_yx: Tuple[int, int],
    head_r: float,
    theta0_deg: float,
    cfg: Config,
    *,
    threshold_quantile: float = 0.60,
    threshold_floor: float = 0.10,
    fail_limit: Optional[int] = None,
    global_angle_span_deg: Optional[int] = None,
    max_dist_scale: Optional[float] = None,
) -> Tuple[float, list[Tuple[float, float]]]:
    h, w = resp_norm.shape
    hy, hx = head_yx

    y = float(hy + 1.02 * head_r * np.sin(np.deg2rad(theta0_deg)))
    x = float(hx + 1.02 * head_r * np.cos(np.deg2rad(theta0_deg)))
    cur_theta = float(theta0_deg)
    points: list[Tuple[float, float]] = [(y, x)]

    thr_low = max(float(threshold_floor), float(np.quantile(resp_norm, threshold_quantile)))
    # Small images tend to overrun into background texture; use a slightly shorter max trace distance by default.
    max_scale_base = cfg.tail_trace_max_dist_scale if max_dist_scale is None else float(max_dist_scale)
    max_scale = max_scale_base * (0.94 if min(h, w) < 300 else 1.0)
    max_dist = max_scale * head_r
    # Small cropped images often place the tail tip close to frame edges.
    margin_scale = 0.45 if min(h, w) < 300 else 1.0
    border_margin = max(2, int(min(h, w) * cfg.tail_trace_border_margin_frac * margin_scale))
    fail_limit_eff = cfg.tail_trace_fail_limit if fail_limit is None else int(fail_limit)
    global_span_eff = cfg.tail_trace_global_angle_span_deg if global_angle_span_deg is None else int(global_angle_span_deg)
    fail = 0
    best_far = math.hypot(y - hy, x - hx)
    best_idx = 0

    for _ in range(cfg.tail_trace_max_steps):
        best = None  # (score, ny, nx, ntheta, local, dist)
        for dtheta in range(
            -cfg.tail_trace_local_angle_span_deg,
            cfg.tail_trace_local_angle_span_deg + 1,
            5,
        ):
            ntheta = cur_theta + dtheta
            if _angle_diff_deg(ntheta, theta0_deg) > global_span_eff:
                continue

            rad = np.deg2rad(ntheta)
            ny = y + cfg.tail_trace_step_px * np.sin(rad)
            nx = x + cfg.tail_trace_step_px * np.cos(rad)
            iy, ix = int(round(ny)), int(round(nx))
            if iy < 2 or iy >= h - 2 or ix < 2 or ix >= w - 2:
                continue
            if min(iy, ix, h - 1 - iy, w - 1 - ix) < border_margin:
                continue

            dist = math.hypot(ny - hy, nx - hx)
            if dist > max_dist:
                continue

            local = float(np.mean(resp_norm[iy - 1:iy + 2, ix - 1:ix + 2]))
            score = local - 0.004 * abs(dtheta)
            cand = (score, ny, nx, ntheta, local, dist)
            if best is None or cand[0] > best[0]:
                best = cand

        if best is None:
            break

        _, ny, nx, ntheta, local, dist = best
        if local < thr_low:
            fail += 1
        else:
            fail = max(0, fail - 1)
        if fail >= fail_limit_eff:
            break

        if len(points) > cfg.tail_trace_loop_back_steps:
            if math.hypot(ny - points[-cfg.tail_trace_loop_back_steps][0], nx - points[-cfg.tail_trace_loop_back_steps][1]) < 1.0:
                break

        y, x, cur_theta = ny, nx, ntheta
        points.append((y, x))

        if dist > best_far:
            best_far = dist
            best_idx = len(points) - 1

    points = points[:best_idx + 1]
    if len(points) < 2:
        raise RuntimeError("Tail tracing failed to produce a valid path.")

    length_px = 0.0
    for (y1, x1), (y2, x2) in zip(points[:-1], points[1:]):
        length_px += math.hypot(y2 - y1, x2 - x1)

    return float(length_px), points


def _estimate_head_from_dark_region(gray: np.ndarray) -> Tuple[Tuple[int, int], float]:
    h, w = gray.shape
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = np.percentile(g[: int(0.85 * h), :], 20)
    bw = (g <= thr)
    bw[: int(0.03 * h), :] = False
    bw[:, : int(0.03 * w)] = False
    bw[:, int(0.97 * w):] = False
    bw[int(0.85 * h):, :] = False
    bw = cv2.morphologyEx((bw.astype(np.uint8) * 255), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1) > 0
    bw = cv2.morphologyEx((bw.astype(np.uint8) * 255), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1) > 0

    lbl = label(bw)
    regs = [r for r in regionprops(lbl) if r.area > 80 and r.centroid[0] < 0.75 * h]
    if not regs:
        raise RuntimeError("Dark-region head fallback failed.")

    rr = max(regs, key=lambda r: r.area)
    comp = (lbl == rr.label).astype(np.uint8)
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    # Keep fallback head radii conservative; very large dark components often absorb tail/background.
    r_cap_frac = 0.11 if min(h, w) >= 500 else 0.16
    r = float(max(16.0, min(float(dist[y, x]), r_cap_frac * min(h, w))))
    return (int(y), int(x)), r


def _path_mean_response(resp_norm: np.ndarray, points: list[Tuple[float, float]]) -> float:
    vals = []
    h, w = resp_norm.shape
    for y, x in points:
        iy, ix = int(round(y)), int(round(x))
        y0, y1 = max(0, iy - 1), min(h, iy + 2)
        x0, x1 = max(0, ix - 1), min(w, ix + 2)
        vals.append(float(np.mean(resp_norm[y0:y1, x0:x1])))
    return float(np.mean(vals)) if vals else 0.0


def measure_phage_tail(
    image_path: str,
    scale_nm: float = 100.0,
    cfg: Optional[Config] = None,
    bar_px_override: Optional[int] = None,
    debug: bool = False,
) -> TailMeasurement:
    cfg = cfg or Config()
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    scale_bar_detection = None
    if bar_px_override is not None:
        bar_px = int(bar_px_override)
    else:
        scale_bar_detection = _find_bottom_scale_bar(img, debug=debug)
        bar_px = int(round(scale_bar_detection.length_px))
    px_per_nm = bar_px / float(scale_nm)

    head_yx, head_r = _detect_head_circle(img, cfg)
    dog_norm, resp_norm = _build_tail_response(img, head_r, cfg)
    theta_deg = _estimate_tail_direction_deg(dog_norm, head_yx, head_r, cfg)

    trace_failed = False
    try:
        tail_px, tail_points = _trace_tail_centerline(resp_norm, head_yx, head_r, theta_deg, cfg)
    except RuntimeError:
        trace_failed = True
        tail_px = 0.0
        tail_points = []

    end_y = tail_points[-1][0] if tail_points else -1.0
    suspicious_initial = (
        (head_yx[0] < 0.12 * img.shape[0])
        and (head_r < 0.12 * min(img.shape))
        and (end_y <= (head_yx[0] + 0.15 * head_r))
    )
    short_initial = (tail_px > 0.0) and (tail_px < 2.8 * head_r)

    if trace_failed or suspicious_initial or short_initial:
        # Fallback for difficult images: dark-region head + multi-direction tracing.
        candidates = [("circle", head_yx, head_r)]
        try:
            dh, dr = _estimate_head_from_dark_region(img)
            candidates.append(("dark", dh, dr))
        except RuntimeError:
            pass

        def _rescue_score(
            px_val: float,
            mean_resp_val: float,
            end_y_val: float,
            center_y_val: float,
            hr_val: float,
        ) -> float:
            if short_initial:
                ratio = px_val / max(hr_val, 1e-6)
                return px_val + 40.0 * mean_resp_val - 70.0 * abs(ratio - 4.0)

            down_pref_val = 40.0 if end_y_val > (center_y_val + 0.20 * hr_val) else -40.0
            return px_val + 60.0 * mean_resp_val + down_pref_val

        best = None  # (score, px, points, head, r, theta_seed)
        if tail_points:
            mean_resp0 = _path_mean_response(resp_norm, tail_points)
            score0 = _rescue_score(tail_px, mean_resp0, tail_points[-1][0], head_yx[0], head_r)
            best = (score0, tail_px, tail_points, head_yx, head_r, theta_deg)

        for _, hyx, hr in candidates:
            dog_i, resp_i = _build_tail_response(img, hr, cfg)
            th_i = _estimate_tail_direction_deg(dog_i, hyx, hr, cfg)
            seeds = [
                th_i,
                (th_i + 180.0) % 360.0,
                (th_i + 60.0) % 360.0,
                (th_i - 60.0) % 360.0,
            ]
            rescue_seed_step = 15 if (trace_failed and min(img.shape) >= 500) else 30
            seeds.extend(float(s) for s in range(0, 360, rescue_seed_step))
            # preserve order while deduping
            uniq = list(dict.fromkeys([float(s % 360.0) for s in seeds]))

            for seed in uniq:
                attempts = [{}]
                if short_initial or trace_failed:
                    attempts.append(
                        {
                            "threshold_quantile": 0.50,
                            "threshold_floor": 0.08,
                            "fail_limit": int(round(cfg.tail_trace_fail_limit * 1.7)),
                            "global_angle_span_deg": max(cfg.tail_trace_global_angle_span_deg, 70),
                            "max_dist_scale": cfg.tail_trace_max_dist_scale * 0.94,
                        }
                    )
                if trace_failed and min(img.shape) >= 500:
                    attempts.append(
                        {
                            "threshold_quantile": 0.48,
                            "threshold_floor": 0.07,
                            "fail_limit": int(round(cfg.tail_trace_fail_limit * 2.4)),
                            "global_angle_span_deg": max(cfg.tail_trace_global_angle_span_deg, 85),
                            "max_dist_scale": 9.5,
                        }
                    )

                for opts in attempts:
                    try:
                        px_i, pts_i = _trace_tail_centerline(resp_i, hyx, hr, seed, cfg, **opts)
                    except RuntimeError:
                        continue

                    end_y_i = pts_i[-1][0]
                    mean_resp = _path_mean_response(resp_i, pts_i)
                    score = _rescue_score(px_i, mean_resp, end_y_i, hyx[0], hr)
                    if best is None or score > best[0]:
                        best = (score, px_i, pts_i, hyx, hr, seed)

        if best is None:
            if trace_failed:
                raise RuntimeError("Tail tracing failed to produce a valid path.")
            # keep initial path if fallback produced no alternatives
            best = (0.0, tail_px, tail_points, head_yx, head_r, theta_deg)

        _, tail_px, tail_points, head_yx, head_r, theta_deg = best

    tail_nm = tail_px / px_per_nm

    if debug:
        start = (int(round(tail_points[0][0])), int(round(tail_points[0][1])))
        end = (int(round(tail_points[-1][0])), int(round(tail_points[-1][1])))
        print(f"[debug] bar_px={bar_px}px; scale_nm={scale_nm}nm => px_per_nm={px_per_nm:.4f}")
        print(f"[debug] head(y,x)={head_yx}; head_r={head_r:.2f}px; tail_theta={theta_deg:.1f} deg")
        print(f"[debug] tail_start(y,x)={start}; tail_end(y,x)={end}")
        print(f"[debug] tail_px={tail_px:.2f}px => tail_nm={tail_nm:.2f}nm")

    return TailMeasurement(
        tail_nm=float(tail_nm),
        tail_px=float(tail_px),
        bar_px=int(bar_px),
        px_per_nm=float(px_per_nm),
        head_yx=(int(head_yx[0]), int(head_yx[1])),
        head_r=float(head_r),
        theta_deg=float(theta_deg),
        tail_points=list(tail_points),
        scale_bbox_xywh=(scale_bar_detection.bbox_xywh if scale_bar_detection is not None else None),
        scale_polarity=(scale_bar_detection.polarity if scale_bar_detection is not None else None),
    )


def measure_phage_tail_length_nm(
    image_path: str,
    scale_nm: float = 100.0,
    cfg: Optional[Config] = None,
    bar_px_override: Optional[int] = None,
    debug: bool = False,
) -> float:
    result = measure_phage_tail(
        image_path=image_path,
        scale_nm=scale_nm,
        cfg=cfg,
        bar_px_override=bar_px_override,
        debug=debug,
    )
    return float(result.tail_nm)


def render_tail_overlay(image_path: str, result: TailMeasurement) -> np.ndarray:
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image for overlay: {image_path}")

    h, w = img_bgr.shape[:2]
    thickness = max(2, int(round(0.004 * min(h, w))))
    line_pts = np.array(
        [[int(round(x)), int(round(y))] for y, x in result.tail_points],
        dtype=np.int32,
    ).reshape(-1, 1, 2)

    if len(line_pts) >= 2:
        cv2.polylines(img_bgr, [line_pts], False, (0, 0, 255), thickness, cv2.LINE_AA)
        sx, sy = int(line_pts[0, 0, 0]), int(line_pts[0, 0, 1])
        ex, ey = int(line_pts[-1, 0, 0]), int(line_pts[-1, 0, 1])
        cv2.circle(img_bgr, (sx, sy), thickness + 1, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img_bgr, (ex, ey), thickness + 1, (0, 255, 255), -1, cv2.LINE_AA)

    hy, hx = result.head_yx
    cv2.circle(img_bgr, (hx, hy), int(round(result.head_r)), (255, 255, 0), max(1, thickness - 1), cv2.LINE_AA)
    cv2.circle(img_bgr, (hx, hy), max(2, thickness), (255, 255, 0), -1, cv2.LINE_AA)

    if result.scale_bbox_xywh is not None:
        x, y, ww, hh = result.scale_bbox_xywh
        cv2.rectangle(img_bgr, (x, y), (x + ww, y + hh), (0, 255, 0), thickness, cv2.LINE_AA)

    text1 = f"Tail: {result.tail_nm:.2f} nm"
    text2 = f"Scale: {result.bar_px}px = {result.bar_px / result.px_per_nm:.0f} nm"
    cv2.putText(img_bgr, text1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_bgr, text2, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
    return img_bgr


def _iter_skeleton_neighbors(y: int, x: int) -> Iterable[tuple[int, int, float]]:
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            weight = math.sqrt(2.0) if dy != 0 and dx != 0 else 1.0
            yield y + dy, x + dx, weight


def _largest_mask_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        raise RuntimeError("Could not find a yellow tail annotation.")

    best_label = None
    best_area = -1
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area > best_area:
            best_label = label_idx
            best_area = area

    if best_label is None or best_area < 20:
        raise RuntimeError("Yellow annotation was detected, but it is too small to measure.")
    return labels == best_label


def _detect_yellow_tail_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15, 70, 80), (45, 255, 255)) > 0
    yellow = cv2.morphologyEx(
        (yellow.astype(np.uint8) * 255),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ) > 0
    yellow = cv2.morphologyEx(
        (yellow.astype(np.uint8) * 255),
        cv2.MORPH_OPEN,
        np.ones((2, 2), dtype=np.uint8),
        iterations=1,
    ) > 0
    return _largest_mask_component(yellow)


def _dijkstra_path(start_idx: int, neighbors: list[list[tuple[int, float]]]) -> tuple[list[float], list[int]]:
    dist = [math.inf] * len(neighbors)
    parent = [-1] * len(neighbors)
    dist[start_idx] = 0.0
    heap: list[tuple[float, int]] = [(0.0, start_idx)]

    while heap:
        cur_dist, node = heapq.heappop(heap)
        if cur_dist != dist[node]:
            continue
        for nxt, weight in neighbors[node]:
            cand = cur_dist + weight
            if cand >= dist[nxt]:
                continue
            dist[nxt] = cand
            parent[nxt] = node
            heapq.heappush(heap, (cand, nxt))

    return dist, parent


def _extract_longest_skeleton_path(mask: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    skeleton = skeletonize(mask)
    points = [tuple(pt) for pt in np.argwhere(skeleton)]
    if len(points) < 2:
        raise RuntimeError("Could not skeletonize the yellow annotation into a measurable path.")

    point_to_idx = {point: idx for idx, point in enumerate(points)}
    neighbors: list[list[tuple[int, float]]] = [[] for _ in points]

    for idx, (y, x) in enumerate(points):
        seen: set[int] = set()
        for ny, nx, weight in _iter_skeleton_neighbors(y, x):
            nxt_idx = point_to_idx.get((ny, nx))
            if nxt_idx is None or nxt_idx in seen:
                continue
            neighbors[idx].append((nxt_idx, weight))
            seen.add(nxt_idx)

    endpoints = [idx for idx, edges in enumerate(neighbors) if len(edges) == 1]
    if len(endpoints) < 2:
        endpoints = list(range(len(points)))

    best_distance = -1.0
    best_start = -1
    best_end = -1
    best_parent: list[int] | None = None

    for start_idx in endpoints:
        dist, parent = _dijkstra_path(start_idx, neighbors)
        for end_idx in endpoints:
            if dist[end_idx] > best_distance and math.isfinite(dist[end_idx]):
                best_distance = dist[end_idx]
                best_start = start_idx
                best_end = end_idx
                best_parent = parent

    if best_parent is None or best_start < 0 or best_end < 0:
        raise RuntimeError("Could not find a valid path along the yellow annotation.")

    path: list[tuple[int, int]] = []
    cur = best_end
    while cur != -1:
        path.append(points[cur])
        if cur == best_start:
            break
        cur = best_parent[cur]
    path.reverse()

    if len(path) < 2 or best_distance <= 0.0:
        raise RuntimeError("The yellow annotation path is too short to measure.")
    return best_distance, path


def _measure_candidate_span(candidate_mask: np.ndarray) -> int:
    best_span = 0
    rows = candidate_mask.shape[0]
    for row_idx in range(rows):
        xs = np.flatnonzero(candidate_mask[row_idx])
        if xs.size == 0:
            continue
        best_span = max(best_span, int(xs[-1] - xs[0] + 1))

    if best_span > 0:
        return best_span

    _, xs = np.where(candidate_mask)
    if xs.size == 0:
        return 0
    return int(xs.max() - xs.min() + 1)


def _find_bottom_scale_bar(gray: np.ndarray, debug: bool = False) -> AnnotatedScaleBarDetection:
    h, w = gray.shape
    y0 = int(round(h * 0.70))
    roi = gray[y0:, :]

    min_width = max(18, int(round(w * 0.04)))
    max_width = int(round(w * 0.45))
    max_height = max(10, int(round(roi.shape[0] * 0.08)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, int(round(w * 0.05))), 1))

    candidates: list[tuple[float, AnnotatedScaleBarDetection]] = []
    percentile_specs = [
        ("bright", roi >= np.percentile(roi, 92)),
        ("dark", roi <= np.percentile(roi, 10)),
    ]

    for polarity, mask in percentile_specs:
        opened = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, ww, hh = cv2.boundingRect(contour)
            if ww < min_width or ww > max_width:
                continue
            if hh < 1 or hh > max_height:
                continue
            if ww / float(hh) < 6.0:
                continue
            if (y + hh) < int(0.25 * roi.shape[0]):
                continue

            candidate_mask = opened[y:y + hh, x:x + ww] > 0
            span_px = _measure_candidate_span(candidate_mask)
            if span_px < min_width:
                continue

            fill = float(cv2.contourArea(contour) / (ww * hh + 1e-6))
            score = span_px + 0.30 * (y + hh) + 25.0 * fill - 2.0 * hh
            detection = AnnotatedScaleBarDetection(
                length_px=float(span_px),
                bbox_xywh=(int(x), int(y + y0), int(ww), int(hh)),
                polarity=polarity,
            )
            candidates.append((score, detection))

    if not candidates:
        raise RuntimeError("Could not find a scale bar in the bottom region of the image.")

    candidates.sort(key=lambda item: item[0], reverse=True)
    detection = candidates[0][1]
    if debug:
        x, y, ww, hh = detection.bbox_xywh
        print(
            "[debug] scale bar (shared): "
            f"x={x}, y={y}, w={detection.length_px:.1f}, h={hh}, polarity={detection.polarity}"
        )
    return detection


def measure_annotated_tail(
    image_path: str | Path,
    scale_nm: float = 100.0,
) -> AnnotatedTailMeasurement:
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    tail_mask = _detect_yellow_tail_mask(image_bgr)
    tail_px, tail_path = _extract_longest_skeleton_path(tail_mask)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    scale_bar = _find_bottom_scale_bar(gray)
    tail_nm = tail_px * float(scale_nm) / scale_bar.length_px

    return AnnotatedTailMeasurement(
        image_path=image_path,
        tail_px=float(tail_px),
        bar_px=float(scale_bar.length_px),
        scale_nm=float(scale_nm),
        tail_nm=float(tail_nm),
        tail_path_yx=tail_path,
        scale_bbox_xywh=scale_bar.bbox_xywh,
        scale_polarity=scale_bar.polarity,
    )


def render_annotated_tail_overlay(result: AnnotatedTailMeasurement) -> np.ndarray:
    img_bgr = cv2.imread(str(result.image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image for overlay: {result.image_path}")

    h, w = img_bgr.shape[:2]
    thickness = max(2, int(round(0.004 * min(h, w))))
    path_xy = np.array(
        [[x, y] for y, x in result.tail_path_yx],
        dtype=np.int32,
    ).reshape(-1, 1, 2)

    if len(path_xy) >= 2:
        cv2.polylines(img_bgr, [path_xy], False, (0, 0, 255), thickness, cv2.LINE_AA)
        sx, sy = path_xy[0, 0]
        ex, ey = path_xy[-1, 0]
        cv2.circle(img_bgr, (int(sx), int(sy)), thickness + 1, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img_bgr, (int(ex), int(ey)), thickness + 1, (0, 255, 255), -1, cv2.LINE_AA)

    x, y, ww, hh = result.scale_bbox_xywh
    cv2.rectangle(img_bgr, (x, y), (x + ww, y + hh), (0, 255, 0), thickness, cv2.LINE_AA)
    cv2.putText(img_bgr, f"Tail: {result.tail_nm:.2f} nm", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        img_bgr,
        f"Scale bar: {result.bar_px:.1f}px = {result.scale_nm:.0f} nm",
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img_bgr


def _show_overlay_window(img_bgr: np.ndarray, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] Could not import matplotlib for display: {exc}")
        return

    plt.figure(figsize=(7, 7))
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


@click.group(cls=OrderedGroup, context_settings=CLICK_CONTEXT_SETTINGS)
@click.version_option(VERSION, "-v", "--version", prog_name="phagescale.py")
def cli() -> None:
    """Measure phage tail length from TEM images."""


@cli.command("measure", context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--image", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Path to input image (png/jpg/tif).")
@click.option("--scale_nm", type=float, default=100.0, show_default=True, help="Scale bar value in nm.")
@click.option("--bar_px_override", type=int, default=None, help="Manual scale bar length in pixels.")
@click.option("--debug", is_flag=True, help="Enable verbose debug output.")
@click.option("--overlay_out", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Path to save image with tail overlay.")
@click.option("--show_overlay", is_flag=True, help="Display the tail overlay at the end of the run.")
def measure_command(
    image: Path,
    scale_nm: float,
    bar_px_override: Optional[int],
    debug: bool,
    overlay_out: Optional[Path],
    show_overlay: bool,
) -> None:
    """Measure capsid diameter and tail length from raw TEM images."""
    try:
        result = measure_phage_tail(
            image_path=str(image),
            scale_nm=scale_nm,
            bar_px_override=bar_px_override,
            debug=debug,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Tail length: {result.tail_nm:.2f} nm")

    if overlay_out is not None or show_overlay:
        out_path = overlay_out if overlay_out is not None else (Path.cwd() / f"{image.stem}_tail_overlay.png")
        overlay = render_tail_overlay(str(image), result)
        cv2.imwrite(str(out_path), overlay)
        click.echo(f"Annotated image: {out_path}")
        if show_overlay:
            _show_overlay_window(overlay, f"Tail length: {result.tail_nm:.2f} nm")


@cli.command("annotated", context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--image", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Path to input image (png/jpg/tif).")
@click.option("--scale_nm", type=float, default=100.0, show_default=True, help="Scale bar value in nm.")
@click.option("--overlay_out", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Path to save image with tail overlay.")
@click.option("--show_overlay", is_flag=True, help="Display the tail overlay at the end of the run.")
def annotated_command(
    image: Path,
    scale_nm: float,
    overlay_out: Optional[Path],
    show_overlay: bool,
) -> None:
    """Measure tail length from yellow-annotated figures."""
    try:
        result = measure_annotated_tail(image_path=image, scale_nm=scale_nm)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Tail length: {result.tail_nm:.2f} nm")

    if overlay_out is not None or show_overlay:
        out_path = (
            overlay_out
            if overlay_out is not None
            else (Path.cwd() / f"{image.stem}_annotated_tail_overlay.png")
        )
        overlay = render_annotated_tail_overlay(result)
        cv2.imwrite(str(out_path), overlay)
        click.echo(f"Annotated image: {out_path}")
        if show_overlay:
            _show_overlay_window(overlay, f"Tail length: {result.tail_nm:.2f} nm")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
