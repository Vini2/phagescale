"""
TEM phage tail length (nm) from a single image.

This tuned version uses:
- robust scale bar detection in the bottom region
- capsid (head) detection with Hough circles + contrast scoring
- direction-guided centerline tracing for the tail
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from skimage.measure import label, regionprops


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
    h, w = gray.shape
    y0 = int(h * (1.0 - cfg.bottom_crop_frac))
    roi = gray[y0:h, :]
    roi_h = roi.shape[0]

    min_len = int(w * cfg.bar_min_len_frac)
    max_len = int(w * cfg.bar_max_len_frac)
    max_h = max(2, int(roi_h * cfg.bar_max_height_frac))

    # Method 1: threshold dark pixels, keep only horizontal structures near bottom corners.
    dark_thr = np.percentile(roi, 12)
    dark_bw = ((roi <= dark_thr).astype(np.uint8) * 255)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, int(w * 0.06)), 1))
    horiz = cv2.morphologyEx(dark_bw, cv2.MORPH_OPEN, h_kernel)
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)), iterations=1)

    contours, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        if ww < min_len or ww > max_len:
            continue
        if hh <= 0 or hh > max_h:
            continue
        aspect = ww / float(hh)
        if aspect < 6.0:
            continue
        if (y + hh) < int(0.45 * roi_h):
            continue
        edge_dist = min(x, w - (x + ww))
        if edge_dist > int(0.35 * w):
            continue

        score = ww - 0.55 * edge_dist + 0.20 * (y + hh)
        candidates.append((score, ww, x, y, hh, edge_dist))

    if candidates:
        candidates.sort(reverse=True, key=lambda t: t[0])
        _, bar_px, x, y, hh, edge_dist = candidates[0]
        if debug:
            print(f"[debug] scale bar (morph): x={x}, y={y}, w={bar_px}, h={hh}, edge={edge_dist} (ROI)")
        return int(bar_px)

    # Method 2: Hough fallback with bottom + corner gating.
    kw = _odd(int(w * cfg.blackhat_kernel_w_frac), minimum=21)
    kh = _odd(int(cfg.blackhat_kernel_h), minimum=5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    bh = cv2.morphologyEx(cv2.equalizeHist(roi), cv2.MORPH_BLACKHAT, kernel)
    _, bw = cv2.threshold(cv2.GaussianBlur(bh, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(bw, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=cfg.bar_hough_thresh,
        minLineLength=max(min_len, cfg.bar_hough_min_line_len),
        maxLineGap=cfg.bar_hough_min_line_gap,
    )

    best_line = None
    if lines is not None:
        for (x1, y1, x2, y2) in lines[:, 0]:
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length < min_len or length > max_len:
                continue
            if abs(dy) > 3:
                continue
            y_mid = 0.5 * (y1 + y2)
            if y_mid < 0.45 * roi_h:
                continue

            edge_dist = min(min(x1, x2), w - max(x1, x2))
            if edge_dist > 0.35 * w:
                continue

            score = length - 0.55 * edge_dist + 0.20 * y_mid
            if best_line is None or score > best_line[0]:
                best_line = (score, length, x1, y1, x2, y2, edge_dist)

    if best_line is not None:
        _, length, x1, y1, x2, y2, edge_dist = best_line
        if debug:
            print(f"[debug] scale bar (hough): ({x1},{y1})-({x2},{y2}) len={length:.1f}px edge={edge_dist:.1f} (ROI)")
        return int(round(length))

    raise RuntimeError(
        "Could not find scale bar automatically. Use --bar_px_override or tune bottom_crop_frac/bar_* settings."
    )


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


def measure_phage_tail_length_nm(
    image_path: str,
    scale_nm: float = 50.0,
    cfg: Optional[Config] = None,
    bar_px_override: Optional[int] = None,
    debug: bool = False,
) -> float:
    cfg = cfg or Config()
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    bar_px = int(bar_px_override) if bar_px_override is not None else _find_scale_bar_px(img, cfg, debug=debug)
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

    return float(tail_nm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to TEM image (png/jpg/tif)")
    ap.add_argument("--scale_nm", type=float, default=50.0, help="Scale bar value in nm (e.g. 100)")
    ap.add_argument("--bar_px_override", type=int, default=None, help="Manual scale bar length in pixels")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    tail_nm = measure_phage_tail_length_nm(
        image_path=args.image,
        scale_nm=args.scale_nm,
        bar_px_override=args.bar_px_override,
        debug=args.debug,
    )
    print(f"Tail length: {tail_nm:.2f} nm")


if __name__ == "__main__":
    main()
