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
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple
from zipfile import ZIP_DEFLATED, ZipFile
import xml.etree.ElementTree as ET

import click
import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize


CLICK_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}
VERSION = "0.1.0"
SCALE_BAR_OVERLAY_COLOR_BGR = (0, 165, 255)


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


LegacyConfig = Config

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
    capsid_diameter_px: float
    capsid_diameter_nm: float
    capsid_center_xy: tuple[float, float]
    capsid_radius_px: float


XLSX_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
XLSX_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XLSX_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
XML_NS = "http://www.w3.org/XML/1998/namespace"
XLSX_NS = {"a": XLSX_MAIN_NS}

ET.register_namespace("", XLSX_MAIN_NS)
ET.register_namespace("r", XLSX_REL_NS)


def _xlsx_qn(tag: str) -> str:
    return f"{{{XLSX_MAIN_NS}}}{tag}"


def _xlsx_col_index_from_letters(letters: str) -> int:
    idx = 0
    for ch in letters:
        if not ch.isalpha():
            continue
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    if idx <= 0:
        raise ValueError(f"Invalid Excel column reference: {letters!r}")
    return idx - 1


def _xlsx_col_letters(col_idx: int) -> str:
    if col_idx < 0:
        raise ValueError(f"Excel column index must be >= 0, got {col_idx}")

    parts: list[str] = []
    cur = col_idx + 1
    while cur > 0:
        cur, rem = divmod(cur - 1, 26)
        parts.append(chr(ord("A") + rem))
    return "".join(reversed(parts))


def _normalize_header_key(value: object) -> str:
    return "".join(ch.lower() for ch in str(value).strip() if ch.isalnum())


def _dedupe_headers(headers: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    deduped: list[str] = []
    for idx, header in enumerate(headers, start=1):
        base = (header or "").strip() or f"Column {idx}"
        count = counts.get(base, 0)
        counts[base] = count + 1
        deduped.append(base if count == 0 else f"{base} ({count + 1})")
    return deduped


def _coerce_xlsx_scalar(text: str) -> object:
    raw = text.strip()
    if raw == "":
        return ""

    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        num = float(raw)
    except ValueError:
        return raw

    if math.isfinite(num) and num.is_integer():
        return int(num)
    return num


def _read_xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> object:
    cell_type = cell.attrib.get("t")

    if cell_type == "inlineStr":
        return "".join(text.text or "" for text in cell.iterfind(".//a:t", XLSX_NS))

    value_elem = cell.find(_xlsx_qn("v"))
    if value_elem is None or value_elem.text is None:
        return ""

    raw = value_elem.text
    if cell_type == "s":
        shared_idx = int(raw)
        if shared_idx < 0 or shared_idx >= len(shared_strings):
            raise ValueError(f"Shared string index out of range: {shared_idx}")
        return shared_strings[shared_idx]
    if cell_type in {"str", "d"}:
        return raw
    if cell_type == "b":
        return raw == "1"

    return _coerce_xlsx_scalar(raw)


def _read_xlsx_rows(xlsx_path: Path, sheet_name: str | None = None) -> tuple[list[str], list[dict[str, object]]]:
    with ZipFile(xlsx_path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("a:si", XLSX_NS):
                shared_strings.append("".join(text.text or "" for text in item.iterfind(".//a:t", XLSX_NS)))

        workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
        rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_target_by_id = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels_root
            if rel.attrib.get("Type", "").endswith("/worksheet")
        }

        sheets = workbook_root.find(_xlsx_qn("sheets"))
        if sheets is None or not list(sheets):
            raise RuntimeError(f"No worksheets found in {xlsx_path}")

        selected_sheet = None
        if sheet_name is None:
            selected_sheet = list(sheets)[0]
        else:
            desired = sheet_name.strip()
            for sheet in sheets:
                if sheet.attrib.get("name", "").strip() == desired:
                    selected_sheet = sheet
                    break
            if selected_sheet is None:
                available = ", ".join(sheet.attrib.get("name", "") for sheet in sheets)
                raise RuntimeError(f"Worksheet {sheet_name!r} was not found in {xlsx_path}. Available sheets: {available}")

        rel_id = selected_sheet.attrib.get(f"{{{XLSX_REL_NS}}}id")
        if not rel_id or rel_id not in rel_target_by_id:
            raise RuntimeError(f"Could not resolve worksheet XML for {selected_sheet.attrib.get('name', '<unknown>')}")

        target = rel_target_by_id[rel_id]
        if not target.startswith("xl/"):
            target = f"xl/{target.lstrip('/')}"

        sheet_root = ET.fromstring(archive.read(target))
        rows = sheet_root.findall(".//a:sheetData/a:row", XLSX_NS)
        if not rows:
            return [], []

        header_row = None
        data_row_maps: list[dict[int, object]] = []
        for row in rows:
            row_map: dict[int, object] = {}
            for cell in row.findall("a:c", XLSX_NS):
                ref = cell.attrib.get("r", "")
                letters = "".join(ch for ch in ref if ch.isalpha())
                if not letters:
                    continue
                row_map[_xlsx_col_index_from_letters(letters)] = _read_xlsx_cell_value(cell, shared_strings)
            if not row_map:
                continue
            if header_row is None:
                header_row = row_map
            else:
                data_row_maps.append(row_map)

        if header_row is None:
            return [], []

        max_header_idx = max(header_row)
        headers = _dedupe_headers(
            [str(header_row.get(col_idx, "")).strip() or f"Column {col_idx + 1}" for col_idx in range(max_header_idx + 1)]
        )

        records: list[dict[str, object]] = []
        for row_map in data_row_maps:
            if row_map and max(row_map) >= len(headers):
                for col_idx in range(len(headers), max(row_map) + 1):
                    headers.append(f"Column {col_idx + 1}")
            record = {header: row_map.get(col_idx, "") for col_idx, header in enumerate(headers)}
            if all(value in {"", None} for value in record.values()):
                continue
            records.append(record)

        return headers, records


def _match_header_name(headers: list[str], desired_header: str) -> str:
    if desired_header in headers:
        return desired_header

    desired_key = _normalize_header_key(desired_header)
    for header in headers:
        if _normalize_header_key(header) == desired_key:
            return header

    available = ", ".join(headers)
    raise RuntimeError(f"Column {desired_header!r} was not found. Available columns: {available}")


def _is_excel_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))


def _build_simple_xlsx_sheet(headers: list[str], rows: list[dict[str, object]]) -> bytes:
    worksheet = ET.Element(_xlsx_qn("worksheet"))
    sheet_data = ET.SubElement(worksheet, _xlsx_qn("sheetData"))

    ordered_rows: list[list[object]] = [headers]
    for record in rows:
        ordered_rows.append([record.get(header, "") for header in headers])

    for row_idx, values in enumerate(ordered_rows, start=1):
        row_elem = ET.SubElement(sheet_data, _xlsx_qn("row"), {"r": str(row_idx)})
        for col_idx, value in enumerate(values):
            if value in {"", None}:
                continue
            cell_ref = f"{_xlsx_col_letters(col_idx)}{row_idx}"
            cell_elem = ET.SubElement(row_elem, _xlsx_qn("c"), {"r": cell_ref})
            if _is_excel_number(value):
                ET.SubElement(cell_elem, _xlsx_qn("v")).text = f"{float(value):.15g}"
            else:
                cell_elem.set("t", "inlineStr")
                inline = ET.SubElement(cell_elem, _xlsx_qn("is"))
                text_elem = ET.SubElement(inline, _xlsx_qn("t"))
                text_value = str(value)
                if text_value != text_value.strip():
                    text_elem.set(f"{{{XML_NS}}}space", "preserve")
                text_elem.text = text_value

    return ET.tostring(worksheet, encoding="utf-8", xml_declaration=True)


def _write_xlsx_rows(
    output_path: Path,
    headers: list[str],
    rows: list[dict[str, object]],
    *,
    sheet_name: str = "Measurements",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_sheet_name = (sheet_name or "Measurements").strip()[:31] or "Measurements"
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    workbook_root = ET.Element(_xlsx_qn("workbook"))
    sheets_elem = ET.SubElement(workbook_root, _xlsx_qn("sheets"))
    ET.SubElement(
        sheets_elem,
        _xlsx_qn("sheet"),
        {
            "name": safe_sheet_name,
            "sheetId": "1",
            f"{{{XLSX_REL_NS}}}id": "rId1",
        },
    )
    workbook_xml = ET.tostring(workbook_root, encoding="utf-8", xml_declaration=True)
    sheet_xml = _build_simple_xlsx_sheet(headers, rows)

    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>
"""
    package_rels = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{XLSX_PACKAGE_REL_NS}">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""
    workbook_rels = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{XLSX_PACKAGE_REL_NS}">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""
    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1">
    <font><sz val="11"/><name val="Calibri"/><family val="2"/></font>
  </fonts>
  <fills count="2">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
  </fills>
  <borders count="1">
    <border><left/><right/><top/><bottom/><diagonal/></border>
  </borders>
  <cellStyleXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>
  </cellStyleXfs>
  <cellXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
  </cellXfs>
  <cellStyles count="1">
    <cellStyle name="Normal" xfId="0" builtinId="0"/>
  </cellStyles>
</styleSheet>
"""
    core_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>phagescale.py</dc:creator>
  <cp:lastModifiedBy>phagescale.py</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified>
</cp:coreProperties>
"""
    app_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>phagescale.py</Application>
  <HeadingPairs>
    <vt:vector size="2" baseType="variant">
      <vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant>
      <vt:variant><vt:i4>1</vt:i4></vt:variant>
    </vt:vector>
  </HeadingPairs>
  <TitlesOfParts>
    <vt:vector size="1" baseType="lpstr">
      <vt:lpstr>{safe_sheet_name}</vt:lpstr>
    </vt:vector>
  </TitlesOfParts>
</Properties>
"""

    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", package_rels)
        archive.writestr("docProps/app.xml", app_xml)
        archive.writestr("docProps/core.xml", core_xml)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", styles_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


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
        cv2.rectangle(img_bgr, (x, y), (x + ww, y + hh), SCALE_BAR_OVERLAY_COLOR_BGR, thickness, cv2.LINE_AA)

    text1 = f"Capsid: {2 * result.head_r * result.px_per_nm:.2f} nm"
    text2 = f"Tail: {result.tail_nm:.2f} nm"
    text3 = f"Scale: {result.bar_px}px = {result.bar_px / result.px_per_nm:.0f} nm"
    cv2.putText(img_bgr, text1, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_bgr, text2, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_bgr, text3, (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
    return img_bgr


def _iter_skeleton_neighbors(y: int, x: int) -> Iterable[tuple[int, int, float]]:
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            weight = math.sqrt(2.0) if dy != 0 and dx != 0 else 1.0
            yield y + dy, x + dx, weight


def _largest_mask_component(mask: np.ndarray, *, label_name: str) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        raise RuntimeError(f"Could not find a {label_name} annotation.")

    best_label = None
    best_area = -1
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area > best_area:
            best_label = label_idx
            best_area = area

    if best_label is None or best_area < 20:
        raise RuntimeError(f"The {label_name} annotation was detected, but it is too small to measure.")
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
    return _largest_mask_component(yellow, label_name="yellow tail")


def _detect_magenta_capsid_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # Magenta/pink annotations can shift slightly after JPEG compression, so keep the hue band tolerant.
    magenta = cv2.inRange(hsv, (130, 40, 40), (175, 255, 255)) > 0
    magenta = cv2.morphologyEx(
        (magenta.astype(np.uint8) * 255),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ) > 0
    magenta = cv2.morphologyEx(
        (magenta.astype(np.uint8) * 255),
        cv2.MORPH_OPEN,
        np.ones((2, 2), dtype=np.uint8),
        iterations=1,
    ) > 0
    return _largest_mask_component(magenta, label_name="magenta capsid")


def _compute_capsid_diameter_px(mask: np.ndarray) -> tuple[float, tuple[float, float], float]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, (0.0, 0.0), 0.0
    # Get all points from the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) < 3:
        return 0.0, (0.0, 0.0), 0.0
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    diameter = 2 * radius
    return diameter, (x, y), radius


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


def _find_green_annotated_scale_bar(image_bgr: np.ndarray) -> AnnotatedScaleBarDetection | None:
    h, w = image_bgr.shape[:2]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Green scale-bar annotations are drawn as thin bright guide lines.
    green_mask = cv2.inRange(hsv, (40, 80, 80), (80, 255, 255))
    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    green_mask = cv2.morphologyEx(
        green_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)),
        iterations=1,
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(green_mask, connectivity=8)
    if num_labels <= 1:
        return None

    min_length = max(18, int(round(min(h, w) * 0.035)))
    max_length = int(round(max(h, w) * 0.98))
    max_thickness = max(6, int(round(min(h, w) * 0.05)))

    best: tuple[float, AnnotatedScaleBarDetection] | None = None
    for label_idx in range(1, num_labels):
        x, y, ww, hh, area = (
            int(stats[label_idx, cv2.CC_STAT_LEFT]),
            int(stats[label_idx, cv2.CC_STAT_TOP]),
            int(stats[label_idx, cv2.CC_STAT_WIDTH]),
            int(stats[label_idx, cv2.CC_STAT_HEIGHT]),
            int(stats[label_idx, cv2.CC_STAT_AREA]),
        )
        if area < 20:
            continue
        if ww < 1 or hh < 1:
            continue
        if ww > max_length and hh > max_length:
            continue

        component_mask = labels[y:y + hh, x:x + ww] == label_idx
        span_x = _measure_candidate_span(component_mask)
        span_y = _measure_candidate_span(component_mask.T)
        long_span = max(span_x, span_y)
        short_span = max(1, min(span_x, span_y))
        if long_span < min_length or long_span > max_length:
            continue
        if short_span > max_thickness:
            continue
        if long_span / float(short_span) < 6.0:
            continue

        edge_dist = float(min(x, y, max(0, w - (x + ww)), max(0, h - (y + hh))))
        edge_bonus = max(0.0, 0.20 * min(h, w) - edge_dist)
        score = float(long_span) + 4.0 * (long_span / float(short_span)) + 0.20 * edge_bonus - 1.5 * short_span

        detection = AnnotatedScaleBarDetection(
            length_px=float(long_span),
            bbox_xywh=(x, y, ww, hh),
            polarity="green",
        )
        if best is None or score > best[0]:
            best = (score, detection)

    return None if best is None else best[1]


def _score_scale_bar_candidate(
    gray: np.ndarray,
    detection: AnnotatedScaleBarDetection,
    *,
    prefer_bottom: bool,
) -> float:
    h, w = gray.shape
    x, y, ww, hh = detection.bbox_xywh

    y0 = max(0, y - 4)
    y1 = min(h, y + hh + 4)
    x0 = max(0, x - 4)
    x1 = min(w, x + ww + 4)
    neighborhood = gray[y0:y1, x0:x1].astype(np.float32)
    candidate = gray[y:y + hh, x:x + ww].astype(np.float32)

    contrast = 0.0
    if neighborhood.size > 0 and candidate.size > 0:
        mask = np.ones(neighborhood.shape, dtype=bool)
        mask[(y - y0):(y - y0 + hh), (x - x0):(x - x0 + ww)] = False
        context = neighborhood[mask]
        if context.size > 0:
            candidate_mean = float(np.mean(candidate))
            context_mean = float(np.mean(context))
            if detection.polarity == "dark":
                contrast = context_mean - candidate_mean
            else:
                contrast = candidate_mean - context_mean

    edge_dist = float(min(x, y, max(0, w - (x + ww)), max(0, h - (y + hh))))
    edge_bonus = max(0.0, 0.18 * min(h, w) - edge_dist)
    bottom_bonus = max(0.0, (y + hh) - 0.72 * h) if prefer_bottom else 0.0
    aspect = detection.length_px / max(float(hh), 1.0)
    return (
        float(detection.length_px)
        + 1.1 * contrast
        + 0.30 * edge_bonus
        + 0.12 * bottom_bonus
        + 2.5 * aspect
        - 1.8 * hh
    )


def _collect_scale_bar_candidates(
    gray: np.ndarray,
    *,
    y_start: int,
    y_end: int,
    prefer_bottom: bool,
) -> list[tuple[float, AnnotatedScaleBarDetection]]:
    h, w = gray.shape
    roi = gray[max(0, y_start):min(h, y_end), :]
    if roi.size == 0:
        return []

    min_width = max(18, int(round(w * 0.035)))
    max_width_frac = 0.55 if prefer_bottom else 0.72
    max_width = int(round(w * max_width_frac))
    max_height = max(
        10,
        min(
            int(round(roi.shape[0] * 0.12)),
            int(round(h * 0.05)),
        ),
    )
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, int(round(w * 0.05))), 1))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(round(w * 0.01))), 3))

    candidates: list[tuple[float, AnnotatedScaleBarDetection]] = []
    percentile_specs = [
        ("bright", roi >= np.percentile(roi, 92)),
        ("dark", roi <= np.percentile(roi, 10)),
    ]

    for polarity, mask in percentile_specs:
        cleaned = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, close_kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, ww, hh = cv2.boundingRect(contour)
            if ww < min_width or ww > max_width:
                continue
            if hh < 1 or hh > max_height:
                continue
            if ww / float(max(hh, 1)) < 4.5:
                continue

            candidate_mask = cleaned[y:y + hh, x:x + ww] > 0
            span_px = _measure_candidate_span(candidate_mask)
            if span_px < min_width:
                continue

            detection = AnnotatedScaleBarDetection(
                length_px=float(span_px),
                bbox_xywh=(int(x), int(y + y_start), int(ww), int(hh)),
                polarity=polarity,
            )
            score = _score_scale_bar_candidate(gray, detection, prefer_bottom=prefer_bottom)
            candidates.append((score, detection))

    return candidates


def _find_scale_bar_by_hough(gray: np.ndarray) -> AnnotatedScaleBarDetection | None:
    h, w = gray.shape
    min_length = max(18, int(round(w * 0.035)))
    max_length = int(round(w * 0.72))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=math.pi / 180.0,
        threshold=32,
        minLineLength=min_length,
        maxLineGap=max(4, int(round(w * 0.01))),
    )
    if lines is None:
        return None

    best: tuple[float, AnnotatedScaleBarDetection] | None = None
    context_radius = max(5, int(round(min(h, w) * 0.01)))

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, line)
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < min_length or length > max_length:
            continue

        angle_deg = abs(math.degrees(math.atan2(dy, dx)))
        if min(angle_deg, abs(180.0 - angle_deg)) > 6.0:
            continue

        x_min, x_max = sorted((x1, x2))
        y_min, y_max = sorted((y1, y2))
        border_dist = min(y_min, h - 1 - y_max, x_min, w - 1 - x_max)
        if border_dist < 2:
            continue

        y0 = max(0, y_min - context_radius)
        y1_ctx = min(h, y_max + context_radius + 1)
        x0 = max(0, x_min - context_radius)
        x1_ctx = min(w, x_max + context_radius + 1)
        region = gray[y0:y1_ctx, x0:x1_ctx].astype(np.float32)
        if region.size == 0:
            continue

        line_mask = np.zeros(region.shape, dtype=np.uint8)
        cv2.line(
            line_mask,
            (x_min - x0, int(round((y1 + y2) / 2.0)) - y0),
            (x_max - x0, int(round((y1 + y2) / 2.0)) - y0),
            255,
            1,
            cv2.LINE_AA,
        )
        line_pixels = region[line_mask > 0]
        context_mask = cv2.dilate(line_mask, np.ones((5, 5), dtype=np.uint8), iterations=1) > 0
        context_pixels = region[context_mask & (line_mask == 0)]
        if line_pixels.size == 0 or context_pixels.size == 0:
            continue

        polarity = "dark" if float(np.mean(line_pixels)) < float(np.mean(context_pixels)) else "bright"
        detection = AnnotatedScaleBarDetection(
            length_px=float(length),
            bbox_xywh=(int(x_min), int(y_min), int(max(1, round(length))), int(max(1, y_max - y_min + 1))),
            polarity=polarity,
        )
        score = _score_scale_bar_candidate(gray, detection, prefer_bottom=False) + 12.0
        if best is None or score > best[0]:
            best = (score, detection)

    return None if best is None else best[1]


def _find_bottom_scale_bar(gray: np.ndarray, debug: bool = False) -> AnnotatedScaleBarDetection:
    h, w = gray.shape
    candidates: list[tuple[float, AnnotatedScaleBarDetection]] = []
    candidates.extend(_collect_scale_bar_candidates(gray, y_start=int(round(h * 0.68)), y_end=h, prefer_bottom=True))
    candidates.extend(_collect_scale_bar_candidates(gray, y_start=0, y_end=h, prefer_bottom=False))

    hough_detection = _find_scale_bar_by_hough(gray)
    if hough_detection is not None:
        candidates.append((_score_scale_bar_candidate(gray, hough_detection, prefer_bottom=False) + 12.0, hough_detection))

    if not candidates:
        raise RuntimeError("Could not find a scale bar in the image.")

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

    capsid_mask = _detect_magenta_capsid_mask(image_bgr)
    capsid_diameter_px, capsid_center_xy, capsid_radius_px = _compute_capsid_diameter_px(capsid_mask)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    scale_bar = _find_green_annotated_scale_bar(image_bgr)
    if scale_bar is None:
        scale_bar = _find_bottom_scale_bar(gray)
    tail_nm = tail_px * float(scale_nm) / scale_bar.length_px
    capsid_diameter_nm = capsid_diameter_px * float(scale_nm) / scale_bar.length_px

    return AnnotatedTailMeasurement(
        image_path=image_path,
        tail_px=float(tail_px),
        bar_px=float(scale_bar.length_px),
        scale_nm=float(scale_nm),
        tail_nm=float(tail_nm),
        tail_path_yx=tail_path,
        scale_bbox_xywh=scale_bar.bbox_xywh,
        scale_polarity=scale_bar.polarity,
        capsid_diameter_px=capsid_diameter_px,
        capsid_diameter_nm=capsid_diameter_nm,
        capsid_center_xy=capsid_center_xy,
        capsid_radius_px=capsid_radius_px,
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

    # Draw capsid circle
    if result.capsid_radius_px > 0:
        cx, cy = result.capsid_center_xy
        cv2.circle(img_bgr, (int(cx), int(cy)), int(result.capsid_radius_px), (255, 255, 0), thickness, cv2.LINE_AA)

    x, y, ww, hh = result.scale_bbox_xywh
    cv2.rectangle(img_bgr, (x, y), (x + ww, y + hh), SCALE_BAR_OVERLAY_COLOR_BGR, thickness, cv2.LINE_AA)
    cv2.putText(img_bgr, f"Capsid: {result.capsid_diameter_nm:.2f} nm", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img_bgr, f"Tail: {result.tail_nm:.2f} nm", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        img_bgr,
        f"Scale bar: {result.bar_px:.1f}px = {result.scale_nm:.0f} nm",
        (12, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return img_bgr


def _parse_required_float(value: object, *, field_name: str, row_number: int) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
    else:
        text = str(value).strip()
        if text == "":
            raise RuntimeError(f"Row {row_number}: {field_name} is blank.")
        try:
            numeric = float(text)
        except ValueError as exc:
            raise RuntimeError(f"Row {row_number}: {field_name} value {value!r} is not numeric.") from exc

    if not math.isfinite(numeric):
        raise RuntimeError(f"Row {row_number}: {field_name} value {value!r} is not finite.")
    return numeric


def _build_image_lookup(images_dir: Path) -> tuple[
    list[Path],
    dict[str, list[Path]],
    dict[str, list[Path]],
]:
    files = [path for path in images_dir.iterdir() if path.is_file()]
    by_lower_name: dict[str, list[Path]] = {}
    by_lower_stem: dict[str, list[Path]] = {}
    for path in files:
        by_lower_name.setdefault(path.name.lower(), []).append(path)
        by_lower_stem.setdefault(path.stem.lower(), []).append(path)
    return files, by_lower_name, by_lower_stem


def _resolve_batch_image_path(
    images_dir: Path,
    image_name: str,
    *,
    image_files: list[Path],
    by_lower_name: dict[str, list[Path]],
    by_lower_stem: dict[str, list[Path]],
) -> Path:
    direct_path = images_dir / image_name
    if direct_path.is_file():
        return direct_path

    name_matches = by_lower_name.get(image_name.lower(), [])
    if len(name_matches) == 1:
        return name_matches[0]

    image_stem = Path(image_name).stem.lower()
    stem_matches = by_lower_stem.get(image_stem, [])
    if len(stem_matches) == 1:
        return stem_matches[0]

    prefix_matches = [path for path in image_files if path.stem.lower().startswith(image_stem)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    raise FileNotFoundError(f"image was not found: {direct_path}")


def measure_annotated_batch(
    *,
    images_dir: Path,
    metadata_xlsx: Path,
    output_xlsx: Path,
    sheet_name: str | None = None,
    image_col: str = "File name",
    scale_col: str = "Scale bar measurement (nm)",
    overlay_dir: Path | None = None,
    fail_fast: bool = False,
) -> tuple[int, int]:
    headers, records = _read_xlsx_rows(metadata_xlsx, sheet_name=sheet_name)
    if not headers or not records:
        raise RuntimeError(f"No data rows were found in {metadata_xlsx}")

    image_header = _match_header_name(headers, image_col)
    scale_header = _match_header_name(headers, scale_col)
    image_files, by_lower_name, by_lower_stem = _build_image_lookup(images_dir)

    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    output_headers = list(headers)
    extra_headers = [
        "Metadata row",
        "Image path",
        "Measurement status",
        "Measurement error",
        "Detected scale bar (px)",
        "Detected scale bar polarity",
        "Capsid diameter (px)",
        "Capsid diameter (nm)",
        "Tail length (px)",
        "Tail length (nm)",
        "Overlay path",
    ]
    for header in extra_headers:
        if header not in output_headers:
            output_headers.append(header)

    output_rows: list[dict[str, object]] = []
    success_count = 0
    failure_count = 0

    for row_idx, record in enumerate(records, start=2):
        output_record = {header: record.get(header, "") for header in headers}
        output_record.update({
            "Metadata row": row_idx,
            "Image path": "",
            "Measurement status": "error",
            "Measurement error": "",
            "Detected scale bar (px)": "",
            "Detected scale bar polarity": "",
            "Capsid diameter (px)": "",
            "Capsid diameter (nm)": "",
            "Tail length (px)": "",
            "Tail length (nm)": "",
            "Overlay path": "",
        })

        image_name = str(record.get(image_header, "")).strip()
        try:
            if image_name == "":
                raise RuntimeError(f"Row {row_idx}: {image_header} is blank.")

            image_path = _resolve_batch_image_path(
                images_dir,
                image_name,
                image_files=image_files,
                by_lower_name=by_lower_name,
                by_lower_stem=by_lower_stem,
            )
            output_record["Image path"] = str(image_path)

            scale_nm = _parse_required_float(record.get(scale_header, ""), field_name=scale_header, row_number=row_idx)
            result = measure_annotated_tail(image_path=image_path, scale_nm=scale_nm)

            output_record["Measurement status"] = "ok"
            output_record["Detected scale bar (px)"] = float(result.bar_px)
            output_record["Detected scale bar polarity"] = result.scale_polarity or ""
            output_record["Capsid diameter (px)"] = float(result.capsid_diameter_px)
            output_record["Capsid diameter (nm)"] = float(result.capsid_diameter_nm)
            output_record["Tail length (px)"] = float(result.tail_px)
            output_record["Tail length (nm)"] = float(result.tail_nm)

            if overlay_dir is not None:
                overlay_path = overlay_dir / f"{Path(image_name).stem}_annotated_overlay.png"
                overlay = render_annotated_tail_overlay(result)
                cv2.imwrite(str(overlay_path), overlay)
                output_record["Overlay path"] = str(overlay_path)

            success_count += 1
        except Exception as exc:
            failure_count += 1
            error_text = str(exc)
            if not error_text.startswith(f"Row {row_idx}:"):
                error_text = f"Row {row_idx}: {error_text}"
            output_record["Measurement error"] = error_text
            if fail_fast:
                raise
        output_rows.append(output_record)

    _write_xlsx_rows(output_xlsx, output_headers, output_rows, sheet_name="Annotated measurements")
    return success_count, failure_count


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


EPS = 1e-6


@dataclass(frozen=True)
class LandmarkSpec:
    kind: str
    value: float
    weight: float
    search_radius_px: int


@dataclass(frozen=True)
class FitConfig:
    capsid_landmarks: int = 16
    tail_landmarks: int = 14
    mean_tail_length_units: float = 4.1
    capsid_radius_scale: float = 0.20
    tail_length_scale: float = 0.62
    bend_scale: float = 0.34
    distal_bend_scale: float = 0.20
    prior_sigmas: tuple[float, float, float, float] = (0.75, 0.95, 0.85, 0.95)
    pose_anchor_sigmas: tuple[float, float, float, float] = (20.0, 20.0, 0.40, 14.0)
    kernel_schedule_px: tuple[float, ...] = (7.0, 4.5, 2.5, 1.2)
    iterations_per_level: int = 6
    damping: float = 0.9
    max_parameter_step_norm: float = 16.0


@dataclass(frozen=True)
class PhageShapeModel:
    name: str
    specs: tuple[LandmarkSpec, ...]
    center_index: int
    capsid_ring_indices: tuple[int, ...]
    neck_index: int
    tail_path_indices: tuple[int, ...]
    tip_index: int


@dataclass(frozen=True)
class FittedPhageGeometry:
    capsid_circle_xyr: tuple[float, float, float]
    tail_polyline_xy: np.ndarray


@dataclass
class PhageCLMResult:
    image_path: Path
    scale_nm: float
    bar_px: float
    px_per_nm: float
    capsid_diameter_px: float
    capsid_diameter_nm: float
    tail_length_px: float
    tail_length_nm: float
    head_center_xy: tuple[float, float]
    theta_deg: float
    params: np.ndarray
    model: PhageShapeModel
    geometry: FittedPhageGeometry
    landmark_xy: np.ndarray
    specs: list[LandmarkSpec]
    scale_bbox_xywh: tuple[int, int, int, int] | None
    scale_polarity: str | None


def _rotation_matrix(theta_rad: float) -> np.ndarray:
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _build_phage_shape_model(cfg: FitConfig) -> PhageShapeModel:
    specs: list[LandmarkSpec] = [LandmarkSpec("center", 0.0, 2.8, 10)]

    for angle in np.linspace(0.0, 2.0 * math.pi, cfg.capsid_landmarks, endpoint=False):
        specs.append(LandmarkSpec("capsid_ring", float(angle), 1.9, 11))

    center_index = 0
    capsid_ring_indices = tuple(range(1, 1 + cfg.capsid_landmarks))

    specs.append(LandmarkSpec("neck", 0.0, 2.1, 10))
    neck_index = len(specs) - 1

    for t in np.linspace(0.10, 0.92, cfg.tail_landmarks):
        specs.append(LandmarkSpec("tail", float(t), 1.3, 12))
    specs.append(LandmarkSpec("tip", 1.0, 1.7, 14))
    tip_index = len(specs) - 1

    return PhageShapeModel(
        name="phage_capsid_tail_rlms",
        specs=tuple(specs),
        center_index=center_index,
        capsid_ring_indices=capsid_ring_indices,
        neck_index=neck_index,
        tail_path_indices=tuple([neck_index] + list(range(neck_index + 1, len(specs)))),
        tip_index=tip_index,
    )


def _shape_in_object_frame(params: np.ndarray, specs: list[LandmarkSpec], cfg: FitConfig) -> np.ndarray:
    _, _, _, _, q_radius, q_tail, q_bend_mid, q_bend_distal = params
    radius_units = max(0.55, 1.0 + cfg.capsid_radius_scale * float(q_radius))
    tail_units = max(1.6, cfg.mean_tail_length_units * (1.0 + cfg.tail_length_scale * float(q_tail)))
    bend_mid_units = cfg.bend_scale * float(q_bend_mid)
    bend_distal_units = cfg.distal_bend_scale * float(q_bend_distal)

    points = []
    for spec in specs:
        if spec.kind == "center":
            points.append((0.0, 0.0))
            continue

        if spec.kind == "capsid_ring":
            ang = spec.value
            points.append((radius_units * math.cos(ang), radius_units * math.sin(ang)))
            continue

        if spec.kind == "neck":
            points.append((1.03 * radius_units, 0.0))
            continue

        if spec.kind in {"tail", "tip"}:
            t = float(spec.value)
            x = 1.03 * radius_units + tail_units * t
            support = 4.0 * t * (1.0 - t)
            asymmetry = support * (2.0 * t - 1.0)
            y = bend_mid_units * support + bend_distal_units * asymmetry
            points.append((x, y))
            continue

        raise ValueError(f"Unknown landmark kind: {spec.kind}")

    return np.asarray(points, dtype=np.float64)


def _landmark_positions_xy(params: np.ndarray, specs: list[LandmarkSpec], cfg: FitConfig) -> np.ndarray:
    tx, ty, theta_rad, scale_px, *_ = params
    obj = _shape_in_object_frame(params, specs, cfg)
    world = obj @ _rotation_matrix(theta_rad).T
    world *= float(scale_px)
    world[:, 0] += float(tx)
    world[:, 1] += float(ty)
    return world


def _numerical_jacobian(params: np.ndarray, specs: list[LandmarkSpec], cfg: FitConfig) -> np.ndarray:
    base = _landmark_positions_xy(params, specs, cfg).reshape(-1)
    jac = np.zeros((base.size, params.size), dtype=np.float64)
    eps_values = np.array([1e-2, 1e-2, 2e-4, 2e-3, 2e-3, 2e-3, 2e-3, 2e-3], dtype=np.float64)

    for idx, eps in enumerate(eps_values):
        step = np.zeros_like(params)
        step[idx] = eps
        plus = _landmark_positions_xy(params + step, specs, cfg).reshape(-1)
        minus = _landmark_positions_xy(params - step, specs, cfg).reshape(-1)
        jac[:, idx] = (plus - minus) / (2.0 * eps)
    return jac


def _make_capsid_response(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    gx = cv2.Scharr(blur, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(blur, cv2.CV_32F, 0, 1)
    grad = cv2.magnitude(gx, gy)

    dark = 255.0 - cv2.GaussianBlur(gray, (0, 0), 2.0).astype(np.float32)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))).astype(np.float32)

    center_resp = _normalize01(0.75 * dark + 0.25 * blackhat)
    ring_resp = _normalize01(0.68 * grad + 0.32 * blackhat)
    return center_resp, ring_resp


def _sample_patch(
    response: np.ndarray,
    x: float,
    y: float,
    search_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = response.shape
    cx = int(round(x))
    cy = int(round(y))
    x0 = max(0, cx - search_radius)
    x1 = min(w, cx + search_radius + 1)
    y0 = max(0, cy - search_radius)
    y1 = min(h, cy + search_radius + 1)

    patch = response[y0:y1, x0:x1].astype(np.float64)
    xs = np.arange(x0, x0 + patch.shape[1], dtype=np.float64)
    ys = np.arange(y0, y0 + patch.shape[0], dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return patch, grid_x, grid_y


def _mean_shift_landmark(
    response: np.ndarray,
    current_xy: np.ndarray,
    sigma_px: float,
    search_radius: int,
) -> tuple[np.ndarray, float]:
    patch, grid_x, grid_y = _sample_patch(response, current_xy[0], current_xy[1], search_radius)
    if patch.size == 0:
        return current_xy.copy(), 0.0

    alpha = patch - float(patch.min())
    if float(alpha.max()) <= EPS:
        return current_xy.copy(), 0.0

    dx = grid_x - float(current_xy[0])
    dy = grid_y - float(current_xy[1])
    kernel = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma_px * sigma_px))
    weights = alpha * kernel + EPS
    total = float(weights.sum())
    if total <= EPS:
        return current_xy.copy(), 0.0

    new_x = float((weights * grid_x).sum() / total)
    new_y = float((weights * grid_y).sum() / total)
    confidence = float(alpha.max())
    return np.array([new_x, new_y], dtype=np.float64), confidence


def _estimate_initial_tail_length_px(
    tail_response: np.ndarray,
    head_center_xy: tuple[float, float],
    head_radius_px: float,
    theta_rad: float,
) -> float:
    h, w = tail_response.shape
    cx, cy = head_center_xy
    max_dist = min(w, h) * 0.75
    distances = np.arange(max(3.0, 1.05 * head_radius_px), max_dist, 1.0, dtype=np.float64)
    vals = []
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    for dist in distances:
        x = cx + dist * cos_t
        y = cy + dist * sin_t
        ix = int(round(x))
        iy = int(round(y))
        if ix < 2 or ix >= w - 2 or iy < 2 or iy >= h - 2:
            vals.append(0.0)
            continue
        vals.append(float(np.mean(tail_response[iy - 1:iy + 2, ix - 1:ix + 2])))

    if not vals:
        return 4.0 * head_radius_px

    arr = np.asarray(vals, dtype=np.float64)
    smooth = cv2.GaussianBlur(arr.reshape(1, -1).astype(np.float32), (1, 0), 1.2).reshape(-1)
    threshold = max(0.18, 0.48 * float(np.max(smooth)))
    active = np.where(smooth >= threshold)[0]
    if active.size == 0:
        return 4.0 * head_radius_px

    tip_dist = float(distances[int(active[-1])])
    return max(1.8 * head_radius_px, tip_dist - 1.03 * head_radius_px)


def _capsid_contrast_score(gray: np.ndarray, head_yx: tuple[int, int], head_r: float) -> float:
    y, x = head_yx
    yy, xx = np.indices(gray.shape)
    d = np.sqrt((yy - y) ** 2 + (xx - x) ** 2)
    inner = gray[d <= 0.72 * head_r]
    ring = gray[(d >= 0.92 * head_r) & (d <= 1.25 * head_r)]
    outer = gray[(d >= 1.45 * head_r) & (d <= 1.90 * head_r)]
    if inner.size < 30 or ring.size < 30 or outer.size < 30:
        return -999.0
    return float(np.mean(inner) - np.mean(ring) + 0.55 * (np.mean(outer) - np.mean(ring)))


def _initial_params(
    gray: np.ndarray,
    specs: list[LandmarkSpec],
    cfg: FitConfig,
    image_path: Optional[Path] = None,
    bar_px_override: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    del specs
    legacy_cfg = LegacyConfig()
    h, w = gray.shape

    if image_path is not None:
        try:
            legacy_result = measure_phage_tail(
                image_path=str(image_path),
                scale_nm=100.0,
                bar_px_override=(int(round(bar_px_override)) if bar_px_override is not None else None),
                debug=False,
            )
            _, tail_resp = _build_tail_response(gray, float(legacy_result.head_r), legacy_cfg)
            q_tail0 = np.clip(
                (legacy_result.tail_px / max(legacy_result.head_r, EPS) / cfg.mean_tail_length_units) - 1.0,
                -1.25,
                1.5,
            ) / max(cfg.tail_length_scale, EPS)
            params0 = np.array(
                [
                    float(legacy_result.head_yx[1]),
                    float(legacy_result.head_yx[0]),
                    math.radians(float(legacy_result.theta_deg)),
                    float(legacy_result.head_r),
                    0.0,
                    q_tail0,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            )
            return params0, tail_resp
        except Exception:
            pass

    candidates: list[tuple[tuple[int, int], float]] = []
    try:
        candidates.append(_detect_head_circle(gray, legacy_cfg))
    except RuntimeError:
        pass
    try:
        candidates.append(_estimate_head_from_dark_region(gray))
    except RuntimeError:
        pass

    if not candidates:
        raise RuntimeError("Could not initialize the phage head.")

    best = None
    for head_yx, head_r in candidates:
        dog_resp, tail_resp = _build_tail_response(gray, float(head_r), legacy_cfg)
        head_center_xy = (float(head_yx[1]), float(head_yx[0]))
        theta_seed_deg = _estimate_tail_direction_deg(dog_resp, head_yx, float(head_r), legacy_cfg)

        theta_options = [float(theta_seed_deg), float((theta_seed_deg + 180.0) % 360.0)]
        best_theta_rad = 0.0
        tail_px0 = -1.0
        for theta_deg in theta_options:
            theta_rad = math.radians(theta_deg)
            tail_px = _estimate_initial_tail_length_px(tail_resp, head_center_xy, float(head_r), theta_rad)
            if tail_px > tail_px0:
                tail_px0 = tail_px
                best_theta_rad = theta_rad

        center_penalty = 0.0
        if head_yx[0] < 0.12 * h or head_yx[0] > 0.82 * h:
            center_penalty += 40.0
        if head_yx[1] < 0.12 * w or head_yx[1] > 0.88 * w:
            center_penalty += 25.0

        score = float(
            tail_px0
            + 0.20 * head_r
            + 3.0 * _capsid_contrast_score(gray, head_yx, float(head_r))
            - center_penalty
        )
        if best is None or score > best[0]:
            best = (score, head_yx, head_r, best_theta_rad, tail_px0, tail_resp)

    assert best is not None
    _, head_yx, head_r, theta_rad, tail_px0, tail_resp = best
    head_center_xy = (float(head_yx[1]), float(head_yx[0]))
    q_tail0 = np.clip((tail_px0 / max(head_r, EPS) / cfg.mean_tail_length_units) - 1.0, -1.25, 1.5) / max(cfg.tail_length_scale, EPS)

    params0 = np.array(
        [
            head_center_xy[0],
            head_center_xy[1],
            theta_rad,
            float(head_r),
            0.0,
            q_tail0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    return params0, tail_resp


def _fit_clm(
    gray: np.ndarray,
    specs: list[LandmarkSpec],
    cfg: FitConfig,
    image_path: Optional[Path] = None,
    bar_px_override: Optional[float] = None,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center_resp, ring_resp = _make_capsid_response(gray)
    params, tail_resp = _initial_params(gray, specs, cfg, image_path=image_path, bar_px_override=bar_px_override)
    params_anchor = params.copy()

    prior_inv_vars = np.array([0.0, 0.0, 0.0, 0.0] + [1.0 / (sigma * sigma) for sigma in cfg.prior_sigmas], dtype=np.float64)
    pose_anchor_inv_vars = np.array(
        [
            1.0 / (cfg.pose_anchor_sigmas[0] * cfg.pose_anchor_sigmas[0]),
            1.0 / (cfg.pose_anchor_sigmas[1] * cfg.pose_anchor_sigmas[1]),
            1.0 / (cfg.pose_anchor_sigmas[2] * cfg.pose_anchor_sigmas[2]),
            1.0 / (cfg.pose_anchor_sigmas[3] * cfg.pose_anchor_sigmas[3]),
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )

    for sigma_px in cfg.kernel_schedule_px:
        for _ in range(cfg.iterations_per_level):
            current = _landmark_positions_xy(params, specs, cfg)
            shifted = np.zeros_like(current)
            confidences = np.zeros(len(specs), dtype=np.float64)

            for idx, spec in enumerate(specs):
                if spec.kind == "center":
                    response = center_resp
                elif spec.kind == "capsid_ring":
                    response = ring_resp
                else:
                    response = tail_resp

                adaptive_radius = max(4, min(spec.search_radius_px, int(round(0.24 * float(params[3]) + 3.0))))
                shifted[idx], conf = _mean_shift_landmark(
                    response=response,
                    current_xy=current[idx],
                    sigma_px=sigma_px,
                    search_radius=adaptive_radius,
                )
                confidences[idx] = max(0.10, conf) * spec.weight

            displacement = (shifted - current).reshape(-1)
            jac = _numerical_jacobian(params, specs, cfg)

            lhs = np.zeros((params.size, params.size), dtype=np.float64)
            rhs = np.zeros(params.size, dtype=np.float64)
            for idx in range(len(specs)):
                j_i = jac[2 * idx:2 * idx + 2, :]
                d_i = displacement[2 * idx:2 * idx + 2]
                w_i = confidences[idx]
                lhs += w_i * (j_i.T @ j_i)
                rhs += w_i * (j_i.T @ d_i)

            reg = np.diag(prior_inv_vars + pose_anchor_inv_vars)
            lhs += reg
            rhs -= np.diag(prior_inv_vars) @ params
            rhs -= np.diag(pose_anchor_inv_vars) @ (params - params_anchor)

            try:
                delta = np.linalg.solve(lhs + 1e-6 * np.eye(lhs.shape[0]), rhs)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(lhs + 1e-6 * np.eye(lhs.shape[0]), rhs, rcond=None)[0]

            step_norm = float(np.linalg.norm(delta))
            if step_norm > cfg.max_parameter_step_norm:
                delta *= cfg.max_parameter_step_norm / step_norm

            params = params + cfg.damping * delta
            params[2] = math.atan2(math.sin(params[2]), math.cos(params[2]))
            params[3] = max(6.0, float(params[3]))
            params[4:] = np.clip(params[4:], -2.5, 2.5)

            if float(np.linalg.norm(cfg.damping * delta)) < 0.02:
                break

        if debug:
            landmark_xy = _landmark_positions_xy(params, specs, cfg)
            tail_len_px = _polyline_length(_tail_polyline_xy_from_landmarks(landmark_xy, specs))
            capsid_diam_px = _capsid_diameter_px_from_params(params, cfg)
            print(
                "[debug] sigma="
                f"{sigma_px:.2f}px tx={params[0]:.2f} ty={params[1]:.2f} "
                f"theta={math.degrees(params[2]):.2f}deg scale={params[3]:.2f}px "
                f"capsid_diam={capsid_diam_px:.2f}px tail={tail_len_px:.2f}px"
            )

    return params, tail_resp, ring_resp


def _capsid_diameter_px_from_params(params: np.ndarray, cfg: FitConfig) -> float:
    q_radius = float(params[4])
    scale_px = float(params[3])
    radius_units = max(0.55, 1.0 + cfg.capsid_radius_scale * q_radius)
    return 2.0 * radius_units * scale_px


def _tail_length_px_from_params(params: np.ndarray, cfg: FitConfig) -> float:
    q_tail = float(params[5])
    scale_px = float(params[3])
    tail_units = max(1.6, cfg.mean_tail_length_units * (1.0 + cfg.tail_length_scale * q_tail))
    return tail_units * scale_px


def _tail_polyline_xy_from_landmarks(landmark_xy: np.ndarray, specs: list[LandmarkSpec]) -> np.ndarray:
    tail_points = [point for point, spec in zip(landmark_xy, specs) if spec.kind in {"neck", "tail", "tip"}]
    if not tail_points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(tail_points, dtype=np.float64)


def _polyline_length(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 2:
        return 0.0
    diffs = np.diff(points_xy, axis=0)
    seg_lengths = np.sqrt(np.sum(diffs * diffs, axis=1))
    return float(np.sum(seg_lengths))


def _fit_circle_least_squares(points_xy: np.ndarray) -> tuple[float, float, float]:
    if points_xy.shape[0] < 3:
        raise RuntimeError("Need at least three boundary points to fit a circle.")

    a = np.column_stack(
        [
            2.0 * points_xy[:, 0],
            2.0 * points_xy[:, 1],
            np.ones(points_xy.shape[0], dtype=np.float64),
        ]
    )
    b = points_xy[:, 0] ** 2 + points_xy[:, 1] ** 2
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy, c = sol
    radius = math.sqrt(max(EPS, float(c + cx * cx + cy * cy)))
    return float(cx), float(cy), float(radius)


def _refine_capsid_circle(
    ring_resp: np.ndarray,
    center_xy: tuple[float, float],
    radius_px: float,
    theta_rad: float,
) -> tuple[float, float, float]:
    h, w = ring_resp.shape
    cx0, cy0 = center_xy
    theta_deg = math.degrees(theta_rad)

    radii = np.arange(max(6.0, 0.68 * radius_px), max(0.68 * radius_px + 1.0, 1.18 * radius_px), 0.5, dtype=np.float32)
    boundary_points: list[tuple[float, float]] = []

    for angle_deg in np.arange(0.0, 360.0, 8.0):
        delta = abs(((angle_deg - theta_deg + 180.0) % 360.0) - 180.0)
        if delta < 42.0:
            continue

        ang = math.radians(float(angle_deg))
        xs = cx0 + radii * math.cos(ang)
        ys = cy0 + radii * math.sin(ang)
        valid = (xs >= 1.0) & (xs < w - 1.0) & (ys >= 1.0) & (ys < h - 1.0)
        if np.count_nonzero(valid) < max(4, radii.size // 3):
            continue

        xs_valid = xs[valid].astype(np.float32).reshape(1, -1)
        ys_valid = ys[valid].astype(np.float32).reshape(1, -1)
        profile = cv2.remap(ring_resp.astype(np.float32), xs_valid, ys_valid, interpolation=cv2.INTER_LINEAR).reshape(-1)
        radii_valid = radii[valid].astype(np.float64)
        if profile.size == 0:
            continue

        dist_penalty = np.abs(radii_valid - radius_px) / max(radius_px, EPS)
        score = profile.astype(np.float64) - 0.08 * dist_penalty
        best_idx = int(np.argmax(score))
        if score[best_idx] < 0.12:
            continue

        best_r = float(radii_valid[best_idx])
        boundary_points.append((cx0 + best_r * math.cos(ang), cy0 + best_r * math.sin(ang)))

    if len(boundary_points) < 8:
        return float(cx0), float(cy0), float(radius_px)

    pts = np.asarray(boundary_points, dtype=np.float64)
    for _ in range(2):
        cx, cy, radius = _fit_circle_least_squares(pts)
        residual = np.abs(np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2) - radius)
        keep = residual <= max(2.0, float(np.median(residual) + 1.5 * np.std(residual)))
        if np.count_nonzero(keep) < 8:
            break
        pts = pts[keep]

    cx, cy, radius = _fit_circle_least_squares(pts)
    return float(cx), float(cy), float(radius)


def fit_phage_clm(
    image_path: Path,
    scale_nm: float = 100.0,
    bar_px_override: Optional[float] = None,
    debug: bool = False,
) -> PhageCLMResult:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if bar_px_override is not None:
        bar_px = float(bar_px_override)
        scale_bbox = None
        scale_polarity = None
    else:
        scale_bar = _find_bottom_scale_bar(gray, debug=debug)
        bar_px = float(scale_bar.length_px)
        scale_bbox = scale_bar.bbox_xywh
        scale_polarity = scale_bar.polarity

    if bar_px <= 0.0:
        raise RuntimeError("Scale bar length must be positive.")

    cfg = FitConfig()
    model = _build_phage_shape_model(cfg)
    specs = list(model.specs)
    params, _, ring_resp = _fit_clm(
        gray,
        specs,
        cfg,
        image_path=image_path,
        bar_px_override=bar_px_override,
        debug=debug,
    )

    refined_cx, refined_cy, refined_radius = _refine_capsid_circle(
        ring_resp=ring_resp,
        center_xy=(float(params[0]), float(params[1])),
        radius_px=0.5 * _capsid_diameter_px_from_params(params, cfg),
        theta_rad=float(params[2]),
    )
    params[0] = refined_cx
    params[1] = refined_cy
    radius_units = max(EPS, refined_radius / max(float(params[3]), EPS))
    params[4] = np.clip((radius_units - 1.0) / cfg.capsid_radius_scale, -2.5, 2.5)

    landmark_xy = _landmark_positions_xy(params, specs, cfg)
    tail_polyline_xy = _tail_polyline_xy_from_landmarks(landmark_xy, specs)
    geometry = FittedPhageGeometry(
        capsid_circle_xyr=(float(refined_cx), float(refined_cy), float(refined_radius)),
        tail_polyline_xy=tail_polyline_xy,
    )

    px_per_nm = float(bar_px / scale_nm)
    capsid_diameter_px = 2.0 * float(refined_radius)
    tail_length_px = _polyline_length(tail_polyline_xy)
    theta_deg = float(math.degrees(params[2]))

    result = PhageCLMResult(
        image_path=image_path,
        scale_nm=float(scale_nm),
        bar_px=float(bar_px),
        px_per_nm=px_per_nm,
        capsid_diameter_px=float(capsid_diameter_px),
        capsid_diameter_nm=float(capsid_diameter_px / px_per_nm),
        tail_length_px=float(tail_length_px),
        tail_length_nm=float(tail_length_px / px_per_nm),
        head_center_xy=(float(params[0]), float(params[1])),
        theta_deg=theta_deg,
        params=params,
        model=model,
        geometry=geometry,
        landmark_xy=landmark_xy,
        specs=specs,
        scale_bbox_xywh=scale_bbox,
        scale_polarity=scale_polarity,
    )
    return result


def render_clm_overlay(result: PhageCLMResult) -> np.ndarray:
    image_bgr = cv2.imread(str(result.image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image for overlay: {result.image_path}")

    h, w = image_bgr.shape[:2]
    thickness = max(2, int(round(0.004 * min(h, w))))

    center = None
    capsid_pts: list[tuple[int, int]] = []

    for (x, y), spec in zip(result.landmark_xy, result.specs):
        pt = (int(round(x)), int(round(y)))
        if spec.kind == "center":
            center = pt
        elif spec.kind == "capsid_ring":
            capsid_pts.append(pt)

    if capsid_pts:
        contour = np.array(capsid_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image_bgr, [contour], True, (0, 255, 255), thickness, cv2.LINE_AA)
        for pt in capsid_pts:
            cv2.circle(image_bgr, pt, max(1, thickness - 1), (0, 255, 255), -1, cv2.LINE_AA)

    # circle_x, circle_y, circle_r = result.geometry.capsid_circle_xyr
    # cv2.circle(
    #     image_bgr,
    #     (int(round(circle_x)), int(round(circle_y))),
    #     int(round(circle_r)),
    #     (255, 255, 0),
    #     thickness,
    #     cv2.LINE_AA,
    # )

    if result.geometry.tail_polyline_xy.shape[0] > 0:
        tail_pts = [
            (int(round(x)), int(round(y)))
            for x, y in result.geometry.tail_polyline_xy
        ]
        tail_arr = np.array(tail_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image_bgr, [tail_arr], False, (0, 0, 255), thickness, cv2.LINE_AA)
        if len(tail_pts) >= 2:
            cv2.circle(image_bgr, tail_pts[0], max(1, thickness), (255, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(image_bgr, tail_pts[-1], max(1, thickness), (0, 255, 0), -1, cv2.LINE_AA)

    if result.scale_bbox_xywh is not None:
        x, y, ww, hh = result.scale_bbox_xywh
        cv2.rectangle(image_bgr, (x, y), (x + ww, y + hh), SCALE_BAR_OVERLAY_COLOR_BGR, thickness, cv2.LINE_AA)

    text = [
        f"Capsid diameter: {result.capsid_diameter_nm:.2f} nm",
        f"Tail length: {result.tail_length_nm:.2f} nm",
        f"Scale bar: {result.bar_px:.1f}px = {result.scale_nm:.0f} nm",
    ]
    for idx, line in enumerate(text):
        cv2.putText(
            image_bgr,
            line,
            (12, 28 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return image_bgr


@click.group(cls=OrderedGroup, context_settings=CLICK_CONTEXT_SETTINGS)
@click.version_option(VERSION, "-v", "--version", prog_name="phagescale.py")
def cli() -> None:
    """Measure phage capsid diameter and tail length from TEM images."""


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
    """Measure from raw TEM images."""
    try:
        result = measure_phage_tail(
            image_path=str(image),
            scale_nm=scale_nm,
            bar_px_override=bar_px_override,
            debug=debug,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Capsid diameter: {2 * result.head_r * result.px_per_nm:.2f} nm")
    click.echo(f"Tail length: {result.tail_nm:.2f} nm")

    if overlay_out is not None or show_overlay:
        out_path = overlay_out if overlay_out is not None else (Path.cwd() / f"{image.stem}_tail_overlay.png")
        overlay = render_tail_overlay(str(image), result)
        cv2.imwrite(str(out_path), overlay)
        click.echo(f"Annotated image: {out_path}")
        if show_overlay:
            _show_overlay_window(overlay, f"Capsid: {2 * result.head_r * result.px_per_nm:.2f} nm, Tail: {result.tail_nm:.2f} nm")


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
    """Measure from yellow/magenta-annotated figures."""
    try:
        result = measure_annotated_tail(image_path=image, scale_nm=scale_nm)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Capsid diameter: {result.capsid_diameter_nm:.2f} nm")
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
            _show_overlay_window(overlay, f"Capsid: {result.capsid_diameter_nm:.2f} nm, Tail: {result.tail_nm:.2f} nm")


@cli.command("annotated-batch", context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--images_dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path), help="Directory containing annotated images.")
@click.option("--metadata_xlsx", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Path to the metadata workbook (.xlsx).")
@click.option("--output_xlsx", required=True, type=click.Path(dir_okay=False, path_type=Path), help="Path to the output workbook (.xlsx).")
@click.option("--sheet_name", default=None, help="Worksheet name to read. Defaults to the first sheet.")
@click.option("--image_col", default="File name", show_default=True, help="Column containing the annotated image filename.")
@click.option("--scale_col", default="Scale bar measurement (nm)", show_default=True, help="Column containing the scale bar size in nm.")
@click.option("--overlay_dir", type=click.Path(file_okay=False, path_type=Path), default=None, help="Optional directory to save overlay images for successful measurements.")
@click.option("--fail_fast", is_flag=True, help="Stop on the first error instead of recording failures in the output workbook.")
def annotated_batch_command(
    images_dir: Path,
    metadata_xlsx: Path,
    output_xlsx: Path,
    sheet_name: Optional[str],
    image_col: str,
    scale_col: str,
    overlay_dir: Optional[Path],
    fail_fast: bool,
) -> None:
    """Measure all annotated images listed in an Excel sheet."""
    try:
        success_count, failure_count = measure_annotated_batch(
            images_dir=images_dir,
            metadata_xlsx=metadata_xlsx,
            output_xlsx=output_xlsx,
            sheet_name=sheet_name,
            image_col=image_col,
            scale_col=scale_col,
            overlay_dir=overlay_dir,
            fail_fast=fail_fast,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Output workbook: {output_xlsx}")
    click.echo(f"Measured images: {success_count}")
    click.echo(f"Failed images: {failure_count}")
    if overlay_dir is not None:
        click.echo(f"Overlay directory: {overlay_dir}")


@cli.command("clm", context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--image", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Path to input image (png/jpg/tif).")
@click.option("--scale_nm", type=float, default=100.0, show_default=True, help="Scale bar value in nm.")
@click.option("--bar_px_override", type=float, default=None, help="Manual scale bar length in pixels.")
@click.option("--overlay_out", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Path to save the fitted overlay image.")
@click.option("--show_overlay", is_flag=True, help="Display the fitted overlay at the end of the run.")
@click.option("--debug", is_flag=True, help="Enable verbose debug output.")
def clm_command(
    image: Path,
    scale_nm: float,
    bar_px_override: Optional[float],
    overlay_out: Optional[Path],
    show_overlay: bool,
    debug: bool,
) -> None:
    """Measure with the fitted CLM phage model."""
    try:
        result = fit_phage_clm(
            image_path=image,
            scale_nm=scale_nm,
            bar_px_override=bar_px_override,
            debug=debug,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Capsid diameter: {result.capsid_diameter_nm:.2f} nm")
    click.echo(f"Tail length: {result.tail_length_nm:.2f} nm")

    if overlay_out is not None or show_overlay:
        out_path = overlay_out if overlay_out is not None else (Path.cwd() / f"{image.stem}_clm_overlay.png")
        overlay = render_clm_overlay(result)
        cv2.imwrite(str(out_path), overlay)
        click.echo(f"Overlay image: {out_path}")
        if show_overlay:
            _show_overlay_window(overlay, f"Capsid {result.capsid_diameter_nm:.1f} nm | Tail {result.tail_length_nm:.1f} nm")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
