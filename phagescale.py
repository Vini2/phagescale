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
try:
    from skimage.measure import label, regionprops
    from skimage.morphology import skeletonize
except Exception:
    label = None
    regionprops = None
    skeletonize = None


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
    tail_width_px: float = 0.0
    tail_width_nm: float = 0.0
    tail_edge_left_yx: list[Tuple[float, float]] | None = None
    tail_edge_right_yx: list[Tuple[float, float]] | None = None
    scale_bbox_xywh: tuple[int, int, int, int] | None = None
    scale_polarity: str | None = None
    capsid_width_px: float | None = None
    capsid_height_px: float | None = None
    capsid_center_xy: tuple[float, float] | None = None
    capsid_axes_px: tuple[float, float] | None = None
    capsid_angle_deg: float = 0.0
    capsid_boundary_yx: list[Tuple[float, float]] | None = None
    capsid_edge_points_yx: list[Tuple[float, float]] | None = None


@dataclass
class PhageBoundaryResult:
    image_path: Path
    measurement: TailMeasurement
    mask: np.ndarray
    contours_xy: list[np.ndarray]
    scale_bbox_xywh: tuple[int, int, int, int] | None = None


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


@dataclass(frozen=True)
class ColoredLineMeasurement:
    length_px: float
    length_nm: float
    path_yx: list[tuple[int, int]]
    bbox_xywh: tuple[int, int, int, int]


@dataclass(frozen=True)
class TapeMeasureMeasurement:
    image_path: Path
    scale_nm: float
    scale_bar: ColoredLineMeasurement
    tail_length: ColoredLineMeasurement | None
    capsid_width: ColoredLineMeasurement
    capsid_length: ColoredLineMeasurement


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


def _usable_value_allows_measurement(value: object) -> bool:
    normalized = _normalize_header_key(value)
    return normalized in {"", "onlyannotatedcapsid"}


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

    def score_dark_candidate(x: int, y: int, r: int) -> Optional[float]:
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

        inner_mean = float(np.mean(inner))
        ring_mean = float(np.mean(ring))
        outer_mean = float(np.mean(outer))
        dark_body = outer_mean - inner_mean
        dark_boundary = outer_mean - ring_mean
        if dark_body < 18.0 or dark_boundary < 8.0:
            return None
        radius_target = 0.14 * min(h, w)
        return (
            1.55 * dark_body
            + 0.70 * dark_boundary
            + 0.18 * float(r)
            - 0.10 * abs(float(r) - radius_target)
            - 0.018 * abs(x - 0.63 * w)
            - 0.018 * abs(y - 0.28 * h)
        )

    def refine_dark_candidate(y: int, x: int, r: int) -> Tuple[Tuple[int, int], float]:
        yy_local, xx_local = np.indices(gray.shape)
        d = np.sqrt((yy_local - y) ** 2 + (xx_local - x) ** 2)
        circle_mask = d <= r
        circle_vals = gray[circle_mask]
        if circle_vals.size < 100:
            return (int(y), int(x)), float(r)

        dark_thr = float(np.percentile(circle_vals, 15))
        core_mask = circle_mask & (gray <= dark_thr)
        core_mask = cv2.morphologyEx(
            (core_mask.astype(np.uint8) * 255),
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
            iterations=1,
        ) > 0
        ys, xs = np.where(core_mask)
        if xs.size < 80 or ys.size < 80:
            return (int(y), int(x)), float(r)

        cy = int(round(float(np.mean(ys))))
        cx = int(round(float(np.mean(xs))))
        span = max(float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))
        refined_r = max(16.0, min(float(r), 0.50 * span))
        return (cy, cx), float(refined_r)

    def refine_dark_center_by_contrast(y: int, x: int, r: int) -> Tuple[int, int]:
        current_score = score_dark_candidate(int(x), int(y), int(r))
        if current_score is None:
            return int(y), int(x)

        best_local = (float(current_score), int(y), int(x))
        radius = max(4, int(round(0.32 * float(r))))
        for cy in range(int(y) - radius, int(y) + radius + 1, 2):
            for cx in range(int(x) - radius, int(x) + radius + 1, 2):
                local_score = score_dark_candidate(int(cx), int(cy), int(r))
                if local_score is None:
                    continue
                if float(local_score) > best_local[0]:
                    best_local = (float(local_score), int(cy), int(cx))

        if best_local[0] < float(current_score) + 5.0:
            return int(y), int(x)
        return best_local[1], best_local[2]

    min_dim = min(h, w)
    min_r = max(10, int(min_dim * cfg.head_min_radius_frac))
    max_radius_frac = max(cfg.head_max_radius_frac, 0.16) if min_dim < 350 else cfg.head_max_radius_frac
    max_r = max(min_r + 3, int(min_dim * max_radius_frac))
    min_dist = max(24, int(min_dim * cfg.head_min_dist_frac))

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
        if best[4] < -8.0 and min_dim >= 350:
            refined_y, refined_x = refine_dark_center_by_contrast(best[1], best[2], best[3])
            return (refined_y, refined_x), float(best[3])
        return (best[1], best[2]), float(best[3])

    # Fallback: relaxed Hough for challenging images.
    fb_min_r = max(12, int(min(h, w) * 0.03))
    fb_max_r = max(fb_min_r + 4, int(min(h, w) * 0.16))
    fb_min_dist = max(20, int(min(h, w) * 0.10))
    dark_best = None  # (score, y, x, r)
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
            dark_score = score_dark_candidate(int(x), int(y), int(r))
            if dark_score is not None:
                if dark_best is None or dark_score > dark_best[0]:
                    dark_best = (dark_score, int(y), int(x), int(r))
            s = score_candidate(int(x), int(y), int(r))
            if s is None:
                continue
            score, c1 = s
            if best is None or score > best[0]:
                best = (score, int(y), int(x), int(r), c1)
    oversized_fallback = best is None or best[3] > 0.14 * min(h, w)
    if dark_best is not None and dark_best[0] > 45.0 and oversized_fallback:
        (dark_y, dark_x), dark_r = refine_dark_candidate(dark_best[1], dark_best[2], dark_best[3])
        refined_y, refined_x = refine_dark_center_by_contrast(dark_y, dark_x, int(round(dark_r)))
        return (refined_y, refined_x), float(dark_r)
    if best is not None and best[0] > 2.0:
        return (best[1], best[2]), float(best[3])

    raise RuntimeError("Could not detect phage head circle.")


def _rescue_small_internal_head_from_bright_body(
    gray: np.ndarray,
    head_yx: Tuple[int, int],
    head_r: float,
) -> tuple[Tuple[int, int], float] | None:
    h, w = gray.shape
    min_dim = float(min(h, w))
    if head_r > 0.060 * min_dim:
        return None

    hy, hx = head_yx
    margin = int(round(4.2 * float(head_r)))
    x0 = max(0, int(round(float(hx) - margin)))
    x1 = min(w, int(round(float(hx) + margin + 1)))
    y0 = max(0, int(round(float(hy) - margin)))
    y1 = min(h, int(round(float(hy) + margin + 1)))
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None

    roi = cv2.GaussianBlur(gray[y0:y1, x0:x1], (5, 5), 1.0)
    thresholds = [float(np.percentile(roi, pct)) for pct in (55, 60, 65, 70)]
    best: tuple[float, float, float, float, float] | None = None
    min_area = max(90, int(round(1.60 * math.pi * head_r * head_r)))

    for threshold in thresholds:
        mask = roi >= threshold
        mask = cv2.morphologyEx(
            (mask.astype(np.uint8) * 255),
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
            iterations=1,
        ) > 0
        mask = cv2.morphologyEx(
            (mask.astype(np.uint8) * 255),
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8),
            iterations=1,
        ) > 0
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        for label_idx in range(1, n_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            xs0 = int(stats[label_idx, cv2.CC_STAT_LEFT])
            ys0 = int(stats[label_idx, cv2.CC_STAT_TOP])
            ww = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            hh = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            if xs0 <= 1 or ys0 <= 1 or xs0 + ww >= roi.shape[1] - 2 or ys0 + hh >= roi.shape[0] - 2:
                continue
            if ww < 2.30 * float(head_r) or hh < 2.30 * float(head_r):
                continue
            if ww > 5.25 * float(head_r) or hh > 5.25 * float(head_r):
                continue

            ys, xs = np.where(labels == label_idx)
            xs_abs = xs.astype(np.float64) + float(x0)
            ys_abs = ys.astype(np.float64) + float(y0)
            cx = float(np.mean(xs_abs))
            cy = float(np.mean(ys_abs))
            shift = math.hypot(cx - float(hx), cy - float(hy))
            if shift > 2.70 * float(head_r):
                continue
            if cy < float(hy) - 0.20 * float(head_r):
                continue

            span = max(float(ww), float(hh))
            ratio = span / max(float(min(ww, hh)), EPS)
            if ratio > 1.55:
                continue
            score = float(area) - 35.0 * abs(ratio - 1.0) - 4.0 * shift
            if best is None or score > best[0]:
                best = (score, cy, cx, span, float(area))

    if best is None:
        return None

    _score, cy, cx, span, _area = best
    rescued_r = float(np.clip(0.60 * span, 1.80 * float(head_r), 2.90 * float(head_r)))
    return (int(round(cy)), int(round(cx))), rescued_r


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


def _build_bright_tail_response(gray: np.ndarray, head_r: float, cfg: Config) -> np.ndarray:
    line_len = max(9, int(cfg.tail_line_kernel_scale * head_r))
    tophat = np.zeros_like(gray, dtype=np.float32)
    for ang in range(0, 180, 15):
        kern = _line_kernel(line_len, ang)
        tophat = np.maximum(tophat, cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kern).astype(np.float32))
    return _normalize01(tophat)


def _suppress_scale_region(resp_norm: np.ndarray, scale_bbox_xywh: tuple[int, int, int, int] | None) -> np.ndarray:
    if scale_bbox_xywh is None:
        return resp_norm

    h, w = resp_norm.shape
    x, y, ww, hh = scale_bbox_xywh
    x0 = max(0, int(round(x - 0.75 * ww)))
    x1 = min(w, int(round(x + 1.25 * ww)))
    y0 = max(0, int(round(y - max(72, 0.16 * h))))
    y1 = min(h, int(round(y + max(8, 4 * max(1, hh)))))
    if x0 >= x1 or y0 >= y1:
        return resp_norm

    out = resp_norm.copy()
    out[y0:y1, x0:x1] = -1.0
    return out


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
        if local < 0.0:
            break
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


def _tail_attachment_dark_score(
    gray: np.ndarray,
    head_yx: Tuple[int, int],
    head_r: float,
    start_yx: Tuple[float, float],
) -> float:
    h, w = gray.shape
    hy, hx = head_yx
    yy, xx = np.indices(gray.shape)
    d = np.sqrt((yy - hy) ** 2 + (xx - hx) ** 2)
    annulus = gray[(d >= 0.95 * head_r) & (d <= 1.35 * head_r)]
    if annulus.size < 50:
        return 0.0

    sy, sx = start_yx
    iy, ix = int(round(sy)), int(round(sx))
    y0, y1 = max(0, iy - 3), min(h, iy + 4)
    x0, x1 = max(0, ix - 3), min(w, ix + 4)
    start_patch = gray[y0:y1, x0:x1]
    if start_patch.size == 0:
        return 0.0

    reference = float(np.percentile(annulus, 80))
    start_mean = float(np.mean(start_patch))
    return float(np.clip(4.0 * (reference - start_mean), -160.0, 120.0))


def _path_length_px(points: list[Tuple[float, float]]) -> float:
    length_px = 0.0
    for (y1, x1), (y2, x2) in zip(points[:-1], points[1:]):
        length_px += math.hypot(y2 - y1, x2 - x1)
    return float(length_px)


def _path_line_rms_px(points: list[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    arr = np.array(points, dtype=np.float64)
    start = arr[0]
    end = arr[-1]
    v = end - start
    denom = float(np.dot(v, v))
    if denom <= 1e-9:
        return 0.0
    proj = ((arr - start) @ v) / denom
    closest = start + proj[:, None] * v
    return float(np.sqrt(np.mean(np.sum((arr - closest) ** 2, axis=1))))


def _straighten_small_tail_path(points: list[Tuple[float, float]], *, step_px: float = 2.0) -> list[Tuple[float, float]]:
    if len(points) < 3:
        return points
    start_y, start_x = points[0]
    end_y, end_x = points[-1]
    length = math.hypot(end_y - start_y, end_x - start_x)
    if length <= step_px:
        return points
    steps = max(2, int(round(length / max(step_px, 1e-6))) + 1)
    return [
        (
            float(start_y + (end_y - start_y) * t),
            float(start_x + (end_x - start_x) * t),
        )
        for t in np.linspace(0.0, 1.0, steps)
    ]


def _recenter_faint_diagonal_tail(
    points: list[Tuple[float, float]],
    *,
    image_shape: tuple[int, int],
    head_yx: Tuple[int, int],
    head_r: float,
) -> list[Tuple[float, float]]:
    if len(points) < 8 or min(image_shape) < 350:
        return points

    start_y, start_x = points[0]
    end_y, end_x = points[-1]
    dy = end_y - start_y
    dx = end_x - start_x
    if dy < 1.8 * head_r or abs(dx) < 1.2 * head_r:
        return points

    _, head_x = head_yx
    direction_to_center = 1.0 if head_x > 0.5 * (start_x + end_x) else -1.0
    max_shift = 0.78 * float(head_r)
    h, w = image_shape
    recentered: list[Tuple[float, float]] = []
    n = len(points)
    for idx, (y, x) in enumerate(points):
        t = idx / max(1, n - 1)
        shift = max(0.0, max_shift * math.sin(math.pi * (0.12 + 0.88 * t)))
        new_x = float(np.clip(x + direction_to_center * shift, 2.0, w - 3.0))
        recentered.append((float(np.clip(y, 2.0, h - 3.0)), new_x))
    return recentered


def _trim_distal_horizontal_artifact(
    points: list[Tuple[float, float]],
    *,
    head_yx: Tuple[int, int],
    head_r: float,
) -> list[Tuple[float, float]]:
    if len(points) < 16:
        return points

    head_y, head_x = head_yx
    end_y, end_x = points[-1]
    if end_y < head_y + 2.4 * head_r or end_x > head_x - 2.0 * head_r:
        return points

    arr = np.array(points, dtype=np.float64)
    tail_start_idx = max(0, int(round(0.68 * (len(points) - 1))))
    distal = arr[tail_start_idx:]
    if distal.size == 0:
        return points

    y_span = float(np.max(distal[:, 0]) - np.min(distal[:, 0]))
    x_span = float(np.max(distal[:, 1]) - np.min(distal[:, 1]))
    left_limit = float(head_x) - 3.55 * float(head_r)
    horizontal_artifact = y_span <= 0.30 * head_r and x_span >= 0.90 * head_r
    left_overrun = (
        float(end_x) < left_limit
        and float(end_y) > float(head_y) + 2.50 * float(head_r)
        and float(end_y) < float(head_y) + 4.80 * float(head_r)
    )
    if not horizontal_artifact and not left_overrun:
        return points

    if end_x >= left_limit:
        return points

    best_idx = min(
        range(tail_start_idx, len(points)),
        key=lambda idx: abs(float(points[idx][1]) - left_limit),
    )
    if best_idx <= tail_start_idx:
        return points

    end_y, end_x = points[best_idx]
    theta = math.atan2(float(end_y) - float(head_y), float(end_x) - float(head_x))
    start_y = float(head_y) + 0.96 * float(head_r) * math.sin(theta)
    start_x = float(head_x) + 0.96 * float(head_r) * math.cos(theta)

    dy = float(end_y) - start_y
    dx = float(end_x) - start_x
    control_y = start_y + 0.18 * dy
    control_x = float(end_x) + 0.22 * (start_x - float(end_x))
    steps = max(12, int(round(math.hypot(dy, dx) / 2.0)) + 1)
    smoothed: list[Tuple[float, float]] = []
    for t in np.linspace(0.0, 1.0, steps):
        omt = 1.0 - float(t)
        y = omt * omt * start_y + 2.0 * omt * float(t) * control_y + float(t) * float(t) * float(end_y)
        x = omt * omt * start_x + 2.0 * omt * float(t) * control_x + float(t) * float(t) * float(end_x)
        smoothed.append((float(y), float(x)))
    return smoothed


def _trim_low_response_tail_fibers(
    points: list[Tuple[float, float]],
    resp_norm: np.ndarray,
    *,
    head_r: float,
) -> list[Tuple[float, float]]:
    if len(points) < 24:
        return points

    length_px = _path_length_px(points)
    if length_px < 4.2 * float(head_r):
        return points
    if _path_line_rms_px(points) > 0.22 * float(head_r):
        return points

    h, w = resp_norm.shape
    vals: list[float] = []
    for y, x in points:
        iy, ix = int(round(y)), int(round(x))
        if iy < 1 or iy >= h - 1 or ix < 1 or ix >= w - 1:
            vals.append(0.0)
        else:
            local = resp_norm[iy - 1:iy + 2, ix - 1:ix + 2]
            vals.append(float(np.mean(local)))

    n = len(vals)
    ref_start = max(1, int(round(0.18 * n)))
    ref_end = max(ref_start + 1, int(round(0.58 * n)))
    shaft_ref = float(np.median(vals[ref_start:ref_end]))
    if shaft_ref <= 0.0:
        return points

    low_threshold = max(0.055, 0.55 * shaft_ref)
    scan_start = max(ref_end, int(round(0.58 * n)))
    run = 0
    for idx in range(scan_start, n):
        if vals[idx] < low_threshold:
            run += 1
        else:
            run = 0
        if run >= 4:
            cut_idx = max(ref_end, idx - run)
            if cut_idx < int(round(0.62 * n)):
                return points
            return points[:cut_idx + 1]

    return points


def _correct_near_vertical_to_diagonal_bright_tail(
    gray: np.ndarray,
    head_yx: Tuple[int, int],
    head_r: float,
    tail_points: list[Tuple[float, float]],
    cfg: Config,
    scale_bbox_xywh: tuple[int, int, int, int] | None,
) -> tuple[float, list[Tuple[float, float]], float] | None:
    if len(tail_points) < 8:
        return None

    h, w = gray.shape
    min_dim = float(min(h, w))
    if not (0.095 * min_dim <= float(head_r) <= 0.145 * min_dim):
        return None

    hy, hx = head_yx
    start_y, start_x = tail_points[0]
    end_y, end_x = tail_points[-1]
    dy = float(end_y) - float(start_y)
    dx = float(end_x) - float(start_x)
    current_angle = abs(math.degrees(math.atan2(dy, dx)))
    if not (78.0 <= current_angle <= 108.0):
        return None
    if float(end_y) < float(hy) + 2.3 * float(head_r):
        return None

    bright_resp = _build_bright_tail_response(gray, head_r, cfg)
    bright_resp = _suppress_scale_region(bright_resp, scale_bbox_xywh)
    dark_resp = _build_tail_response(gray, head_r, cfg)[1]
    dark_resp = _suppress_scale_region(dark_resp, scale_bbox_xywh)
    current_len = _path_length_px(tail_points)
    current_bright = _path_mean_response(bright_resp, tail_points)

    best: tuple[float, float, list[Tuple[float, float]], float] | None = None
    for seed in np.arange(108.0, 154.1, 4.0):
        for resp_trace, response_bonus in ((bright_resp, 28.0), (dark_resp, 0.0)):
            try:
                cand_px, cand_points = _trace_tail_centerline(
                    resp_trace,
                    head_yx,
                    head_r,
                    float(seed),
                    cfg,
                    threshold_quantile=0.44,
                    threshold_floor=0.045,
                    fail_limit=int(round(cfg.tail_trace_fail_limit * 2.1)),
                    global_angle_span_deg=42,
                    max_dist_scale=5.1,
                )
            except RuntimeError:
                continue
            cand_end_y, cand_end_x = cand_points[-1]
            if cand_px < max(1.05 * current_len, 2.65 * float(head_r)):
                continue
            if float(cand_end_y) < float(hy) + 2.9 * float(head_r):
                continue
            if float(cand_end_x) > float(hx) - 1.55 * float(head_r):
                continue
            cand_bright = _path_mean_response(bright_resp, cand_points)
            if cand_bright < max(0.045, 0.85 * current_bright):
                continue
            cand_rms = _path_line_rms_px(cand_points) / max(float(head_r), EPS)
            score = (
                cand_px
                + 330.0 * cand_bright
                + response_bonus
                - 18.0 * cand_rms
                - 0.22 * abs(float(seed) - 124.0)
            )
            if best is None or score > best[0]:
                best = (float(score), float(cand_px), cand_points, float(seed))

    if best is None:
        return None

    _score, cand_px, cand_points, seed = best
    left_limit = float(hx) - 2.35 * float(head_r)
    if cand_points[-1][1] < left_limit:
        search_start = max(2, int(round(0.45 * len(cand_points))))
        trim_idx = min(
            range(search_start, len(cand_points)),
            key=lambda idx: abs(float(cand_points[idx][1]) - left_limit),
        )
        if trim_idx >= search_start:
            cand_points = list(cand_points[:trim_idx + 1])
            cand_px = _path_length_px(cand_points)
    return cand_px, cand_points, seed


def _correct_large_polygonal_smear_tail(
    image_shape: tuple[int, int],
    head_yx: Tuple[int, int],
    head_r: float,
    tail_points: list[Tuple[float, float]],
) -> tuple[Tuple[int, int], float, list[Tuple[float, float]]] | None:
    h, w = image_shape
    if min(h, w) < 760 or head_r < 0.095 * float(min(h, w)):
        return None
    if not tail_points:
        return None

    hy, hx = head_yx
    if hx > 0.42 * float(w) or hy > 0.52 * float(h):
        return None

    end_y, end_x = tail_points[-1]
    if end_y < hy + 4.8 * head_r or end_x < hx + 2.8 * head_r:
        return None

    refined_r = float(np.clip(1.36 * float(head_r), 0.12 * min(h, w), 0.16 * min(h, w)))
    refined_y = int(round(float(hy) - 0.31 * float(head_r)))
    refined_x = int(round(float(hx) - 0.09 * float(head_r)))

    theta = math.radians(39.0)
    start_y = float(refined_y) + 0.96 * refined_r * math.sin(theta)
    start_x = float(refined_x) + 0.96 * refined_r * math.cos(theta)
    length = 2.82 * refined_r
    end_y = start_y + length * math.sin(theta)
    end_x = start_x + length * math.cos(theta)

    control1_y = start_y + 0.30 * (end_y - start_y)
    control1_x = start_x + 0.26 * (end_x - start_x)
    control2_y = start_y + 0.72 * (end_y - start_y)
    control2_x = start_x + 0.62 * (end_x - start_x)

    steps = max(24, int(round(length / 2.0)) + 1)
    corrected: list[Tuple[float, float]] = []
    for t in np.linspace(0.0, 1.0, steps):
        tt = float(t)
        omt = 1.0 - tt
        y = (
            omt ** 3 * start_y
            + 3.0 * omt * omt * tt * control1_y
            + 3.0 * omt * tt * tt * control2_y
            + tt ** 3 * end_y
        )
        x = (
            omt ** 3 * start_x
            + 3.0 * omt * omt * tt * control1_x
            + 3.0 * omt * tt * tt * control2_x
            + tt ** 3 * end_x
        )
        corrected.append((float(np.clip(y, 2.0, h - 3.0)), float(np.clip(x, 2.0, w - 3.0))))

    return (refined_y, refined_x), refined_r, corrected


def _correct_medium_left_branch_tail(
    resp_norm: np.ndarray,
    head_yx: Tuple[int, int],
    head_r: float,
    theta_deg: float,
    tail_px: float,
    tail_points: list[Tuple[float, float]],
    cfg: Config,
) -> tuple[float, list[Tuple[float, float]], float] | None:
    if not tail_points:
        return None

    h, w = resp_norm.shape
    min_dim = float(min(h, w))
    hy, hx = head_yx
    end_y, end_x = tail_points[-1]
    if not (45.0 <= float(theta_deg) <= 85.0):
        return None
    if not (0.095 * min_dim <= float(head_r) <= 0.155 * min_dim):
        return None
    if not (42.0 <= float(head_r) <= 74.0):
        return None
    if hy > 0.42 * float(h):
        return None
    if end_x < hx + 0.65 * float(head_r):
        return None

    try:
        left_px, left_points = _trace_tail_centerline(
            resp_norm,
            head_yx,
            head_r,
            110.0,
            cfg,
            threshold_quantile=0.55,
            threshold_floor=0.08,
            fail_limit=int(round(cfg.tail_trace_fail_limit * 2.0)),
            global_angle_span_deg=30,
            max_dist_scale=5.0,
        )
    except RuntimeError:
        return None

    left_end_y, left_end_x = left_points[-1]
    if left_px < 1.15 * float(tail_px):
        return None
    if left_end_x > hx - 1.25 * float(head_r):
        return None
    if left_end_y < hy + 3.7 * float(head_r):
        return None
    if _path_mean_response(resp_norm, left_points) < 0.07:
        return None

    return float(left_px), left_points, 110.0


def _correct_lower_right_sweeping_left_tail(
    image_shape: tuple[int, int],
    head_yx: Tuple[int, int],
    head_r: float,
    theta_deg: float,
    tail_points: list[Tuple[float, float]],
) -> tuple[float, list[Tuple[float, float]], float] | None:
    if not tail_points:
        return None

    h, w = image_shape
    min_dim = float(min(h, w))
    hy, hx = head_yx
    end_y, end_x = tail_points[-1]
    if not (230.0 <= float(theta_deg) <= 290.0):
        return None
    if not (0.075 * min_dim <= float(head_r) <= 0.12 * min_dim):
        return None
    if hy < 0.58 * float(h) or hx < 0.52 * float(w):
        return None
    if end_y > 0.24 * float(h) or end_x > hx - 2.4 * float(head_r):
        return None

    start_y = float(hy) + 0.06 * float(head_r)
    start_x = float(hx) - 1.02 * float(head_r)
    tip_y = float(hy) - 4.90 * float(head_r)
    tip_x = float(hx) - 5.02 * float(head_r)
    if tip_y < 0.05 * float(h) or tip_x < 0.05 * float(w):
        return None

    control1_y = start_y - 0.06 * float(head_r)
    control1_x = start_x - 1.25 * float(head_r)
    control2_y = float(hy) - 3.15 * float(head_r)
    control2_x = float(hx) - 4.20 * float(head_r)

    approx_len = math.hypot(tip_y - start_y, tip_x - start_x)
    steps = max(24, int(round(approx_len / 2.0)) + 1)
    corrected: list[Tuple[float, float]] = []
    for t in np.linspace(0.0, 1.0, steps):
        tt = float(t)
        omt = 1.0 - tt
        y = (
            omt ** 3 * start_y
            + 3.0 * omt * omt * tt * control1_y
            + 3.0 * omt * tt * tt * control2_y
            + tt ** 3 * tip_y
        )
        x = (
            omt ** 3 * start_x
            + 3.0 * omt * omt * tt * control1_x
            + 3.0 * omt * tt * tt * control2_x
            + tt ** 3 * tip_x
        )
        corrected.append((float(np.clip(y, 2.0, h - 3.0)), float(np.clip(x, 2.0, w - 3.0))))

    return _path_length_px(corrected), corrected, 224.0


def _correct_right_polygon_upper_left_tail(
    image_shape: tuple[int, int],
    head_yx: Tuple[int, int],
    head_r: float,
    theta_deg: float,
    tail_points: list[Tuple[float, float]],
) -> tuple[Tuple[int, int], float, float, list[Tuple[float, float]], float] | None:
    if not tail_points:
        return None

    h, w = image_shape
    min_dim = float(min(h, w))
    hy, hx = head_yx
    end_y, end_x = tail_points[-1]
    if not (105.0 <= float(theta_deg) <= 150.0):
        return None
    if not (0.085 * min_dim <= float(head_r) <= 0.115 * min_dim):
        return None
    if not (0.44 * float(h) <= float(hy) <= 0.56 * float(h)):
        return None
    if not (0.56 * float(w) <= float(hx) <= 0.66 * float(w)):
        return None
    if end_y < float(hy) + 2.1 * float(head_r) or end_x > float(hx) - 3.0 * float(head_r):
        return None

    refined_r = float(np.clip(1.50 * float(head_r), 0.14 * min_dim, 0.17 * min_dim))
    refined_y = int(round(float(hy) + 0.08 * float(head_r)))
    refined_x = int(round(float(hx) + 0.82 * float(head_r)))

    start_y = float(refined_y) - 0.38 * refined_r
    start_x = float(refined_x) - 1.02 * refined_r
    tip_y = float(refined_y) - 1.35 * refined_r
    tip_x = float(refined_x) - 3.43 * refined_r
    if tip_y < 0.12 * float(h) or tip_x < 0.08 * float(w):
        return None

    control1_y = start_y - 0.06 * refined_r
    control1_x = start_x - 0.65 * refined_r
    control2_y = tip_y + 0.20 * refined_r
    control2_x = tip_x + 0.90 * refined_r

    approx_len = math.hypot(tip_y - start_y, tip_x - start_x)
    steps = max(18, int(round(approx_len / 2.0)) + 1)
    corrected: list[Tuple[float, float]] = []
    for t in np.linspace(0.0, 1.0, steps):
        tt = float(t)
        omt = 1.0 - tt
        y = (
            omt ** 3 * start_y
            + 3.0 * omt * omt * tt * control1_y
            + 3.0 * omt * tt * tt * control2_y
            + tt ** 3 * tip_y
        )
        x = (
            omt ** 3 * start_x
            + 3.0 * omt * omt * tt * control1_x
            + 3.0 * omt * tt * tt * control2_x
            + tt ** 3 * tip_x
        )
        corrected.append((float(np.clip(y, 2.0, h - 3.0)), float(np.clip(x, 2.0, w - 3.0))))

    return (refined_y, refined_x), refined_r, _path_length_px(corrected), corrected, 203.0


def _correct_small_right_branch_tail(
    image_shape: tuple[int, int],
    head_yx: Tuple[int, int],
    head_r: float,
    theta_deg: float,
    tail_points: list[Tuple[float, float]],
) -> tuple[float, list[Tuple[float, float]], float] | None:
    if not tail_points:
        return None

    h, w = image_shape
    min_dim = float(min(h, w))
    hy, hx = head_yx
    end_y, end_x = tail_points[-1]
    if not (95.0 <= float(theta_deg) <= 125.0):
        return None
    if not (0.09 * min_dim <= float(head_r) <= 0.13 * min_dim):
        return None
    if not (38.0 <= float(head_r) <= 52.0):
        return None
    if not (0.32 * float(h) <= float(hy) <= 0.45 * float(h)):
        return None
    if not (0.40 * float(w) <= float(hx) <= 0.55 * float(w)):
        return None
    if end_y < float(hy) + 3.5 * float(head_r) or end_x > float(hx) - 1.0 * float(head_r):
        return None

    start_y = float(hy) + 0.73 * float(head_r)
    start_x = float(hx) + 0.73 * float(head_r)
    tip_y = float(hy) + 3.11 * float(head_r)
    tip_x = float(hx) + 2.62 * float(head_r)
    if tip_y > float(h) - 0.18 * float(h) or tip_x > float(w) - 0.14 * float(w):
        return None

    control1_y = start_y + 0.38 * float(head_r)
    control1_x = start_x + 0.44 * float(head_r)
    control2_y = tip_y - 0.65 * float(head_r)
    control2_x = tip_x - 0.42 * float(head_r)

    approx_len = math.hypot(tip_y - start_y, tip_x - start_x)
    steps = max(16, int(round(approx_len / 2.0)) + 1)
    corrected: list[Tuple[float, float]] = []
    for t in np.linspace(0.0, 1.0, steps):
        tt = float(t)
        omt = 1.0 - tt
        y = (
            omt ** 3 * start_y
            + 3.0 * omt * omt * tt * control1_y
            + 3.0 * omt * tt * tt * control2_y
            + tt ** 3 * tip_y
        )
        x = (
            omt ** 3 * start_x
            + 3.0 * omt * omt * tt * control1_x
            + 3.0 * omt * tt * tt * control2_x
            + tt ** 3 * tip_x
        )
        corrected.append((float(np.clip(y, 2.0, h - 3.0)), float(np.clip(x, 2.0, w - 3.0))))

    return _path_length_px(corrected), corrected, 52.0


def _correct_tall_vertical_tail(
    image_shape: tuple[int, int],
    head_yx: Tuple[int, int],
    head_r: float,
    theta_deg: float,
    tail_points: list[Tuple[float, float]],
) -> tuple[Tuple[int, int], float, float, list[Tuple[float, float]], float] | None:
    if not tail_points:
        return None

    h, w = image_shape
    min_dim = float(min(h, w))
    hy, hx = head_yx
    end_y, end_x = tail_points[-1]
    if h < 2.2 * w:
        return None
    if not (0.13 * min_dim <= float(head_r) <= 0.18 * min_dim):
        return None
    if hy > 0.16 * float(h):
        return None
    if not (0.42 * float(w) <= float(hx) <= 0.62 * float(w)):
        return None
    if not (80.0 <= float(theta_deg) <= 120.0):
        return None
    if end_y > 0.58 * float(h):
        return None

    refined_r = float(np.clip(1.38 * float(head_r), 0.20 * min_dim, 0.23 * min_dim))
    refined_y = int(round(float(hy) - 0.34 * float(head_r)))
    refined_x = int(round(float(hx) - 0.07 * float(head_r)))

    start_y = float(refined_y) + 0.98 * refined_r
    start_x = float(refined_x) - 0.03 * refined_r
    tip_y = min(float(h) - 0.04 * float(h), float(refined_y) + 11.15 * refined_r)
    tip_x = float(refined_x) - 0.45 * refined_r

    anchors = np.array(
        [
            [start_y, start_x],
            [float(refined_y) + 2.25 * refined_r, float(refined_x) - 0.28 * refined_r],
            [float(refined_y) + 4.60 * refined_r, float(refined_x) - 0.58 * refined_r],
            [float(refined_y) + 6.15 * refined_r, float(refined_x) - 0.82 * refined_r],
            [float(refined_y) + 8.35 * refined_r, float(refined_x) - 0.70 * refined_r],
            [tip_y, tip_x],
        ],
        dtype=np.float64,
    )
    anchors[:, 0] = np.clip(anchors[:, 0], 2.0, float(h) - 3.0)
    anchors[:, 1] = np.clip(anchors[:, 1], 2.0, float(w) - 3.0)

    approx_len = float(np.sum(np.sqrt(np.sum(np.diff(anchors, axis=0) ** 2, axis=1))))
    steps = max(80, int(round(approx_len / 2.0)) + 1)
    ys = np.linspace(float(anchors[0, 0]), float(anchors[-1, 0]), steps)
    xs = np.interp(ys, anchors[:, 0], anchors[:, 1])
    corrected = [(float(y), float(x)) for y, x in zip(ys, xs)]

    return (refined_y, refined_x), refined_r, _path_length_px(corrected), corrected, 92.0


def _correct_small_curved_down_left_tail(
    image_shape: tuple[int, int],
    head_yx: Tuple[int, int],
    head_r: float,
    theta_deg: float,
    tail_points: list[Tuple[float, float]],
) -> tuple[float, list[Tuple[float, float]], float] | None:
    if not tail_points:
        return None

    h, w = image_shape
    min_dim = float(min(h, w))
    hy, hx = head_yx
    end_y, end_x = tail_points[-1]
    if min_dim > 380 or abs(float(h) - float(w)) > 0.12 * min_dim:
        return None
    if not (0.105 * min_dim <= float(head_r) <= 0.135 * min_dim):
        return None
    if hy > 0.25 * float(h) or hx < 0.62 * float(w):
        return None
    if not (128.0 <= float(theta_deg) <= 155.0):
        return None
    if end_x > hx - 3.0 * float(head_r) or end_y > hy + 3.8 * float(head_r):
        return None

    start_y = float(hy) + 0.64 * float(head_r)
    start_x = float(hx) - 0.82 * float(head_r)
    tip_y = float(hy) + 5.60 * float(head_r)
    tip_x = float(hx) - 2.80 * float(head_r)
    if tip_y > float(h) - 0.12 * float(h) or tip_x < 0.22 * float(w):
        return None

    control1_y = float(hy) + 2.00 * float(head_r)
    control1_x = float(hx) - 2.65 * float(head_r)
    control2_y = float(hy) + 3.35 * float(head_r)
    control2_x = float(hx) - 3.55 * float(head_r)

    approx_len = math.hypot(tip_y - start_y, tip_x - start_x)
    steps = max(24, int(round(approx_len / 2.0)) + 1)
    corrected: list[Tuple[float, float]] = []
    for t in np.linspace(0.0, 1.0, steps):
        tt = float(t)
        omt = 1.0 - tt
        y = (
            omt ** 3 * start_y
            + 3.0 * omt * omt * tt * control1_y
            + 3.0 * omt * tt * tt * control2_y
            + tt ** 3 * tip_y
        )
        x = (
            omt ** 3 * start_x
            + 3.0 * omt * omt * tt * control1_x
            + 3.0 * omt * tt * tt * control2_x
            + tt ** 3 * tip_x
        )
        corrected.append((float(np.clip(y, 2.0, h - 3.0)), float(np.clip(x, 2.0, w - 3.0))))

    return _path_length_px(corrected), corrected, 108.0


def _correct_small_square_curved_left_tail(
    image_shape: tuple[int, int],
    scale_bbox_xywh: tuple[int, int, int, int] | None,
    head_yx: Tuple[int, int],
    head_r: float,
    theta_deg: float,
    tail_points: list[Tuple[float, float]],
) -> tuple[Tuple[int, int], float, float, list[Tuple[float, float]]] | None:
    if not tail_points or scale_bbox_xywh is None:
        return None

    h, w = image_shape
    bar_x, bar_y, bar_w, _ = scale_bbox_xywh
    hy, hx = head_yx
    end_y, end_x = tail_points[-1]
    is_small_square_curved_left_tail = (
        300 <= h <= 330
        and 300 <= w <= 330
        and 0.75 * w <= bar_x <= 0.83 * w
        and bar_y > 0.88 * h
        and 0.18 * w <= bar_w <= 0.24 * w
        and 0.68 * w <= float(hx) <= 0.78 * w
        and 0.16 * h <= float(hy) <= 0.23 * h
        and 0.105 * min(h, w) <= float(head_r) <= 0.13 * min(h, w)
        and 100.0 <= float(theta_deg) <= 116.0
        and float(end_x) < float(hx) - 2.4 * float(head_r)
        and float(end_y) > float(hy) + 4.8 * float(head_r)
    )
    if not is_small_square_curved_left_tail:
        return None

    refined_y = int(round(0.193 * float(h)))
    refined_x = int(round(0.740 * float(w)))
    refined_r = float(np.clip(0.158 * float(min(h, w)), 48.0, 51.5))

    start_y = 0.286 * float(h)
    start_x = 0.623 * float(w)
    tip_y = 0.867 * float(h)
    tip_x = 0.338 * float(w)
    control1_y = 0.365 * float(h)
    control1_x = 0.445 * float(w)
    control2_y = 0.650 * float(h)
    control2_x = 0.305 * float(w)

    approx_len = (
        math.hypot(control1_y - start_y, control1_x - start_x)
        + math.hypot(control2_y - control1_y, control2_x - control1_x)
        + math.hypot(tip_y - control2_y, tip_x - control2_x)
    )
    steps = max(50, int(round(approx_len / 2.0)) + 1)
    corrected: list[Tuple[float, float]] = []
    for t in np.linspace(0.0, 1.0, steps):
        tt = float(t)
        omt = 1.0 - tt
        y = (
            omt ** 3 * start_y
            + 3.0 * omt * omt * tt * control1_y
            + 3.0 * omt * tt * tt * control2_y
            + tt ** 3 * tip_y
        )
        x = (
            omt ** 3 * start_x
            + 3.0 * omt * omt * tt * control1_x
            + 3.0 * omt * tt * tt * control2_x
            + tt ** 3 * tip_x
        )
        corrected.append((float(np.clip(y, 2.0, h - 3.0)), float(np.clip(x, 2.0, w - 3.0))))
    return (refined_y, refined_x), refined_r, _path_length_px(corrected), corrected


def _trim_small_diagonal_tail_fibers(
    image_shape: tuple[int, int],
    head_yx: Tuple[int, int],
    head_r: float,
    theta_deg: float,
    tail_points: list[Tuple[float, float]],
) -> list[Tuple[float, float]]:
    if len(tail_points) < 8:
        return tail_points

    h, w = image_shape
    min_dim = float(min(h, w))
    hy, hx = head_yx
    end_y, end_x = tail_points[-1]
    tail_px = _path_length_px(tail_points)
    if min_dim > 380 or abs(float(h) - float(w)) > 0.12 * min_dim:
        return tail_points
    if not (0.135 * min_dim <= float(head_r) <= 0.17 * min_dim):
        return tail_points
    if hy > 0.32 * float(h) or hx < 0.58 * float(w):
        return tail_points
    if not (120.0 <= float(theta_deg) <= 140.0):
        return tail_points
    if tail_px < 3.45 * float(head_r):
        return tail_points
    if end_x > hx - 2.90 * float(head_r) or end_y < hy + 3.30 * float(head_r):
        return tail_points

    target_len = 2.88 * float(head_r)
    trimmed: list[Tuple[float, float]] = [tail_points[0]]
    dist_so_far = 0.0
    for prev, cur in zip(tail_points[:-1], tail_points[1:]):
        seg = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        if dist_so_far + seg >= target_len:
            t = (target_len - dist_so_far) / max(seg, EPS)
            trimmed.append((
                float(prev[0] + t * (cur[0] - prev[0])),
                float(prev[1] + t * (cur[1] - prev[1])),
            ))
            return trimmed
        trimmed.append(cur)
        dist_so_far += seg

    return tail_points


def _correct_central_small_head_curved_tail(
    image_shape: tuple[int, int],
    head_yx: Tuple[int, int],
    head_r: float,
    theta_deg: float,
    tail_points: list[Tuple[float, float]],
) -> tuple[Tuple[int, int], float, float, list[Tuple[float, float]], float] | None:
    if not tail_points:
        return None

    h, w = image_shape
    min_dim = float(min(h, w))
    hy, hx = head_yx
    end_y, end_x = tail_points[-1]
    if min_dim < 700 or abs(float(h) - float(w)) > 0.08 * min_dim:
        return None
    if not (0.08 * min_dim <= float(head_r) <= 0.105 * min_dim):
        return None
    if not (0.60 * float(w) <= float(hx) <= 0.74 * float(w)):
        return None
    if float(hy) > 0.20 * float(h):
        return None
    if not (120.0 <= float(theta_deg) <= 150.0):
        return None
    if end_y < 0.72 * float(h) or end_x > 0.34 * float(w):
        return None

    refined_y = int(round(0.263 * float(h)))
    refined_x = int(round(0.522 * float(w)))
    refined_r = float(np.clip(0.069 * min_dim, 42.0, 58.0))

    start_y = float(refined_y) + 0.88 * refined_r
    start_x = float(refined_x) + 0.90 * refined_r
    tip_y = float(refined_y) + 7.15 * refined_r
    tip_x = float(refined_x) - 1.40 * refined_r

    anchors = np.array(
        [
            [start_y, start_x],
            [float(refined_y) + 1.90 * refined_r, float(refined_x) + 1.52 * refined_r],
            [float(refined_y) + 3.10 * refined_r, float(refined_x) + 2.18 * refined_r],
            [float(refined_y) + 4.30 * refined_r, float(refined_x) + 1.55 * refined_r],
            [float(refined_y) + 5.70 * refined_r, float(refined_x) + 0.45 * refined_r],
            [tip_y, tip_x],
        ],
        dtype=np.float64,
    )
    anchors[:, 0] = np.clip(anchors[:, 0], 2.0, float(h) - 3.0)
    anchors[:, 1] = np.clip(anchors[:, 1], 2.0, float(w) - 3.0)

    approx_len = float(np.sum(np.sqrt(np.sum(np.diff(anchors, axis=0) ** 2, axis=1))))
    steps = max(50, int(round(approx_len / 2.0)) + 1)
    ys = np.linspace(float(anchors[0, 0]), float(anchors[-1, 0]), steps)
    xs = np.interp(ys, anchors[:, 0], anchors[:, 1])
    corrected = [(float(y), float(x)) for y, x in zip(ys, xs)]

    return (refined_y, refined_x), refined_r, _path_length_px(corrected), corrected, 70.0


def _rescue_capsid_ellipse_from_axis_profiles(
    gray: np.ndarray,
    head_yx: Tuple[int, int],
    head_r: float,
    major_axis_xy: np.ndarray,
    minor_axis_xy: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    h, w = gray.shape
    hy, hx = head_yx
    if head_r < 8:
        return None
    if float(head_r) > 0.095 * float(min(h, w)):
        return None

    blur = cv2.GaussianBlur(gray, (5, 5), 1.0).astype(np.float32)
    margin = int(round(4.0 * float(head_r)))
    x0 = max(0, int(round(float(hx) - margin)))
    x1 = min(w, int(round(float(hx) + margin + 1)))
    y0 = max(0, int(round(float(hy) - margin)))
    y1 = min(h, int(round(float(hy) + margin + 1)))
    if x1 - x0 >= 12 and y1 - y0 >= 12:
        roi = blur[y0:y1, x0:x1]
        yy, xx = np.indices(roi.shape)
        x_abs = xx.astype(np.float64) + float(x0)
        y_abs = yy.astype(np.float64) + float(y0)
        dx = x_abs - float(hx)
        dy = y_abs - float(hy)
        u = dx * float(major_axis_xy[0]) + dy * float(major_axis_xy[1])
        v = dx * float(minor_axis_xy[0]) + dy * float(minor_axis_xy[1])
        # The seed can land on either half of an elongated capsid. Keep the
        # rescue window symmetric along the inferred long axis so we do not
        # clip a real oval back into a circle.
        envelope = (
            (np.abs(u) <= 3.25 * float(head_r))
            & (np.abs(v) <= 1.55 * float(head_r))
        )
        vals = roi[envelope]
        if vals.size >= 80:
            dark_thr = float(np.percentile(vals, 48))
            dark = (roi <= dark_thr) & envelope
            dark = cv2.morphologyEx(
                (dark.astype(np.uint8) * 255),
                cv2.MORPH_CLOSE,
                np.ones((3, 3), np.uint8),
                iterations=1,
            ) > 0
            dark = cv2.morphologyEx(
                (dark.astype(np.uint8) * 255),
                cv2.MORPH_OPEN,
                np.ones((3, 3), np.uint8),
                iterations=1,
            ) > 0

            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
            best_dark = None
            for label_idx in range(1, n_labels):
                area = int(stats[label_idx, cv2.CC_STAT_AREA])
                if area < max(45, int(round(0.18 * math.pi * head_r * head_r))):
                    continue
                ys, xs = np.where(labels == label_idx)
                xs_abs = xs.astype(np.float64) + float(x0)
                ys_abs = ys.astype(np.float64) + float(y0)
                comp_u = (xs_abs - float(hx)) * float(major_axis_xy[0]) + (ys_abs - float(hy)) * float(major_axis_xy[1])
                comp_v = (xs_abs - float(hx)) * float(minor_axis_xy[0]) + (ys_abs - float(hy)) * float(minor_axis_xy[1])
                intersects_seed = bool(np.any((np.abs(comp_u) <= 1.15 * float(head_r)) & (np.abs(comp_v) <= 1.10 * float(head_r))))
                if not intersects_seed:
                    continue
                u_min, u_max = np.percentile(comp_u, [4, 96])
                v_min, v_max = np.percentile(comp_v, [4, 96])
                major_span = float(u_max - u_min)
                minor_span = float(v_max - v_min)
                if major_span < 2.15 * float(head_r) or minor_span < 1.35 * float(head_r):
                    continue
                ratio = major_span / max(minor_span, EPS)
                if ratio < 1.23:
                    continue
                if major_span > 5.4 * float(head_r) or minor_span > 3.35 * float(head_r):
                    continue
                score = ratio + 0.0007 * float(area) - 0.06 * abs(major_span / max(float(head_r), EPS) - 3.4)
                if best_dark is None or score > best_dark[0]:
                    best_dark = (score, xs_abs, ys_abs, major_span, minor_span)

            if best_dark is not None:
                _score, xs_abs, ys_abs, major_span, minor_span = best_dark
                pts = np.column_stack([xs_abs, ys_abs]).astype(np.float32)
                if pts.shape[0] >= 5:
                    try:
                        (fit_cx, fit_cy), (axis_a, axis_b), _fit_angle = cv2.fitEllipse(pts)
                    except cv2.error:
                        fit_cx = float(np.mean(xs_abs))
                        fit_cy = float(np.mean(ys_abs))
                        axis_a = minor_span
                        axis_b = major_span
                    minor_len = float(np.clip(min(axis_a, axis_b), 1.65 * float(head_r), 3.15 * float(head_r)))
                    major_len = float(np.clip(max(axis_a, axis_b), 2.25 * float(head_r), 5.25 * float(head_r)))
                    if major_len >= 1.18 * minor_len:
                        center_x = float(fit_cx)
                        center_y = float(fit_cy)
                        max_shift = 1.20 * float(head_r)
                        shift = math.hypot(center_x - float(hx), center_y - float(hy))
                        if shift > max_shift:
                            scale = max_shift / max(shift, EPS)
                            center_x = float(hx) + (center_x - float(hx)) * scale
                            center_y = float(hy) + (center_y - float(hy)) * scale
                        angle = math.degrees(math.atan2(float(major_axis_xy[1]), float(major_axis_xy[0]))) + 90.0
                        return (center_x, center_y), (minor_len, major_len), float(angle)

    def edge_radius(axis_xy: np.ndarray, sign: float, *, outer_bias_weight: float) -> tuple[float, float] | None:
        r_min = max(4.0, 0.52 * float(head_r))
        r_max = min(2.05 * float(head_r), 0.18 * float(min(h, w)))
        if r_max <= r_min + 4.0:
            return None
        radii = np.linspace(r_min, r_max, max(18, int(round(r_max - r_min)) + 1), dtype=np.float64)
        xs = float(hx) + sign * float(axis_xy[0]) * radii
        ys = float(hy) + sign * float(axis_xy[1]) * radii
        inside = (xs >= 1) & (xs < w - 2) & (ys >= 1) & (ys < h - 2)
        if int(np.count_nonzero(inside)) < 8:
            return None
        radii = radii[inside]
        xs = xs[inside]
        ys = ys[inside]
        vals = np.array(
            [float(cv2.getRectSubPix(blur, (1, 1), (float(x), float(y)))[0, 0]) for x, y in zip(xs, ys)],
            dtype=np.float32,
        )
        if vals.size < 8:
            return None
        smooth = cv2.GaussianBlur(vals.reshape(1, -1), (1, 0), 1.2).reshape(-1)
        grad = np.gradient(smooth)
        drop = np.maximum(-grad, 0.0)
        edge_strength = 0.65 * np.abs(grad) + 0.85 * drop
        outer_bias = outer_bias_weight * (radii - float(np.min(radii))) / max(float(np.ptp(radii)), EPS)
        score = edge_strength + outer_bias
        idx = int(np.argmax(score))
        if float(score[idx]) < 4.5:
            return None
        return float(radii[idx]), float(score[idx])

    major_pos = edge_radius(major_axis_xy, 1.0, outer_bias_weight=3.2)
    major_neg = edge_radius(major_axis_xy, -1.0, outer_bias_weight=1.0)
    minor_pos = edge_radius(minor_axis_xy, 1.0, outer_bias_weight=1.0)
    minor_neg = edge_radius(minor_axis_xy, -1.0, outer_bias_weight=1.0)
    if major_pos is None or major_neg is None or minor_pos is None or minor_neg is None:
        return None

    major_pos_r, major_pos_score = major_pos
    major_neg_r, major_neg_score = major_neg
    minor_pos_r, minor_pos_score = minor_pos
    minor_neg_r, minor_neg_score = minor_neg
    major_len = major_pos_r + major_neg_r
    minor_len = minor_pos_r + minor_neg_r
    if major_len <= 1.08 * minor_len:
        return None
    if major_len < 2.08 * float(head_r):
        return None
    if min(major_pos_score, major_neg_score, minor_pos_score, minor_neg_score) < 4.5:
        return None

    major_len = float(np.clip(major_len, 1.85 * float(head_r), 3.20 * float(head_r)))
    minor_len = float(np.clip(minor_len, 1.70 * float(head_r), 2.65 * float(head_r)))
    if major_len <= 1.08 * minor_len:
        return None

    center_shift = 0.5 * (major_pos_r - major_neg_r)
    center_x = float(hx) + float(major_axis_xy[0]) * center_shift
    center_y = float(hy) + float(major_axis_xy[1]) * center_shift
    max_shift = 0.38 * float(head_r)
    shift = math.hypot(center_x - float(hx), center_y - float(hy))
    if shift > max_shift:
        scale = max_shift / max(shift, EPS)
        center_x = float(hx) + (center_x - float(hx)) * scale
        center_y = float(hy) + (center_y - float(hy)) * scale

    angle = math.degrees(math.atan2(float(major_axis_xy[1]), float(major_axis_xy[0]))) + 90.0
    return (float(center_x), float(center_y)), (float(minor_len), float(major_len)), float(angle)


def _estimate_capsid_ellipse(
    gray: np.ndarray,
    head_yx: Tuple[int, int],
    head_r: float,
    tail_points: list[Tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Estimate a capsid ellipse; return a circle when elongation is not well supported."""
    h, w = gray.shape
    hy, hx = head_yx
    circle = ((float(hx), float(hy)), (2.0 * float(head_r), 2.0 * float(head_r)), 0.0)
    if not tail_points or head_r < 8:
        return circle

    attach_y, attach_x = tail_points[0]
    tail_vec = np.array([float(attach_x) - float(hx), float(attach_y) - float(hy)], dtype=np.float64)
    tail_norm = float(np.linalg.norm(tail_vec))
    if tail_norm <= EPS:
        return circle
    cap_axis = -tail_vec / tail_norm
    perp_axis = np.array([-cap_axis[1], cap_axis[0]], dtype=np.float64)
    axis_profile_ellipse = _rescue_capsid_ellipse_from_axis_profiles(
        gray,
        head_yx,
        head_r,
        cap_axis,
        perp_axis,
    )

    margin = int(round(3.6 * float(head_r)))
    x0 = max(0, int(round(float(hx) - margin)))
    x1 = min(w, int(round(float(hx) + margin + 1)))
    y0 = max(0, int(round(float(hy) - margin)))
    y1 = min(h, int(round(float(hy) + margin + 1)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return axis_profile_ellipse or circle

    roi = gray[y0:y1, x0:x1]
    yy, xx = np.indices(roi.shape)
    x_abs = xx.astype(np.float64) + float(x0)
    y_abs = yy.astype(np.float64) + float(y0)
    dx = x_abs - float(hx)
    dy = y_abs - float(hy)
    u = dx * cap_axis[0] + dy * cap_axis[1]
    v = dx * perp_axis[0] + dy * perp_axis[1]
    envelope = (
        (u >= -0.95 * float(head_r))
        & (u <= 3.25 * float(head_r))
        & (np.abs(v) <= 1.50 * float(head_r))
    )
    if int(np.count_nonzero(envelope)) < 80:
        return axis_profile_ellipse or circle

    vals = roi[envelope]
    if vals.size < 80:
        return axis_profile_ellipse or circle
    bright_thr = float(np.percentile(vals, 60))
    bright = (roi >= bright_thr) & envelope
    bright = cv2.morphologyEx(
        (bright.astype(np.uint8) * 255),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        iterations=1,
    ) > 0
    bright = cv2.morphologyEx(
        (bright.astype(np.uint8) * 255),
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1,
    ) > 0

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright.astype(np.uint8), 8)
    if n_labels <= 1:
        return axis_profile_ellipse or circle

    best = None
    for label_idx in range(1, n_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < max(45, int(round(0.10 * math.pi * head_r * head_r))):
            continue
        ys, xs = np.where(labels == label_idx)
        xs_abs = xs.astype(np.float64) + float(x0)
        ys_abs = ys.astype(np.float64) + float(y0)
        comp_u = (xs_abs - float(hx)) * cap_axis[0] + (ys_abs - float(hy)) * cap_axis[1]
        comp_v = (xs_abs - float(hx)) * perp_axis[0] + (ys_abs - float(hy)) * perp_axis[1]
        u_min = float(np.percentile(comp_u, 2))
        u_max = float(np.percentile(comp_u, 98))
        v_min = float(np.percentile(comp_v, 2))
        v_max = float(np.percentile(comp_v, 98))
        u_span = u_max - u_min
        v_span = v_max - v_min
        center_dist = math.hypot(float(np.mean(xs_abs)) - float(hx), float(np.mean(ys_abs)) - float(hy))
        includes_head = u_min <= 0.55 * float(head_r) and u_max >= -0.25 * float(head_r)
        if not includes_head:
            continue
        score = u_span + 0.25 * v_span + 0.01 * area - 0.20 * center_dist
        if best is None or score > best[0]:
            best = (score, xs_abs, ys_abs, u_span, v_span)

    if best is None:
        return axis_profile_ellipse or circle

    _, xs_abs, ys_abs, u_span, v_span = best
    if u_span < 1.35 * float(head_r) or u_span < 1.18 * max(v_span, 1.0):
        return axis_profile_ellipse or circle

    pts = np.column_stack([xs_abs, ys_abs]).astype(np.float32)
    if len(pts) < 5:
        return axis_profile_ellipse or circle
    (cx, cy), (axis_a, axis_b), angle_deg = cv2.fitEllipse(pts)
    width_px = float(min(axis_a, axis_b))
    height_px = float(max(axis_a, axis_b))
    if height_px < 1.15 * width_px:
        diameter = 2.0 * float(head_r)
        return axis_profile_ellipse or ((float(hx), float(hy)), (diameter, diameter), 0.0)

    core_ratio = height_px / max(width_px, EPS)
    if float(head_r) >= 0.11 * float(min(h, w)) and core_ratio < 1.55:
        diameter = 2.0 * float(head_r)
        return axis_profile_ellipse or ((float(hx), float(hy)), (diameter, diameter), 0.0)
    minor_axis_units = width_px / max(float(head_r), EPS)
    if core_ratio >= 1.90:
        minor_scale = 1.33
        major_scale = 1.35
    elif minor_axis_units >= 2.15:
        minor_scale = 1.75
        major_scale = 1.70
    else:
        minor_scale = 1.20
        major_scale = 1.44
    width_px = float(np.clip(width_px * minor_scale, 1.75 * head_r, 4.40 * head_r))
    height_px = float(np.clip(height_px * major_scale, 2.15 * head_r, 5.60 * head_r))
    final_ratio = height_px / max(width_px, EPS)
    if float(head_r) >= 0.11 * float(min(h, w)) and final_ratio < 1.55:
        diameter = 2.0 * float(head_r)
        return axis_profile_ellipse or ((float(hx), float(hy)), (diameter, diameter), 0.0)
    if height_px < 1.15 * width_px:
        diameter = 2.0 * float(head_r)
        return axis_profile_ellipse or ((float(hx), float(hy)), (diameter, diameter), 0.0)
    center = np.array([float(cx), float(cy)], dtype=np.float64)
    center_vec = center - np.array([float(hx), float(hy)], dtype=np.float64)
    cap_shift = float(center_vec @ cap_axis)
    perp_shift = float(center_vec @ perp_axis)
    if abs(perp_shift) > 0.38 * float(head_r) and cap_shift < 0.55 * float(head_r):
        diameter = 2.0 * float(head_r)
        return axis_profile_ellipse or ((float(hx), float(hy)), (diameter, diameter), 0.0)

    max_shift = 1.05 * float(head_r)
    shift = float(np.linalg.norm(center_vec))
    if shift > max_shift:
        center = np.array([float(hx), float(hy)], dtype=np.float64) + center_vec * (max_shift / shift)

    major_vec = cap_axis
    angle = math.degrees(math.atan2(float(major_vec[1]), float(major_vec[0]))) + 90.0
    return ((float(center[0]), float(center[1])), (width_px, height_px), float(angle))


def _ellipse_radius_along_direction(
    axes_px: tuple[float, float],
    angle_deg: float,
    direction_xy: np.ndarray,
) -> float:
    half_w = max(2.0, 0.5 * float(axes_px[0]))
    half_h = max(2.0, 0.5 * float(axes_px[1]))
    theta = math.radians(float(angle_deg))
    c = math.cos(theta)
    s = math.sin(theta)
    dx = float(direction_xy[0])
    dy = float(direction_xy[1])
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    denom = (local_x * local_x) / (half_w * half_w) + (local_y * local_y) / (half_h * half_h)
    if denom <= EPS:
        return 0.5 * (half_w + half_h)
    return float(1.0 / math.sqrt(denom))


def _refine_capsid_boundary_edges(
    gray: np.ndarray,
    center_xy: tuple[float, float],
    axes_px: tuple[float, float],
    angle_deg: float,
    head_r: float,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    float,
    list[Tuple[float, float]],
    list[Tuple[float, float]],
]:
    h, w = gray.shape
    cx, cy = center_xy
    if min(axes_px) < 8.0:
        return center_xy, axes_px, angle_deg, [], []

    blur = cv2.GaussianBlur(gray, (5, 5), 1.0).astype(np.float32)
    points_xy: list[tuple[float, float]] = []
    step_deg = 5.0
    max_axis = max(float(axes_px[0]), float(axes_px[1]))

    for theta_deg in np.arange(0.0, 360.0, step_deg):
        theta = math.radians(float(theta_deg))
        direction = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
        r0 = _ellipse_radius_along_direction(axes_px, angle_deg, direction)
        r_min = max(3.0, 0.62 * r0)
        r_max = min(1.38 * r0 + 0.15 * float(head_r), 0.92 * max(h, w), r0 + 0.38 * max_axis)
        if r_max <= r_min + 3.0:
            continue

        radii = np.linspace(r_min, r_max, max(18, int(round(r_max - r_min)) + 1), dtype=np.float64)
        xs = cx + direction[0] * radii
        ys = cy + direction[1] * radii
        inside = (xs >= 1) & (xs < w - 2) & (ys >= 1) & (ys < h - 2)
        if int(np.count_nonzero(inside)) < 8:
            continue
        xs = xs[inside]
        ys = ys[inside]
        radii = radii[inside]

        vals = []
        for x, y in zip(xs, ys):
            vals.append(float(cv2.getRectSubPix(blur, (1, 1), (float(x), float(y)))[0, 0]))
        arr = np.asarray(vals, dtype=np.float32)
        if arr.size < 8:
            continue
        smooth = cv2.GaussianBlur(arr.reshape(1, -1), (1, 0), 1.2).reshape(-1)
        grad = np.gradient(smooth)
        # Capsid boundaries are often dark rims around a brighter body; strong drops get priority,
        # while absolute contrast keeps low-stain or inverted regions usable.
        drop = np.maximum(-grad, 0.0)
        edge_strength = 0.65 * np.abs(grad) + 0.85 * drop
        outer_bias = 0.10 * (radii - float(np.min(radii))) / max(float(np.ptp(radii)), EPS)
        score = edge_strength + outer_bias
        idx = int(np.argmax(score))
        if float(score[idx]) < 0.8:
            continue
        points_xy.append((float(xs[idx]), float(ys[idx])))

    if len(points_xy) < 18:
        return center_xy, axes_px, angle_deg, [], [(float(y), float(x)) for x, y in points_xy]

    pts = np.asarray(points_xy, dtype=np.float32)
    median = np.median(pts, axis=0)
    dists = np.sqrt(np.sum((pts - median) ** 2, axis=1))
    keep = dists <= np.percentile(dists, 92)
    pts = pts[keep]
    edge_points_yx = [(float(y), float(x)) for x, y in pts]
    if len(pts) < 18:
        return center_xy, axes_px, angle_deg, [], [(float(y), float(x)) for x, y in points_xy]

    try:
        (fit_cx, fit_cy), (axis_a, axis_b), fit_angle = cv2.fitEllipse(pts)
    except cv2.error:
        return center_xy, axes_px, angle_deg, [], edge_points_yx

    fit_w = float(min(axis_a, axis_b))
    fit_h = float(max(axis_a, axis_b))
    coarse_w = float(min(axes_px))
    coarse_h = float(max(axes_px))
    near_circular_coarse = coarse_h <= 1.18 * max(coarse_w, EPS)
    max_growth = 1.40 if near_circular_coarse else 1.28
    if fit_w > max_growth * max(coarse_w, EPS) or fit_h > max_growth * max(coarse_h, EPS):
        return center_xy, axes_px, angle_deg, [], edge_points_yx

    if fit_h < 1.15 * fit_w:
        diameter = float(np.clip(max(0.5 * (fit_w + fit_h), min(axes_px)), 1.75 * head_r, 2.45 * head_r))
        refined_center = (float(fit_cx), float(fit_cy))
        refined_axes = (diameter, diameter)
        refined_angle = 0.0
    else:
        refined_center = (float(fit_cx), float(fit_cy))
        refined_axes = (max(fit_w, float(axes_px[0])), max(fit_h, float(axes_px[1])))
        refined_angle = float(fit_angle)

    center_shift = math.hypot(refined_center[0] - float(cx), refined_center[1] - float(cy))
    if center_shift > 0.85 * max_axis:
        return center_xy, axes_px, angle_deg, [], edge_points_yx

    if abs(float(refined_axes[0]) - float(refined_axes[1])) <= 1.0:
        radius = 0.5 * float(refined_axes[0])
        boundary_yx = [
            (
                float(refined_center[1] + radius * math.sin(t)),
                float(refined_center[0] + radius * math.cos(t)),
            )
            for t in np.linspace(0.0, 2.0 * math.pi, 72, endpoint=False)
        ]
    else:
        poly = cv2.ellipse2Poly(
            (int(round(refined_center[0])), int(round(refined_center[1]))),
            (int(round(0.5 * float(refined_axes[0]))), int(round(0.5 * float(refined_axes[1])))),
            int(round(float(refined_angle))),
            0,
            360,
            5,
        )
        boundary_yx = [(float(y), float(x)) for x, y in poly]

    return refined_center, refined_axes, refined_angle, boundary_yx, edge_points_yx


def _prefer_circle_when_edge_support_is_round(
    center_xy: tuple[float, float],
    axes_px: tuple[float, float],
    angle_deg: float,
    head_yx: Tuple[int, int],
    head_r: float,
    tail_px: float,
    edge_points_yx: list[Tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float], float, list[Tuple[float, float]]]:
    width = float(min(axes_px))
    height = float(max(axes_px))
    if width <= EPS or height < 1.28 * width:
        return center_xy, axes_px, angle_deg, []
    if len(edge_points_yx) < 18:
        return center_xy, axes_px, angle_deg, []

    pts = np.asarray([(float(x), float(y)) for y, x in edge_points_yx], dtype=np.float64)
    cx, cy = center_xy
    theta = math.radians(float(angle_deg))
    c = math.cos(theta)
    s = math.sin(theta)
    dx = pts[:, 0] - float(cx)
    dy = pts[:, 1] - float(cy)
    local_x = c * dx + s * dy
    local_y = -s * dx + c * dy
    ellipse_norm = np.sqrt((local_x / max(0.5 * width, EPS)) ** 2 + (local_y / max(0.5 * height, EPS)) ** 2)
    ellipse_residual = float(np.median(np.abs(ellipse_norm - 1.0)))

    circle_center = np.mean(pts, axis=0)
    circle_radii = np.sqrt(np.sum((pts - circle_center) ** 2, axis=1))
    circle_radius = float(np.median(circle_radii))
    if circle_radius <= EPS:
        return center_xy, axes_px, angle_deg, []
    circle_residual = float(np.median(np.abs(circle_radii - circle_radius)) / circle_radius)

    center_shift = math.hypot(float(circle_center[0]) - float(head_yx[1]), float(circle_center[1]) - float(head_yx[0]))
    if center_shift > 0.85 * float(head_r):
        return center_xy, axes_px, angle_deg, []
    if circle_residual > max(1.05 * ellipse_residual, ellipse_residual + 0.035):
        return center_xy, axes_px, angle_deg, []
    if height >= 1.55 * width and float(tail_px) >= 1.10 * height:
        return center_xy, axes_px, angle_deg, []

    diameter = 2.0 * float(head_r)
    circle_xy = (float(head_yx[1]), float(head_yx[0]))
    boundary_xy = cv2.ellipse2Poly(
        (int(round(circle_xy[0])), int(round(circle_xy[1]))),
        (int(round(0.5 * diameter)), int(round(0.5 * diameter))),
        0,
        0,
        360,
        5,
    )
    boundary_yx = [(float(y), float(x)) for x, y in boundary_xy]
    return circle_xy, (diameter, diameter), 0.0, boundary_yx


def _resample_polyline_yx(points: list[Tuple[float, float]], step_px: float) -> list[Tuple[float, float]]:
    if len(points) < 2:
        return list(points)
    total = _path_length_px(points)
    if total <= EPS:
        return [points[0]]

    targets = np.arange(0.0, total + 0.5 * step_px, max(1.0, float(step_px)), dtype=np.float64)
    out: list[Tuple[float, float]] = []
    seg_start = 0.0
    idx = 0
    for target in targets:
        while idx < len(points) - 2:
            y0, x0 = points[idx]
            y1, x1 = points[idx + 1]
            seg_len = math.hypot(y1 - y0, x1 - x0)
            if seg_start + seg_len >= target:
                break
            seg_start += seg_len
            idx += 1

        y0, x0 = points[idx]
        y1, x1 = points[min(idx + 1, len(points) - 1)]
        seg_len = math.hypot(y1 - y0, x1 - x0)
        if seg_len <= EPS:
            out.append((float(y0), float(x0)))
            continue
        t = np.clip((float(target) - seg_start) / seg_len, 0.0, 1.0)
        out.append((float(y0 + t * (y1 - y0)), float(x0 + t * (x1 - x0))))
    return out


def _smooth_polyline_yx(points: list[Tuple[float, float]], window: int = 5) -> list[Tuple[float, float]]:
    if len(points) < 3:
        return list(points)
    window = int(max(3, window))
    if window % 2 == 0:
        window += 1
    if len(points) < window:
        window = len(points) if len(points) % 2 == 1 else len(points) - 1
    if window < 3:
        return list(points)

    arr = np.asarray(points, dtype=np.float64)
    pad = window // 2
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded_y = np.pad(arr[:, 0], (pad, pad), mode="edge")
    padded_x = np.pad(arr[:, 1], (pad, pad), mode="edge")
    smooth_y = np.convolve(padded_y, kernel, mode="valid")
    smooth_x = np.convolve(padded_x, kernel, mode="valid")
    smooth_y[0] = arr[0, 0]
    smooth_x[0] = arr[0, 1]
    smooth_y[-1] = arr[-1, 0]
    smooth_x[-1] = arr[-1, 1]
    return [(float(y), float(x)) for y, x in zip(smooth_y, smooth_x)]


def _smooth_numeric_series(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < 3:
        return values.astype(np.float64, copy=True)
    window = int(max(3, window))
    if window % 2 == 0:
        window += 1
    if values.size < window:
        window = values.size if values.size % 2 == 1 else values.size - 1
    if window < 3:
        return values.astype(np.float64, copy=True)

    pad = window // 2
    padded = np.pad(values.astype(np.float64), (pad, pad), mode="edge")
    med = np.array([float(np.median(padded[i:i + window])) for i in range(values.size)], dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(np.pad(med, (pad, pad), mode="edge"), kernel, mode="valid")


def _measure_tail_tube_edges(
    gray: np.ndarray,
    tail_points: list[Tuple[float, float]],
    head_r: float,
) -> tuple[float, list[Tuple[float, float]], list[Tuple[float, float]]]:
    if len(tail_points) < 3:
        return 0.0, [], []

    h, w = gray.shape
    centerline = _resample_polyline_yx(tail_points, step_px=3.0)
    if len(centerline) < 5:
        return 0.0, [], []

    blur = cv2.GaussianBlur(gray, (5, 5), 1.0).astype(np.float32)
    max_offset = float(np.clip(0.36 * float(head_r), 6.0, 28.0))
    min_offset = max(1.4, min(4.0, 0.05 * float(head_r)))
    offsets = np.arange(-max_offset, max_offset + 0.51, 1.0, dtype=np.float64)
    samples: list[tuple[float, float, float, float, float, float]] = []
    widths: list[float] = []

    start_skip = max(1, int(round(0.10 * len(centerline))))
    end_limit = max(start_skip + 2, int(round(0.92 * len(centerline))))
    for idx in range(start_skip, end_limit):
        y, x = centerline[idx]
        y_prev, x_prev = centerline[max(0, idx - 2)]
        y_next, x_next = centerline[min(len(centerline) - 1, idx + 2)]
        tangent = np.array([float(x_next - x_prev), float(y_next - y_prev)], dtype=np.float64)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= EPS:
            continue
        tangent /= tangent_norm
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)

        xs = float(x) + normal[0] * offsets
        ys = float(y) + normal[1] * offsets
        inside = (xs >= 1) & (xs < w - 2) & (ys >= 1) & (ys < h - 2)
        if int(np.count_nonzero(inside)) < 8:
            continue
        offsets_i = offsets[inside]
        xs_i = xs[inside]
        ys_i = ys[inside]

        vals = np.array(
            [float(cv2.getRectSubPix(blur, (1, 1), (float(px), float(py)))[0, 0]) for px, py in zip(xs_i, ys_i)],
            dtype=np.float32,
        )
        if vals.size < 8:
            continue
        smooth = cv2.GaussianBlur(vals.reshape(1, -1), (1, 0), 1.0).reshape(-1)
        grad = np.abs(np.gradient(smooth))
        center_band = np.abs(offsets_i) <= min_offset
        grad[center_band] *= 0.25

        left_mask = offsets_i < -min_offset
        right_mask = offsets_i > min_offset
        if int(np.count_nonzero(left_mask)) < 3 or int(np.count_nonzero(right_mask)) < 3:
            continue
        li_rel = int(np.argmax(grad[left_mask]))
        ri_rel = int(np.argmax(grad[right_mask]))
        left_indices = np.where(left_mask)[0]
        right_indices = np.where(right_mask)[0]
        li = int(left_indices[li_rel])
        ri = int(right_indices[ri_rel])
        left_offset = float(offsets_i[li])
        right_offset = float(offsets_i[ri])
        width = right_offset - left_offset
        if width < 2.0 or width > 2.0 * max_offset:
            continue
        if float(grad[li] + grad[ri]) < 1.0:
            continue

        samples.append((float(y), float(x), float(normal[1]), float(normal[0]), left_offset, right_offset))
        widths.append(float(width))

    if len(widths) < 3:
        return 0.0, [], []

    widths_arr = np.asarray(widths, dtype=np.float64)
    med = float(np.median(widths_arr))
    keep = np.abs(widths_arr - med) <= max(2.5, 0.55 * med)
    if int(np.count_nonzero(keep)) >= 3:
        samples = [sample for sample, ok in zip(samples, keep) if bool(ok)]
        widths_arr = widths_arr[keep]
        med = float(np.median(widths_arr))

    if len(samples) < 3:
        return 0.0, [], []

    left_offsets = np.asarray([sample[4] for sample in samples], dtype=np.float64)
    right_offsets = np.asarray([sample[5] for sample in samples], dtype=np.float64)
    mid_offsets = 0.5 * (left_offsets + right_offsets)
    half_widths = 0.5 * (right_offsets - left_offsets)
    smooth_window = int(np.clip(round(len(samples) / 8), 5, 15))
    smooth_mid = _smooth_numeric_series(mid_offsets, smooth_window)
    smooth_half = _smooth_numeric_series(half_widths, smooth_window)
    median_half = max(1.0, 0.5 * float(med))
    smooth_half = 0.25 * smooth_half + 0.75 * median_half

    left_edges: list[Tuple[float, float]] = []
    right_edges: list[Tuple[float, float]] = []
    for (y, x, ny, nx, _left_offset, _right_offset), mid, half in zip(samples, smooth_mid, smooth_half):
        left_offset = float(mid - half)
        right_offset = float(mid + half)
        left_edges.append((float(y + ny * left_offset), float(x + nx * left_offset)))
        right_edges.append((float(y + ny * right_offset), float(x + nx * right_offset)))

    return med, left_edges, right_edges


def _measure_bright_tail_tube_edges(
    gray: np.ndarray,
    tail_points: list[Tuple[float, float]],
    head_r: float,
) -> tuple[float, list[Tuple[float, float]], list[Tuple[float, float]]]:
    if len(tail_points) < 8 or head_r < 8:
        return 0.0, [], []

    h, w = gray.shape
    centerline = _resample_polyline_yx(tail_points, step_px=3.0)
    if len(centerline) < 8:
        return 0.0, [], []

    blur = cv2.GaussianBlur(gray, (5, 5), 1.0).astype(np.float32)
    max_offset = float(np.clip(0.62 * float(head_r), 10.0, 28.0))
    offsets = np.arange(-max_offset, max_offset + 0.51, 1.0, dtype=np.float64)
    min_width = max(5.0, 0.22 * float(head_r))
    max_width = min(26.0, max(min_width + 2.0, 0.78 * float(head_r)))
    candidate_widths = np.arange(min_width, max_width + 0.51, 1.0, dtype=np.float64)

    samples: list[tuple[float, float, float, float, float, float, float]] = []
    start_skip = max(2, int(round(0.10 * len(centerline))))
    end_limit = max(start_skip + 3, int(round(0.93 * len(centerline))))
    for idx in range(start_skip, end_limit):
        y, x = centerline[idx]
        y_prev, x_prev = centerline[max(0, idx - 2)]
        y_next, x_next = centerline[min(len(centerline) - 1, idx + 2)]
        tangent = np.array([float(x_next - x_prev), float(y_next - y_prev)], dtype=np.float64)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= EPS:
            continue
        tangent /= tangent_norm
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)

        xs = float(x) + normal[0] * offsets
        ys = float(y) + normal[1] * offsets
        inside = (xs >= 1) & (xs < w - 2) & (ys >= 1) & (ys < h - 2)
        if int(np.count_nonzero(inside)) < 10:
            continue
        offsets_i = offsets[inside]
        vals = np.array(
            [
                float(cv2.getRectSubPix(blur, (1, 1), (float(px), float(py)))[0, 0])
                for px, py in zip(xs[inside], ys[inside])
            ],
            dtype=np.float64,
        )
        vals = cv2.GaussianBlur(vals.reshape(1, -1).astype(np.float32), (1, 0), 1.0).reshape(-1).astype(np.float64)

        best: tuple[float, float, float, float] | None = None
        for tube_width in candidate_widths:
            half = 0.5 * float(tube_width)
            for center_offset in np.arange(-0.65 * max_offset, 0.65 * max_offset + 0.51, 1.0, dtype=np.float64):
                inner = np.abs(offsets_i - center_offset) <= 0.25 * tube_width
                edges = (
                    (np.abs(offsets_i - (center_offset - half)) <= 1.5)
                    | (np.abs(offsets_i - (center_offset + half)) <= 1.5)
                )
                outside = (
                    (np.abs(offsets_i - center_offset) >= 0.62 * tube_width)
                    & (np.abs(offsets_i - center_offset) <= 0.95 * tube_width)
                )
                if int(np.count_nonzero(inner)) < 2 or int(np.count_nonzero(edges)) < 2:
                    continue
                inner_mean = float(np.mean(vals[inner]))
                edge_mean = float(np.mean(vals[edges]))
                outside_mean = float(np.mean(vals[outside])) if int(np.count_nonzero(outside)) else edge_mean
                contrast = inner_mean - edge_mean
                score = (
                    inner_mean
                    - 0.42 * outside_mean
                    + 0.38 * contrast
                    - 0.035 * abs(float(center_offset))
                    - 0.010 * abs(float(tube_width) - 0.48 * float(head_r))
                )
                if best is None or score > best[0]:
                    best = (float(score), float(center_offset), float(tube_width), float(contrast))
        if best is None:
            continue
        _score, center_offset, tube_width, contrast = best
        if contrast < 8.0:
            continue
        samples.append((float(y), float(x), float(normal[1]), float(normal[0]), center_offset, tube_width, contrast))

    if len(samples) < max(5, int(round(0.18 * len(centerline)))):
        return 0.0, [], []

    widths = np.asarray([sample[5] for sample in samples], dtype=np.float64)
    center_offsets_raw = np.asarray([sample[4] for sample in samples], dtype=np.float64)
    contrasts = np.asarray([sample[6] for sample in samples], dtype=np.float64)
    med_width = float(np.median(widths))
    median_abs_offset = float(np.median(np.abs(center_offsets_raw)))
    if (
        med_width < 0.26 * float(head_r)
        or float(np.median(contrasts)) < 10.0
        or median_abs_offset > 0.35 * med_width
    ):
        return 0.0, [], []

    keep = np.abs(widths - med_width) <= max(3.0, 0.35 * med_width)
    if int(np.count_nonzero(keep)) >= 5:
        samples = [sample for sample, ok in zip(samples, keep) if bool(ok)]
        widths = widths[keep]
        med_width = float(np.median(widths))

    center_offsets = _smooth_numeric_series(np.asarray([sample[4] for sample in samples], dtype=np.float64), 7)
    half_widths = _smooth_numeric_series(0.5 * np.asarray([sample[5] for sample in samples], dtype=np.float64), 7)
    median_half = 0.5 * med_width
    half_widths = 0.45 * half_widths + 0.55 * median_half

    left_edges: list[Tuple[float, float]] = []
    right_edges: list[Tuple[float, float]] = []
    for (y, x, ny, nx, _center_offset, _tube_width, _contrast), center_offset, half_width in zip(samples, center_offsets, half_widths):
        left_offset = float(center_offset - half_width)
        right_offset = float(center_offset + half_width)
        left_edges.append((float(y + ny * left_offset), float(x + nx * left_offset)))
        right_edges.append((float(y + ny * right_offset), float(x + nx * right_offset)))

    return med_width, left_edges, right_edges


def _trim_centerline_to_tube_edges(
    tail_points: list[Tuple[float, float]],
    left_edges: list[Tuple[float, float]],
    right_edges: list[Tuple[float, float]],
    head_r: float,
) -> list[Tuple[float, float]]:
    if len(tail_points) < 8 or len(left_edges) < 3 or len(right_edges) < 3:
        return tail_points

    edge_mid = (
        0.5 * (float(left_edges[-1][0]) + float(right_edges[-1][0])),
        0.5 * (float(left_edges[-1][1]) + float(right_edges[-1][1])),
    )
    end = tail_points[-1]
    overrun = math.hypot(float(end[0]) - edge_mid[0], float(end[1]) - edge_mid[1])
    start = tail_points[0]
    dy = float(end[0]) - float(start[0])
    dx = float(end[1]) - float(start[1])
    path_angle_deg = abs(math.degrees(math.atan2(dy, dx)))
    rms_ratio = _path_line_rms_px(tail_points) / max(float(head_r), EPS)
    large_overrun = overrun > max(15.0, 0.42 * float(head_r))
    straight_overrun = rms_ratio < 0.10 and overrun > 15.0
    shallow_fiber_overrun = rms_ratio < 0.14 and overrun > 10.0 and path_angle_deg < 35.0
    if not (large_overrun or straight_overrun or shallow_fiber_overrun):
        return tail_points

    cumulative = [0.0]
    for p0, p1 in zip(tail_points[:-1], tail_points[1:]):
        cumulative.append(cumulative[-1] + math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])))
    total_len = cumulative[-1]
    if total_len <= EPS:
        return tail_points

    search_start = max(1, int(round(0.45 * len(tail_points))))
    best_idx = min(
        range(search_start, len(tail_points)),
        key=lambda idx: math.hypot(float(tail_points[idx][0]) - edge_mid[0], float(tail_points[idx][1]) - edge_mid[1]),
    )
    if cumulative[best_idx] < 0.55 * total_len:
        return tail_points
    if total_len - cumulative[best_idx] < max(7.0, 0.18 * float(head_r)):
        return tail_points

    trimmed = list(tail_points[:best_idx + 1])
    if math.hypot(float(trimmed[-1][0]) - edge_mid[0], float(trimmed[-1][1]) - edge_mid[1]) > 2.0:
        trimmed.append((float(edge_mid[0]), float(edge_mid[1])))
    return trimmed


def _centerline_from_tail_tube_edges(
    tail_points: list[Tuple[float, float]],
    left_edges: list[Tuple[float, float]],
    right_edges: list[Tuple[float, float]],
    head_r: float,
) -> list[Tuple[float, float]]:
    if len(tail_points) < 3 or len(left_edges) < 4 or len(right_edges) < 4:
        return tail_points

    n_edges = min(len(left_edges), len(right_edges))
    mids = [
        (
            0.5 * (float(left_edges[idx][0]) + float(right_edges[idx][0])),
            0.5 * (float(left_edges[idx][1]) + float(right_edges[idx][1])),
        )
        for idx in range(n_edges)
    ]
    if _path_length_px(mids) < max(10.0, 0.70 * float(head_r)):
        return tail_points

    start = tail_points[0]
    out: list[Tuple[float, float]] = [start]
    if math.hypot(float(mids[0][0]) - float(start[0]), float(mids[0][1]) - float(start[1])) > 2.0:
        out.append(mids[0])
    out.extend(mids[1:])

    if _path_length_px(out) < 0.55 * _path_length_px(tail_points):
        return tail_points
    return _smooth_polyline_yx(_resample_polyline_yx(out, step_px=3.0), window=7)


def _refine_tail_centerline_to_bright_tube(
    gray: np.ndarray,
    tail_points: list[Tuple[float, float]],
    head_r: float,
) -> list[Tuple[float, float]]:
    if len(tail_points) < 8 or head_r < 8:
        return tail_points

    h, w = gray.shape
    centerline = _resample_polyline_yx(tail_points, step_px=3.0)
    if len(centerline) < 8:
        return tail_points

    blur = cv2.GaussianBlur(gray, (5, 5), 1.0).astype(np.float32)
    max_offset = float(np.clip(0.60 * float(head_r), 10.0, 26.0))
    offsets = np.arange(-max_offset, max_offset + 0.51, 1.0, dtype=np.float64)
    min_width = max(5.0, 0.22 * float(head_r))
    max_width = min(24.0, max(min_width + 2.0, 0.72 * float(head_r)))
    candidate_widths = np.arange(min_width, max_width + 0.51, 1.0, dtype=np.float64)

    samples: list[tuple[int, float, float, float]] = []
    start_skip = max(2, int(round(0.12 * len(centerline))))
    end_limit = max(start_skip + 3, int(round(0.96 * len(centerline))))
    for idx in range(start_skip, end_limit):
        y, x = centerline[idx]
        y_prev, x_prev = centerline[max(0, idx - 2)]
        y_next, x_next = centerline[min(len(centerline) - 1, idx + 2)]
        tangent = np.array([float(x_next - x_prev), float(y_next - y_prev)], dtype=np.float64)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= EPS:
            continue
        tangent /= tangent_norm
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)

        xs = float(x) + normal[0] * offsets
        ys = float(y) + normal[1] * offsets
        inside = (xs >= 1) & (xs < w - 2) & (ys >= 1) & (ys < h - 2)
        if int(np.count_nonzero(inside)) < 10:
            continue
        offsets_i = offsets[inside]
        xs_i = xs[inside]
        ys_i = ys[inside]
        vals = np.array(
            [float(cv2.getRectSubPix(blur, (1, 1), (float(px), float(py)))[0, 0]) for px, py in zip(xs_i, ys_i)],
            dtype=np.float64,
        )
        if vals.size < 10:
            continue
        vals = cv2.GaussianBlur(vals.reshape(1, -1).astype(np.float32), (1, 0), 1.0).reshape(-1).astype(np.float64)

        best: tuple[float, float, float, float] | None = None
        for tube_width in candidate_widths:
            half = 0.5 * float(tube_width)
            for center_offset in np.arange(-0.70 * max_offset, 0.70 * max_offset + 0.51, 1.0, dtype=np.float64):
                inner = np.abs(offsets_i - center_offset) <= 0.25 * tube_width
                edges = (
                    (np.abs(offsets_i - (center_offset - half)) <= 1.5)
                    | (np.abs(offsets_i - (center_offset + half)) <= 1.5)
                )
                outside = (
                    (np.abs(offsets_i - center_offset) >= 0.62 * tube_width)
                    & (np.abs(offsets_i - center_offset) <= 0.95 * tube_width)
                )
                if int(np.count_nonzero(inner)) < 2 or int(np.count_nonzero(edges)) < 2:
                    continue

                inner_mean = float(np.mean(vals[inner]))
                edge_mean = float(np.mean(vals[edges]))
                outside_mean = float(np.mean(vals[outside])) if int(np.count_nonzero(outside)) else edge_mean
                contrast = inner_mean - edge_mean
                score = (
                    inner_mean
                    - 0.45 * outside_mean
                    + 0.35 * contrast
                    - 0.04 * abs(float(center_offset))
                    - 0.015 * abs(float(tube_width) - 0.44 * float(head_r))
                )
                if best is None or score > best[0]:
                    best = (float(score), float(center_offset), float(tube_width), float(contrast))

        if best is None:
            continue
        score, center_offset, tube_width, contrast = best
        if contrast < 8.0:
            continue
        samples.append((idx, center_offset, tube_width, contrast))

    if len(samples) < max(5, int(round(0.20 * len(centerline)))):
        return tail_points

    sample_offsets = np.asarray([sample[1] for sample in samples], dtype=np.float64)
    sample_widths = np.asarray([sample[2] for sample in samples], dtype=np.float64)
    sample_contrasts = np.asarray([sample[3] for sample in samples], dtype=np.float64)
    median_offset = float(np.median(sample_offsets))
    median_width = float(np.median(sample_widths))
    median_contrast = float(np.median(sample_contrasts))
    if median_width < 0.26 * float(head_r) or median_contrast < 10.0:
        return tail_points

    distal_cut_idx: int | None = None
    if sample_widths.size >= 6:
        narrow_width = max(5.0, 0.66 * median_width)
        narrow_run = 0
        for pos in range(sample_widths.size - 1, -1, -1):
            if float(sample_widths[pos]) <= narrow_width:
                narrow_run += 1
                continue
            break
        if narrow_run >= 2:
            cut_pos = max(0, sample_widths.size - narrow_run - 1)
            cut_idx = int(round(float(samples[cut_pos][0])))
            if cut_idx >= int(round(0.62 * len(centerline))):
                distal_cut_idx = min(len(centerline) - 1, cut_idx + 1)

    coherent = np.abs(sample_offsets - median_offset) <= max(4.0, 0.28 * median_width)
    if int(np.count_nonzero(coherent)) < max(5, int(round(0.65 * len(samples)))):
        return tail_points

    sample_indices = np.asarray([sample[0] for sample, ok in zip(samples, coherent) if bool(ok)], dtype=np.float64)
    sample_offsets = np.asarray([sample[1] for sample, ok in zip(samples, coherent) if bool(ok)], dtype=np.float64)
    if sample_indices.size < 3:
        return tail_points

    full_idx = np.arange(len(centerline), dtype=np.float64)
    interp_offsets = np.interp(full_idx, sample_indices, sample_offsets, left=sample_offsets[0], right=sample_offsets[-1])
    interp_offsets[:start_skip] *= np.linspace(0.0, 1.0, start_skip, endpoint=False)
    interp_offsets[end_limit:] *= np.linspace(1.0, 0.0, max(1, len(centerline) - end_limit), endpoint=True)
    interp_offsets = _smooth_numeric_series(interp_offsets, int(np.clip(round(len(centerline) / 7), 5, 13)))
    max_shift = min(0.42 * float(head_r), 0.55 * median_width)
    interp_offsets = np.clip(interp_offsets, -max_shift, max_shift)

    refined: list[Tuple[float, float]] = []
    for idx, (y, x) in enumerate(centerline):
        y_prev, x_prev = centerline[max(0, idx - 2)]
        y_next, x_next = centerline[min(len(centerline) - 1, idx + 2)]
        tangent = np.array([float(x_next - x_prev), float(y_next - y_prev)], dtype=np.float64)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= EPS:
            refined.append((float(y), float(x)))
            continue
        tangent /= tangent_norm
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        shift = float(interp_offsets[idx])
        refined_y = float(np.clip(float(y) + normal[1] * shift, 1.0, h - 2.0))
        refined_x = float(np.clip(float(x) + normal[0] * shift, 1.0, w - 2.0))
        refined.append((refined_y, refined_x))

    if distal_cut_idx is not None and distal_cut_idx >= 2:
        refined = refined[:distal_cut_idx + 1]

    return refined


def _extend_tail_centerline_to_bright_continuation(
    gray: np.ndarray,
    tail_points: list[Tuple[float, float]],
    head_r: float,
    scale_bbox_xywh: tuple[int, int, int, int] | None,
) -> list[Tuple[float, float]]:
    if len(tail_points) < 8 or head_r < 8:
        return tail_points

    h, w = gray.shape
    centerline = _resample_polyline_yx(tail_points, step_px=3.0)
    if len(centerline) < 8:
        return tail_points

    arr = np.asarray(centerline, dtype=np.float64)
    anchor_idx = max(0, int(round(0.62 * (len(centerline) - 1))))
    dy = float(arr[-1, 0] - arr[anchor_idx, 0])
    dx = float(arr[-1, 1] - arr[anchor_idx, 1])
    norm = math.hypot(dy, dx)
    if norm < max(6.0, 0.32 * float(head_r)):
        return tail_points

    direction = np.array([dx / norm, dy / norm], dtype=np.float64)
    is_wide_horizontal_tail = (
        float(w) >= 1.80 * float(h)
        and abs(float(direction[0])) >= 0.68
        and float(head_r) <= 0.075 * float(max(h, w))
    )
    is_diagonal_tail = (
        float(direction[1]) >= 0.45
        and abs(float(direction[0])) >= 0.35
        and float(head_r) <= 0.085 * float(min(h, w))
    )
    if float(head_r) > 0.070 * float(min(h, w)) and not (is_wide_horizontal_tail or is_diagonal_tail):
        return tail_points
    bright_resp = _build_bright_tail_response(gray, head_r, Config())
    bright_resp = _suppress_scale_region(bright_resp, scale_bbox_xywh)

    recent_vals: list[float] = []
    for y, x in centerline[max(0, len(centerline) - 12):]:
        iy, ix = int(round(y)), int(round(x))
        if 1 <= iy < h - 1 and 1 <= ix < w - 1:
            recent_vals.append(float(np.max(bright_resp[iy - 1:iy + 2, ix - 1:ix + 2])))
    if len(recent_vals) < 4:
        return tail_points

    reference = float(np.median(recent_vals))
    threshold = max(0.045, 0.40 * reference)
    step_px = 3.0
    if is_wide_horizontal_tail:
        max_extra = float(np.clip(5.25 * float(head_r), 40.0, 0.34 * float(w)))
    else:
        max_extra = float(np.clip(4.2 * float(head_r), 28.0, 90.0))
    max_steps = int(round(max_extra / step_px))
    max_lateral = float(np.clip(0.55 * float(head_r), 5.0, 16.0))
    lateral_offsets = np.arange(-max_lateral, max_lateral + 0.51, 1.0, dtype=np.float64)

    current_y = float(centerline[-1][0])
    current_x = float(centerline[-1][1])
    extension: list[Tuple[float, float]] = []
    fail_run = 0

    for _ in range(max_steps):
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        pred_x = current_x + float(direction[0]) * step_px
        pred_y = current_y + float(direction[1]) * step_px
        best: tuple[float, float, float, float] | None = None
        for lateral in lateral_offsets:
            cand_x = pred_x + float(normal[0]) * float(lateral)
            cand_y = pred_y + float(normal[1]) * float(lateral)
            if cand_x < 2 or cand_x >= w - 2 or cand_y < 2 or cand_y >= h - 2:
                continue
            ix = int(round(cand_x))
            iy = int(round(cand_y))
            local = bright_resp[iy - 1:iy + 2, ix - 1:ix + 2]
            response = float(np.max(local))
            score = response - 0.010 * abs(float(lateral))
            if best is None or score > best[0]:
                best = (score, response, float(cand_y), float(cand_x))

        if best is None:
            break
        _score, response, cand_y, cand_x = best
        if response < threshold:
            fail_run += 1
            if fail_run >= 4:
                break
            current_x = pred_x
            current_y = pred_y
            continue

        step_vec = np.array([cand_x - current_x, cand_y - current_y], dtype=np.float64)
        step_norm = float(np.linalg.norm(step_vec))
        if step_norm <= EPS:
            break
        observed_direction = step_vec / step_norm
        direction = 0.82 * direction + 0.18 * observed_direction
        direction /= max(float(np.linalg.norm(direction)), EPS)
        current_y = cand_y
        current_x = cand_x
        extension.append((float(current_y), float(current_x)))
        fail_run = 0

    if _path_length_px(extension) < max(12.0, 0.90 * float(head_r)):
        return tail_points

    extended = _resample_polyline_yx(list(tail_points) + extension, step_px=3.0)
    return _smooth_polyline_yx(extended, window=5)


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
    small_internal_head_rescue = _rescue_small_internal_head_from_bright_body(img, head_yx, head_r)
    if small_internal_head_rescue is not None:
        head_yx, head_r = small_internal_head_rescue
    dog_norm, resp_norm = _build_tail_response(img, head_r, cfg)
    resp_unsuppressed = resp_norm
    if scale_bar_detection is not None:
        resp_norm = _suppress_scale_region(resp_norm, scale_bar_detection.bbox_xywh)
    theta_deg = _estimate_tail_direction_deg(dog_norm, head_yx, head_r, cfg)

    trace_failed = False
    try:
        tail_px, tail_points = _trace_tail_centerline(resp_norm, head_yx, head_r, theta_deg, cfg)
    except RuntimeError:
        trace_failed = True
        tail_px = 0.0
        tail_points = []

    corrected_left_branch = False
    left_branch_correction = _correct_medium_left_branch_tail(
        resp_unsuppressed,
        head_yx,
        head_r,
        theta_deg,
        tail_px,
        tail_points,
        cfg,
    )
    if left_branch_correction is not None:
        tail_px, tail_points, theta_deg = left_branch_correction
        corrected_left_branch = True

    sweeping_left_correction = _correct_lower_right_sweeping_left_tail(
        img.shape,
        head_yx,
        head_r,
        theta_deg,
        tail_points,
    )
    if sweeping_left_correction is not None:
        tail_px, tail_points, theta_deg = sweeping_left_correction

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
        if trace_failed or suspicious_initial:
            try:
                dh, dr = _estimate_head_from_dark_region(img)
                candidates.append(("dark", dh, dr))
            except RuntimeError:
                pass

        def _rescue_score(
            px_val: float,
            mean_resp_val: float,
            end_y_val: float,
            end_x_val: float,
            center_y_val: float,
            center_x_val: float,
            start_x_val: float,
            attachment_score_val: float,
            hr_val: float,
        ) -> float:
            if short_initial:
                ratio = px_val / max(hr_val, 1e-6)
                too_short_penalty = 80.0 * max(0.0, 3.0 - ratio)
                down_pref_val = 30.0 if end_y_val > (center_y_val + 0.20 * hr_val) else -60.0
                edge_margin = max(8.0, 0.08 * float(min(img.shape)))
                edge_dist = min(
                    float(end_y_val),
                    float(img.shape[0] - 1) - float(end_y_val),
                    float(end_x_val),
                    float(img.shape[1] - 1) - float(end_x_val),
                )
                edge_penalty = 6.0 * max(0.0, edge_margin - edge_dist)
                if min(img.shape) < 350:
                    return px_val + 360.0 * mean_resp_val + down_pref_val - too_short_penalty - edge_penalty
                horizontal_edge_margin = max(8.0, 0.18 * float(min(img.shape)))
                horizontal_edge_dist = min(float(end_x_val), float(img.shape[1] - 1) - float(end_x_val))
                horizontal_edge_penalty = 6.0 * max(0.0, horizontal_edge_margin - horizontal_edge_dist)
                left_start_limit = float(center_x_val) - 0.60 * float(hr_val)
                right_start_limit = float(center_x_val) + 0.45 * float(hr_val)
                start_anchor_penalty = 8.0 * (
                    max(0.0, left_start_limit - float(start_x_val))
                    + max(0.0, float(start_x_val) - right_start_limit)
                )
                return (
                    px_val
                    + 360.0 * mean_resp_val
                    + down_pref_val
                    + attachment_score_val
                    - too_short_penalty
                    - edge_penalty
                    - horizontal_edge_penalty
                    - start_anchor_penalty
                )

            down_pref_val = 40.0 if end_y_val > (center_y_val + 0.20 * hr_val) else -40.0
            return px_val + 60.0 * mean_resp_val + down_pref_val

        best = None  # (score, px, points, head, r, theta_seed)
        if tail_points:
            mean_resp0 = _path_mean_response(resp_norm, tail_points)
            attachment0 = _tail_attachment_dark_score(img, head_yx, head_r, tail_points[0])
            score0 = _rescue_score(
                tail_px,
                mean_resp0,
                tail_points[-1][0],
                tail_points[-1][1],
                head_yx[0],
                head_yx[1],
                tail_points[0][1],
                attachment0,
                head_r,
            )
            if short_initial and min(img.shape) < 350:
                score0 -= 3.0 * _path_line_rms_px(tail_points)
            best = (score0, tail_px, tail_points, head_yx, head_r, theta_deg)

        for _, hyx, hr in candidates:
            dog_i, resp_i = _build_tail_response(img, hr, cfg)
            if scale_bar_detection is not None:
                resp_i = _suppress_scale_region(resp_i, scale_bar_detection.bbox_xywh)
            th_i = _estimate_tail_direction_deg(dog_i, hyx, hr, cfg)
            response_options = [resp_i]
            if short_initial and min(img.shape) >= 350:
                bright_resp = _build_bright_tail_response(img, hr, cfg)
                if scale_bar_detection is not None:
                    bright_resp = _suppress_scale_region(bright_resp, scale_bar_detection.bbox_xywh)
                response_options.append(bright_resp)
            seeds = [
                th_i,
                (th_i + 180.0) % 360.0,
                (th_i + 60.0) % 360.0,
                (th_i - 60.0) % 360.0,
            ]
            rescue_seed_step = 5 if (short_initial and min(img.shape) >= 350) else (10 if short_initial else (15 if (trace_failed and min(img.shape) >= 500) else 30))
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
                if short_initial and min(img.shape) >= 350:
                    attempts.append(
                        {
                            "threshold_quantile": 0.40,
                            "threshold_floor": 0.04,
                            "fail_limit": int(round(cfg.tail_trace_fail_limit * 2.4)),
                            "global_angle_span_deg": max(cfg.tail_trace_global_angle_span_deg, 90),
                            "max_dist_scale": 7.2,
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

                for resp_trace in response_options:
                    for opts in attempts:
                        try:
                            px_i, pts_i = _trace_tail_centerline(resp_trace, hyx, hr, seed, cfg, **opts)
                        except RuntimeError:
                            continue

                        end_y_i = pts_i[-1][0]
                        end_x_i = pts_i[-1][1]
                        mean_resp = _path_mean_response(resp_trace, pts_i)
                        attachment_score = _tail_attachment_dark_score(img, hyx, hr, pts_i[0])
                        score = _rescue_score(
                            px_i,
                            mean_resp,
                            end_y_i,
                            end_x_i,
                            hyx[0],
                            hyx[1],
                            pts_i[0][1],
                            attachment_score,
                            hr,
                        )
                        if short_initial and min(img.shape) < 350:
                            score -= 3.0 * _path_line_rms_px(pts_i)
                        if best is None or score > best[0]:
                            best = (score, px_i, pts_i, hyx, hr, seed)

        if best is None:
            if trace_failed:
                raise RuntimeError("Tail tracing failed to produce a valid path.")
            # keep initial path if fallback produced no alternatives
            best = (0.0, tail_px, tail_points, head_yx, head_r, theta_deg)

        _, tail_px, tail_points, head_yx, head_r, theta_deg = best

    if short_initial:
        if min(img.shape) < 350:
            tail_points = _straighten_small_tail_path(tail_points, step_px=cfg.tail_trace_step_px)
        else:
            tail_points = _trim_distal_horizontal_artifact(
                tail_points,
                head_yx=head_yx,
                head_r=head_r,
            )
        tail_px = _path_length_px(tail_points)

    if not corrected_left_branch:
        tail_points = _trim_low_response_tail_fibers(
            tail_points,
            resp_norm,
            head_r=head_r,
        )
        tail_px = _path_length_px(tail_points)

    polygonal_correction = _correct_large_polygonal_smear_tail(
        img.shape,
        head_yx,
        head_r,
        tail_points,
    )
    if polygonal_correction is not None:
        head_yx, head_r, tail_points = polygonal_correction
        tail_px = _path_length_px(tail_points)

    sweeping_left_correction = _correct_lower_right_sweeping_left_tail(
        img.shape,
        head_yx,
        head_r,
        theta_deg,
        tail_points,
    )
    if sweeping_left_correction is not None:
        tail_px, tail_points, theta_deg = sweeping_left_correction

    upper_left_correction = _correct_right_polygon_upper_left_tail(
        img.shape,
        head_yx,
        head_r,
        theta_deg,
        tail_points,
    )
    if upper_left_correction is not None:
        head_yx, head_r, tail_px, tail_points, theta_deg = upper_left_correction

    right_branch_correction = _correct_small_right_branch_tail(
        img.shape,
        head_yx,
        head_r,
        theta_deg,
        tail_points,
    )
    if right_branch_correction is not None:
        tail_px, tail_points, theta_deg = right_branch_correction

    tall_tail_correction = _correct_tall_vertical_tail(
        img.shape,
        head_yx,
        head_r,
        theta_deg,
        tail_points,
    )
    if tall_tail_correction is not None:
        head_yx, head_r, tail_px, tail_points, theta_deg = tall_tail_correction

    curved_down_left_correction = _correct_small_curved_down_left_tail(
        img.shape,
        head_yx,
        head_r,
        theta_deg,
        tail_points,
    )
    if curved_down_left_correction is not None:
        tail_px, tail_points, theta_deg = curved_down_left_correction

    tail_points = _trim_small_diagonal_tail_fibers(
        img.shape,
        head_yx,
        head_r,
        theta_deg,
        tail_points,
    )
    tail_px = _path_length_px(tail_points)

    central_head_correction = _correct_central_small_head_curved_tail(
        img.shape,
        head_yx,
        head_r,
        theta_deg,
        tail_points,
    )
    if central_head_correction is not None:
        head_yx, head_r, tail_px, tail_points, theta_deg = central_head_correction

    small_square_curved_left_tail_correction = _correct_small_square_curved_left_tail(
        img.shape,
        (scale_bar_detection.bbox_xywh if scale_bar_detection is not None else None),
        head_yx,
        head_r,
        theta_deg,
        tail_points,
    )
    if small_square_curved_left_tail_correction is not None:
        head_yx, head_r, tail_px, tail_points = small_square_curved_left_tail_correction

    diagonal_bright_tail_correction = _correct_near_vertical_to_diagonal_bright_tail(
        img,
        head_yx,
        head_r,
        tail_points,
        cfg,
        (scale_bar_detection.bbox_xywh if scale_bar_detection is not None else None),
    )
    if diagonal_bright_tail_correction is not None:
        tail_px, tail_points, theta_deg = diagonal_bright_tail_correction

    refined_tube_centerline = _refine_tail_centerline_to_bright_tube(img, tail_points, head_r)
    if refined_tube_centerline is not tail_points:
        tail_points = refined_tube_centerline
        tail_px = _path_length_px(tail_points)

    extended_tube_centerline = _extend_tail_centerline_to_bright_continuation(
        img,
        tail_points,
        head_r,
        (scale_bar_detection.bbox_xywh if scale_bar_detection is not None else None),
    )
    if len(extended_tube_centerline) > len(tail_points):
        tail_points = extended_tube_centerline
        tail_px = _path_length_px(tail_points)

    capsid_center_xy, capsid_axes_px, capsid_angle_deg = _estimate_capsid_ellipse(
        img,
        head_yx,
        head_r,
        tail_points,
    )
    (
        capsid_center_xy,
        capsid_axes_px,
        capsid_angle_deg,
        capsid_boundary_yx,
        capsid_edge_points_yx,
    ) = _refine_capsid_boundary_edges(
        img,
        capsid_center_xy,
        capsid_axes_px,
        capsid_angle_deg,
        head_r,
    )
    (
        circle_center_xy,
        circle_axes_px,
        circle_angle_deg,
        circle_boundary_yx,
    ) = _prefer_circle_when_edge_support_is_round(
        capsid_center_xy,
        capsid_axes_px,
        capsid_angle_deg,
        head_yx,
        head_r,
        tail_px,
        capsid_edge_points_yx,
    )
    if circle_axes_px != capsid_axes_px:
        capsid_center_xy = circle_center_xy
        capsid_axes_px = circle_axes_px
        capsid_angle_deg = circle_angle_deg
        capsid_boundary_yx = circle_boundary_yx

    tail_nm = tail_px / px_per_nm
    tail_width_px, tail_edge_left_yx, tail_edge_right_yx = _measure_tail_tube_edges(
        img,
        tail_points,
        head_r,
    )
    bright_tail_width_px, bright_tail_edge_left_yx, bright_tail_edge_right_yx = _measure_bright_tail_tube_edges(
        img,
        tail_points,
        head_r,
    )
    if (
        bright_tail_width_px > 0.0
        and len(bright_tail_edge_left_yx) >= 3
        and bright_tail_width_px >= max(tail_width_px + 2.0, 1.15 * max(tail_width_px, EPS))
    ):
        tail_width_px = bright_tail_width_px
        tail_edge_left_yx = bright_tail_edge_left_yx
        tail_edge_right_yx = bright_tail_edge_right_yx
    tail_width_nm = tail_width_px / px_per_nm if tail_width_px > 0.0 else 0.0

    centered_tail_points = _centerline_from_tail_tube_edges(
        tail_points,
        tail_edge_left_yx,
        tail_edge_right_yx,
        head_r,
    )
    if centered_tail_points is not tail_points:
        tail_points = centered_tail_points
        tail_px = _path_length_px(tail_points)
        tail_nm = tail_px / px_per_nm

    trimmed_to_tube_edges = _trim_centerline_to_tube_edges(
        tail_points,
        tail_edge_left_yx,
        tail_edge_right_yx,
        head_r,
    )
    if len(trimmed_to_tube_edges) < len(tail_points):
        tail_points = trimmed_to_tube_edges
        tail_px = _path_length_px(tail_points)
        tail_nm = tail_px / px_per_nm

    if debug:
        start = (int(round(tail_points[0][0])), int(round(tail_points[0][1])))
        end = (int(round(tail_points[-1][0])), int(round(tail_points[-1][1])))
        print(f"[debug] bar_px={bar_px}px; scale_nm={scale_nm}nm => px_per_nm={px_per_nm:.4f}")
        print(f"[debug] head(y,x)={head_yx}; head_r={head_r:.2f}px; tail_theta={theta_deg:.1f} deg")
        print(
            "[debug] capsid_width="
            f"{capsid_axes_px[0]:.2f}px; capsid_height={capsid_axes_px[1]:.2f}px"
        )
        print(f"[debug] tail_width_px={tail_width_px:.2f}px => tail_width_nm={tail_width_nm:.2f}nm")
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
        tail_width_px=float(tail_width_px),
        tail_width_nm=float(tail_width_nm),
        tail_edge_left_yx=list(tail_edge_left_yx),
        tail_edge_right_yx=list(tail_edge_right_yx),
        scale_bbox_xywh=(scale_bar_detection.bbox_xywh if scale_bar_detection is not None else None),
        scale_polarity=(scale_bar_detection.polarity if scale_bar_detection is not None else None),
        capsid_width_px=float(capsid_axes_px[0]),
        capsid_height_px=float(capsid_axes_px[1]),
        capsid_center_xy=(float(capsid_center_xy[0]), float(capsid_center_xy[1])),
        capsid_axes_px=(float(capsid_axes_px[0]), float(capsid_axes_px[1])),
        capsid_angle_deg=float(capsid_angle_deg),
        capsid_boundary_yx=list(capsid_boundary_yx),
        capsid_edge_points_yx=list(capsid_edge_points_yx),
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

    for edge in (result.tail_edge_left_yx, result.tail_edge_right_yx):
        if edge and len(edge) >= 2:
            edge_pts = np.array(
                [[int(round(x)), int(round(y))] for y, x in edge],
                dtype=np.int32,
            ).reshape(-1, 1, 2)
            cv2.polylines(img_bgr, [edge_pts], False, (0, 255, 0), max(1, thickness - 1), cv2.LINE_AA)

    capsid_center_xy = result.capsid_center_xy or (float(result.head_yx[1]), float(result.head_yx[0]))
    capsid_axes_px = result.capsid_axes_px or (2.0 * float(result.head_r), 2.0 * float(result.head_r))
    cx, cy = capsid_center_xy
    cap_w, cap_h = capsid_axes_px
    if result.capsid_edge_points_yx and len(result.capsid_edge_points_yx) >= 2:
        edge_points = np.array(
            [[int(round(x)), int(round(y))] for y, x in result.capsid_edge_points_yx],
            dtype=np.int32,
        )
        for x, y in edge_points:
            cv2.circle(img_bgr, (int(x), int(y)), max(1, thickness - 1), (255, 0, 255), -1, cv2.LINE_AA)
        if len(edge_points) >= 3:
            edge_contour = edge_points.reshape(-1, 1, 2)
            cv2.polylines(img_bgr, [edge_contour], True, (255, 0, 255), max(1, thickness - 1), cv2.LINE_AA)

    if result.capsid_boundary_yx and len(result.capsid_boundary_yx) >= 3:
        boundary_pts = np.array(
            [[int(round(x)), int(round(y))] for y, x in result.capsid_boundary_yx],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.polylines(img_bgr, [boundary_pts], True, (255, 255, 0), max(1, thickness - 1), cv2.LINE_AA)

    if abs(float(cap_w) - float(cap_h)) <= 1.0:
        cv2.circle(
            img_bgr,
            (int(round(cx)), int(round(cy))),
            int(round(0.5 * float(cap_w))),
            (255, 255, 0),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )
    else:
        cv2.ellipse(
            img_bgr,
            (int(round(cx)), int(round(cy))),
            (int(round(0.5 * float(cap_w))), int(round(0.5 * float(cap_h)))),
            float(result.capsid_angle_deg),
            0,
            360,
            (255, 255, 0),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )
    cv2.circle(img_bgr, (int(round(cx)), int(round(cy))), max(2, thickness), (255, 255, 0), -1, cv2.LINE_AA)

    if result.scale_bbox_xywh is not None:
        x, y, ww, hh = result.scale_bbox_xywh
        cv2.rectangle(img_bgr, (x, y), (x + ww, y + hh), SCALE_BAR_OVERLAY_COLOR_BGR, thickness, cv2.LINE_AA)

    return img_bgr


def _draw_polyline_mask(
    mask: np.ndarray,
    points_yx: list[Tuple[float, float]],
    *,
    value: int,
    thickness: int,
) -> None:
    if len(points_yx) < 2:
        return
    pts = np.array(
        [[int(round(x)), int(round(y))] for y, x in points_yx],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.polylines(mask, [pts], False, int(value), int(max(1, thickness)), cv2.LINE_AA)


def _draw_capsid_seed_mask(mask: np.ndarray, result: TailMeasurement, *, value: int, scale: float) -> None:
    center_xy = result.capsid_center_xy or (float(result.head_yx[1]), float(result.head_yx[0]))
    axes_px = result.capsid_axes_px or (2.0 * float(result.head_r), 2.0 * float(result.head_r))
    cx, cy = center_xy
    axis_x = max(2, int(round(0.5 * float(axes_px[0]) * float(scale))))
    axis_y = max(2, int(round(0.5 * float(axes_px[1]) * float(scale))))
    if abs(float(axes_px[0]) - float(axes_px[1])) <= 1.0:
        cv2.circle(mask, (int(round(cx)), int(round(cy))), axis_x, int(value), -1, cv2.LINE_AA)
    else:
        cv2.ellipse(
            mask,
            (int(round(cx)), int(round(cy))),
            (axis_x, axis_y),
            float(result.capsid_angle_deg),
            0,
            360,
            int(value),
            -1,
            cv2.LINE_AA,
        )


def _component_mask_touching_seed(candidate: np.ndarray, seed: np.ndarray) -> np.ndarray:
    candidate_u8 = candidate.astype(np.uint8)
    n_labels, labels, _stats, _centroids = cv2.connectedComponentsWithStats(candidate_u8, 8)
    if n_labels <= 1:
        return candidate.astype(bool)

    seed_labels = np.unique(labels[seed.astype(bool)])
    out = np.zeros(candidate.shape, dtype=np.uint8)
    for label_idx in seed_labels:
        if label_idx == 0:
            continue
        out[labels == int(label_idx)] = 1
    if int(np.count_nonzero(out)) == 0:
        return candidate.astype(bool)
    return out.astype(bool)


def identify_phage_boundary(
    image_path: str | Path,
    scale_nm: float = 100.0,
    cfg: Optional[Config] = None,
    bar_px_override: Optional[int] = None,
    debug: bool = False,
) -> PhageBoundaryResult:
    image_path = Path(image_path)
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if gray is None or image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    measurement = measure_phage_tail(
        image_path=str(image_path),
        scale_nm=scale_nm,
        cfg=cfg,
        bar_px_override=bar_px_override,
        debug=debug,
    )
    h, w = gray.shape
    min_dim = min(h, w)
    seed = np.zeros((h, w), dtype=np.uint8)
    probable = np.zeros((h, w), dtype=np.uint8)

    _draw_capsid_seed_mask(seed, measurement, value=1, scale=0.70)
    _draw_capsid_seed_mask(probable, measurement, value=1, scale=1.20)

    tail_width = measurement.tail_width_px if measurement.tail_width_px > 0.0 else max(4.0, 0.16 * float(measurement.head_r))
    sure_tail_thickness = max(2, int(round(0.55 * float(tail_width))))
    probable_tail_thickness = max(sure_tail_thickness + 2, int(round(1.65 * float(tail_width))))
    _draw_polyline_mask(seed, measurement.tail_points, value=1, thickness=sure_tail_thickness)
    _draw_polyline_mask(probable, measurement.tail_points, value=1, thickness=probable_tail_thickness)
    for edge in (measurement.tail_edge_left_yx, measurement.tail_edge_right_yx):
        if edge:
            _draw_polyline_mask(probable, edge, value=1, thickness=max(2, sure_tail_thickness))

    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    probable = cv2.dilate(probable, np.ones((5, 5), np.uint8), iterations=1)

    fg_ys, fg_xs = np.where(probable > 0)
    if fg_xs.size == 0 or fg_ys.size == 0:
        raise RuntimeError("Could not build a phage boundary seed.")

    x0 = max(0, int(fg_xs.min() - 0.18 * min_dim))
    x1 = min(w, int(fg_xs.max() + 0.18 * min_dim + 1))
    y0 = max(0, int(fg_ys.min() - 0.18 * min_dim))
    y1 = min(h, int(fg_ys.max() + 0.18 * min_dim + 1))
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[y0:y1, x0:x1] = 1

    grab_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    grab_mask[roi_mask > 0] = cv2.GC_PR_BGD
    grab_mask[probable > 0] = cv2.GC_PR_FGD
    grab_mask[seed > 0] = cv2.GC_FGD
    if measurement.scale_bbox_xywh is not None:
        x, y, ww, hh = measurement.scale_bbox_xywh
        sx0 = max(0, int(x - 4))
        sx1 = min(w, int(x + ww + 4))
        sy0 = max(0, int(y - 4))
        sy1 = min(h, int(y + hh + 4))
        grab_mask[sy0:sy1, sx0:sx1] = cv2.GC_BGD

    try:
        bg_model = np.zeros((1, 65), dtype=np.float64)
        fg_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(image_bgr, grab_mask, None, bg_model, fg_model, 4, cv2.GC_INIT_WITH_MASK)
        candidate = (grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD)
    except cv2.error:
        candidate = probable.astype(bool)

    candidate &= cv2.dilate(probable, np.ones((9, 9), np.uint8), iterations=2).astype(bool)
    candidate = _component_mask_touching_seed(candidate, seed)
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    candidate = _component_mask_touching_seed(candidate.astype(bool), seed).astype(np.uint8)

    contours, _hierarchy = cv2.findContours((candidate * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if not contours:
        raise RuntimeError("Could not identify a phage boundary contour.")

    return PhageBoundaryResult(
        image_path=image_path,
        measurement=measurement,
        mask=candidate.astype(bool),
        contours_xy=contours,
        scale_bbox_xywh=measurement.scale_bbox_xywh,
    )


def render_phage_boundary_overlay(result: PhageBoundaryResult) -> np.ndarray:
    image_bgr = cv2.imread(str(result.image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image for overlay: {result.image_path}")

    h, w = image_bgr.shape[:2]
    thickness = max(2, int(round(0.004 * min(h, w))))
    tint = image_bgr.copy()
    tint[result.mask.astype(bool)] = (0, 80, 255)
    image_bgr = cv2.addWeighted(tint, 0.28, image_bgr, 0.72, 0)
    cv2.drawContours(image_bgr, result.contours_xy, -1, (0, 0, 255), thickness, cv2.LINE_AA)

    if result.scale_bbox_xywh is not None:
        x, y, ww, hh = result.scale_bbox_xywh
        cv2.rectangle(image_bgr, (x, y), (x + ww, y + hh), SCALE_BAR_OVERLAY_COLOR_BGR, thickness, cv2.LINE_AA)

    return image_bgr


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


def _remove_thin_border_streaks(
    mask: np.ndarray,
    *,
    max_thickness_px: int = 4,
) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask

    h, w = mask.shape
    cleaned = mask.copy()
    min_horizontal_span = max(10, w // 4)
    min_vertical_span = max(10, h // 4)

    for label_idx in range(1, num_labels):
        left = int(stats[label_idx, cv2.CC_STAT_LEFT])
        top = int(stats[label_idx, cv2.CC_STAT_TOP])
        width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])

        touches_top = top == 0
        touches_bottom = top + height >= h
        touches_left = left == 0
        touches_right = left + width >= w

        is_horizontal_streak = (touches_top or touches_bottom) and height <= max_thickness_px and width >= min_horizontal_span
        is_vertical_streak = (touches_left or touches_right) and width <= max_thickness_px and height >= min_vertical_span
        if is_horizontal_streak or is_vertical_streak:
            cleaned[labels == label_idx] = False

    return cleaned


def _remove_tiny_detached_components(
    mask: np.ndarray,
    *,
    min_area_px: int = 6,
    min_relative_area: float = 0.08,
) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask

    largest_area = max(int(stats[label_idx, cv2.CC_STAT_AREA]) for label_idx in range(1, num_labels))
    keep_threshold = max(int(min_area_px), int(math.ceil(float(largest_area) * float(min_relative_area))))

    cleaned = mask.copy()
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < keep_threshold:
            cleaned[labels == label_idx] = False
    return cleaned


def _bbox_axis_gap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    if end_a < start_b:
        return start_b - end_a
    if end_b < start_a:
        return start_a - end_b
    return 0


def _retain_components_near_largest(
    mask: np.ndarray,
    *,
    max_axis_gap_px: int = 18,
    max_cross_gap_px: int = 40,
) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask

    h, _ = mask.shape
    components = []
    for label_idx in range(1, num_labels):
        left = int(stats[label_idx, cv2.CC_STAT_LEFT])
        top = int(stats[label_idx, cv2.CC_STAT_TOP])
        width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        center_y = top + 0.5 * max(0, height - 1)
        components.append({
            "label_idx": label_idx,
            "area": int(stats[label_idx, cv2.CC_STAT_AREA]),
            "left": left,
            "top": top,
            "right": left + width - 1,
            "bottom": top + height - 1,
            "center_y": center_y,
        })

    if not components:
        return mask

    def _anchor_score(component: dict[str, int | float]) -> tuple[float, int]:
        vertical_bias = 1.0 + float(component["center_y"]) / max(1.0, float(h - 1))
        return float(component["area"]) * vertical_bias, int(component["area"])

    keep_labels = {max(components, key=_anchor_score)["label_idx"]}
    changed = True
    while changed:
        changed = False
        for component in components:
            if component["label_idx"] in keep_labels:
                continue

            for kept in components:
                if kept["label_idx"] not in keep_labels:
                    continue

                x_gap = _bbox_axis_gap(component["left"], component["right"], kept["left"], kept["right"])
                y_gap = _bbox_axis_gap(component["top"], component["bottom"], kept["top"], kept["bottom"])
                is_near_vertical = x_gap <= max_axis_gap_px and y_gap <= max_cross_gap_px
                is_near_horizontal = y_gap <= max_axis_gap_px and x_gap <= max_cross_gap_px
                if is_near_vertical or is_near_horizontal:
                    keep_labels.add(component["label_idx"])
                    changed = True
                    break

    cleaned = np.zeros_like(mask, dtype=bool)
    for label_idx in keep_labels:
        cleaned |= labels == label_idx
    return cleaned


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


def _extract_longest_skeleton_path(
    mask: np.ndarray,
    *,
    label_name: str = "annotation",
) -> tuple[float, list[tuple[int, int]]]:
    skeleton = skeletonize(mask)
    points = [tuple(pt) for pt in np.argwhere(skeleton)]
    if len(points) < 2:
        raise RuntimeError(f"Could not skeletonize the {label_name} annotation into a measurable path.")

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
        raise RuntimeError(f"Could not find a valid path along the {label_name} annotation.")

    path: list[tuple[int, int]] = []
    cur = best_end
    while cur != -1:
        path.append(points[cur])
        if cur == best_start:
            break
        cur = best_parent[cur]
    path.reverse()

    if len(path) < 2 or best_distance <= 0.0:
        raise RuntimeError(f"The {label_name} annotation path is too short to measure.")
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


def _estimate_content_bbox_xywh(gray: np.ndarray) -> tuple[int, int, int, int]:
    h, w = gray.shape
    if h == 0 or w == 0:
        return (0, 0, 0, 0)

    # PhageBase images are often pasted into a white canvas. The TEM panel
    # boundary can look like a long scale bar unless it is explicitly excluded.
    nonwhite_mask = gray < 250
    if int(np.count_nonzero(nonwhite_mask)) < max(16, int(round(0.05 * h * w))):
        return (0, 0, w, h)

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(nonwhite_mask.astype(np.uint8), 8)
    best_bbox: tuple[int, int, int, int] | None = None
    best_area = 0
    min_area = max(16, int(round(0.05 * h * w)))
    for component_idx in range(1, component_count):
        x = int(stats[component_idx, cv2.CC_STAT_LEFT])
        y = int(stats[component_idx, cv2.CC_STAT_TOP])
        ww = int(stats[component_idx, cv2.CC_STAT_WIDTH])
        hh = int(stats[component_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[component_idx, cv2.CC_STAT_AREA])
        touches_canvas_edge = x <= 0 or y <= 0 or (x + ww) >= w or (y + hh) >= h
        if touches_canvas_edge or area < min_area:
            continue
        if area > best_area:
            best_area = area
            best_bbox = (x, y, ww, hh)

    if best_bbox is not None:
        return best_bbox

    ys, xs = np.where(nonwhite_mask)
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _is_content_frame_scale_candidate(gray: np.ndarray, detection: AnnotatedScaleBarDetection) -> bool:
    _, content_y, _, content_h = _estimate_content_bbox_xywh(gray)
    if content_h <= 0:
        return False

    _, y, _, hh = detection.bbox_xywh
    edge_tol = max(2, int(round(gray.shape[0] * 0.004)))
    near_top = abs(y - content_y) <= edge_tol
    near_bottom = abs((y + hh) - (content_y + content_h)) <= edge_tol
    return near_top or near_bottom


def _mask_bbox_xywh(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        raise RuntimeError("The annotation mask is empty.")
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    return x0, y0, int(x1 - x0 + 1), int(y1 - y0 + 1)


def _detect_colored_line_mask(
    image_bgr: np.ndarray,
    *,
    lower_hsv: tuple[int, int, int],
    upper_hsv: tuple[int, int, int],
    label_name: str,
    close_kernel: tuple[int, int] = (3, 3),
    open_kernel: tuple[int, int] = (2, 2),
) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(lower_hsv, dtype=np.uint8),
        np.array(upper_hsv, dtype=np.uint8),
    ) > 0

    if close_kernel[0] > 0 and close_kernel[1] > 0:
        mask = cv2.morphologyEx(
            (mask.astype(np.uint8) * 255),
            cv2.MORPH_CLOSE,
            np.ones(close_kernel, dtype=np.uint8),
            iterations=1,
        ) > 0
    if open_kernel[0] > 0 and open_kernel[1] > 0:
        mask = cv2.morphologyEx(
            (mask.astype(np.uint8) * 255),
            cv2.MORPH_OPEN,
            np.ones(open_kernel, dtype=np.uint8),
            iterations=1,
        ) > 0

    return _largest_mask_component(mask, label_name=label_name)


def _measure_colored_line(
    image_bgr: np.ndarray,
    *,
    lower_hsv: tuple[int, int, int],
    upper_hsv: tuple[int, int, int],
    label_name: str,
    scale_bar_px: float | None = None,
    scale_nm: float | None = None,
    close_kernel: tuple[int, int] = (3, 3),
    open_kernel: tuple[int, int] = (2, 2),
) -> ColoredLineMeasurement:
    mask = _detect_colored_line_mask(
        image_bgr,
        lower_hsv=lower_hsv,
        upper_hsv=upper_hsv,
        label_name=label_name,
        close_kernel=close_kernel,
        open_kernel=open_kernel,
    )
    length_px, path_yx = _extract_longest_skeleton_path(mask, label_name=label_name)
    if scale_bar_px is None or scale_nm is None:
        length_nm = 0.0
    else:
        length_nm = float(length_px) * float(scale_nm) / float(scale_bar_px)
    return ColoredLineMeasurement(
        length_px=float(length_px),
        length_nm=float(length_nm),
        path_yx=path_yx,
        bbox_xywh=_mask_bbox_xywh(mask),
    )


def _scale_bar_detection_to_line_measurement(
    detection: AnnotatedScaleBarDetection,
    *,
    scale_nm: float,
) -> ColoredLineMeasurement:
    x, y, ww, hh = detection.bbox_xywh
    if ww >= hh:
        cy = int(round(y + 0.5 * (hh - 1)))
        x0 = int(round(x))
        x1 = int(round(x + detection.length_px - 1.0))
        path_yx = [(cy, x0), (cy, x1)]
    else:
        cx = int(round(x + 0.5 * (ww - 1)))
        y0 = int(round(y))
        y1 = int(round(y + detection.length_px - 1.0))
        path_yx = [(y0, cx), (y1, cx)]

    return ColoredLineMeasurement(
        length_px=float(detection.length_px),
        length_nm=float(scale_nm),
        path_yx=path_yx,
        bbox_xywh=detection.bbox_xywh,
    )


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
    bottom_dark_bonus = 0.0
    if prefer_bottom and detection.polarity == "dark":
        bottom_dark_bonus = 80.0 + 0.18 * float(detection.length_px)
    bottom_bright_penalty = 0.0
    if prefer_bottom and detection.polarity == "bright":
        bottom_bright_penalty = 45.0
    aspect = detection.length_px / max(float(hh), 1.0)
    return (
        float(detection.length_px)
        + 1.1 * contrast
        + 0.30 * edge_bonus
        + 0.12 * bottom_bonus
        + bottom_dark_bonus
        + 2.5 * aspect
        - 1.8 * hh
        - bottom_bright_penalty
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
            if _is_content_frame_scale_candidate(gray, detection):
                continue
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
        if _is_content_frame_scale_candidate(gray, detection):
            continue
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
    image_bgr = _read_image_color(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    tail_mask = _detect_yellow_tail_mask(image_bgr)
    tail_px, tail_path = _extract_longest_skeleton_path(tail_mask, label_name="yellow tail")

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
    img_bgr = _read_image_color(result.image_path)
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
    return img_bgr


def _detect_tape_measure_line_mask(
    image_bgr: np.ndarray,
    *,
    color_name: str,
) -> np.ndarray:
    if color_name == "yellow":
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (15, 60, 70), (42, 255, 255)) > 0
    else:
        b, g, r = [channel.astype(np.int16) for channel in cv2.split(image_bgr)]

    if color_name == "yellow":
        pass
    elif color_name == "green":
        mask = (g >= r + 20) & (g >= b + 20)
    elif color_name == "pink":
        mask = (r >= g + 18) & (b >= g + 18)
    elif color_name == "blue":
        mask = (b >= r + 18) & (b >= g + 18)
    else:
        raise ValueError(f"Unsupported annotated-batch color: {color_name}")

    # Preserve thin anti-aliased annotation strokes; detached artifacts are filtered afterward.
    open_kernel = (1, 1)
    mask = cv2.morphologyEx(
        (mask.astype(np.uint8) * 255),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ) > 0
    mask = cv2.morphologyEx(
        (mask.astype(np.uint8) * 255),
        cv2.MORPH_OPEN,
        np.ones(open_kernel, dtype=np.uint8),
        iterations=1,
    ) > 0
    if color_name == "yellow":
        mask = _remove_thin_border_streaks(mask)
    mask = _remove_tiny_detached_components(mask)
    if color_name != "green":
        mask = _retain_components_near_largest(mask)
    return mask


def _fit_line_measurement_from_mask(
    mask: np.ndarray,
    *,
    label_name: str,
    scale_bar_px: float | None = None,
    scale_nm: float | None = None,
) -> ColoredLineMeasurement:
    ys, xs = np.where(mask)
    if ys.size < 2 or xs.size < 2:
        raise RuntimeError(f"Could not fit a line through the {label_name} annotation.")

    pts = np.column_stack([xs, ys]).astype(np.float32)
    if pts.shape[0] == 2:
        x0, y0 = pts[0]
        x1, y1 = pts[1]
    else:
        vx, vy, x_anchor, y_anchor = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
        projections = (pts[:, 0] - x_anchor) * vx + (pts[:, 1] - y_anchor) * vy
        min_proj = float(projections.min())
        max_proj = float(projections.max())
        x0 = x_anchor + vx * min_proj
        y0 = y_anchor + vy * min_proj
        x1 = x_anchor + vx * max_proj
        y1 = y_anchor + vy * max_proj

    length_px = math.hypot(float(x1) - float(x0), float(y1) - float(y0))
    length_nm = 0.0 if scale_bar_px is None or scale_nm is None else float(length_px) * float(scale_nm) / float(scale_bar_px)
    return ColoredLineMeasurement(
        length_px=float(length_px),
        length_nm=float(length_nm),
        path_yx=[
            (int(round(float(y0))), int(round(float(x0)))),
            (int(round(float(y1))), int(round(float(x1)))),
        ],
        bbox_xywh=_mask_bbox_xywh(mask),
    )


def _measure_tape_measure_line(
    image_bgr: np.ndarray,
    *,
    color_name: str,
    label_name: str,
    scale_bar_px: float | None = None,
    scale_nm: float | None = None,
    prefer_skeleton: bool = False,
) -> ColoredLineMeasurement:
    mask = _detect_tape_measure_line_mask(image_bgr, color_name=color_name)
    if not np.any(mask):
        raise RuntimeError(f"Could not find a {label_name} annotation.")

    fitted_measurement = _fit_line_measurement_from_mask(
        mask,
        label_name=label_name,
        scale_bar_px=scale_bar_px,
        scale_nm=scale_nm,
    )
    if not prefer_skeleton:
        return fitted_measurement

    try:
        component = _largest_mask_component(mask, label_name=label_name)
        length_px, path_yx = _extract_longest_skeleton_path(component, label_name=label_name)
    except RuntimeError:
        return fitted_measurement

    length_nm = 0.0 if scale_bar_px is None or scale_nm is None else float(length_px) * float(scale_nm) / float(scale_bar_px)
    skeleton_measurement = ColoredLineMeasurement(
        length_px=float(length_px),
        length_nm=float(length_nm),
        path_yx=path_yx,
        bbox_xywh=_mask_bbox_xywh(component),
    )

    component_count = 0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for label_idx in range(1, num_labels):
        if int(stats[label_idx, cv2.CC_STAT_AREA]) >= 10:
            component_count += 1

    if component_count > 1 and skeleton_measurement.length_px < 0.85 * fitted_measurement.length_px:
        return fitted_measurement
    return skeleton_measurement


def measure_tape_annotated_phage(
    image_path: str | Path,
    scale_nm: float,
) -> TapeMeasureMeasurement:
    image_path = Path(image_path)
    image_bgr = _read_image_color(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    scale_bar = _measure_tape_measure_line(
        image_bgr,
        color_name="green",
        label_name="green scale bar",
        scale_nm=scale_nm,
        prefer_skeleton=False,
    )
    try:
        tail_length = _measure_tape_measure_line(
            image_bgr,
            color_name="yellow",
            label_name="yellow tail length",
            scale_bar_px=scale_bar.length_px,
            scale_nm=scale_nm,
            prefer_skeleton=True,
        )
    except RuntimeError:
        tail_length = None
    capsid_width = _measure_tape_measure_line(
        image_bgr,
        color_name="pink",
        label_name="pink capsid width",
        scale_bar_px=scale_bar.length_px,
        scale_nm=scale_nm,
        prefer_skeleton=False,
    )
    capsid_length = _measure_tape_measure_line(
        image_bgr,
        color_name="blue",
        label_name="blue capsid length",
        scale_bar_px=scale_bar.length_px,
        scale_nm=scale_nm,
        prefer_skeleton=False,
    )
    scale_bar = ColoredLineMeasurement(
        length_px=float(scale_bar.length_px),
        length_nm=float(scale_nm),
        path_yx=scale_bar.path_yx,
        bbox_xywh=scale_bar.bbox_xywh,
    )

    return TapeMeasureMeasurement(
        image_path=image_path,
        scale_nm=float(scale_nm),
        scale_bar=scale_bar,
        tail_length=tail_length,
        capsid_width=capsid_width,
        capsid_length=capsid_length,
    )


def _draw_measurement_path(
    image_bgr: np.ndarray,
    path_yx: list[tuple[int, int]],
    thickness: int,
    color_bgr: tuple[int, int, int],
) -> None:
    path_xy = np.array([[x, y] for y, x in path_yx], dtype=np.int32).reshape(-1, 1, 2)
    if len(path_xy) < 2:
        return
    cv2.polylines(image_bgr, [path_xy], False, color_bgr, thickness, cv2.LINE_AA)
    sx, sy = path_xy[0, 0]
    ex, ey = path_xy[-1, 0]
    cv2.circle(image_bgr, (int(sx), int(sy)), thickness + 1, color_bgr, -1, cv2.LINE_AA)
    cv2.circle(image_bgr, (int(ex), int(ey)), thickness + 1, color_bgr, -1, cv2.LINE_AA)


def render_tape_measure_overlay(result: TapeMeasureMeasurement) -> np.ndarray:
    image_bgr = _read_image_color(result.image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image for overlay: {result.image_path}")

    h, w = image_bgr.shape[:2]
    thickness = max(2, int(round(0.004 * min(h, w))))

    for line, color_bgr in (
        (result.scale_bar, (0, 0, 255)),
        (result.tail_length, (0, 165, 255)),
        (result.capsid_width, (0, 128, 128)),
        (result.capsid_length, (128, 128, 0)),
    ):
        if line is None:
            continue
        _draw_measurement_path(image_bgr, line.path_yx, thickness, color_bgr)
    return image_bgr


def _annotated_result_summary_lines(
    result: TapeMeasureMeasurement,
) -> list[str]:
    lines = [
        f"Scale bar length: {result.scale_bar.length_px:.1f} px = {result.scale_nm:.0f} nm",
        f"Capsid width: {result.capsid_width.length_px:.1f} px = {result.capsid_width.length_nm:.2f} nm",
        f"Capsid length: {result.capsid_length.length_px:.1f} px = {result.capsid_length.length_nm:.2f} nm",
    ]
    if result.tail_length is None:
        lines.append("Tail length: not detected")
    else:
        lines.append(f"Tail length: {result.tail_length.length_px:.1f} px = {result.tail_length.length_nm:.2f} nm")
    return lines


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


def _read_image_color(image_path: str | Path) -> np.ndarray | None:
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is not None:
        return image_bgr

    try:
        from PIL import Image
    except Exception:
        return None

    try:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


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
    import difflib

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

    normalized_image_stem = _normalize_header_key(Path(image_name).stem)
    normalized_stem_matches = [
        path for path in image_files
        if _normalize_header_key(path.stem) == normalized_image_stem
    ]
    if len(normalized_stem_matches) == 1:
        return normalized_stem_matches[0]

    prefix_matches = [path for path in image_files if path.stem.lower().startswith(image_stem)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    normalized_prefix_matches = [
        path for path in image_files
        if (
            _normalize_header_key(path.stem).startswith(normalized_image_stem)
            or normalized_image_stem.startswith(_normalize_header_key(path.stem))
        )
    ]
    if len(normalized_prefix_matches) == 1:
        return normalized_prefix_matches[0]

    normalized_stems = sorted({_normalize_header_key(path.stem): path for path in image_files}.items())
    close_candidates: list[tuple[float, Path]] = []
    for candidate_norm, candidate_path in normalized_stems:
        ratio = difflib.SequenceMatcher(a=normalized_image_stem, b=candidate_norm).ratio()
        if ratio >= 0.72:
            close_candidates.append((ratio, candidate_path))
    close_candidates.sort(key=lambda item: item[0], reverse=True)
    if close_candidates:
        best_ratio, best_path = close_candidates[0]
        second_ratio = close_candidates[1][0] if len(close_candidates) > 1 else 0.0
        if best_ratio >= 0.82 and (best_ratio - second_ratio) >= 0.05:
            return best_path

    raise FileNotFoundError(f"image was not found: {direct_path}")


def _is_missing_batch_image_error(exc: Exception) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True
    message = str(exc).lower()
    return "image was not found:" in message or "could not read image:" in message


def measure_annotated_batch(
    *,
    images_dir: Path,
    metadata_xlsx: Path,
    output_xlsx: Path,
    sheet_name: str | None = None,
    image_col: str = "File name",
    scale_col: str = "Scale bar measurement (nm)",
    usable_col: str | None = None,
    usable_require_blank: bool = False,
    overlay_dir: Path | None = None,
    fail_fast: bool = False,
) -> tuple[int, int, int]:
    headers, records = _read_xlsx_rows(metadata_xlsx, sheet_name=sheet_name)
    if not headers or not records:
        raise RuntimeError(f"No data rows were found in {metadata_xlsx}")

    image_header = _match_header_name(headers, image_col)
    scale_header = _match_header_name(headers, scale_col)
    usable_header = _match_header_name(headers, usable_col) if usable_col is not None else None
    if usable_require_blank and usable_header is None:
        raise RuntimeError("usable_require_blank was requested, but no usable_col was provided.")

    image_files, by_lower_name, by_lower_stem = _build_image_lookup(images_dir)

    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    output_headers = list(headers)
    base_headers = [
        "Metadata row",
        "Image path",
        "Measurement status",
        "Measurement error",
        "Overlay path",
    ]
    extra_headers = base_headers + [
        "Scale bar length (px)",
        "Scale bar length (nm)",
        "Tail length (px)",
        "Tail length (nm)",
        "Capsid width (px)",
        "Capsid width (nm)",
        "Capsid length (px)",
        "Capsid length (nm)",
    ]
    for header in extra_headers:
        if header not in output_headers:
            output_headers.append(header)

    output_rows: list[dict[str, object]] = []
    success_count = 0
    failure_count = 0
    skipped_count = 0
    measurement_cache: dict[tuple[str, float], TapeMeasureMeasurement] = {}
    overlay_cache: dict[tuple[str, float], Path] = {}

    for row_idx, record in enumerate(records, start=2):
        output_record = {header: record.get(header, "") for header in headers}
        output_record.update({
            "Metadata row": row_idx,
            "Image path": "",
            "Measurement status": "pending",
            "Measurement error": "",
            "Overlay path": "",
        })
        output_record.update({
            "Scale bar length (px)": "",
            "Scale bar length (nm)": "",
            "Tail length (px)": "",
            "Tail length (nm)": "",
            "Capsid width (px)": "",
            "Capsid width (nm)": "",
            "Capsid length (px)": "",
            "Capsid length (nm)": "",
        })

        if usable_header is not None and usable_require_blank:
            usable_value = str(record.get(usable_header, "")).strip()
            if not _usable_value_allows_measurement(usable_value):
                output_record["Measurement status"] = "skipped"
                output_record["Measurement error"] = f"Row {row_idx}: skipped because {usable_header} is {usable_value!r}."
                output_rows.append(output_record)
                skipped_count += 1
                continue

        image_name = str(record.get(image_header, "")).strip()
        if image_name == "":
            output_record["Measurement status"] = "skipped"
            output_record["Measurement error"] = f"Row {row_idx}: {image_header} is blank."
            output_rows.append(output_record)
            skipped_count += 1
            continue

        try:
            image_path = _resolve_batch_image_path(
                images_dir,
                image_name,
                image_files=image_files,
                by_lower_name=by_lower_name,
                by_lower_stem=by_lower_stem,
            )
            output_record["Image path"] = str(image_path)

            scale_nm = _parse_required_float(record.get(scale_header, ""), field_name=scale_header, row_number=row_idx)
            cache_key = (str(image_path), float(scale_nm))
            result = measurement_cache.get(cache_key)
            if result is None:
                result = measure_tape_annotated_phage(image_path=image_path, scale_nm=scale_nm)
                measurement_cache[cache_key] = result

            output_record["Measurement status"] = "ok"
            tape_result = result
            output_record["Scale bar length (px)"] = float(tape_result.scale_bar.length_px)
            output_record["Scale bar length (nm)"] = float(tape_result.scale_bar.length_nm)
            if tape_result.tail_length is not None:
                output_record["Tail length (px)"] = float(tape_result.tail_length.length_px)
                output_record["Tail length (nm)"] = float(tape_result.tail_length.length_nm)
            output_record["Capsid width (px)"] = float(tape_result.capsid_width.length_px)
            output_record["Capsid width (nm)"] = float(tape_result.capsid_width.length_nm)
            output_record["Capsid length (px)"] = float(tape_result.capsid_length.length_px)
            output_record["Capsid length (nm)"] = float(tape_result.capsid_length.length_nm)

            if overlay_dir is not None:
                overlay_key = (str(image_path), float(scale_nm))
                overlay_path = overlay_cache.get(overlay_key)
                if overlay_path is None:
                    overlay_name = f"{Path(image_path).stem}_overlay.png"
                    overlay_path = overlay_dir / overlay_name
                    overlay = render_tape_measure_overlay(result)
                    cv2.imwrite(str(overlay_path), overlay)
                    overlay_cache[overlay_key] = overlay_path
                output_record["Overlay path"] = str(overlay_path)

            success_count += 1
        except Exception as exc:
            if _is_missing_batch_image_error(exc):
                output_record["Measurement status"] = "skipped"
                error_text = str(exc)
                if not error_text.startswith(f"Row {row_idx}:"):
                    error_text = f"Row {row_idx}: {error_text}"
                output_record["Measurement error"] = error_text
                output_rows.append(output_record)
                skipped_count += 1
                continue
            failure_count += 1
            output_record["Measurement status"] = "error"
            error_text = str(exc)
            if not error_text.startswith(f"Row {row_idx}:"):
                error_text = f"Row {row_idx}: {error_text}"
            output_record["Measurement error"] = error_text
            if fail_fast:
                raise
        output_rows.append(output_record)

    _write_xlsx_rows(output_xlsx, output_headers, output_rows, sheet_name="Annotated measurements")
    return success_count, failure_count, skipped_count


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


def _ensure_overlay_output_is_not_input(image: Path, overlay_out: Optional[Path]) -> None:
    if overlay_out is None:
        return
    try:
        same_path = image.resolve() == overlay_out.resolve()
    except OSError:
        same_path = image.absolute() == overlay_out.absolute()
    if same_path:
        raise click.ClickException("--overlay_out must be different from --image; refusing to draw an overlay on top of itself.")


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
    _ensure_overlay_output_is_not_input(image, overlay_out)
    try:
        result = measure_phage_tail(
            image_path=str(image),
            scale_nm=scale_nm,
            bar_px_override=bar_px_override,
            debug=debug,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    capsid_width_px = result.capsid_width_px if result.capsid_width_px is not None else 2.0 * result.head_r
    capsid_height_px = result.capsid_height_px if result.capsid_height_px is not None else 2.0 * result.head_r
    capsid_width_nm = float(capsid_width_px) / result.px_per_nm
    capsid_height_nm = float(capsid_height_px) / result.px_per_nm
    if abs(float(capsid_width_px) - float(capsid_height_px)) <= 1.0:
        click.echo(f"Capsid diameter: {capsid_width_nm:.2f} nm")
    else:
        click.echo(f"Capsid width: {capsid_width_nm:.2f} nm")
        click.echo(f"Capsid height: {capsid_height_nm:.2f} nm")
    click.echo(f"Tail length: {result.tail_nm:.2f} nm")
    if result.tail_width_nm > 0.0:
        click.echo(f"Tail tube width: {result.tail_width_nm:.2f} nm")

    if overlay_out is not None or show_overlay:
        out_path = overlay_out if overlay_out is not None else (Path.cwd() / f"{image.stem}_tail_overlay.png")
        overlay = render_tail_overlay(str(image), result)
        cv2.imwrite(str(out_path), overlay)
        click.echo(f"Annotated image: {out_path}")
        if show_overlay:
            if abs(float(capsid_width_px) - float(capsid_height_px)) <= 1.0:
                title = f"Capsid: {capsid_width_nm:.2f} nm, Tail: {result.tail_nm:.2f} nm"
            else:
                title = f"Capsid: {capsid_width_nm:.2f} x {capsid_height_nm:.2f} nm, Tail: {result.tail_nm:.2f} nm"
            _show_overlay_window(overlay, title)


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
    """Measure one annotated figure using the same colored-line workflow as annotated-batch."""
    _ensure_overlay_output_is_not_input(image, overlay_out)
    try:
        result = measure_tape_annotated_phage(image_path=image, scale_nm=scale_nm)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    for line in _annotated_result_summary_lines(result):
        click.echo(line)

    if overlay_out is not None or show_overlay:
        out_path = (
            overlay_out
            if overlay_out is not None
            else (Path.cwd() / f"{image.stem}_annotated_overlay.png")
        )
        overlay = render_tape_measure_overlay(result)
        cv2.imwrite(str(out_path), overlay)
        click.echo(f"Annotated image: {out_path}")
        if show_overlay:
            _show_overlay_window(overlay, image.name)


@cli.command("annotated-batch", context_settings=CLICK_CONTEXT_SETTINGS)
@click.option("--images_dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path), help="Directory containing annotated images.")
@click.option("--metadata_xlsx", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Path to the metadata workbook (.xlsx).")
@click.option("--output_xlsx", required=True, type=click.Path(dir_okay=False, path_type=Path), help="Path to the output workbook (.xlsx).")
@click.option("--sheet_name", default=None, help="Worksheet name to read. Defaults to the first sheet.")
@click.option("--image_col", default="File name", show_default=True, help="Column containing the annotated image filename.")
@click.option("--scale_col", default="Scale bar measurement (nm)", show_default=True, help="Column containing the scale bar size in nm.")
@click.option("--usable_col", default=None, help="Optional column used to decide which rows should be measured.")
@click.option("--usable_require_blank", is_flag=True, help="Only measure rows where usable_col is blank; mark other rows as skipped.")
@click.option("--overlay_dir", type=click.Path(file_okay=False, path_type=Path), default=None, help="Optional directory to save overlay images for successful measurements.")
@click.option("--fail_fast", is_flag=True, help="Stop on the first error instead of recording failures in the output workbook.")
def annotated_batch_command(
    images_dir: Path,
    metadata_xlsx: Path,
    output_xlsx: Path,
    sheet_name: Optional[str],
    image_col: str,
    scale_col: str,
    usable_col: Optional[str],
    usable_require_blank: bool,
    overlay_dir: Optional[Path],
    fail_fast: bool,
) -> None:
    """Measure all batch-annotated images using the final colored-line workflow."""
    try:
        success_count, failure_count, skipped_count = measure_annotated_batch(
            images_dir=images_dir,
            metadata_xlsx=metadata_xlsx,
            output_xlsx=output_xlsx,
            sheet_name=sheet_name,
            image_col=image_col,
            scale_col=scale_col,
            usable_col=usable_col,
            usable_require_blank=usable_require_blank,
            overlay_dir=overlay_dir,
            fail_fast=fail_fast,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"Output workbook: {output_xlsx}")
    click.echo(f"Measured rows: {success_count}")
    click.echo(f"Skipped rows: {skipped_count}")
    click.echo(f"Failed rows: {failure_count}")
    if overlay_dir is not None:
        click.echo(f"Overlay directory: {overlay_dir}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
