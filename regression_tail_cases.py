#!/usr/bin/env python3
"""
Regression checks for known phage tail-length cases.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

from phagescale import measure_phage_tail_length_nm


DEFAULT_IMAGE_ROOT = Path("/Users/vijinimallawaarachchi/Documents/Data/Phagebase/Phagebase_Images")


@dataclass(frozen=True)
class RegressionCase:
    name: str
    file_name: str
    scale_nm: float
    check: str  # "min" | "approx"
    expected_nm: float
    tolerance_nm: float = 0.0

    def evaluate(self, measured_nm: float) -> tuple[bool, str]:
        if self.check == "min":
            ok = measured_nm >= self.expected_nm
            detail = f">= {self.expected_nm:.1f} nm"
            return ok, detail
        if self.check == "approx":
            delta = abs(measured_nm - self.expected_nm)
            ok = delta <= self.tolerance_nm
            detail = f"{self.expected_nm:.1f} +/- {self.tolerance_nm:.1f} nm"
            return ok, detail
        raise ValueError(f"Unknown check type: {self.check}")


CASES = [
    RegressionCase("T4", "T4.png", 50.0, "min", 100.0),
    RegressionCase("KIL1", "KIL1.png", 50.0, "min", 100.0),
    RegressionCase("DLP3", "DLP3.png", 100.0, "approx", 202.0, 10.0),
    RegressionCase("MarsHill", "MarsHill.jpeg", 100.0, "approx", 233.0, 12.0),
    RegressionCase("phiCD5", "\u03c6CD5.png", 100.0, "approx", 126.0, 8.0),
    RegressionCase("ph004f", "ph004f.png", 100.0, "approx", 102.5, 8.0),
    RegressionCase("H70", "H70.png", 50.0, "approx", 165.0, 15.0),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run regression checks for phage tail-length measurements.")
    ap.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help=f"Directory containing the test images (default: {DEFAULT_IMAGE_ROOT})",
    )
    ap.add_argument("--debug", action="store_true", help="Enable verbose debug output from phagescale.")
    ap.add_argument(
        "--fail_on_missing",
        action="store_true",
        help="Fail if a listed regression image is not found (default: missing files are skipped).",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root

    print(f"Regression root: {root}")
    print("-" * 72)

    passed = 0
    failed = 0
    skipped = 0

    for case in CASES:
        image_path = root / case.file_name
        if not image_path.exists():
            skipped += 1
            status = "FAIL" if args.fail_on_missing else "SKIP"
            print(f"[{status}] {case.name:<8} missing file: {image_path}")
            if args.fail_on_missing:
                failed += 1
            continue

        measured_nm = measure_phage_tail_length_nm(
            image_path=str(image_path),
            scale_nm=case.scale_nm,
            debug=args.debug,
        )
        ok, expected_text = case.evaluate(measured_nm)
        if ok:
            passed += 1
            print(f"[PASS] {case.name:<8} measured={measured_nm:7.2f} nm expected {expected_text}")
        else:
            failed += 1
            print(f"[FAIL] {case.name:<8} measured={measured_nm:7.2f} nm expected {expected_text}")

    print("-" * 72)
    print(f"Summary: pass={passed} fail={failed} skip={skipped}")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
