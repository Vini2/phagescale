# phagescale

PhageScale: measuring phage tail length from TEM images

## Install

```bash
pip install opencv-python numpy
```

## Run

```bash
python phagescale.py --image /path/to/image.png --scale_nm 100 --debug
```

- `--scale_nm` is the numeric value printed for the scale bar (for example `100` for `100 nm`).
- If auto scale-bar detection fails, pass `--bar_px_override` with a manually measured bar length in pixels.

## Regression Checks

Run the built-in regression suite for known calibration images:

```bash
python regression_tail_cases.py
```

Options:
- `--root /path/to/Phagebase_Images` to point to a different image directory.
- `--debug` to print per-image detection diagnostics.
- `--fail_on_missing` to fail when any regression image is missing.
