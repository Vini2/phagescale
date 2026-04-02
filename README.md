# PhageScale: measuring phage dimensions from TEM images

## Install

```bash
pip install click opencv-python numpy scikit-image matplotlib
```

## Run

The CLI has two subcommands:

- `measure` for raw TEM images, using automatic head and tail detection.
- `annotated` for figures where the tail has already been marked in yellow.

Global options:

- `-v` or `--version` shows the CLI version
- `-h` or `--help` shows help

Both subcommands support:

- `--image` to input path of the image
- `--scale_nm` to input scale-bar length in nm
- `--overlay_out` for an output overlay image path
- `--show_overlay` to display the overlay
- printing the measured tail length to `stdout`

### Measuring from raw images

```bash
python phagescale.py measure --image /path/to/image.png --scale_nm 100 --debug
```

- `--scale_nm` is the numeric value printed for the scale bar, for example `100` for `100 nm`.
- If auto scale-bar detection fails, pass `--bar_px_override` with a manually measured bar length in pixels.
- `--overlay_out /path/to/output.png` saves an annotated image with the traced tail.
- `--show_overlay` displays the annotated image at the end of the run, and also saves it in the current working directory if `--overlay_out` is not provided.

Example with overlay:

```bash
python phagescale.py measure --image /path/to/image.png --scale_nm 100 --overlay_out /path/to/annotated.png --show_overlay
```

### Measuring from annotated figures

For figures where the tail has already been marked in yellow:

```bash
python phagescale.py annotated --image /path/to/figure.png --scale_nm 100
```

You can also save or display an overlay:

```bash
python phagescale.py annotated --image /path/to/figure.png --scale_nm 100 --overlay_out /path/to/annotated.png --show_overlay
```

## Regression Checks

Run the built-in regression suite for known calibration images:

```bash
python regression_tail_cases.py
```

Options:
- `--root /path/to/Phagebase_Images` to point to a different image directory.
- `--debug` to print per-image detection diagnostics.
- `--fail_on_missing` to fail when any regression image is missing.

**Warning**: this is still under construction and heavy testing. Results might be incorrect.