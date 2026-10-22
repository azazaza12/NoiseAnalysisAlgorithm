# Spectral Noise Analysis

[![tests](https://github.com/azazaza12/NoiseAnalysisAlgorithm/actions/workflows/tests.yml/badge.svg)](https://github.com/azazaza12/NoiseAnalysisAlgorithm/actions/workflows/tests.yml)

A desktop tool and signal-processing pipeline for detecting acoustic anomalies
in spectral noise logging data. It reads LAS files, isolates significant regions
on the depth-frequency panel, classifies the detected noise, and exports reviewed
results to XLSX or LAS.

## What it does

- reads low- and high-frequency acoustic channels from LAS files;
- suppresses background drift with median and Gaussian filtering;
- segments candidate regions using thresholding and morphological operations;
- detects, filters, and merges contours into anomaly intervals;
- classifies intervals as reservoir flow, behind-casing channeling, or borehole noise;
- provides a Tkinter interface for reviewing, adding, resizing, and deleting detections;
- exports interval metadata and amplitudes to XLSX or a binary indicator curve to LAS.

## Processing pipeline

```text
LAS channels
  -> median drift removal
  -> Gaussian smoothing
  -> percentile thresholding
  -> erosion and dilation
  -> contour detection
  -> peak and overlap analysis
  -> depth-frequency rectangles
  -> manual review and export
```

The detailed methodology is available in
[`Algorithm_description.pdf`](Algorithm_description.pdf).

## Run locally

The project requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python main.py
```

Select a `.las` file in the application. Detected intervals appear as rectangles
on the spectral panel and can be adjusted before export.

## Tests

```bash
uv run pytest -q
```

The test suite covers data loading, filtering, contour processing, interval
merging, classification, and report generation. GitHub Actions runs it on every
push and pull request.

## Project structure

```text
main.py             # Tkinter review interface
start_program.py    # LAS loading and detection pipeline
functions.py        # signal processing, contour analysis, and exports
constants.py        # channel-specific thresholds
plotting.py         # standalone visualization helpers
tests/              # unit and integration tests
```

## Current limitations

- Detection thresholds are calibrated for the supported LF/HF channel layouts.
- The repository does not include proprietary field data; bring your own LAS file.
- The desktop interface requires a graphical environment.

## License

[MIT](LICENSE)
