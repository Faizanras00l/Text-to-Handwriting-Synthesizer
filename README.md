# Text to Handwriting Converter

A Python-based neural handwriting synthesis engine that converts plain text into realistic, human-like handwriting. Output is rendered as scalable SVG or printable PDF, on blank or notebook-style lined paper.

---

## Live Demo

**[https://my-inkflow.vercel.app/](https://my-inkflow.vercel.app/)**

**InkFlow** is a production web application built entirely on this engine. It ships two independent tools, both powered by the same underlying model:

| Tool | Description |
|---|---|
| **Calligrapher** | Generates decorative calligraphy-style lettering from text input |
| **Text to Handwriting Converter** | Converts plain text into multi-page realistic handwritten documents with full control over style, paper type, and neatness |

These are two different applications of the same core model. Developers can use this repository as a foundation to build their own handwriting tools — for education, content generation, accessibility, or creative applications.

- Frontend deployed on [Vercel](https://vercel.com)
- Backend (model inference API) deployed on [Hugging Face Spaces](https://huggingface.co/spaces)

---

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Project Structure](#project-structure)
4. [Module Reference](#module-reference)
5. [Quickstart](#quickstart)
6. [All Parameters Explained](#all-parameters-explained)
7. [Customization Guide](#customization-guide)
8. [Output Formats](#output-formats)
9. [Paper Modes](#paper-modes)
10. [Tech Stack](#tech-stack)
11. [Requirements](#requirements)
12. [Credits](#credits)

---

## Overview

This is an open-source **Text to Handwriting Converter** built on a Mixture Density Recurrent Neural Network (MD-RNN), trained on the [IAM On-Line Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database). The model learns the stroke-level mechanics of human handwriting — pen lifts, pressure curves, character spacing, and natural imperfection — and reproduces them on demand for any input text.

The engine is tool-agnostic. [InkFlow](https://my-inkflow.vercel.app/) is one example of what can be built on top of it, but the same core model can power any handwriting-related application.

Output is rendered as vector SVG (or optionally converted to PDF), making it infinitely scalable and print-ready.

**Key features:**

- 13 distinct handwriting styles (style index 0–12), each derived from real human penmanship
- Bias control to tune handwriting from loose and natural to clean and precise
- Blank paper (dynamic canvas) and lined/ruled paper (notebook-style) modes
- SVG and PDF output
- Smart text wrapping that never breaks mid-word
- Multi-page support — long texts paginate automatically
- Fully interactive CLI — run with no arguments for a guided prompt

---

## How It Works

The system is a sequence-to-sequence generative model:

1. Input text is encoded as a sequence of character indices using a fixed 75-character alphabet.
2. An optional style seed — real stroke data from the chosen style — is prepended to prime the model's state.
3. A 3-layer LSTM with a soft attention mechanism over the character sequence generates a series of **(Δx, Δy, end-of-stroke)** triplets representing relative pen movements.
4. These stroke offsets are converted to absolute coordinates, denoised with a Savitzky-Golay filter, aligned to correct for global slant, and scaled.
5. The final stroke paths are rendered onto an SVG canvas — either blank or ruled.

The attention mechanism ensures the model tracks which character it is currently writing and advances naturally through the input. The output distribution at each step is a **Gaussian Mixture Model (GMM)**, which allows the model to produce natural variation in stroke trajectories rather than a single deterministic output.

---

## Project Structure

```
text-to-handwriting/
├── inference/
│   ├── __init__.py
│   ├── generate_cli.py       # Main entry point
│   ├── synthesizer.py        # Hand class — core orchestration layer
│   └── canvas.py             # Stroke manipulation and SVG utilities
│
├── networks/
│   ├── __init__.py
│   ├── lstm_layer.py         # Top-level RNN model (rnn class)
│   ├── lstm_cell.py          # Custom LSTM + Attention cell
│   ├── rnn_operations.py     # Low-level RNN loop implementations
│   ├── base_network.py       # TF model base: session, checkpointing, training
│   └── network_utils.py      # Dense layer helpers and tensor utilities
│
├── resources/
│   ├── checkpoints/          # Pre-trained model weights (TensorFlow checkpoints)
│   └── styles/               # Style seed files (.npy) — stroke and character data
│
├── results/
│   ├── output/               # Generated SVG and PDF files
│   └── logs/                 # TensorFlow session logs
│
├── requirements.txt
└── README.md
```

---

## Module Reference

### `inference/generate_cli.py`

The main entry point. Run this file directly from the command line. Responsibilities:

- Accepts CLI arguments (`--text`, `--style`, `--bias`, `--paper`, `--format`, `--output`) or launches an interactive session if none are provided.
- Defines all page layout constants at the top of the file — this is where page size, margins, line count, and character limits are configured.
- Runs `smart_wrap()` to break long paragraphs at word boundaries before passing them to the model.
- Delegates to `Hand.write()` in `synthesizer.py` to produce the SVG output.
- Optionally converts SVG files to a single multi-page PDF.

### `inference/synthesizer.py`

The core orchestration layer, containing the `Hand` class.

| Method | Purpose |
|---|---|
| `__init__()` | Loads the pre-trained RNN model from `resources/checkpoints/` |
| `write()` | Public API — accepts text, style, bias, and page parameters; returns a list of generated filenames |
| `_sample()` | Runs the TensorFlow session to generate raw stroke offsets |
| `_draw_lined()` | Renders strokes onto a ruled notebook-style SVG canvas |
| `_draw_blank()` | Renders strokes onto a dynamic blank white canvas |

The `write()` method handles wrapping long lines to fit within margins, paginating output across multiple SVG files, and extending bias/style lists to match line count automatically.

### `inference/canvas.py`

A utility module for stroke processing. No classes — standalone functions only.

| Function | Description |
|---|---|
| `align(coords)` | Corrects global tilt/slant using linear regression |
| `denoise(coords)` | Applies a Savitzky-Golay smoothing filter |
| `offsets_to_coords(offsets)` | Converts relative pen movements to absolute coordinates |
| `coords_to_offsets(coords)` | Inverse of above |
| `encode_ascii(string)` | Maps characters to integer indices using the model alphabet |
| `interpolate(coords)` | Cubic spline up-sampling for smoother curves |
| `normalize(offsets)` | Scales strokes to median unit norm |

The `alphabet` list defines the 75 characters the model was trained on and supports.

### `networks/lstm_layer.py`

Defines the `rnn` class (inherits from `TFBaseModel`). This is the top-level TensorFlow model. It defines all placeholders, instantiates the `LSTMAttentionCell`, builds the forward pass, defines the Negative Log-Likelihood loss over the GMM output, and exposes `sampled_sequence` — the tensor fetched during inference.

### `networks/lstm_cell.py`

Defines `LSTMAttentionCell`, a custom TF RNN cell implementing a 3-layer LSTM with soft windowed attention.

- **LSTM 1** receives the previous attention window vector and the current input.
- The **attention mechanism** computes a soft alignment over the character sequence using Gaussian windows, producing a context vector.
- **LSTM 2 and 3** receive the context vector alongside LSTM 1's output.
- `output_function()` samples the next pen stroke from the GMM distribution.
- `termination_condition()` detects end-of-sequence to stop generation cleanly.

### `networks/rnn_operations.py`

Provides two custom RNN loop functions built on top of `raw_rnn`:

- `rnn_free_run()` — autoregressive loop where each output is fed back as the next input (inference/sampling).
- `rnn_teacher_force()` — loop using ground truth inputs at each step (training).

### `networks/base_network.py`

`TFBaseModel` is the training scaffold. It handles graph construction, TF session management, training loops with early stopping and multi-stage learning rates, checkpoint saving and restoring, and logging.

At inference time, only `restore()` and `session.run()` are used.

### `networks/network_utils.py`

Lightweight TF utilities:

- `dense_layer()` — fully-connected layer with optional bias, batch normalization, dropout, and activation.
- `time_distributed_dense_layer()` — the same operation applied across every timestep of a sequence.
- `shape()` / `rank()` — tensor shape helpers.

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Interactive mode (step-by-step prompts)
python inference/generate_cli.py

# Single command: blank paper, style 3, bias 1.0
python inference/generate_cli.py --text "Hello, World!" --style 3 --bias 1.0 --paper blank

# Lined paper, PDF output, custom filename
python inference/generate_cli.py --text "Meeting notes." --style 7 --bias 0.85 --paper lined --format pdf --output meeting_notes
```

Output files are saved to `results/output/`.

---

## All Parameters Explained

| Argument | Type | Default | Description |
|---|---|---|---|
| `--text` | `str` | *(interactive prompt)* | The text to synthesize. Use `\n` for explicit line breaks. |
| `--style` | `int` (0–12) | `0` | Handwriting style index. Each index corresponds to a different person's penmanship. |
| `--bias` | `float` (0.1–1.5) | `1.0` | Controls the balance between naturalness and precision. See the Bias section below. |
| `--paper` | `blank` / `lined` | `blank` | Paper type. `lined` renders notebook-style ruled lines beneath the handwriting. |
| `--format` | `svg` / `pdf` | `svg` | Output file format. |
| `--output` | `str` | `output` | Base filename for the output (no extension — added automatically). |

---

## Customization Guide

All page layout settings are defined as named constants at the top of `inference/generate_cli.py`. No model code needs to be modified.

### Page Dimensions

```python
PAGE_WIDTH = 1860          # Width in pixels (A4-like proportion)
LINES_PER_PAGE = 17        # Lines per page (applies to lined paper mode)
LINE_GAP = 125             # Line spacing in pixels — do not change this value
# Page height is derived automatically:
# PAGE_HEIGHT = (LINES_PER_PAGE * LINE_GAP) + TOP_MARGIN + BOTTOM_MARGIN
```

> **Note:** `LINE_GAP` is coupled to the model's stroke scale factor. Adjust `LINES_PER_PAGE` to control how many lines appear per page; the height will adjust accordingly.

### Lines Per Page

```python
LINES_PER_PAGE = 20    # More lines, taller page
LINES_PER_PAGE = 12    # Fewer lines, more spacious layout
```

### Margins

```python
LEFT_MARGIN = 150       # Distance from the left edge, in pixels
RIGHT_MARGIN = 150      # Distance from the right edge
TOP_MARGIN = 250        # Distance from the top
BOTTOM_MARGIN = 250     # Distance from the bottom
```

Wider margins produce a more formal, letter-like result. Narrower margins maximize the writing area.

### Handwriting Quality (Bias)

The `bias` parameter is a temperature-like coefficient applied to the model's GMM output distribution. Higher values narrow the distribution, producing more consistent but less natural strokes. Lower values widen it, producing more human variation but with occasional character irregularities.

| Bias | Character |
|---|---|
| `0.5` | Artistic — maximum natural variation, loose strokes |
| `0.65` | Casual — relaxed, informal feel |
| `0.75` | Balanced — readable with natural character |
| `0.85` | Neat — clean with subtle variation |
| `1.0` | Professional — clean and uniform (recommended default) |
| `1.2` | Ultra-clean — minimal variation |
| `1.5` | Maximum precision — near-mechanical consistency |

### Handwriting Styles (0–12)

There are 13 style seeds stored in `resources/styles/`. Each seed is derived from a real person's handwriting sample in the IAM dataset. Passing a style index primes the LSTM state with that person's pen dynamics before generation begins.

To select a style:

```bash
python inference/generate_cli.py --style 5
```

Experiment with different indices to find the style that suits your use case.

### Text Wrapping

```python
MAX_CHARS_PER_LINE = 55    # Soft wrap limit for blank paper mode
```

The model produces optimal quality output at 40–55 characters per line. Values above 75 will raise a validation error in blank paper mode. In lined mode, the wrap limit is calculated automatically based on page width and margins.

---

## Output Formats

| Format | Description | Use Case |
|---|---|---|
| SVG | Scalable vector graphics, editable in Inkscape or Illustrator | Digital use, further editing |
| PDF | Multi-page document compiled from SVG pages | Printing, sharing |

Multi-page documents are named sequentially: `output_p1.svg`, `output_p2.svg`, and so on.

---

## Paper Modes

| Mode | Description |
|---|---|
| Blank | Clean white canvas with dynamic height based on line count. Handwriting is horizontally centered. |
| Lined | Fixed-size page with blue horizontal ruled lines, a double pink header line, and gray vertical margin guides. Long texts paginate automatically. |

---

## Tech Stack

| Technology | Role |
|---|---|
| Python 3.11 | Core language |
| TensorFlow 1.x (compat.v1) | Neural network inference |
| TensorFlow Probability | Multivariate Gaussian sampling for stroke generation |
| NumPy | Numerical stroke processing |
| svgwrite | SVG canvas rendering |
| SciPy | Savitzky-Golay smoothing, cubic spline interpolation |
| svglib + reportlab | SVG to PDF conversion |

---

## Requirements

```
tensorflow
tensorflow-probability
numpy
svgwrite
scipy
matplotlib
svglib
reportlab
```

```bash
pip install -r requirements.txt
```

The pre-trained model checkpoint files must be present in `resources/checkpoints/` for inference to work. These are not included in the repository due to file size — download them separately and place them in that directory.

---

## Credits

- **Synthesis Model** — Based on the research by Alex Graves: [*Generating Sequences With Recurrent Neural Networks*](https://arxiv.org/abs/1308.0850)
- **Dataset** — [IAM On-Line Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database), University of Bern
