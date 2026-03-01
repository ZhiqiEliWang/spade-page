---
title: SPADE
emoji: 💻
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
license: mit
---

## What this repo provides

1. `app.py`: Hugging Face Space backend pipeline (`x -> detector -> explainer`)
2. `docs/`: GitHub Pages demo frontend that calls the Space API

## Backend (Hugging Face Space)

This repo is configured as a Gradio Space (`sdk: gradio`).

### API endpoint

- Endpoint name: `/pipeline`
- Input: `text` (string)
- Output (streamed updates): `[detector_output_json_or_partial, explainer_markdown_or_partial]`

### Replace with your paper models

In `app.py`, replace:

- `run_detector(text)` with your detector inference
- `run_explainer(text, detector_output)` with your explainer inference

The pipeline order is already implemented in `pipeline(text)`.

## Frontend (GitHub Project Page)

The static site lives in `docs/`:

- `docs/index.html`
- `docs/app.js`
- `docs/config.js`
- `docs/styles.css`

### Configure Space ID

The GitHub Pages frontend in `docs/` is fixed to:

`ZhiqiEliWang/SPADE`

`docs/app.js` always connects to that Space when running the pipeline.
It uses `app.submit("/pipeline", { text })` to render streamed output chunks on the page.

### Enable GitHub Pages

In GitHub repo settings:

1. Open `Settings -> Pages`
2. Set source to `Deploy from a branch`
3. Select branch `main` and folder `/docs`
4. Save

Your demo will be published at:

`https://<your-github-username>.github.io/<repo-name>/`

## Local run (optional)

```bash
pip install -r requirements.txt
python app.py
```

Then open `docs/index.html` in a local static server if needed.
