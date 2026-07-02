# dl_genai_t12026

Full working application for the Introduction to DL and GenAI T12026 course project.

The original repository contains Kaggle notebooks and reports for the Messy Mashup
music genre classification task. This version adds a runnable browser application
that analyzes an audio file, extracts signal features, predicts one of the project
genres, visualizes the waveform/spectrum, and exports a JSON report.

## Run the app

Open this file in a browser:

```text
app/index.html
```

No Python packages or build step are required. The app runs fully in the browser
using the Web Audio API.

## Use it

1. Choose an audio file, or click `Demo Mix`.
2. Review the top genre prediction, confidence, metrics, waveform, and spectrum.
3. Change model mode or analysis window to compare results.
4. Click `Download Report` to export the prediction and metrics as JSON.

## Project genres

The app predicts among the same 10 labels used by the notebooks:

```text
blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
```

## Repository contents

- `app/` - complete static web application.
- `milestones/` - original Kaggle notebooks and report artifact.
- `23f2002033_DG_T12026.pdf` - project document.

## Notes

The in-browser classifier is a lightweight, deterministic baseline built from
audio features inspired by the notebooks: RMS energy, estimated tempo, zero
crossing rate, spectral centroid, frequency-band balance, and dynamics. The
notebooks remain available for training heavier Kaggle models such as the
scratch neural baseline, Wav2Vec2 experiment, and AST experiment.
