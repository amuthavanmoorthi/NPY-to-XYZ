# NPY to XYZ Converter

A small Flask website for converting NumPy `.npy` point data into plain-text `.xyz` files.

## What it expects

- Numeric arrays shaped like `(N, 3)`
- Higher-dimensional numeric arrays where the last dimension is at least `3`
- Structured arrays with `x`, `y`, and `z` fields

The converter flattens the data into rows and writes the first three values of each point as:

```text
x y z
```

## Run locally

```powershell
python app.py
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Deploy on Render free tier

This repo includes a [render.yaml](./render.yaml) blueprint for Render's free web service tier.

Files used for deployment:

- `render.yaml`
- `.python-version`
- `requirements.txt`

Render setup:

1. Push this folder to a GitHub, GitLab, or Bitbucket repository.
2. Sign in to Render and create a new Blueprint or Web Service from that repo.
3. Render will install dependencies with `pip install -r requirements.txt`.
4. Render will start the app with `gunicorn --bind 0.0.0.0:$PORT app:app`.
5. After the build finishes, the app will be public on a `*.onrender.com` URL.
