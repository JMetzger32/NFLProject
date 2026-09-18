"""Render Web Service entry point.

Today this only serves the static site (index.html + site/data.json +
site/weeks/*.json) — the same content GitHub Pages already serves. A Static Site
service on Render would have been the leaner fit for that alone; this exists
because the plan is to grow it into a real backend later (regenerating data.json
on request, running weekly_picks.py on a schedule, grading completed weeks, ...).
Nothing below assumes that yet — it's deliberately just the seam to build onto.

Local run:
    python server.py                     # http://localhost:8000

Render (Web Service):
    Build Command: pip install -r requirements-web.txt
    Start Command: gunicorn server:app --bind 0.0.0.0:$PORT
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, send_from_directory

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"

app = Flask(__name__, static_folder=None)


@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.get("/site/<path:filename>")
def site_files(filename):
    # Matches every path the frontend actually fetches: site/data.json directly,
    # and each week chip's site/weeks/<season>_wk<NN>.json.
    return send_from_directory(SITE, filename)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "data_json_present": (SITE / "data.json").exists()}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
