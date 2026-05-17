#!/usr/bin/env bash
# Render a notebook to HTML with constrained max-width for legibility.
# Usage: scripts/render_html.sh [notebook] [output.html]
set -euo pipefail
NB="${1:-notebooks/03_report.ipynb}"
OUT="${2:-outputs/report.html}"
TMP=$(mktemp)

uv run jupyter nbconvert --to html "$NB" --embed-images --stdout > "$TMP"

uv run python - "$TMP" "$OUT" <<'PY'
import sys
from pathlib import Path
tmp, out = sys.argv[1], sys.argv[2]
html = Path(tmp).read_text()
css = (
    "<style>"
    "body{max-width:80rem;margin:0 auto;padding:1rem 2rem;}"
    ".jp-Notebook{max-width:80rem;margin:0 auto;}"
    "img{max-width:100%;height:auto;}"
    "table{display:block;overflow-x:auto;}"
    "figure{width:80%;margin:1.5rem auto;text-align:center;}"
    "figure img{width:100%;border:1px solid #ccc;border-radius:4px;box-shadow:0 1px 4px rgba(0,0,0,0.08);}"
    "figcaption{font-size:0.875rem;color:#555;margin-top:0.5rem;font-style:italic;text-align:center;}"
    ".jp-OutputArea-output pre,.output_text pre{background:#f4f6f9;border-left:3px solid #9467bd;padding:0.5rem 0.75rem;border-radius:3px;color:#1a2733;font-size:0.85rem;line-height:1.45;}"
    "</style>"
)
html = html.replace("</head>", css + "</head>", 1)
Path(out).write_text(html)
print(f"wrote {out}")
PY
