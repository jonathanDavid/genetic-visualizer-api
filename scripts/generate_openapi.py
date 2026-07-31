"""Export the OpenAPI spec to openapi.json for the Pages docs site.

Run:  python scripts/generate_openapi.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402

out = Path(__file__).resolve().parents[1] / "openapi.json"
spec = app.openapi()
out.write_text(json.dumps(spec, indent=2), encoding="utf-8")
print(f"wrote {out} - {len(spec['paths'])} paths")
