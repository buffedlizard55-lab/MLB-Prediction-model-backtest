"""HTTP helpers for talking to official MLB hosts.

Two transports:

* ``fetch_url``      – direct HTTPS GET (works on any machine with normal
                       egress to statsapi.mlb.com / baseballsavant.mlb.com).
* ``extract_payload``– parses a JSON/CSV payload out of a page-rendered
                       capture (the ``fetch_page`` tool renders API JSON as a
                       markdown code fence; Savant CSV as a markdown table).
                       Cached captures live under data/raw/page_capture/.

Every byte we keep from an official response is stored verbatim in
``data/raw/`` so the whole pipeline stays auditable.
"""

from __future__ import annotations

import gzip
import io
import json
import re
import time
import urllib.request
from pathlib import Path

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 mlb-predict/0.1"
)

DEFAULT_TIMEOUT = 30.0
DEFAULT_DELAY = 0.35  # be polite to MLB's public endpoints


def fetch_url(url: str, timeout: float = DEFAULT_TIMEOUT, retries: int = 3,
              backoff: float = 1.6) -> bytes:
    """GET a URL and return the raw response bytes (with retries)."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as err:  # noqa: BLE001 - retry on any transport error
            last_err = err
            if attempt < retries - 1:
                time.sleep(backoff ** (attempt + 1))
    raise RuntimeError(f"fetch failed after {retries} tries: {url}: {last_err}")


def save_raw(raw_dir: Path, rel_path: str, payload: bytes,
             compress: bool = True) -> Path:
    """Persist raw response bytes under raw_dir (optionally gzipped)."""
    dest = raw_dir / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    if compress and not str(dest).endswith(".gz"):
        dest = dest.with_name(dest.name + ".gz")
    if str(dest).endswith(".gz"):
        with gzip.open(dest, "wb") as fh:
            fh.write(payload)
    else:
        dest.write_bytes(payload)
    return dest


def load_raw(path: Path) -> bytes:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as fh:
            return fh.read()
    return path.read_bytes()


# --------------------------------------------------------------------------
# Page-capture parsing (used when responses were captured via fetch_page)
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.DOTALL)


def extract_json_payload(text: str) -> dict | list:
    """Pull the JSON object out of a page-captured markdown document.

    Handles: bare JSON, ```json fenced JSON, or a JSON blob embedded in
    surrounding prose. Raises ValueError if nothing parseable is found.
    """
    candidates: list[str] = []
    candidates.extend(m.group(1) for m in _FENCE_RE.finditer(text))
    candidates.append(text)
    for cand in candidates:
        cand = cand.strip()
        if not cand:
            continue
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
        # Trim to the first '{' / '[' ... last '}' / ']' span.
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            i, j = cand.find(open_ch), cand.rfind(close_ch)
            if 0 <= i < j:
                try:
                    return json.loads(cand[i:j + 1])
                except json.JSONDecodeError:
                    continue
    raise ValueError("no JSON payload found in page capture")


def markdown_table_to_csv(text: str) -> str:
    """Convert a markdown pipe table (how fetch_page renders Savant CSVs)
    back into plain CSV text."""
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        # Skip alignment rows like | --- | --- |
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue
        if cells and cells[0] == "---":
            continue
        # Quote cells containing commas so CSV stays parseable.
        out_cells = []
        for c in cells:
            if "," in c and not (c.startswith('"') and c.endswith('"')):
                c = '"' + c.replace('"', '""') + '"'
            out_cells.append(c)
        lines.append(",".join(out_cells))
    return "\n".join(lines) + "\n"


def extract_csv_payload(text: str) -> str:
    """Pull CSV out of a page capture: either a fenced block or a
    markdown-rendered table."""
    for m in _FENCE_RE.finditer(text):
        body = m.group(1).strip()
        if "," in body and "|" not in body.splitlines()[0]:
            return body + "\n"
    table = markdown_table_to_csv(text)
    if table.strip():
        return table
    raise ValueError("no CSV payload found in page capture")
