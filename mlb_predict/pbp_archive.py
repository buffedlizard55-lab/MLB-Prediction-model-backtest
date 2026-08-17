"""Adapter around the vendored MLB-PBP package (buffedlizard55-lab/MLB-PBP).

MLB-PBP archives the *complete official play-by-play* for any game or date
range, straight from ``statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live``,
with SHA-256 manifests and resume support. It is stdlib-only Python, so it
runs anywhere with direct MLB egress (your laptop / CI — not this sandbox).

Usage on such a machine:

    scripts/sync_sources.sh          # clones/updates vendor/MLB-PBP
    python -m mlb_predict.pbp_archive one 746865
    python -m mlb_predict.pbp_archive range 2025-04-01 2025-04-07

which delegates to:

    python -m mlb_pbp game 746865 --format raw
    python -m mlb_pbp archive --start ... --end ... --output data/raw/pbp

This adapter only locates and launch the vendored package; it never
re-implements or fabricates any of its verified output.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR_PBP = REPO_ROOT / "vendor" / "MLB-PBP"
PBP_ARCHIVE_DIR = REPO_ROOT / "data" / "raw" / "pbp"


def _mlb_pbp_cmd() -> list[str]:
    if not (VENDOR_PBP / "mlb_pbp" / "__init__.py").exists():
        raise RuntimeError(
            "vendor/MLB-PBP not found. Run scripts/sync_sources.sh first."
        )
    return [sys.executable, "-m", "mlb_pbp"]


def run_cli(*args: str, cwd: Path | None = None) -> int:
    """Run ``python -m mlb_pbp <args>`` inside vendor/MLB-PBP."""
    cmd = _mlb_pbp_cmd() + list(args)
    return subprocess.call(cmd, cwd=cwd or VENDOR_PBP)


def archive_game(game_pk: int) -> int:
    """Fetch one game's official live feed (raw JSON report)."""
    return run_cli("game", str(game_pk), "--format", "raw")


def archive_range(start: str, end: str | None = None,
                  output: Path | None = None, workers: int = 2,
                  request_delay: float = 0.2) -> int:
    """Fetch every qualifying official game feed in [start, end].

    This is the full PBP collection path: scores, every pitch, every
    play, line scores — the same feeds the backtester's ground truth is
    cross-checked against.
    """
    out = output or PBP_ARCHIVE_DIR
    out.mkdir(parents=True, exist_ok=True)
    args = ["archive", "--start", start, "--output", str(out),
            "--formats", "text,raw,events", "--workers", str(workers),
            "--request-delay", str(request_delay)]
    if end:
        args += ["--end", end]
    return run_cli(*args)


def verify_archive(archive_dir: Path | None = None) -> int:
    """Run MLB-PBP's own digest verification over the archive."""
    return run_cli("verify", str(archive_dir or PBP_ARCHIVE_DIR))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    one = sub.add_parser("one", help="fetch one game feed")
    one.add_argument("game_pk", type=int)
    rng = sub.add_parser("range", help="fetch a date range of game feeds")
    rng.add_argument("start")
    rng.add_argument("end", nargs="?")
    rng.add_argument("--output", type=Path, default=None)
    ver = sub.add_parser("verify", help="verify archive digests")
    ver.add_argument("archive", nargs="?", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "one":
        return archive_game(args.game_pk)
    if args.cmd == "range":
        return archive_range(args.start, args.end, args.output)
    return verify_archive(args.archive)


if __name__ == "__main__":
    sys.exit(main())
