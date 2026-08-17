"""Parse raw 2023 schedule page-captures -> verified scores -> ingest.

One-off helper for the 2023 historical ingest. For every
data/raw/page_capture/scores_2023/sched_<w0>_<w1>.txt capture:
  * extract the JSON payload,
  * keep only regular-season games with both scores present (postponed
    games without scores stay missing; makeup games have their own gamePk),
  * validate (known team ids, sane scores, unique game_pk, date in window,
    not already in the transcript),
  * write a compact CSV and feed it through `python -m mlb_predict
    ingest-scores` so the transcript + provenance ledger stay exact.

Team ids are read from the official teams reference (data/ref/teams_official.csv).
"""

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlb_predict.http_io import extract_json_payload  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SEASON = sys.argv[1] if len(sys.argv) > 1 else "2023"
CAP_DIR = REPO / "data" / "raw" / "page_capture" / f"scores_{SEASON}"
TRANSCRIPT = REPO / "data" / "raw" / "scores_transcript.csv"

VALID_IDS = set()
teams = (REPO / "data" / "ref" / "teams_official.csv").read_text().splitlines()[1:]
for line in teams:
    if line.strip():
        VALID_IDS.add(int(line.split(",")[0]))
assert len(VALID_IDS) == 30, VALID_IDS

seen_pks = set()
if TRANSCRIPT.exists():
    for line in TRANSCRIPT.read_text().splitlines()[1:]:
        if line.strip():
            seen_pks.add(int(line.split(",")[0]))

FNAME_RE = re.compile(r"sched_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.txt$")


def main() -> int:
    captures = sorted(CAP_DIR.glob("sched_*.txt"))
    if not captures:
        print("no captures found")
        return 1
    total_new = 0
    for cap in captures:
        m = FNAME_RE.search(cap.name)
        w0, w1 = m.group(1), m.group(2)
        text = cap.read_text()
        doc = None
        try:
            doc = extract_json_payload(text)
        except ValueError:
            pass
        if doc is None or "dates" not in doc:
            # The page renderer drops the JSON's final "]}" immediately
            # before the closing fence (observed on every schedule capture;
            # the game data itself is complete). Restore the two structural
            # delimiters and retry — anything more is rejected loudly.
            m2 = re.search(r"```[a-zA-Z]*\n(.*?)```", text, re.DOTALL)
            if not m2:
                raise SystemExit(f"no fenced payload in {cap.name}")
            body = m2.group(1).strip()
            try:
                doc = json.loads(body + "]}")
            except json.JSONDecodeError as err:
                raise SystemExit(f"unrepairable capture {cap.name}: {err}")
        rows = []
        n_no_score = 0
        for day in doc.get("dates", []):
            for g in day.get("games", []):
                away = g.get("teams", {}).get("away", {})
                home = g.get("teams", {}).get("home", {})
                a_id = (away.get("team") or {}).get("id")
                h_id = (home.get("team") or {}).get("id")
                a_s, h_s = away.get("score"), home.get("score")
                date = g.get("officialDate", "")
                if a_s is None or h_s is None:
                    n_no_score += 1
                    continue  # postponed/suspended without a result: stays missing
                if g["gamePk"] in seen_pks:
                    continue  # already ingested on a previous run
                # validations
                assert a_id in VALID_IDS and h_id in VALID_IDS, (g["gamePk"], a_id, h_id)
                assert a_id != h_id, g["gamePk"]
                assert 0 <= int(a_s) <= 99 and 0 <= int(h_s) <= 99, g["gamePk"]
                assert int(a_s) != int(h_s), f"tie in R game {g['gamePk']}"
                assert w0 <= date <= w1, (date, w0, w1)
                assert g["gamePk"] not in seen_pks, f"duplicate game_pk {g['gamePk']}"
                seen_pks.add(int(g["gamePk"]))
                rows.append((g["gamePk"], date, a_id, int(a_s), h_id, int(h_s)))
        url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1"
               f"&startDate={w0}&endDate={w1}&gameTypes=R"
               f"&fields=dates,games,gamePk,officialDate,teams,away,home,team,id,score")
        csv_path = cap.with_name(cap.name.replace("sched_", "parsed_").replace(".txt", ".csv"))
        with csv_path.open("w") as fh:
            for r in rows:
                fh.write(",".join(str(x) for x in r) + "\n")
        print(f"{cap.name}: {len(rows)} games with scores "
              f"({n_no_score} postponed/no-result skipped) -> {csv_path.name}")
        if rows:
            r = subprocess.run(
                [sys.executable, "-m", "mlb_predict", "ingest-scores", str(csv_path),
                 "--source-url", url], cwd=REPO, capture_output=True, text=True)
            print("   ", r.stdout.strip() or r.stderr.strip())
            assert r.returncode == 0, r.stderr
            total_new += len(rows)
    print(f"TOTAL new games ingested: {total_new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
