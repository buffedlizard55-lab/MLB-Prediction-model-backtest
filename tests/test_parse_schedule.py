import unittest

import pandas as pd

from mlb_predict.parse_schedule import payload_to_records, payloads_to_frame


def _game(pk, a_score, h_score, a_win, status="Final", gtype="R"):
    return {
        "gamePk": pk, "season": "2025", "gameType": gtype,
        "officialDate": "2025-05-01", "gameDate": "2025-05-01T23:05:00Z",
        "status": {"abstractGameState": status},
        "teams": {
            "away": {"team": {"id": 110, "name": "Away Team"},
                     "score": a_score, "isWinner": a_win},
            "home": {"team": {"id": 111, "name": "Home Team"},
                     "score": h_score, "isWinner": (not a_win)},
        },
    }


class ParseScheduleTests(unittest.TestCase):
    def test_final_regular_season_game_parses(self):
        rows = payload_to_records({"dates": [{"games": [_game(1, 3, 5, False)]}]})
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["home_win"], 1)
        self.assertEqual(r["total_runs"], 8)
        self.assertEqual(r["run_diff_home"], 2)

    def test_non_final_and_non_regular_rejected(self):
        payload = {"dates": [{"games": [
            _game(2, 3, 5, False, status="Preview"),
            _game(3, 3, 5, False, gtype="S"),
        ]}]}
        self.assertEqual(payload_to_records(payload), [])

    def test_inconsistent_winner_flag_rejected(self):
        g = _game(4, 3, 5, False)
        g["teams"]["away"]["isWinner"] = True   # contradicts scores
        g["teams"]["home"]["isWinner"] = True
        self.assertEqual(payload_to_records({"dates": [{"games": [g]}]}), [])

    def test_missing_score_rejected(self):
        g = _game(5, 3, 5, False)
        g["teams"]["home"]["score"] = None
        self.assertEqual(payload_to_records({"dates": [{"games": [g]}]}), [])

    def test_dedupe_and_sort(self):
        p1 = {"dates": [{"games": [_game(10, 1, 2, False)]}]}
        p2 = {"dates": [{"games": [_game(10, 1, 2, False),
                                   _game(9, 4, 0, True)]}]}
        df = payloads_to_frame([p1, p2])
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df["game_pk"]), [9, 10])


if __name__ == "__main__":
    unittest.main()
