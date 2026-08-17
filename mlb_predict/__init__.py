"""MLB betting-outcome prediction model with walk-forward backtesting.

Data sources (official MLB hosts, via the two upstream repos):
  * buffedlizard55-lab/MLB-PBP      -> scoreboard / schedule / full PBP live feeds
  * karagemop466-tech/StatcastMLB   -> Baseball Savant Statcast + leaderboards

Method: correlation-driven selective input filtering + Monte Carlo game
simulation + walk-forward evaluation. Zero fabricated data: every number
in the pipeline traces back to an official MLB response on disk.
"""

__version__ = "0.1.0"
