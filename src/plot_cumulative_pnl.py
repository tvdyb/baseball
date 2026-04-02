#!/usr/bin/env python3
"""
Plot cumulative P&L from betting 1 contract per game on the side
where the model sees edge over Kalshi.

On Kalshi, 1 contract costs the market price (in cents) and pays $1 if correct.
- Bet YES (model_prob > kalshi_prob): pay kalshi_home_prob, win $1 if home wins
  P&L = (1 - cost) if win, -cost if lose
- Bet NO (model_prob < kalshi_prob): pay (1 - kalshi_home_prob), win $1 if away wins
  P&L = (1 - cost) if win, -cost if lose

Usage:
    python src/plot_cumulative_pnl.py
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    df = pd.read_csv(DATA_DIR / "full_2025_comparison.csv")
    df = df.dropna(subset=["kalshi_home_prob"]).copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    edge = df["model_home_prob"] - df["kalshi_home_prob"]
    kalshi_p = df["kalshi_home_prob"].values
    outcome = df["home_win"].values

    # Bet 1 contract on the side the model favors
    pnl = []
    for i in range(len(df)):
        if edge.iloc[i] > 0:
            # Bet YES home: cost = kalshi_p, pays $1 if home wins
            cost = kalshi_p[i]
            pnl.append((1 - cost) if outcome[i] == 1 else -cost)
        else:
            # Bet NO home (= YES away): cost = 1 - kalshi_p, pays $1 if away wins
            cost = 1 - kalshi_p[i]
            pnl.append((1 - cost) if outcome[i] == 0 else -cost)

    pnl = np.array(pnl)
    cum_pnl = np.cumsum(pnl)
    dates = df["game_date"].values

    # Stats
    n_bets = len(pnl)
    total_risked = sum(kalshi_p[i] if edge.iloc[i] > 0 else 1 - kalshi_p[i] for i in range(n_bets))
    total_pnl = cum_pnl[-1]
    roi = total_pnl / total_risked
    win_rate = (pnl > 0).mean()
    max_dd = 0
    peak = cum_pnl[0]
    for v in cum_pnl:
        peak = max(peak, v)
        max_dd = min(max_dd, v - peak)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, cum_pnl, linewidth=1.5, color="#2563eb")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.fill_between(dates, cum_pnl, 0,
                    where=cum_pnl >= 0, alpha=0.15, color="#2563eb")
    ax.fill_between(dates, cum_pnl, 0,
                    where=cum_pnl < 0, alpha=0.15, color="#dc2626")

    ax.set_title("Cumulative P&L: 1 Contract Per Game on Model-Favored Side",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative P&L ($)", fontsize=11)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # Stats box
    stats_text = (
        f"Total P&L: ${total_pnl:+,.2f}\n"
        f"ROI: {roi:+.1%}\n"
        f"Bets: {n_bets:,}\n"
        f"Win rate: {win_rate:.1%}\n"
        f"Max drawdown: ${max_dd:,.2f}"
    )
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="gray", alpha=0.9))

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = DATA_DIR / "cumulative_pnl.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")
    print(f"\n{stats_text}")


if __name__ == "__main__":
    main()
