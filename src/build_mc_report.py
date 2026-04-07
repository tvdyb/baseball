#!/usr/bin/env python3
"""
Build a multi-page strategy report PDF for the 7th-Inning MC Strategy.
In-game Monte Carlo simulation with phase-dependent Kelly rebalancing on Polymarket.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path("/Users/wilsonw/baseball")
OUT_PDF = BASE / "outputs" / "mc_strategy_report.pdf"
INGAME_CSV = BASE / "data" / "audit" / "sim_vs_kalshi_ingame_2025.csv"

# ── Constants ──────────────────────────────────────────────────────────────
PHASE_CONFIDENCE = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,
                    6: 0.15, 7: 0.40, 8: 0.65, 9: 1.00}
EXTRAS_CONFIDENCE = 1.00
KELLY_FRAC = 0.25
MIN_EDGE = 0.03
TAKER_FEE_RATE = 0.03
BANKROLL = 10_000.0
FIG_SIZE = (16, 9)

# ── Color scheme ───────────────────────────────────────────────────────────
BG_COLOR = "#0e1117"
CARD_COLOR = "#1a1d23"
TEXT_COLOR = "#e6e6e6"
ACCENT = "#4fc3f7"
GREEN = "#66bb6a"
RED = "#ef5350"
GOLD = "#ffd54f"
MUTED = "#888888"
PURPLE = "#ab47bc"


def setup_fig(title=None):
    fig = plt.figure(figsize=FIG_SIZE, facecolor=BG_COLOR)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if title:
        ax.text(0.5, 0.93, title, ha="center", va="top",
                fontsize=28, fontweight="bold", color=TEXT_COLOR,
                family="sans-serif")
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING & KELLY SIMULATION
# ══════════════════════════════════════════════════════════════════════════

def load_ingame_data():
    return pd.read_csv(INGAME_CSV)


def simulate_kelly_rebalancing(df):
    """Simulate phase-dependent Kelly rebalancing from inning 7+."""
    # Sort games chronologically (not by game_pk which is an arbitrary ID)
    game_dates = df.groupby("game_pk")["game_date"].first().reset_index()
    game_dates = game_dates.sort_values("game_date")
    games = game_dates["game_pk"].tolist()
    game_results = []
    bankroll = BANKROLL
    bankroll_history = [bankroll]

    for gpk in games:
        game = df[df["game_pk"] == gpk].sort_values("game_progress")
        home_win = game.iloc[0]["home_win"]

        position = 0.0
        cost_basis = 0.0
        traded = False

        for _, state in game.iterrows():
            inning = int(state["inning"])
            phase_conf = PHASE_CONFIDENCE.get(inning, EXTRAS_CONFIDENCE if inning >= 10 else 0.0)
            if phase_conf == 0:
                continue

            sim_wp = state["sim_home_wp"]
            mkt_price = state["kalshi_home_prob"]
            edge = sim_wp - mkt_price

            if abs(edge) < MIN_EDGE:
                target = 0.0
            elif edge > 0:
                kelly = edge / (1 - mkt_price)
                target = bankroll * KELLY_FRAC * phase_conf * kelly / 100
            else:
                kelly = abs(edge) / mkt_price
                target = -bankroll * KELLY_FRAC * phase_conf * kelly / 100

            max_pos = bankroll * 0.10 / 100
            target = np.clip(target, -max_pos, max_pos)
            delta = target - position
            if abs(delta) < 0.01:
                continue

            fee = abs(delta) * TAKER_FEE_RATE * mkt_price * (1 - mkt_price) * 100
            cost_basis += delta * mkt_price * 100 + fee
            position = target
            traded = True

        if traded and position != 0:
            settlement = position * 100 if home_win == 1 else 0
            pnl = settlement - cost_basis
            bankroll += pnl
            bankroll_history.append(bankroll)
            won = pnl > 0
            side = "HOME" if position > 0 else "AWAY"
            # Get last state with edge for summary
            late_states = game[game["inning"] >= 7].sort_values("game_progress")
            if len(late_states) > 0:
                last = late_states.iloc[-1]
                final_edge = last["edge"] if position > 0 else -last["edge"]
                mkt = last["kalshi_home_prob"] if position > 0 else 1 - last["kalshi_home_prob"]
                sim = last["sim_home_wp"] if position > 0 else 1 - last["sim_home_wp"]
            else:
                final_edge = 0
                mkt = 0.5
                sim = 0.5
            game_results.append({
                "game_pk": gpk,
                "game_date": game.iloc[0]["game_date"],
                "home_team": game.iloc[0]["home_team"],
                "away_team": game.iloc[0]["away_team"],
                "side": side,
                "mkt_price": mkt,
                "sim_prob": sim,
                "edge": abs(final_edge),
                "pnl": pnl,
                "won": int(won),
                "bankroll": bankroll,
                "final_position": position,
                "home_win": home_win,
            })

    return pd.DataFrame(game_results), bankroll_history


# ══════════════════════════════════════════════════════════════════════════
# SLIDES
# ══════════════════════════════════════════════════════════════════════════

def slide_title(pdf, results, bankroll_history):
    fig, ax = setup_fig()
    ax.text(0.5, 0.72, "7th Inning+ MC Strategy", ha="center", va="center",
            fontsize=52, fontweight="bold", color=TEXT_COLOR, family="sans-serif")
    ax.text(0.5, 0.60, "In-Game Kelly Rebalancing on Polymarket",
            ha="center", va="center", fontsize=30, color=ACCENT, family="sans-serif")

    n_games = len(results)
    total_pnl = results["pnl"].sum()
    final_br = bankroll_history[-1]
    ret = (final_br / BANKROLL - 1) * 100
    sharpe = results["pnl"].mean() / results["pnl"].std() * np.sqrt(n_games) if results["pnl"].std() > 0 else 0
    wr = results["won"].mean() * 100

    stats = [
        (f"{n_games}", "Games Traded"),
        (f"{wr:.0f}%", "Win Rate"),
        (f"{ret:+.1f}%", "Return"),
        (f"${total_pnl:+,.0f}", "P&L ($10K)"),
        (f"{sharpe:.2f}", "Sharpe Ratio"),
    ]
    for i, (val, label) in enumerate(stats):
        x = 0.10 + i * 0.175
        color = GREEN if "+" in str(val) or (val.replace("%", "").replace("$", "").replace(",", "").replace(".", "").lstrip("+-").isdigit() and float(val.replace("%", "").replace("$", "").replace(",", "")) > 50) else TEXT_COLOR
        ax.text(x, 0.38, val, ha="center", va="center",
                fontsize=32, fontweight="bold", color=color, family="sans-serif")
        ax.text(x, 0.31, label, ha="center", va="center",
                fontsize=14, color=MUTED, family="sans-serif")

    ax.text(0.5, 0.18, "2025 Season Backtest  |  198 games  |  3,521 state points",
            ha="center", va="center", fontsize=16, color=MUTED, family="sans-serif")
    ax.text(0.5, 0.10, "Quarter-Kelly  |  3% min edge  |  3% taker fee",
            ha="center", va="center", fontsize=12, color=MUTED, family="sans-serif", style="italic")

    pdf.savefig(fig)
    plt.close(fig)


def slide_mc_architecture(pdf):
    """Explain the MC simulation engine."""
    fig, ax = setup_fig("Monte Carlo Simulation Engine")

    sections = [
        ("At-Bat Level Markov Chain", [
            "Each plate appearance: sample outcome from 11-category distribution",
            "Outcomes: K, BB, HBP, 1B, 2B, 3B, HR, DP, ground out, fly out, line out",
            "Base-running transitions from empirical (outcome, base state, outs) matrix",
        ]),
        ("Matchup-Aware Distributions", [
            "Multi-output neural net predicts PA outcome distribution per batter vs SP",
            "Arsenal-based: pitcher pitch mix/velocity/movement vs batter tendencies",
            "Bayesian shrinkage toward league average for low-sample matchups",
        ]),
        ("In-Game State Injection", [
            "At each half-inning: fetch live score, outs, baserunners, pitcher state",
            "Simulate forward from CURRENT state (not pregame) -- 3,000 iterations",
            "Handles bullpen transitions, pinch hitters, extra innings",
        ]),
    ]

    y = 0.82
    for title, bullets in sections:
        ax.text(0.06, y, title, fontsize=18, fontweight="bold",
                color=GOLD, va="center", family="sans-serif")
        y -= 0.04
        for b in bullets:
            ax.text(0.09, y, f"  {b}", fontsize=13, color=TEXT_COLOR,
                    va="center", family="sans-serif")
            y -= 0.035
        y -= 0.025

    # Flow diagram
    boxes = ["Live Game\nState", "PA Outcome\nDistributions", "Markov Chain\nSimulation", "Win Prob\nDistribution", "Edge vs\nPolymarket"]
    box_w = 0.14
    box_h = 0.06
    y_flow = 0.10
    for i, label in enumerate(boxes):
        x = 0.08 + i * 0.185
        rect = FancyBboxPatch((x, y_flow - box_h / 2), box_w, box_h,
                               boxstyle="round,pad=0.008", facecolor=CARD_COLOR,
                               edgecolor=ACCENT, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y_flow, label, ha="center", va="center",
                fontsize=10, color=TEXT_COLOR, family="sans-serif")
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + box_w + 0.015, y_flow),
                        xytext=(x + box_w + 0.04, y_flow),
                        arrowprops=dict(arrowstyle="<-", color=ACCENT, lw=2))

    pdf.savefig(fig)
    plt.close(fig)


def slide_phase_analysis(pdf, df):
    """By-inning log loss comparison: sim vs market."""
    fig = plt.figure(figsize=FIG_SIZE, facecolor=BG_COLOR)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_COLOR)

    innings = []
    sim_lls = []
    mkt_lls = []
    ns = []
    for inn in range(1, 10):
        sub = df[df["inning"] == inn]
        if len(sub) < 10:
            continue
        y = sub["home_win"].values
        sim_ll = log_loss(y, np.clip(sub["sim_home_wp"], 0.01, 0.99))
        mkt_ll = log_loss(y, np.clip(sub["kalshi_home_prob"], 0.01, 0.99))
        innings.append(inn)
        sim_lls.append(sim_ll)
        mkt_lls.append(mkt_ll)
        ns.append(len(sub))

    x = np.arange(len(innings))
    w = 0.35
    bars1 = ax.bar(x - w / 2, sim_lls, w, color=ACCENT, alpha=0.85, label="MC Simulator")
    bars2 = ax.bar(x + w / 2, mkt_lls, w, color=GOLD, alpha=0.85, label="Kalshi Market")

    # Highlight innings where sim beats market
    for i, (s, m) in enumerate(zip(sim_lls, mkt_lls)):
        if s < m:
            ax.bar(x[i] - w / 2, s, w, color=GREEN, alpha=0.85)

    # Labels
    for i, (s, m, n) in enumerate(zip(sim_lls, mkt_lls, ns)):
        delta = s - m
        color = GREEN if delta < 0 else RED
        label = f"{delta:+.4f}"
        y_pos = max(s, m) + 0.008
        ax.text(x[i], y_pos, label, ha="center", fontsize=10, fontweight="bold", color=color)
        ax.text(x[i], min(s, m) - 0.015, f"n={n}", ha="center", fontsize=9, color=MUTED)

    # Phase confidence overlay
    ax2 = ax.twinx()
    conf_vals = [PHASE_CONFIDENCE.get(inn, 0) for inn in innings]
    ax2.plot(x, [c * 100 for c in conf_vals], color=PURPLE, linewidth=2.5,
             marker="s", markersize=8, label="Phase Confidence %", zorder=5)
    ax2.set_ylabel("Phase Confidence %", fontsize=13, color=PURPLE)
    ax2.set_ylim(-5, 110)
    ax2.tick_params(axis="y", colors=PURPLE, labelsize=11)
    ax2.spines["right"].set_color(PURPLE)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Inn {i}" for i in innings], fontsize=13, color=MUTED)
    ax.set_ylabel("Log Loss (lower = better)", fontsize=14, color=MUTED)
    ax.set_title("Model vs Market by Inning -- Where the Edge Lives",
                 fontsize=24, fontweight="bold", color=TEXT_COLOR, pad=15, family="sans-serif")
    ax.tick_params(axis="y", colors=MUTED, labelsize=11)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=12,
              facecolor=CARD_COLOR, edgecolor=MUTED, labelcolor=TEXT_COLOR, loc="upper right")

    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax2.spines["top"].set_visible(False)

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def slide_strategy_overview(pdf):
    """Explain the phase-dependent Kelly strategy."""
    fig, ax = setup_fig("Phase-Dependent Kelly Rebalancing")

    bullets = [
        ("When", "Innings 7-9+ only -- model worse than market in early innings"),
        ("Signal", "MC simulator win probability vs Polymarket mid-quote"),
        ("Edge", "Simulator WP - Polymarket price (3% minimum threshold)"),
        ("Sizing", "Quarter-Kelly scaled by phase confidence: 7th=40%, 8th=65%, 9th=100%"),
        ("Rebalancing", "Adjust position at each half-inning boundary (buy/sell contracts)"),
        ("Fees", "Polymarket taker fee: 3% x price x (1 - price), near-zero at extremes"),
    ]

    y = 0.80
    for label, desc in bullets:
        ax.text(0.08, y, label, fontsize=17, fontweight="bold",
                color=ACCENT, va="center", family="sans-serif")
        ax.text(0.28, y, desc, fontsize=15, color=TEXT_COLOR,
                va="center", family="sans-serif")
        y -= 0.065

    # Why late innings work box
    y_box = 0.28
    box_h = 0.25
    rect = FancyBboxPatch((0.06, y_box - box_h / 2), 0.88, box_h,
                            boxstyle="round,pad=0.015", facecolor=CARD_COLOR,
                            edgecolor=GOLD, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(0.5, y_box + 0.08, "Why the Edge Concentrates in Late Innings",
            ha="center", fontsize=18, fontweight="bold", color=GOLD, family="sans-serif")
    reasons = [
        "Early innings: enormous uncertainty, team strength dominates -- market is efficient",
        "Late innings: score is known, remaining outs are few -- sim's state model adds value",
        "Polymarket prices reflect pregame opinions that don't update fully mid-game",
        "Extreme prices (85%+) in late innings have near-zero taker fees -- cheap to trade",
    ]
    for i, r in enumerate(reasons):
        ax.text(0.10, y_box + 0.03 - i * 0.04, f"  {r}", fontsize=13,
                color=TEXT_COLOR, va="center", family="sans-serif")

    pdf.savefig(fig)
    plt.close(fig)


def slide_trade_table(pdf, results):
    """Table of every game traded with P&L and cumulative P&L."""
    if len(results) == 0:
        return

    bets = results.sort_values("game_date").reset_index(drop=True)
    bets["cum_pnl"] = bets["pnl"].cumsum()
    bets["game_date"] = pd.to_datetime(bets["game_date"])

    headers = ["Date", "Matchup", "Side", "Market", "Sim", "Edge", "W/L", "P&L", "Cum P&L", "Bankroll"]
    max_rows = 16
    pages = [bets.iloc[i:i + max_rows] for i in range(0, len(bets), max_rows)]

    for page_idx, page_bets in enumerate(pages):
        suffix = "" if page_idx == 0 else " (cont.)"
        fig, ax = setup_fig(f"Every Trade{suffix}")

        n = len(page_bets)
        row_h = 0.68 / (n + 1)
        y_start = 0.83
        col_x = [0.02, 0.10, 0.30, 0.40, 0.50, 0.59, 0.68, 0.76, 0.85, 0.93]

        for j, h in enumerate(headers):
            ax.text(col_x[j], y_start, h, fontsize=11, fontweight="bold",
                    color=ACCENT, va="center", family="sans-serif")

        for i, (_, r) in enumerate(page_bets.iterrows()):
            y = y_start - (i + 1) * row_h
            bg_color = GREEN if r["won"] else RED
            rect = FancyBboxPatch((0.01, y - row_h * 0.4), 0.98, row_h * 0.8,
                                   boxstyle="round,pad=0.003",
                                   facecolor=bg_color, alpha=0.12, edgecolor="none")
            ax.add_patch(rect)

            team_str = f"{r['away_team']}@{r['home_team']}"
            vals = [
                r["game_date"].strftime("%m/%d"),
                team_str,
                r["side"],
                f"{r['mkt_price']:.2f}",
                f"{r['sim_prob']:.2f}",
                f"{r['edge']:.1%}",
                "W" if r["won"] else "L",
                f"${r['pnl']:+,.0f}",
                f"${r['cum_pnl']:+,.0f}",
                f"${r['bankroll']:,.0f}",
            ]
            for j, v in enumerate(vals):
                if j == 6:  # W/L
                    c = GREEN if r["won"] else RED
                elif j == 8:  # Cum P&L
                    c = GREEN if r["cum_pnl"] > 0 else RED
                else:
                    c = TEXT_COLOR
                ax.text(col_x[j], y, v, fontsize=10, color=c, va="center", family="sans-serif")

        pdf.savefig(fig)
        plt.close(fig)


def slide_cumulative_pnl(pdf, results, bankroll_history):
    """Cumulative P&L curve."""
    fig = plt.figure(figsize=FIG_SIZE, facecolor=BG_COLOR)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_COLOR)

    pnl_vals = [b - BANKROLL for b in bankroll_history[1:]]
    results = results.sort_values("game_date").reset_index(drop=True)

    ax.fill_between(range(len(pnl_vals)), pnl_vals, 0,
                     where=[p >= 0 for p in pnl_vals], color=GREEN, alpha=0.3, interpolate=True)
    ax.fill_between(range(len(pnl_vals)), pnl_vals, 0,
                     where=[p < 0 for p in pnl_vals], color=RED, alpha=0.3, interpolate=True)
    ax.plot(range(len(pnl_vals)), pnl_vals, color=TEXT_COLOR, linewidth=2.5)
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")

    n = len(results)
    if n > 15:
        tick_idx = np.linspace(0, n - 1, min(12, n), dtype=int)
    else:
        tick_idx = range(n)
    dates = pd.to_datetime(results["game_date"])
    ax.set_xticks(list(tick_idx))
    ax.set_xticklabels([dates.iloc[i].strftime("%m/%d") for i in tick_idx],
                        rotation=45, fontsize=11, color=MUTED)
    ax.tick_params(axis="y", colors=MUTED, labelsize=12)

    total_pnl = pnl_vals[-1] if pnl_vals else 0
    ret = total_pnl / BANKROLL * 100
    sharpe = results["pnl"].mean() / results["pnl"].std() * np.sqrt(n) if results["pnl"].std() > 0 else 0
    max_dd = min(np.minimum.accumulate(pnl_vals) - np.maximum.accumulate(
        [0] + pnl_vals[:-1])) if pnl_vals else 0

    # Running max drawdown
    peaks = np.maximum.accumulate([0] + pnl_vals)
    drawdowns = np.array([0] + pnl_vals) - peaks
    max_dd = drawdowns.min()

    ann = (f"$10K bankroll  |  {n} games  |  Return: {ret:+.1f}%  |  "
           f"P&L: ${total_pnl:+,.0f}  |  Sharpe: {sharpe:.2f}  |  Max DD: ${max_dd:,.0f}")
    ax.text(0.5, 1.06, "Cumulative P&L -- Quarter-Kelly Rebalancing",
            transform=ax.transAxes, ha="center",
            fontsize=26, fontweight="bold", color=TEXT_COLOR, family="sans-serif")
    ax.text(0.5, 1.01, ann, transform=ax.transAxes, ha="center",
            fontsize=13, color=GREEN if total_pnl > 0 else RED, family="sans-serif")

    ax.set_ylabel("P&L ($)", fontsize=14, color=MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.92])
    pdf.savefig(fig)
    plt.close(fig)


def slide_game_trace(pdf, df):
    """Show a single game trace: sim vs market WP over innings."""
    # Find a good game: close game with lots of state points and a late swing
    game_counts = df.groupby("game_pk").size()
    candidates = game_counts[game_counts >= 14].index
    if len(candidates) == 0:
        candidates = game_counts.nlargest(5).index

    best_game = None
    best_swing = 0
    for gpk in candidates:
        game = df[df["game_pk"] == gpk].sort_values("game_progress")
        late = game[game["inning"] >= 7]
        if len(late) < 4:
            continue
        swing = abs(late["sim_home_wp"].max() - late["sim_home_wp"].min())
        if swing > best_swing:
            best_swing = swing
            best_game = gpk

    if best_game is None:
        best_game = candidates[0]

    game = df[df["game_pk"] == best_game].sort_values("game_progress")
    home = game.iloc[0]["home_team"]
    away = game.iloc[0]["away_team"]
    gdate = game.iloc[0]["game_date"]
    home_win = game.iloc[0]["home_win"]
    result = f"{home} wins" if home_win else f"{away} wins"

    fig = plt.figure(figsize=FIG_SIZE, facecolor=BG_COLOR)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_COLOR)

    x = range(len(game))
    ax.plot(x, game["sim_home_wp"], color=ACCENT, linewidth=2.5, marker="o",
            markersize=5, label="MC Simulator", zorder=3)
    ax.plot(x, game["kalshi_home_prob"], color=GOLD, linewidth=2.5, marker="s",
            markersize=5, label="Kalshi Market", zorder=3)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle="--", alpha=0.5)

    # Shade the 7th+ inning region
    inn7_start = None
    for i, (_, row) in enumerate(game.iterrows()):
        if row["inning"] >= 7 and inn7_start is None:
            inn7_start = i
    if inn7_start is not None:
        ax.axvspan(inn7_start - 0.5, len(game) - 0.5, alpha=0.08, color=GREEN, label="Trading Zone (7th+)")

    # Inning labels
    prev_inn = 0
    for i, (_, row) in enumerate(game.iterrows()):
        if row["inning"] != prev_inn:
            ax.axvline(i - 0.5, color=MUTED, linewidth=0.5, alpha=0.3)
            ax.text(i, 0.05, f"{int(row['inning'])}", fontsize=9, color=MUTED,
                    ha="center", transform=ax.get_xaxis_transform())
            prev_inn = row["inning"]

    ax.set_xticks(list(x))
    half_labels = [f"{'T' if r['top_bottom'] == 'Top' else 'B'}{int(r['inning'])}"
                   for _, r in game.iterrows()]
    ax.set_xticklabels(half_labels, fontsize=9, color=MUTED, rotation=45)
    ax.set_ylabel("Home Win Probability", fontsize=14, color=MUTED)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="y", colors=MUTED, labelsize=12)

    ax.set_title(f"Game Trace: {away} @ {home} -- {gdate} ({result})",
                 fontsize=24, fontweight="bold", color=TEXT_COLOR, pad=15, family="sans-serif")
    ax.legend(fontsize=12, facecolor=CARD_COLOR, edgecolor=MUTED, labelcolor=TEXT_COLOR, loc="upper right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def slide_edge_by_inning(pdf, df):
    """Distribution of edges by inning (7-9 only)."""
    fig = plt.figure(figsize=FIG_SIZE, facecolor=BG_COLOR)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_COLOR)

    colors = {7: ACCENT, 8: GOLD, 9: GREEN}
    for inn in [7, 8, 9]:
        sub = df[df["inning"] == inn]
        edges = sub["edge"].values
        ax.hist(edges, bins=40, alpha=0.5, color=colors[inn],
                label=f"Inning {inn} (n={len(sub)}, mean={edges.mean():+.3f})", edgecolor="none")

    ax.axvline(0, color=TEXT_COLOR, linewidth=1.5, linestyle="-", alpha=0.5)
    ax.axvline(MIN_EDGE, color=GREEN, linewidth=1.5, linestyle="--", alpha=0.7, label=f"Min edge ({MIN_EDGE:.0%})")
    ax.axvline(-MIN_EDGE, color=RED, linewidth=1.5, linestyle="--", alpha=0.7)

    ax.set_xlabel("Edge (Sim WP - Market Price)", fontsize=14, color=MUTED)
    ax.set_ylabel("Count", fontsize=14, color=MUTED)
    ax.set_title("Edge Distribution by Inning (7th-9th)",
                 fontsize=24, fontweight="bold", color=TEXT_COLOR, pad=15, family="sans-serif")
    ax.tick_params(colors=MUTED, labelsize=11)
    ax.legend(fontsize=12, facecolor=CARD_COLOR, edgecolor=MUTED, labelcolor=TEXT_COLOR)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def slide_results_summary(pdf, results, df):
    """Combined results summary."""
    fig, ax = setup_fig("Backtest Summary")

    n_games = len(results)
    total_pnl = results["pnl"].sum()
    ret = total_pnl / BANKROLL * 100
    wr = results["won"].mean() * 100
    sharpe = results["pnl"].mean() / results["pnl"].std() * np.sqrt(n_games) if results["pnl"].std() > 0 else 0
    avg_pnl = results["pnl"].mean()
    std_pnl = results["pnl"].std()

    # Overall box
    rect = FancyBboxPatch((0.06, 0.55), 0.88, 0.30,
                            boxstyle="round,pad=0.015", facecolor=CARD_COLOR,
                            edgecolor=GREEN, linewidth=2)
    ax.add_patch(rect)
    ax.text(0.5, 0.82, "2025 Season Backtest  (198 games, 3,521 state points)",
            ha="center", fontsize=20, fontweight="bold", color=GREEN, family="sans-serif")

    lines = [
        f"Games Traded: {n_games}   |   Win Rate: {wr:.0f}%   |   Sharpe: {sharpe:.2f}",
        f"Return: {ret:+.1f}% ($10K -> ${BANKROLL + total_pnl:,.0f})   |   Avg P&L: ${avg_pnl:+.1f}/game",
        f"Quarter-Kelly sizing   |   3% min edge   |   3% Polymarket taker fee",
    ]
    for i, line in enumerate(lines):
        c = TEXT_COLOR if i != 1 else (GREEN if total_pnl > 0 else RED)
        ax.text(0.5, 0.73 - i * 0.05, line, ha="center",
                fontsize=15, color=c, family="sans-serif")

    # By-inning log loss table
    ax.text(0.5, 0.48, "By-Inning Log Loss Comparison", ha="center", fontsize=18,
            fontweight="bold", color=GOLD, family="sans-serif")

    headers = ["Inning", "Sim LL", "Market LL", "Delta", "Winner", "Confidence"]
    col_x = [0.10, 0.25, 0.40, 0.55, 0.68, 0.82]
    y = 0.42
    for j, h in enumerate(headers):
        ax.text(col_x[j], y, h, fontsize=12, fontweight="bold", color=ACCENT,
                va="center", family="sans-serif")

    for inn in range(1, 10):
        sub = df[df["inning"] == inn]
        if len(sub) < 10:
            continue
        y -= 0.035
        yt = sub["home_win"].values
        sim_ll = log_loss(yt, np.clip(sub["sim_home_wp"], 0.01, 0.99))
        mkt_ll = log_loss(yt, np.clip(sub["kalshi_home_prob"], 0.01, 0.99))
        delta = sim_ll - mkt_ll
        winner = "SIM" if delta < 0 else "MKT"
        conf = PHASE_CONFIDENCE.get(inn, 0)
        winner_color = GREEN if delta < 0 else RED

        vals = [str(inn), f"{sim_ll:.4f}", f"{mkt_ll:.4f}", f"{delta:+.4f}", winner, f"{conf:.0%}"]
        colors = [TEXT_COLOR, TEXT_COLOR, TEXT_COLOR, winner_color, winner_color, PURPLE]
        for j, (v, c) in enumerate(zip(vals, colors)):
            ax.text(col_x[j], y, v, fontsize=11, color=c, va="center", family="sans-serif")

    pdf.savefig(fig)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("Loading in-game backtest data...")
    df = load_ingame_data()
    print(f"  {len(df)} state points across {df['game_pk'].nunique()} games")

    print("Simulating Kelly rebalancing...")
    results, bankroll_history = simulate_kelly_rebalancing(df)
    print(f"  {len(results)} games traded")
    print(f"  P&L: ${results['pnl'].sum():+,.0f}  ({(bankroll_history[-1] / BANKROLL - 1) * 100:+.1f}%)")
    print(f"  Sharpe: {results['pnl'].mean() / results['pnl'].std() * np.sqrt(len(results)):.2f}")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(OUT_PDF)) as pdf:
        print("Slide 1: Title...")
        slide_title(pdf, results, bankroll_history)

        print("Slide 2: MC Architecture...")
        slide_mc_architecture(pdf)

        print("Slide 3: Phase Analysis...")
        slide_phase_analysis(pdf, df)

        print("Slide 4: Strategy Overview...")
        slide_strategy_overview(pdf)

        print("Slide 5: Edge Distribution...")
        slide_edge_by_inning(pdf, df)

        print("Slide 6: Game Trace...")
        slide_game_trace(pdf, df)

        print("Slide 7: Trade Table...")
        slide_trade_table(pdf, results)

        print("Slide 8: Cumulative P&L...")
        slide_cumulative_pnl(pdf, results, bankroll_history)

        print("Slide 9: Results Summary...")
        slide_results_summary(pdf, results, df)

    print(f"\nReport saved to {OUT_PDF}")


if __name__ == "__main__":
    main()
