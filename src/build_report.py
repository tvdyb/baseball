"""
Build a multi-page strategy report PDF using matplotlib.
MLB Moneyline Edge: Exploiting Polymarket Inefficiencies
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path("/Users/wilsonw/baseball")
OUT_PDF = BASE / "outputs" / "strategy_report.pdf"
BET_LOG_2025 = BASE / "outputs" / "bet_log_2025.csv"
SBR_2026 = BASE / "data" / "odds" / "sbr_ml_2026.parquet"
POLY_CLOSING = BASE / "data" / "polymarket" / "poly_closing_prices.parquet"
NN_FEATURES = BASE / "data" / "features" / "nn_features_2025.parquet"

# ── Constants ──────────────────────────────────────────────────────────────
EDGE_THRESHOLD = 0.03
POLY_FEE = 0.0075
FIG_SIZE = (16, 9)
REG_SEASON_START = "2026-03-25"

# ── Color scheme ───────────────────────────────────────────────────────────
BG_COLOR = "#0e1117"
CARD_COLOR = "#1a1d23"
TEXT_COLOR = "#e6e6e6"
ACCENT = "#4fc3f7"
GREEN = "#66bb6a"
RED = "#ef5350"
GOLD = "#ffd54f"
MUTED = "#888888"

# Team abbreviation mapping (DK -> Poly)
TEAM_MAP = {"ATH": "OAK", "WAS": "WSH"}

def map_team(t):
    return TEAM_MAP.get(t, t)


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
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_2025_poly_bets():
    """Load 2025 bet log, use LGB model prob as signal vs Polymarket."""
    df = pd.read_csv(BET_LOG_2025)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["poly_home", "poly_away", "lgb_home", "lgb_away"]).copy()
    # LGB model prob is the signal (includes DK + pitcher matchup features)
    df["edge_home"] = df["lgb_home"] - df["poly_home"]
    df["edge_away"] = df["lgb_away"] - df["poly_away"]
    bets = []
    for _, r in df.iterrows():
        eh, ea = r["edge_home"], r["edge_away"]
        if eh >= EDGE_THRESHOLD or ea >= EDGE_THRESHOLD:
            if eh >= ea:
                side = "home"
                price = r["poly_home"]
                edge = eh
                signal_prob = r["lgb_home"]
                won = r["result"] == "home_win"
            else:
                side = "away"
                price = r["poly_away"]
                edge = ea
                signal_prob = r["lgb_away"]
                won = r["result"] == "away_win"
            pnl = (1 - price - POLY_FEE) if won else (-price - POLY_FEE)
            kelly_full = edge / (1 - price) if price < 1 else 0
            kelly_quarter = kelly_full * 0.25
            bets.append({
                "date": r["date"], "home": r["home"], "away": r["away"],
                "side": side, "price": price, "signal_prob": signal_prob,
                "edge": edge, "won": won, "pnl": pnl,
                "kelly_full": kelly_full, "kelly_quarter": kelly_quarter,
            })
    return pd.DataFrame(bets)


def load_2026_bets():
    """Match LGB predictions to Polymarket for 2026, apply betting logic."""
    # Load LGB predictions for 2026
    lgb_path = BASE / "data" / "features" / "lgb_predictions_2026.parquet"
    lgb_preds = pd.read_parquet(lgb_path)
    lgb_preds["game_date"] = pd.to_datetime(lgb_preds["game_date"])
    lgb_preds = lgb_preds[lgb_preds["game_date"] >= REG_SEASON_START].copy()

    # Also load SBR for game results (home_score, away_score)
    dk = pd.read_parquet(SBR_2026)
    dk["game_date"] = pd.to_datetime(dk["game_date"])

    poly = pd.read_parquet(POLY_CLOSING)
    poly["game_date"] = pd.to_datetime(poly["game_date"])
    poly = poly[poly["game_date"] >= REG_SEASON_START].copy()
    poly = poly[(poly["poly_team0_prob"] >= 0.18) & (poly["poly_team0_prob"] <= 0.82)]
    poly = poly[(poly["poly_team1_prob"] >= 0.18) & (poly["poly_team1_prob"] <= 0.82)]

    bets = []
    for _, g in lgb_preds.iterrows():
        home = map_team(g["home_team"])
        away = map_team(g["away_team"])
        gd = g["game_date"]

        # Match polymarket
        m = poly[(poly["game_date"] == gd)]
        match = m[(m["team0"] == home) & (m["team1"] == away)]
        if len(match) == 1:
            p_home = match.iloc[0]["poly_team0_prob"]
            p_away = match.iloc[0]["poly_team1_prob"]
            t0_won = match.iloc[0]["team0_won"]
            home_won = t0_won
        else:
            match = m[(m["team0"] == away) & (m["team1"] == home)]
            if len(match) == 1:
                p_home = match.iloc[0]["poly_team1_prob"]
                p_away = match.iloc[0]["poly_team0_prob"]
                t0_won = match.iloc[0]["team0_won"]
                home_won = not t0_won
            else:
                continue

        lgb_home = g["lgb_home_prob"]
        lgb_away = g["lgb_away_prob"]
        eh = lgb_home - p_home
        ea = lgb_away - p_away

        if eh >= EDGE_THRESHOLD or ea >= EDGE_THRESHOLD:
            if eh >= ea:
                side = "home"
                price = p_home
                edge = eh
                signal = lgb_home
                won = bool(home_won)
            else:
                side = "away"
                price = p_away
                edge = ea
                signal = lgb_away
                won = not bool(home_won)
            pnl = (1 - price - POLY_FEE) if won else (-price - POLY_FEE)
            kelly_full = edge / (1 - price) if price < 1 else 0
            kelly_quarter = kelly_full * 0.25

            # Get scores from SBR for result string
            sbr_match = dk[(dk["game_date"] == gd) & (dk["home_team"] == g["home_team"]) & (dk["away_team"] == g["away_team"])]
            if len(sbr_match) > 0:
                hs = int(sbr_match.iloc[0]["home_score"])
                aws = int(sbr_match.iloc[0]["away_score"])
                result_str = f"{g['home_team']} {hs}-{aws} {g['away_team']}"
            else:
                result_str = ""

            bets.append({
                "date": gd, "home": g["home_team"], "away": g["away_team"],
                "side": side, "poly_price": price, "signal_prob": signal,
                "edge": edge, "won": won, "pnl": pnl,
                "kelly_full": kelly_full, "kelly_quarter": kelly_quarter,
                "result_str": result_str,
            })
    return pd.DataFrame(bets)


# ══════════════════════════════════════════════════════════════════════════
# SLIDES — Section 1: Modeling & Methods
# ══════════════════════════════════════════════════════════════════════════

def slide_title(pdf, bets_2025, bets_2026):
    fig, ax = setup_fig()
    ax.text(0.5, 0.72, "MLB Moneyline Edge", ha="center", va="center",
            fontsize=52, fontweight="bold", color=TEXT_COLOR, family="sans-serif")
    ax.text(0.5, 0.60, "Exploiting Polymarket Inefficiencies",
            ha="center", va="center", fontsize=30, color=ACCENT, family="sans-serif")

    total_bets = len(bets_2025) + len(bets_2026)
    total_pnl = bets_2025["pnl"].sum() + bets_2026["pnl"].sum()
    wag_25 = bets_2025["price"].sum() if "price" in bets_2025.columns else 0
    wag_26 = bets_2026["poly_price"].sum() if len(bets_2026) > 0 else 0
    total_wagered = wag_25 + wag_26
    total_wins = bets_2025["won"].sum() + bets_2026["won"].sum()
    win_rate = total_wins / total_bets * 100 if total_bets else 0
    roi = total_pnl / total_wagered * 100 if total_wagered else 0

    stats = [
        (f"{total_bets}", "Total Bets"),
        (f"{win_rate:.1f}%", "Win Rate"),
        (f"{roi:+.1f}%", "ROI"),
        (f"${total_pnl:+.2f}", "P&L (per $1)"),
    ]
    for i, (val, label) in enumerate(stats):
        x = 0.15 + i * 0.215
        color = GREEN if "+" in val or float(val.replace("%","").replace("$","").replace(",","")) > 50 else TEXT_COLOR
        ax.text(x, 0.38, val, ha="center", va="center",
                fontsize=32, fontweight="bold", color=color, family="sans-serif")
        ax.text(x, 0.31, label, ha="center", va="center",
                fontsize=14, color=MUTED, family="sans-serif")

    dates_all = list(bets_2025["date"]) + list(bets_2026["date"])
    d_min = min(dates_all).strftime("%b %d, %Y")
    d_max = max(dates_all).strftime("%b %d, %Y")
    ax.text(0.5, 0.18, f"{d_min}  --  {d_max}", ha="center", va="center",
            fontsize=16, color=MUTED, family="sans-serif")
    ax.text(0.5, 0.10, "April 2026", ha="center", va="center",
            fontsize=12, color=MUTED, family="sans-serif", style="italic")

    pdf.savefig(fig)
    plt.close(fig)


def slide_data_pipeline(pdf):
    """Slide explaining the data pipeline and feature engineering."""
    fig, ax = setup_fig("Data Pipeline & Feature Engineering")

    sections = [
        ("Statcast Pitch-Level Data", [
            "Every pitch thrown in MLB (velocity, spin, movement, location, result)",
            "xRV (expected Run Value) models assign value to each pitch",
            "Decomposed into Stuff, Location, and Sequencing scores per pitcher",
        ]),
        ("Matchup Model (Multi-Output Neural Network)", [
            "Predicts PA outcome distribution (K, BB, 1B, 2B, 3B, HR, ...) for each batter vs SP",
            "Inputs: pitcher arsenal profile (pitch mix, velocity, movement) + batter ID embeddings",
            "9 lineup slots x 11 outcomes = 99 features per side (198 total)",
        ]),
        ("Game-Level Features (LightGBM Input)", [
            "Pitcher eval scores (stuff, location, sequencing) for home & away SP",
            "Bullpen quality (xRV, fatigue), SP rest days, park factors, weather",
            "DraftKings devigged closing moneyline (market consensus probability)",
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

    # Flow diagram at the bottom
    boxes = ["Statcast\nPitch Data", "xRV + Stuff\nModels", "Matchup\nModel", "LightGBM\nWin Prob", "Edge vs\nPolymarket"]
    box_w = 0.14
    box_h = 0.06
    y_flow = 0.10
    for i, label in enumerate(boxes):
        x = 0.08 + i * 0.185
        rect = FancyBboxPatch((x, y_flow - box_h/2), box_w, box_h,
                               boxstyle="round,pad=0.008", facecolor=CARD_COLOR,
                               edgecolor=ACCENT, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + box_w/2, y_flow, label, ha="center", va="center",
                fontsize=10, color=TEXT_COLOR, family="sans-serif")
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + box_w + 0.015, y_flow),
                       xytext=(x + box_w + 0.04, y_flow),
                       arrowprops=dict(arrowstyle="<-", color=ACCENT, lw=2))

    pdf.savefig(fig)
    plt.close(fig)


def slide_pitcher_models(pdf):
    """Show SP stuff and location distributions from nn_features."""
    nn = pd.read_parquet(NN_FEATURES)
    nn = nn.dropna(subset=["home_sp_stuff"])

    for col, title, xlabel in [
        ("stuff", "Pitcher Stuff Model -- Raw Stuff Quality", "Stuff Score (xRV/100 pitches)"),
        ("location", "Pitcher Location Model -- Command Quality", "Location Score (xRV/100 pitches)"),
    ]:
        all_scores = pd.concat([nn[f"home_sp_{col}"], nn[f"away_sp_{col}"]]).dropna()

        fig = plt.figure(figsize=FIG_SIZE, facecolor=BG_COLOR)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG_COLOR)

        ax.hist(all_scores, bins=60, color=ACCENT, alpha=0.7, edgecolor="none")
        mean_val = all_scores.mean()
        std_val = all_scores.std()
        ax.axvline(mean_val, color=GOLD, linewidth=2, linestyle="--", label=f"Mean: {mean_val:.4f}")
        ax.axvline(mean_val + std_val, color=GREEN, linewidth=1.5, linestyle=":", alpha=0.7, label=f"+1 SD: {mean_val+std_val:.4f}")
        ax.axvline(mean_val - std_val, color=RED, linewidth=1.5, linestyle=":", alpha=0.7, label=f"-1 SD: {mean_val-std_val:.4f}")

        ax.set_title(title, fontsize=24, fontweight="bold", color=TEXT_COLOR, pad=15, family="sans-serif")
        ax.set_xlabel(xlabel, fontsize=14, color=MUTED)
        ax.set_ylabel("Count (game-starts)", fontsize=14, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=11)
        ax.legend(fontsize=12, facecolor=CARD_COLOR, edgecolor=MUTED, labelcolor=TEXT_COLOR)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(MUTED)
        ax.spines["bottom"].set_color(MUTED)

        fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])
        pdf.savefig(fig)
        plt.close(fig)


def slide_pa_distribution(pdf):
    """Show PA outcome distribution for a notable matchup."""
    nn = pd.read_parquet(NN_FEATURES)
    nn["game_date"] = pd.to_datetime(nn["game_date"])
    nn_good = nn.dropna(subset=["home_h0_K"])
    notable = nn_good[
        ((nn_good["home_team"] == "PHI") & (nn_good["away_team"] == "LAD")) |
        ((nn_good["home_team"] == "LAD") & (nn_good["away_team"] == "NYY"))
    ]
    if len(notable) == 0:
        notable = nn_good.head(1)
    row = notable.iloc[0]
    game_label = f"{row['away_team']} @ {row['home_team']} -- {row['game_date'].strftime('%b %d, %Y')}"

    outcomes = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "dp", "out_ground", "out_fly", "out_line"]
    outcome_labels = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "DP", "GB Out", "FB Out", "LD Out"]
    vals = [row[f"home_h0_{o}"] for o in outcomes]

    fig = plt.figure(figsize=FIG_SIZE, facecolor=BG_COLOR)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_COLOR)

    colors_bar = [RED if o in ["K", "dp", "out_ground", "out_fly", "out_line"]
                  else GREEN if o in ["1B", "2B", "3B", "HR", "BB", "HBP"]
                  else MUTED for o in outcomes]

    bars = ax.bar(range(len(outcomes)), vals, color=colors_bar, alpha=0.85, edgecolor="none", width=0.65)
    ax.set_xticks(range(len(outcomes)))
    ax.set_xticklabels(outcome_labels, fontsize=13, color=MUTED, rotation=0)
    ax.tick_params(axis="y", colors=MUTED, labelsize=12)
    ax.set_ylabel("Probability", fontsize=14, color=MUTED)

    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", fontsize=10, color=TEXT_COLOR)

    ax.set_title(f"Matchup Model -- Predicted PA Outcomes\n{game_label}\nHome Leadoff Hitter vs Away SP",
                 fontsize=22, fontweight="bold", color=TEXT_COLOR, pad=20, family="sans-serif")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# SLIDES — Section 2: Strategy & Calibration
# ══════════════════════════════════════════════════════════════════════════

def slide_strategy_overview(pdf):
    """Explain the betting strategy logic."""
    fig, ax = setup_fig("Betting Strategy")

    bullets = [
        ("Signal", "LightGBM win prob model (pitcher matchups + DK closing line as feature)"),
        ("Market", "Polymarket MLB moneyline contracts (binary outcomes)"),
        ("Edge", "LGB model probability - Polymarket contract price"),
        ("Threshold", "Only bet when edge > 3% (filters noise, keeps high-conviction)"),
        ("Sizing", "Quarter-Kelly: f = 0.25 x edge / (1 - poly_price)"),
        ("Fees", "Polymarket charges $0.0075 per contract (0.75%)"),
    ]

    y = 0.80
    for label, desc in bullets:
        ax.text(0.08, y, label, fontsize=17, fontweight="bold",
                color=ACCENT, va="center", family="sans-serif")
        ax.text(0.24, y, desc, fontsize=16, color=TEXT_COLOR,
                va="center", family="sans-serif")
        y -= 0.065

    # Why this works box
    y_box = 0.28
    box_h = 0.25
    rect = FancyBboxPatch((0.06, y_box - box_h/2), 0.88, box_h,
                            boxstyle="round,pad=0.015", facecolor=CARD_COLOR,
                            edgecolor=GOLD, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(0.5, y_box + 0.08, "Why Polymarket Is Mispriced", ha="center", fontsize=18,
            fontweight="bold", color=GOLD, family="sans-serif")
    reasons = [
        "Polymarket is a prediction market with retail flow, not sharp bookmakers",
        "DraftKings lines are set by professionals with decades of modeling expertise",
        "Systematic 1-5% mispricing on ~30% of games creates a persistent edge",
        "Low liquidity means prices don't correct quickly before game time",
    ]
    for i, r in enumerate(reasons):
        ax.text(0.10, y_box + 0.03 - i * 0.04, f"  {r}", fontsize=13,
                color=TEXT_COLOR, va="center", family="sans-serif")

    pdf.savefig(fig)
    plt.close(fig)


def slide_calibration(pdf):
    """LGB model calibration using full 2025 test set."""
    df = pd.read_csv(BET_LOG_2025)
    df = df.dropna(subset=["lgb_home"])
    df["home_won"] = (df["result"] == "home_win").astype(int)
    df["pred"] = df["lgb_home"]

    # Use 5 bins for cleaner calibration with ~125 games
    df["bucket"] = pd.qcut(df["pred"], q=5, duplicates="drop")
    cal = df.groupby("bucket", observed=True).agg(
        pred_mean=("pred", "mean"),
        actual_mean=("home_won", "mean"),
        count=("home_won", "count"),
    ).reset_index()

    fig = plt.figure(figsize=FIG_SIZE, facecolor=BG_COLOR)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_COLOR)

    ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1.5, label="Perfect calibration")
    ax.scatter(cal["pred_mean"], cal["actual_mean"], s=cal["count"] * 5,
               color=ACCENT, edgecolor=TEXT_COLOR, linewidth=1, zorder=5)
    ax.plot(cal["pred_mean"], cal["actual_mean"], color=ACCENT, linewidth=2, alpha=0.7)

    for _, r in cal.iterrows():
        ax.annotate(f"n={int(r['count'])}", (r["pred_mean"], r["actual_mean"]),
                    textcoords="offset points", xytext=(0, 12),
                    fontsize=10, color=MUTED, ha="center")

    ax.set_xlabel("LGB Predicted Win Probability", fontsize=14, color=MUTED)
    ax.set_ylabel("Actual Win Rate", fontsize=14, color=MUTED)
    ax.set_title("LGB Model Calibration -- Predicted vs Actual (2025 Test Set)",
                 fontsize=24, fontweight="bold", color=TEXT_COLOR, pad=15, family="sans-serif")
    ax.set_xlim(0.25, 0.75)
    ax.set_ylim(0.25, 0.75)
    ax.tick_params(colors=MUTED, labelsize=11)
    ax.legend(fontsize=13, facecolor=CARD_COLOR, edgecolor=MUTED, labelcolor=TEXT_COLOR, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def slide_edge_distribution(pdf, bets_2025, bets_2026):
    """Histogram of edge sizes, colored by W/L."""
    all_edges_w = list(bets_2025[bets_2025["won"]]["edge"])
    all_edges_l = list(bets_2025[~bets_2025["won"]]["edge"])
    if len(bets_2026) > 0:
        all_edges_w += list(bets_2026[bets_2026["won"]]["edge"])
        all_edges_l += list(bets_2026[~bets_2026["won"]]["edge"])

    fig = plt.figure(figsize=FIG_SIZE, facecolor=BG_COLOR)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_COLOR)

    bins = np.linspace(EDGE_THRESHOLD, max(all_edges_w + all_edges_l) + 0.01, 20)
    ax.hist(all_edges_w, bins=bins, color=GREEN, alpha=0.7, label="Wins", edgecolor="none")
    ax.hist(all_edges_l, bins=bins, color=RED, alpha=0.7, label="Losses", edgecolor="none")

    ax.set_xlabel("Edge Size (Model prob - Poly price)", fontsize=14, color=MUTED)
    ax.set_ylabel("Count", fontsize=14, color=MUTED)
    ax.set_title("Distribution of Edges Taken",
                 fontsize=24, fontweight="bold", color=TEXT_COLOR, pad=15, family="sans-serif")
    ax.axvline(EDGE_THRESHOLD, color=GOLD, linewidth=2, linestyle="--", label=f"Threshold: {EDGE_THRESHOLD:.0%}")
    ax.tick_params(colors=MUTED, labelsize=11)
    ax.legend(fontsize=13, facecolor=CARD_COLOR, edgecolor=MUTED, labelcolor=TEXT_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)

    total = len(all_edges_w) + len(all_edges_l)
    avg_edge = np.mean(all_edges_w + all_edges_l)
    ax.text(0.97, 0.95, f"Total bets: {total}\nAvg edge: {avg_edge:.1%}",
            transform=ax.transAxes, ha="right", va="top", fontsize=13,
            color=TEXT_COLOR, family="sans-serif",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=CARD_COLOR, edgecolor=MUTED))

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# SLIDES — Section 3: Backtest & Live Results
# ══════════════════════════════════════════════════════════════════════════

def slide_cumulative_pnl(pdf, bets, title_str, year_label):
    """Generic cumulative P&L slide."""
    fig = plt.figure(figsize=FIG_SIZE, facecolor=BG_COLOR)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_COLOR)

    bets = bets.sort_values("date").reset_index(drop=True)
    cum_pnl = bets["pnl"].cumsum()
    dates = bets["date"]

    ax.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                     where=cum_pnl >= 0, color=GREEN, alpha=0.3, interpolate=True)
    ax.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                     where=cum_pnl < 0, color=RED, alpha=0.3, interpolate=True)
    ax.plot(range(len(cum_pnl)), cum_pnl, color=TEXT_COLOR, linewidth=2.5)
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")

    n = len(dates)
    if n > 15:
        tick_idx = np.linspace(0, n - 1, min(12, n), dtype=int)
    else:
        tick_idx = range(n)
    ax.set_xticks(list(tick_idx))
    ax.set_xticklabels([dates.iloc[i].strftime("%m/%d") for i in tick_idx],
                        rotation=45, fontsize=11, color=MUTED)
    ax.tick_params(axis="y", colors=MUTED, labelsize=12)

    price_col = "price" if "price" in bets.columns else "poly_price"
    n_bets = len(bets)
    wins = int(bets["won"].sum())
    total_pnl = cum_pnl.iloc[-1]
    wagered = bets[price_col].sum()
    roi = total_pnl / wagered * 100 if wagered else 0

    ann = f"{n_bets} bets  |  {wins}W-{n_bets-wins}L  |  ROI: {roi:+.1f}%  |  P&L: ${total_pnl:+.2f}"
    ax.text(0.5, 1.06, title_str, transform=ax.transAxes, ha="center",
            fontsize=26, fontweight="bold", color=TEXT_COLOR, family="sans-serif")
    ax.text(0.5, 1.01, ann, transform=ax.transAxes, ha="center",
            fontsize=14, color=GREEN if total_pnl > 0 else RED, family="sans-serif")

    ax.set_ylabel("Cumulative P&L ($1 flat bets)", fontsize=14, color=MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.92])
    pdf.savefig(fig)
    plt.close(fig)


def slide_cumulative_pnl_kelly(pdf, bets, title_str):
    """Cumulative P&L with quarter-Kelly sizing."""
    fig = plt.figure(figsize=FIG_SIZE, facecolor=BG_COLOR)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_COLOR)

    bets = bets.sort_values("date").reset_index(drop=True)
    # Kelly-sized P&L: pnl * kelly_quarter * bankroll (normalize to $1000 bankroll)
    bankroll = 1000.0
    running_bankroll = [bankroll]
    for _, r in bets.iterrows():
        bet_size = running_bankroll[-1] * r["kelly_quarter"]
        if r["won"]:
            profit = bet_size * (1 - r.get("poly_price", r.get("price", 0.5))) / r.get("poly_price", r.get("price", 0.5)) - bet_size * POLY_FEE / r.get("poly_price", r.get("price", 0.5))
        else:
            profit = -bet_size - bet_size * POLY_FEE / r.get("poly_price", r.get("price", 0.5))
        # Simpler: kelly-weighted flat P&L
        running_bankroll.append(running_bankroll[-1] + r["pnl"] * bet_size)

    # Actually let's just do it simply: cumulative P&L where each bet is sized at kelly_quarter of current bankroll
    bets_sorted = bets.sort_values("date").reset_index(drop=True)
    bankroll = 1000.0
    bankrolls = [bankroll]
    for _, r in bets_sorted.iterrows():
        bet_amount = bankroll * r["kelly_quarter"]
        price = r.get("poly_price", r.get("price"))
        if r["won"]:
            profit = bet_amount * ((1 - price - POLY_FEE) / price)
        else:
            profit = -bet_amount * (1 + POLY_FEE / price)
        bankroll = bankroll + profit
        bankrolls.append(bankroll)

    dates = bets_sorted["date"]
    pnl_vals = [b - 1000 for b in bankrolls[1:]]

    ax.fill_between(range(len(pnl_vals)), pnl_vals, 0,
                     where=[p >= 0 for p in pnl_vals], color=GREEN, alpha=0.3, interpolate=True)
    ax.fill_between(range(len(pnl_vals)), pnl_vals, 0,
                     where=[p < 0 for p in pnl_vals], color=RED, alpha=0.3, interpolate=True)
    ax.plot(range(len(pnl_vals)), pnl_vals, color=GOLD, linewidth=2.5)
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")

    n = len(dates)
    if n > 15:
        tick_idx = np.linspace(0, n - 1, min(12, n), dtype=int)
    else:
        tick_idx = range(n)
    ax.set_xticks(list(tick_idx))
    ax.set_xticklabels([dates.iloc[i].strftime("%m/%d") for i in tick_idx],
                        rotation=45, fontsize=11, color=MUTED)
    ax.tick_params(axis="y", colors=MUTED, labelsize=12)

    final_bankroll = bankrolls[-1]
    total_return = (final_bankroll / 1000 - 1) * 100

    ann = f"Starting bankroll: $1,000  |  Final: ${final_bankroll:,.0f}  |  Return: {total_return:+.1f}%"
    ax.text(0.5, 1.06, title_str, transform=ax.transAxes, ha="center",
            fontsize=26, fontweight="bold", color=TEXT_COLOR, family="sans-serif")
    ax.text(0.5, 1.01, ann, transform=ax.transAxes, ha="center",
            fontsize=14, color=GREEN if total_return > 0 else RED, family="sans-serif")

    ax.set_ylabel("P&L from $1,000 (Quarter-Kelly)", fontsize=14, color=MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.92])
    pdf.savefig(fig)
    plt.close(fig)


def slide_bet_table(pdf, bets_df, title_prefix, price_col="poly_price"):
    """Generic table of every bet with cumulative PnL and Kelly sizing."""
    if len(bets_df) == 0:
        fig, ax = setup_fig(f"{title_prefix} -- Every Bet")
        ax.text(0.5, 0.5, "No matched bets.",
                ha="center", fontsize=20, color=MUTED)
        pdf.savefig(fig)
        plt.close(fig)
        return

    bets = bets_df.sort_values("date").reset_index(drop=True)
    bets["cum_pnl"] = bets["pnl"].cumsum()

    headers = ["Date", "Matchup", "Side", "Poly", "Model", "Edge", "Kelly", "W/L", "P&L", "Cum P&L"]
    max_rows = 16
    n_rows = len(bets)
    pages = [bets.iloc[i:i+max_rows] for i in range(0, n_rows, max_rows)]

    for page_idx, page_bets in enumerate(pages):
        suffix = "" if page_idx == 0 else " (cont.)"
        fig, ax = setup_fig(f"{title_prefix} -- Every Bet{suffix}")

        n = len(page_bets)
        row_h = 0.68 / (n + 1)
        y_start = 0.83
        col_x = [0.02, 0.10, 0.32, 0.42, 0.51, 0.60, 0.69, 0.78, 0.85, 0.93]

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

            team_str = f"{r['away']}@{r['home']}"
            poly_val = r.get("poly_price", r.get("price", 0))
            vals = [
                r["date"].strftime("%m/%d"),
                team_str,
                r["side"].upper(),
                f"{poly_val:.2f}",
                f"{r['signal_prob']:.2f}",
                f"{r['edge']:.1%}",
                f"{r['kelly_quarter']:.1%}",
                "W" if r["won"] else "L",
                f"{r['pnl']:+.3f}",
                f"{r['cum_pnl']:+.3f}",
            ]
            for j, v in enumerate(vals):
                if j == 7:  # W/L
                    c = GREEN if r["won"] else RED
                elif j == 9:  # Cum P&L
                    c = GREEN if r["cum_pnl"] > 0 else RED
                else:
                    c = TEXT_COLOR
                ax.text(col_x[j], y, v, fontsize=10, color=c, va="center", family="sans-serif")

        pdf.savefig(fig)
        plt.close(fig)


def slide_results_summary(pdf, bets_2025, bets_2026):
    """Combined results summary with key metrics."""
    fig, ax = setup_fig("Results Summary")

    # 2025 stats
    n25 = len(bets_2025)
    w25 = int(bets_2025["won"].sum())
    pnl25 = bets_2025["pnl"].sum()
    wag25 = bets_2025["price"].sum()
    roi25 = pnl25 / wag25 * 100 if wag25 else 0
    avg_edge25 = bets_2025["edge"].mean() * 100
    avg_kelly25 = bets_2025["kelly_quarter"].mean() * 100

    # 2026 stats
    n26 = len(bets_2026)
    w26 = int(bets_2026["won"].sum()) if n26 > 0 else 0
    pnl26 = bets_2026["pnl"].sum() if n26 > 0 else 0
    wag26 = bets_2026["poly_price"].sum() if n26 > 0 else 0
    roi26 = pnl26 / wag26 * 100 if wag26 else 0
    avg_edge26 = bets_2026["edge"].mean() * 100 if n26 > 0 else 0
    avg_kelly26 = bets_2026["kelly_quarter"].mean() * 100 if n26 > 0 else 0

    # Combined
    total_bets = n25 + n26
    total_wins = w25 + w26
    total_pnl = pnl25 + pnl26
    total_wag = wag25 + wag26
    total_roi = total_pnl / total_wag * 100 if total_wag else 0

    # Layout: two boxes side by side + combined at bottom
    y_top = 0.78
    box_h = 0.28
    box_w = 0.40

    for i, (label, color, n, w, pnl, roi, avg_e, avg_k) in enumerate([
        ("2025 Backtest", ACCENT, n25, w25, pnl25, roi25, avg_edge25, avg_kelly25),
        ("2026 Live", GOLD, n26, w26, pnl26, roi26, avg_edge26, avg_kelly26),
    ]):
        x = 0.06 + i * 0.48
        rect = FancyBboxPatch((x, y_top - box_h), box_w, box_h,
                                boxstyle="round,pad=0.015", facecolor=CARD_COLOR,
                                edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + box_w/2, y_top - 0.03, label, ha="center", fontsize=20,
                fontweight="bold", color=color, family="sans-serif")

        lines = [
            f"Bets: {n}   |   Record: {w}-{n-w}   ({w/n*100:.0f}%)" if n > 0 else "No bets",
            f"Flat P&L: ${pnl:+.2f}   |   ROI: {roi:+.1f}%",
            f"Avg Edge: {avg_e:.1f}%   |   Avg Kelly: {avg_k:.1f}%",
        ]
        for li, line in enumerate(lines):
            c = TEXT_COLOR
            if li == 1:
                c = GREEN if pnl > 0 else RED
            ax.text(x + box_w/2, y_top - 0.10 - li * 0.05, line, ha="center",
                    fontsize=14, color=c, family="sans-serif")

    # Combined box
    y_comb = 0.30
    rect = FancyBboxPatch((0.15, y_comb - 0.15), 0.70, 0.18,
                            boxstyle="round,pad=0.015", facecolor=CARD_COLOR,
                            edgecolor=GREEN, linewidth=2)
    ax.add_patch(rect)
    ax.text(0.5, y_comb + 0.01, "Combined", ha="center", fontsize=22,
            fontweight="bold", color=GREEN, family="sans-serif")
    ax.text(0.5, y_comb - 0.05,
            f"{total_bets} bets  |  {total_wins}W-{total_bets-total_wins}L  ({total_wins/total_bets*100:.0f}%)  |  "
            f"P&L: ${total_pnl:+.2f}  |  ROI: {total_roi:+.1f}%",
            ha="center", fontsize=16, color=TEXT_COLOR, family="sans-serif")

    pdf.savefig(fig)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("Loading 2025 bets...")
    bets_2025 = load_2025_poly_bets()
    print(f"  {len(bets_2025)} bets (2025, poly-matched, 3% edge)")

    print("Loading 2026 bets...")
    bets_2026 = load_2026_bets()
    print(f"  {len(bets_2026)} bets (2026 regular season)")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(OUT_PDF)) as pdf:
        # --- Section 1: Modeling & Methods ---
        print("Slide 1: Title...")
        slide_title(pdf, bets_2025, bets_2026)

        print("Slide 2: Data Pipeline...")
        slide_data_pipeline(pdf)

        print("Slide 3-4: Pitcher Models...")
        slide_pitcher_models(pdf)  # Stuff + Location = 2 slides

        print("Slide 5: PA Matchup Model...")
        slide_pa_distribution(pdf)

        # --- Section 2: Strategy & Calibration ---
        print("Slide 6: Strategy Overview...")
        slide_strategy_overview(pdf)

        print("Slide 7: DK Calibration...")
        slide_calibration(pdf)

        print("Slide 8: Edge Distribution...")
        slide_edge_distribution(pdf, bets_2025, bets_2026)

        # --- Section 3: Results ---
        print("Slide 9: 2025 Bet Table...")
        slide_bet_table(pdf, bets_2025, "2025 Backtest", price_col="price")

        print("Slide 10: 2025 Backtest P&L...")
        slide_cumulative_pnl(pdf, bets_2025, "2025 Backtest -- Cumulative P&L (Flat $1 Bets)", "2025")

        print("Slide 11: 2026 Bet Table...")
        slide_bet_table(pdf, bets_2026, "2026 Live Results", price_col="poly_price")

        print("Slide 12: 2026 Cumulative P&L (Flat)...")
        if len(bets_2026) > 0:
            slide_cumulative_pnl(pdf, bets_2026, "2026 Live -- Cumulative P&L (Flat $1 Bets)", "2026")

        print("Slide 12: Combined Kelly P&L...")
        all_bets = pd.concat([
            bets_2025.rename(columns={"price": "poly_price"}),
            bets_2026,
        ], ignore_index=True)
        slide_cumulative_pnl_kelly(pdf, all_bets, "Combined P&L -- Quarter-Kelly Sizing ($1,000 Bankroll)")

        print("Slide 13: Results Summary...")
        slide_results_summary(pdf, bets_2025, bets_2026)

    print(f"\nReport saved to {OUT_PDF}")


if __name__ == "__main__":
    main()
