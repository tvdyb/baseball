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
import matplotlib.patches as mpatches
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


def add_subtitle(ax, text, y=0.87):
    ax.text(0.5, y, text, ha="center", va="top",
            fontsize=16, color=MUTED, family="sans-serif")


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_2025_poly_bets():
    """Load 2025 bet log, recompute P&L against poly prices with 3% edge threshold."""
    df = pd.read_csv(BET_LOG_2025)
    df["date"] = pd.to_datetime(df["date"])
    # Only rows with poly data
    df = df.dropna(subset=["poly_home", "poly_away"]).copy()
    # Recompute edge vs poly
    df["edge_home"] = df["dk_home"] - df["poly_home"]
    df["edge_away"] = df["dk_away"] - df["poly_away"]
    bets = []
    for _, r in df.iterrows():
        eh, ea = r["edge_home"], r["edge_away"]
        if eh >= EDGE_THRESHOLD or ea >= EDGE_THRESHOLD:
            if eh >= ea:
                side = "home"
                price = r["poly_home"]
                edge = eh
                won = r["result"] == "home_win"
            else:
                side = "away"
                price = r["poly_away"]
                edge = ea
                won = r["result"] == "away_win"
            pnl = (1 - price - POLY_FEE) if won else (-price - POLY_FEE)
            bets.append({
                "date": r["date"], "home": r["home"], "away": r["away"],
                "side": side, "price": price, "dk_prob": r[f"dk_{side}"],
                "edge": edge, "won": won, "pnl": pnl,
            })
    return pd.DataFrame(bets)


def load_2026_bets():
    """Match DK lines to Polymarket for 2026 regular season, apply betting logic."""
    dk = pd.read_parquet(SBR_2026)
    dk["game_date"] = pd.to_datetime(dk["game_date"])
    dk = dk[dk["game_date"] >= REG_SEASON_START].copy()
    dk = dk.dropna(subset=["dk_home_prob"])

    poly = pd.read_parquet(POLY_CLOSING)
    poly["game_date"] = pd.to_datetime(poly["game_date"])
    poly = poly[poly["game_date"] >= REG_SEASON_START].copy()
    # Filter resolution leaks
    poly = poly[(poly["poly_team0_prob"] >= 0.18) & (poly["poly_team0_prob"] <= 0.82)]
    poly = poly[(poly["poly_team1_prob"] >= 0.18) & (poly["poly_team1_prob"] <= 0.82)]

    bets = []
    for _, g in dk.iterrows():
        home = map_team(g["home_team"])
        away = map_team(g["away_team"])
        gd = g["game_date"]
        # Try matching poly: team0=home, team1=away OR team0=away, team1=home
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

        dk_home = g["dk_home_prob"]
        dk_away = 1 - dk_home
        eh = dk_home - p_home
        ea = dk_away - p_away

        if eh >= EDGE_THRESHOLD or ea >= EDGE_THRESHOLD:
            if eh >= ea:
                side = "home"
                price = p_home
                edge = eh
                won = bool(home_won)
            else:
                side = "away"
                price = p_away
                edge = ea
                won = not bool(home_won)
            pnl = (1 - price - POLY_FEE) if won else (-price - POLY_FEE)
            bets.append({
                "date": gd, "home": g["home_team"], "away": g["away_team"],
                "side": side, "poly_price": price, "dk_prob": dk_home if side == "home" else dk_away,
                "edge": edge, "won": won, "pnl": pnl,
                "result_str": f"{g['home_team']} {int(g['home_score'])}-{int(g['away_score'])} {g['away_team']}"
            })
    return pd.DataFrame(bets)


# ══════════════════════════════════════════════════════════════════════════
# SLIDES
# ══════════════════════════════════════════════════════════════════════════

def slide_title(pdf, bets_2025, bets_2026):
    fig, ax = setup_fig()
    ax.text(0.5, 0.72, "MLB Moneyline Edge", ha="center", va="center",
            fontsize=52, fontweight="bold", color=TEXT_COLOR, family="sans-serif")
    ax.text(0.5, 0.60, "Exploiting Polymarket Inefficiencies",
            ha="center", va="center", fontsize=30, color=ACCENT, family="sans-serif")

    # Key stats
    total_bets = len(bets_2025) + len(bets_2026)
    total_pnl = bets_2025["pnl"].sum() + bets_2026["pnl"].sum()
    total_wagered = bets_2025["price"].sum() + bets_2026["poly_price"].sum()
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

    # Date range
    dates_all = list(bets_2025["date"]) + list(bets_2026["date"])
    d_min = min(dates_all).strftime("%b %d, %Y")
    d_max = max(dates_all).strftime("%b %d, %Y")
    ax.text(0.5, 0.18, f"{d_min}  —  {d_max}", ha="center", va="center",
            fontsize=16, color=MUTED, family="sans-serif")
    ax.text(0.5, 0.10, "April 2026", ha="center", va="center",
            fontsize=12, color=MUTED, family="sans-serif", style="italic")

    pdf.savefig(fig)
    plt.close(fig)


def slide_strategy_overview(pdf, bets_2025, bets_2026):
    fig, ax = setup_fig("Strategy Overview")

    bullets = [
        "Train a LightGBM model on pitcher matchup features + DraftKings closing moneyline",
        "Compare model probability vs. Polymarket contract price",
        "Buy the underpriced side when edge > 3% threshold",
        "Polymarket fee: $0.0075 per contract (0.75%)",
        "Edge = DK implied probability − Polymarket price",
    ]
    y = 0.78
    for b in bullets:
        ax.text(0.08, y, "•", fontsize=20, color=ACCENT, va="center", family="sans-serif")
        ax.text(0.11, y, b, fontsize=17, color=TEXT_COLOR, va="center", family="sans-serif")
        y -= 0.065

    # Side by side comparison boxes
    y_box = 0.32
    box_h = 0.22
    # 2025 box
    rect1 = FancyBboxPatch((0.06, y_box - box_h/2), 0.40, box_h,
                            boxstyle="round,pad=0.015", facecolor=CARD_COLOR, edgecolor=ACCENT, linewidth=1.5)
    ax.add_patch(rect1)
    n25 = len(bets_2025)
    w25 = bets_2025["won"].sum()
    pnl25 = bets_2025["pnl"].sum()
    wag25 = bets_2025["price"].sum()
    roi25 = pnl25 / wag25 * 100 if wag25 else 0
    ax.text(0.26, y_box + 0.07, "2025 Test Period", ha="center", fontsize=18,
            fontweight="bold", color=ACCENT, family="sans-serif")
    ax.text(0.26, y_box + 0.01, f"{n25} bets  |  {w25}/{n25} wins ({w25/n25*100:.0f}%)",
            ha="center", fontsize=14, color=TEXT_COLOR, family="sans-serif")
    ax.text(0.26, y_box - 0.05, f"ROI: {roi25:+.1f}%   |   P&L: ${pnl25:+.2f}",
            ha="center", fontsize=14, color=GREEN if pnl25 > 0 else RED, family="sans-serif")

    # 2026 box
    rect2 = FancyBboxPatch((0.54, y_box - box_h/2), 0.40, box_h,
                            boxstyle="round,pad=0.015", facecolor=CARD_COLOR, edgecolor=GOLD, linewidth=1.5)
    ax.add_patch(rect2)
    n26 = len(bets_2026)
    if n26 > 0:
        w26 = bets_2026["won"].sum()
        pnl26 = bets_2026["pnl"].sum()
        wag26 = bets_2026["poly_price"].sum()
        roi26 = pnl26 / wag26 * 100 if wag26 else 0
        ax.text(0.74, y_box + 0.07, "2026 Live Period", ha="center", fontsize=18,
                fontweight="bold", color=GOLD, family="sans-serif")
        ax.text(0.74, y_box + 0.01, f"{n26} bets  |  {w26}/{n26} wins ({w26/n26*100:.0f}%)",
                ha="center", fontsize=14, color=TEXT_COLOR, family="sans-serif")
        ax.text(0.74, y_box - 0.05, f"ROI: {roi26:+.1f}%   |   P&L: ${pnl26:+.2f}",
                ha="center", fontsize=14, color=GREEN if pnl26 > 0 else RED, family="sans-serif")
    else:
        ax.text(0.74, y_box, "2026 Live\n(no matched bets yet)", ha="center", fontsize=14,
                color=MUTED, family="sans-serif")

    pdf.savefig(fig)
    plt.close(fig)


def slide_cumulative_pnl(pdf, bets, title_str, year_label):
    """Generic cumulative P&L slide."""
    fig = plt.figure(figsize=FIG_SIZE, facecolor=BG_COLOR)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_COLOR)

    bets = bets.sort_values("date").reset_index(drop=True)
    cum_pnl = bets["pnl"].cumsum()
    dates = bets["date"]

    # Fill green/red
    ax.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                     where=cum_pnl >= 0, color=GREEN, alpha=0.3, interpolate=True)
    ax.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                     where=cum_pnl < 0, color=RED, alpha=0.3, interpolate=True)
    ax.plot(range(len(cum_pnl)), cum_pnl, color=TEXT_COLOR, linewidth=2.5)
    ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")

    # X-axis: show date labels
    n = len(dates)
    if n > 15:
        tick_idx = np.linspace(0, n - 1, min(12, n), dtype=int)
    else:
        tick_idx = range(n)
    ax.set_xticks(list(tick_idx))
    ax.set_xticklabels([dates.iloc[i].strftime("%m/%d") for i in tick_idx],
                        rotation=45, fontsize=11, color=MUTED)
    ax.tick_params(axis="y", colors=MUTED, labelsize=12)

    # Annotation
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

    ax.set_ylabel("Cumulative P&L ($1 bets)", fontsize=14, color=MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.92])
    pdf.savefig(fig)
    plt.close(fig)


def slide_2026_table(pdf, bets_2026):
    """Table of every 2026 bet."""
    fig, ax = setup_fig("2026 Live Results — Every Bet")

    if len(bets_2026) == 0:
        ax.text(0.5, 0.5, "No matched bets for 2026 regular season yet.",
                ha="center", fontsize=20, color=MUTED)
        pdf.savefig(fig)
        plt.close(fig)
        return

    bets = bets_2026.sort_values("date").reset_index(drop=True)
    headers = ["Date", "Matchup", "Side", "Poly", "DK Prob", "Edge", "W/L", "P&L"]
    n_rows = len(bets)

    # Determine how many rows fit
    max_rows = 18
    if n_rows > max_rows:
        # Multiple pages needed
        pages = [bets.iloc[i:i+max_rows] for i in range(0, n_rows, max_rows)]
    else:
        pages = [bets]

    for page_idx, page_bets in enumerate(pages):
        if page_idx > 0:
            fig, ax = setup_fig("2026 Live Results — Every Bet (cont.)")

        n = len(page_bets)
        row_h = 0.70 / (n + 1)
        y_start = 0.83
        col_x = [0.04, 0.15, 0.38, 0.50, 0.59, 0.69, 0.78, 0.88]

        # Header
        for j, h in enumerate(headers):
            ax.text(col_x[j], y_start, h, fontsize=12, fontweight="bold",
                    color=ACCENT, va="center", family="sans-serif")

        for i, (_, r) in enumerate(page_bets.iterrows()):
            y = y_start - (i + 1) * row_h
            bg_color = GREEN if r["won"] else RED
            rect = FancyBboxPatch((0.02, y - row_h * 0.4), 0.96, row_h * 0.8,
                                   boxstyle="round,pad=0.003",
                                   facecolor=bg_color, alpha=0.12, edgecolor="none")
            ax.add_patch(rect)

            team_str = f"{r['away']}@{r['home']}"
            side_str = r["side"].upper()
            vals = [
                r["date"].strftime("%m/%d"),
                team_str,
                side_str,
                f"{r['poly_price']:.2f}",
                f"{r['dk_prob']:.2f}",
                f"{r['edge']:.1%}",
                "W" if r["won"] else "L",
                f"{r['pnl']:+.3f}",
            ]
            for j, v in enumerate(vals):
                c = GREEN if j == 6 and r["won"] else (RED if j == 6 else TEXT_COLOR)
                ax.text(col_x[j], y, v, fontsize=11, color=c, va="center", family="sans-serif")

        pdf.savefig(fig)
        plt.close(fig)


def slide_pa_distribution(pdf):
    """Show PA outcome distribution for a notable matchup."""
    nn = pd.read_parquet(NN_FEATURES)
    nn["game_date"] = pd.to_datetime(nn["game_date"])
    nn_good = nn.dropna(subset=["home_h0_K"])
    # Pick a notable game (LAD @ PHI or similar)
    notable = nn_good[
        ((nn_good["home_team"] == "PHI") & (nn_good["away_team"] == "LAD")) |
        ((nn_good["home_team"] == "LAD") & (nn_good["away_team"] == "NYY"))
    ]
    if len(notable) == 0:
        notable = nn_good.head(1)
    row = notable.iloc[0]
    game_label = f"{row['away_team']} @ {row['home_team']} — {row['game_date'].strftime('%b %d, %Y')}"

    outcomes = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "dp", "out_ground", "out_fly", "out_line"]
    outcome_labels = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "DP", "GB Out", "FB Out", "LD Out"]
    # Home hitter slot 0 vs away SP
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

    # Value labels on bars
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", fontsize=10, color=TEXT_COLOR)

    ax.set_title(f"Matchup Model — Predicted PA Outcomes\n{game_label}\nHome Leadoff Hitter vs Away SP",
                 fontsize=22, fontweight="bold", color=TEXT_COLOR, pad=20, family="sans-serif")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def slide_pitcher_models(pdf):
    """Show SP stuff and location distributions from nn_features."""
    nn = pd.read_parquet(NN_FEATURES)
    nn = nn.dropna(subset=["home_sp_stuff"])

    for col, title, xlabel in [
        ("stuff", "Pitcher Stuff Model — Raw Stuff Quality", "Stuff Score (xRV/100 pitches)"),
        ("location", "Pitcher Location Model — Command Quality", "Location Score (xRV/100 pitches)"),
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


def slide_calibration(pdf, bets_2025):
    """LGB model calibration using bet_log_2025 with dk_home as the predicted probability."""
    df = pd.read_csv(BET_LOG_2025)
    df = df.dropna(subset=["poly_home", "poly_away"])
    # dk_home is the model/market probability; actual result is whether home won
    df["home_won"] = (df["result"] == "home_win").astype(int)
    df["pred"] = df["dk_home"]

    # Bucket into deciles
    df["bucket"] = pd.qcut(df["pred"], q=10, duplicates="drop")
    cal = df.groupby("bucket", observed=True).agg(
        pred_mean=("pred", "mean"),
        actual_mean=("home_won", "mean"),
        count=("home_won", "count"),
    ).reset_index()

    fig = plt.figure(figsize=FIG_SIZE, facecolor=BG_COLOR)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_COLOR)

    ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1.5, label="Perfect calibration")
    ax.scatter(cal["pred_mean"], cal["actual_mean"], s=cal["count"] * 8,
               color=ACCENT, edgecolor=TEXT_COLOR, linewidth=1, zorder=5)
    ax.plot(cal["pred_mean"], cal["actual_mean"], color=ACCENT, linewidth=2, alpha=0.7)

    for _, r in cal.iterrows():
        ax.annotate(f"n={int(r['count'])}", (r["pred_mean"], r["actual_mean"]),
                    textcoords="offset points", xytext=(0, 12),
                    fontsize=10, color=MUTED, ha="center")

    ax.set_xlabel("Predicted Win Probability (DK Closing Line)", fontsize=14, color=MUTED)
    ax.set_ylabel("Actual Win Rate", fontsize=14, color=MUTED)
    ax.set_title("LGB Model Calibration — Predicted vs Actual Win Rate",
                 fontsize=24, fontweight="bold", color=TEXT_COLOR, pad=15, family="sans-serif")
    ax.set_xlim(0.15, 0.85)
    ax.set_ylim(0.15, 0.85)
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

    ax.set_xlabel("Edge Size (DK prob - Poly price)", fontsize=14, color=MUTED)
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
        print("Slide 1: Title...")
        slide_title(pdf, bets_2025, bets_2026)

        print("Slide 2: Strategy Overview...")
        slide_strategy_overview(pdf, bets_2025, bets_2026)

        print("Slide 3: 2025 Cumulative P&L...")
        slide_cumulative_pnl(pdf, bets_2025, "2025 Cumulative P&L (Test Period)", "2025")

        print("Slide 4: 2026 Bet Table...")
        slide_2026_table(pdf, bets_2026)

        print("Slide 5: 2026 Cumulative P&L...")
        if len(bets_2026) > 0:
            slide_cumulative_pnl(pdf, bets_2026, "2026 Cumulative P&L (Live)", "2026")
        else:
            fig, ax = setup_fig("2026 Cumulative P&L (Live)")
            ax.text(0.5, 0.5, "No matched bets yet.", ha="center", fontsize=20, color=MUTED)
            pdf.savefig(fig)
            plt.close(fig)

        print("Slide 6: Pitcher Stuff Model...")
        slide_pitcher_models(pdf)  # This creates slides 6 and 7

        print("Slide 8: PA Outcome Distribution...")
        slide_pa_distribution(pdf)

        print("Slide 9: Model Calibration...")
        slide_calibration(pdf, bets_2025)

        print("Slide 10: Edge Distribution...")
        slide_edge_distribution(pdf, bets_2025, bets_2026)

    print(f"\nReport saved to {OUT_PDF}")


if __name__ == "__main__":
    main()
