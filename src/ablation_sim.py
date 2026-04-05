#!/usr/bin/env python3
"""
Ablation study on MC simulator improvements.

Runs backtest_nrfi_ou with different feature toggles disabled to measure
each improvement's contribution. Each config runs 500 games at 2000 sims.

Configs:
  A: Current (all improvements)
  B: No platoon splits
  C: No first-inning adjustment
  D: No recent form
  E: No eval scores
  F: No platoon + No first-inning
"""

import re
import subprocess
import sys
import textwrap
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
BASE_CMD = [
    sys.executable, str(SRC_DIR / "backtest_nrfi_ou.py"),
    "--season", "2025", "--n-sims", "2000", "--max-games", "500",
]

# Each config is a (name, description, preamble_code) tuple.
# preamble_code is injected before the backtest import to monkey-patch simulate.
CONFIGS = {
    "A": ("All improvements", ""),
    "B": ("No platoon splits", textwrap.dedent("""\
        import simulate as sim
        _orig_batter = sim._compute_batter_outcome_rates
        _orig_pitcher = sim._compute_pitcher_outcome_rates
        def _no_platoon_batter(*args, **kwargs):
            kwargs.pop('vs_hand', None)
            return _orig_batter(*args, **kwargs)
        def _no_platoon_pitcher(*args, **kwargs):
            kwargs.pop('vs_hand', None)
            return _orig_pitcher(*args, **kwargs)
        sim._compute_batter_outcome_rates = _no_platoon_batter
        sim._compute_pitcher_outcome_rates = _no_platoon_pitcher
    """)),
    "C": ("No first-inning adj", textwrap.dedent("""\
        import simulate as sim
        sim._FIRST_INNING_ADJ = {o: 1.0 for o in sim.OUTCOME_ORDER}
    """)),
    "D": ("No recent form", textwrap.dedent("""\
        import simulate as sim
        sim._compute_pitcher_recent_form = lambda *a, **kw: kw.get('season_rates') or (a[3] if len(a) > 3 else None)
    """)),
    "E": ("No eval scores", textwrap.dedent("""\
        import simulate as sim
        _orig_SimConfig = sim.SimConfig
        class _PatchedSimConfig(_orig_SimConfig):
            def __init__(self, **kwargs):
                kwargs.setdefault('eval_sensitivity', 0.0)
                super().__init__(**kwargs)
        sim.SimConfig = _PatchedSimConfig
    """)),
    "F": ("No platoon + No 1st-inn", textwrap.dedent("""\
        import simulate as sim
        _orig_batter = sim._compute_batter_outcome_rates
        _orig_pitcher = sim._compute_pitcher_outcome_rates
        def _no_platoon_batter(*args, **kwargs):
            kwargs.pop('vs_hand', None)
            return _orig_batter(*args, **kwargs)
        def _no_platoon_pitcher(*args, **kwargs):
            kwargs.pop('vs_hand', None)
            return _orig_pitcher(*args, **kwargs)
        sim._compute_batter_outcome_rates = _no_platoon_batter
        sim._compute_pitcher_outcome_rates = _no_platoon_pitcher
        sim._FIRST_INNING_ADJ = {o: 1.0 for o in sim.OUTCOME_ORDER}
    """)),
}


def make_wrapper_script(config_key: str, preamble: str) -> Path:
    """Create a temporary wrapper script that applies patches then runs backtest."""
    script = SRC_DIR / f"_ablation_runner_{config_key}.py"
    code = textwrap.dedent(f"""\
        #!/usr/bin/env python3
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))

        # Apply monkey-patches BEFORE backtest imports SimConfig etc.
        {textwrap.indent(preamble, '        ').strip()}

        from backtest_nrfi_ou import run_backtest
        run_backtest(season=2025, n_sims=2000, max_games=500)
    """)
    script.write_text(code)
    return script


def parse_metrics(output: str) -> dict:
    """Extract key metrics from backtest output."""
    metrics = {}

    # Sim bias
    m = re.search(r"Sim bias \(sim-actual\):\s*([+\-]?\d+\.\d+)", output)
    metrics["ou_bias"] = float(m.group(1)) if m else None

    # Correlation (full set — first occurrence)
    m = re.search(r"^\s*Correlation:\s*([+\-]?\d+\.\d+)", output, re.MULTILINE)
    metrics["ou_corr"] = float(m.group(1)) if m else None

    # NRFI Brier skill score (full set — first occurrence)
    m = re.search(r"Brier skill score:\s*([+\-]?\d+\.\d+)", output)
    metrics["nrfi_brier_skill"] = float(m.group(1)) if m else None

    # Test O/U correlation
    m = re.search(r"Test O/U correlation:\s*([+\-]?\d+\.\d+)", output)
    metrics["test_ou_corr"] = float(m.group(1)) if m else None

    # Test NRFI Brier skill
    m = re.search(r"Test NRFI Brier skill:\s*([+\-]?\d+\.\d+)", output)
    metrics["test_nrfi_brier"] = float(m.group(1)) if m else None

    return metrics


def run_config(key: str) -> dict:
    """Run a single ablation config and return parsed metrics."""
    desc, preamble = CONFIGS[key]
    print(f"\n{'='*70}")
    print(f"CONFIG {key}: {desc}")
    print(f"{'='*70}\n", flush=True)

    if not preamble:
        # Config A: run backtest directly
        proc = subprocess.run(
            BASE_CMD, capture_output=True, text=True, timeout=1200,
            cwd=str(SRC_DIR),
        )
    else:
        # Create wrapper script with monkey-patches
        wrapper = make_wrapper_script(key, preamble)
        proc = subprocess.run(
            [sys.executable, str(wrapper)],
            capture_output=True, text=True, timeout=1200,
            cwd=str(SRC_DIR),
        )
        wrapper.unlink(missing_ok=True)

    output = proc.stdout + "\n" + proc.stderr
    print(output, flush=True)

    if proc.returncode != 0:
        print(f"  *** CONFIG {key} FAILED (rc={proc.returncode}) ***")
        return {}

    return parse_metrics(output)


def main():
    results = {}
    for key in CONFIGS:
        results[key] = run_config(key)

    # Cleanup any leftover runner scripts
    for p in SRC_DIR.glob("_ablation_runner_*.py"):
        p.unlink(missing_ok=True)

    # Print comparison table
    print(f"\n\n{'='*90}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*90}")

    header = (
        f"{'Config':<6} {'Description':<24} "
        f"{'O/U Bias':>9} {'O/U Corr':>9} {'NRFI BSS':>9} "
        f"{'Test Corr':>10} {'Test NRFI':>10}"
    )
    print(header)
    print("-" * len(header))

    for key in CONFIGS:
        desc = CONFIGS[key][0]
        m = results.get(key, {})
        row = f"{key:<6} {desc:<24} "
        for metric in ["ou_bias", "ou_corr", "nrfi_brier_skill", "test_ou_corr", "test_nrfi_brier"]:
            val = m.get(metric)
            if val is not None:
                row += f"{val:>9.4f} "
            else:
                row += f"{'N/A':>9} "
        print(row)

    # Deltas vs baseline (A)
    base = results.get("A", {})
    if base:
        print(f"\nDELTAS vs Config A (positive = better than A):")
        print("-" * len(header))
        for key in list(CONFIGS.keys())[1:]:
            desc = CONFIGS[key][0]
            m = results.get(key, {})
            row = f"{key:<6} {desc:<24} "
            for metric in ["ou_bias", "ou_corr", "nrfi_brier_skill", "test_ou_corr", "test_nrfi_brier"]:
                bval = base.get(metric)
                val = m.get(metric)
                if val is not None and bval is not None:
                    delta = val - bval
                    # For ou_bias, closer to 0 is better, so flip sign
                    if metric == "ou_bias":
                        delta = abs(bval) - abs(val)  # positive = improved (less bias)
                    row += f"{delta:>+9.4f} "
                else:
                    row += f"{'N/A':>9} "
            print(row)

    print(f"\nNote: For O/U Bias delta, positive means less absolute bias (improvement).")
    print(f"      For other metrics, positive means the ablated version is BETTER,")
    print(f"      which means removing that feature HELPED — the feature is HURTING.")


if __name__ == "__main__":
    main()
