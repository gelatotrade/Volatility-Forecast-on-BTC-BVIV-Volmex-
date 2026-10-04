"""Print figures for the paper (vector PDF).

Visual grammar: one categorical order (validated for colour-vision deficiency,
all pairs, three slots), thin marks, hairline solid grid, text in ink tones
only, no dual axes -- stacked panels share time instead.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

FAMILY_COLOR = {"Always-on": BLUE, "VWAP-switch": ORANGE, "VWAP-ratchet": AQUA}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 8,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "lines.linewidth": 1.2,
    "lines.solid_capstyle": "round",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

WIDTH = 6.3  # inches, the paper's text width


def fig_live_market(daily: pd.DataFrame, perp_start: pd.Timestamp, path: str):
    """BTC, the official BVIV index and the Bitfinex BVIV-perp funding rate, 2023-2026."""
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 3.6), sharex=True,
                             gridspec_kw={"hspace": 0.38, "height_ratios": [1, 1, 0.9]})
    ax = axes[0]
    ax.plot(daily.index, daily["btc"], color=BLUE, lw=1.0)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title("BTC (USD, log scale)", color=INK)
    ax = axes[1]
    ax.plot(daily.index, daily["bviv"], color=BLUE, lw=1.0)
    ax.set_title("BVIV, official Volmex index (vol points)", color=INK)
    ax = axes[2]
    f = 100 * daily["funding_8h_mean"].loc[perp_start:]
    ax.axhline(0, color=AXIS, lw=0.8)
    for cap in (0.25, -0.25):
        ax.axhline(cap, color=GRID, lw=0.8)
    ax.fill_between(f.index, 0, f.clip(lower=0), color=ORANGE, lw=0, alpha=0.85, label="longs pay")
    ax.fill_between(f.index, 0, f.clip(upper=0), color=AQUA, lw=0, alpha=0.85, label="longs receive")
    ax.set_ylim(-0.3, 0.3)
    ax.set_title("Bitfinex BVIV-perp funding (% per 8h, daily mean; cap ±0.25%)", color=INK)
    ax.legend(loc="lower left", ncol=2, handlelength=1.0)
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_minor_locator(matplotlib.dates.MonthLocator(bymonth=(4, 7, 10)))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    fig.savefig(path)
    plt.close(fig)


def fig_live_episode(ep: pd.DataFrame, z_enter: float, names: dict[str, str], path: str):
    """The best hedge day at 15-minute resolution: price vs VWAP band, BVIV, hedge sizes."""
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 3.9), sharex=True,
                             gridspec_kw={"hspace": 0.62, "height_ratios": [1.15, 1, 1]})
    ax = axes[0]
    ax.plot(ep.index, ep["close"], color=INK2, lw=0.8, label="BTC close (15m)")
    ax.plot(ep.index, ep["vwap"], color=BLUE, lw=1.2, label="rolling 24h VWAP")
    ax.plot(ep.index, ep["band"], color=ORANGE, lw=1.0, label=f"breakdown band (z = {z_enter:g})".replace("-", "\u2212"))
    trig = ep["z"] < z_enter
    ax.scatter(ep.index[trig], ep["close"][trig], s=6, color=ORANGE, zorder=3, linewidths=0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
    ax.set_title("BTC, its VWAP and the breakdown band (USD)", color=INK, pad=14)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.98), ncol=3, handlelength=1.4, borderaxespad=0)
    ax = axes[1]
    ax.plot(ep.index, ep["bviv"], color=BLUE, lw=1.2, label="Volmex index")
    ax.plot(ep.index, ep["bviv_mid"], color=INK2, lw=0.8, label="Bitfinex perpetual, mid")
    ax.set_title("BVIV (vol points)", color=INK, pad=14)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.98), ncol=2, handlelength=1.4, borderaxespad=0)
    ax = axes[2]
    for key, label in names.items():
        ax.step(ep.index, ep[f"{key}|position"], where="post", color=FAMILY_COLOR[label], lw=1.2, label=label)
    ax.set_ylim(bottom=0)
    ax.set_title("Hedge size for 1 BTC (contracts, \\$1 per vol point)", color=INK, pad=14)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.98), ncol=3, handlelength=1.4, borderaxespad=0)
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator())
    ax.xaxis.set_minor_locator(matplotlib.dates.HourLocator(byhour=(6, 12, 18)))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    fig.savefig(path)
    plt.close(fig)


def fig_live_hedge(daily: pd.DataFrame, key: str, path: str):
    """Cumulative P&L of the always-on hedge leg, split into index P&L, funding and trading costs."""
    d = daily[[f"{key}|hedge_mtm", f"{key}|funding", f"{key}|cost"]].dropna()
    fig, ax = plt.subplots(figsize=(WIDTH, 2.1))
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.plot(d.index, 100 * d[f"{key}|hedge_mtm"].cumsum(), color=BLUE, lw=1.3, label="index P&L of the BVIV position")
    ax.plot(d.index, -100 * d[f"{key}|funding"].cumsum(), color=ORANGE, lw=1.3, label="funding paid (negative = cost)")
    net = d[f"{key}|hedge_mtm"] - d[f"{key}|funding"] - d[f"{key}|cost"]
    ax.plot(d.index, 100 * net.cumsum(), color=INK2, lw=1.0, label="net hedge P&L")
    ax.set_title("Always-on hedge: cumulative P&L (% of BTC notional)", color=INK)
    ax.legend(loc="upper left", handlelength=1.4)
    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
    fig.savefig(path)
    plt.close(fig)


def fig_live_timing(placebo: pd.DataFrame, real: float, grid: pd.DataFrame, path: str):
    """Left: the selected ratchet against every weekly shift of its triggers.
    Right: timing value (ES reduction minus placebo mean) of every ratchet configuration, by core floor."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.1), gridspec_kw={"wspace": 0.3, "width_ratios": [1.1, 1]})
    ax = axes[0]
    ax.hist(placebo["es_red"], bins=24, color=GRID, edgecolor="#fcfcfb", linewidth=0.6)
    ax.axvline(real, color=AQUA, lw=2.0)
    ax.annotate("selected\nratchet", (real, ax.get_ylim()[1] * 0.88), xytext=(-6, 0), textcoords="offset points",
                color=INK, fontsize=7, ha="right", va="top")
    ax.set_title("ES reduction: ratchet vs. its placebos", color=INK)
    ax.set_xlabel("ES$_{97.5}$ reduction (%)")
    ax.set_ylabel("weekly shifts")
    ax = axes[1]
    ax.axhline(0, color=AXIS, lw=0.8)
    floors = sorted(grid["floor"].unique())
    for i, f in enumerate(floors):
        g = grid[grid["floor"] == f]
        jitter = (np.arange(len(g)) / max(len(g) - 1, 1) - 0.5) * 0.35
        ax.scatter(i + jitter, g["timing"], s=12, color=AQUA, alpha=0.75, linewidths=0)
        ax.plot([i - 0.22, i + 0.22], [g["timing"].mean()] * 2, color=INK, lw=1.4)
    ax.set_xticks(range(len(floors)), [f"core {f:g}" for f in floors])
    ax.set_title("Timing value by core hedge", color=INK)
    ax.set_ylabel("ES red. minus placebo (pp)")
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)
