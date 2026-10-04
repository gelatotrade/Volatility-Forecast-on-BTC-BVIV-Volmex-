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
GRID, AXIS, SHADE = "#e1e0d9", "#c3c2b7", "#f0efec"

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


def _shade(ax, mask: pd.Series, label: str | None = None):
    """Shade contiguous True runs of a boolean series."""
    m = mask.astype(int).to_numpy()
    edges = np.flatnonzero(np.diff(np.r_[0, m, 0]))
    for i, (a, b) in enumerate(zip(edges[::2], edges[1::2])):
        ax.axvspan(mask.index[a], mask.index[min(b, len(mask) - 1)], color=SHADE, lw=0,
                   label=label if i == 0 else None, zorder=0)


def fig_market(bars: pd.DataFrame, path: str):
    """BTC and BVIV on one simulated path, latent stress regime shaded."""
    daily = bars.resample("4h").last()
    stress = (bars["regime"] == 1).resample("4h").mean() > 0.5 if "regime" in bars else None
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 2.45), sharex=True, gridspec_kw={"hspace": 0.32})
    for ax, col, title in ((axes[0], "close", "BTC price (USD, log scale)"),
                           (axes[1], "bviv", "BVIV implied volatility (vol points)")):
        if stress is not None:
            _shade(ax, stress, "latent stress regime" if ax is axes[0] else None)
        ax.plot(daily.index, daily[col], color=BLUE, lw=1.0)
        ax.set_title(title, color=INK)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0)))
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1].xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=(1, 4, 7, 10)))
    axes[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
    if stress is not None:
        axes[0].legend(loc="upper left", handlelength=1.2)
    fig.savefig(path)
    plt.close(fig)


def fig_mechanics(bars: pd.DataFrame, sig: pd.DataFrame, results: dict, window: slice, z_enter: float,
                  names: dict[str, str], path: str):
    """A breakdown episode: price vs rolling VWAP and band (top), hedge sizes (bottom)."""
    b, s = bars.loc[window], sig.loc[window]
    lower = s["vwap"] * np.exp(z_enter * s["sigma_day"] / np.sqrt(3.0))
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 3.0), sharex=True, gridspec_kw={"hspace": 0.5, "height_ratios": [1.25, 1]})
    ax = axes[0]
    ax.plot(b.index, b["close"], color=INK2, lw=0.8, label="BTC close (15m)")
    ax.plot(s.index, s["vwap"], color=BLUE, lw=1.2, label="rolling 24h VWAP")
    ax.plot(s.index, lower, color=ORANGE, lw=1.0, label=f"breakdown band (z = {z_enter:g})")
    trig = s["z"] < z_enter
    ax.scatter(b.index[trig], b["close"][trig], s=6, color=ORANGE, zorder=3, linewidths=0)
    ax.set_title("Price, VWAP and the breakdown band", color=INK, pad=14)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.1f}k"))
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.98), ncol=3, handlelength=1.4, borderaxespad=0)
    ax = axes[1]
    for key, label in names.items():
        pos = results[key]["position"].loc[window]
        ax.step(pos.index, pos, where="post", color=FAMILY_COLOR[label], lw=1.2, label=label)
    ax.set_title("Hedge size (BVIV-perp contracts, \\$1 per vol point)", color=INK, pad=14)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.98), ncol=3, handlelength=1.4, borderaxespad=0)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b\n%H:%M"))
    fig.savefig(path)
    plt.close(fig)


def fig_frontier(med: pd.DataFrame, iqr: pd.DataFrame, selected: dict[str, str], path: str):
    """Median ES reduction vs median hedge cost on test paths, per rule family."""
    fig, ax = plt.subplots(figsize=(WIDTH, 2.45))
    fam = med.index.to_series().str.split("|").str[0]
    for key, color, label in (("switch", ORANGE, "VWAP-switch grid"), ("ratchet", AQUA, "VWAP-ratchet grid")):
        pts = med[fam == key]
        ax.scatter(pts["hedge_cost"], pts["es_red"], s=14, color=color, alpha=0.55, linewidths=0, label=label, zorder=2)
    ladder = med[fam == "always"].sort_values("hedge_cost")
    ax.plot(np.r_[0.0, ladder["hedge_cost"]], np.r_[0.0, ladder["es_red"]], color=BLUE, lw=1.4, zorder=3)
    ax.scatter(ladder["hedge_cost"], ladder["es_red"], s=26, color=BLUE, edgecolors="#fcfcfb", linewidths=1.5,
               zorder=4, label="Always-on, scaled 0.5x-3x")
    for k, row in ladder.iterrows():
        ax.annotate(f"{float(k.split('|')[1]):g}x", (row["hedge_cost"], row["es_red"]), xytext=(-4, 6), ha="right",
                    textcoords="offset points", color=INK2, fontsize=7)
    for family, key in selected.items():
        row = med.loc[key]
        color = ORANGE if family == "switch" else AQUA
        ax.errorbar(row["hedge_cost"], row["es_red"], yerr=[[row["es_red"] - iqr.loc[key, "q25"]], [iqr.loc[key, "q75"] - row["es_red"]]],
                    color=color, lw=1.0, capsize=0, zorder=5)
        ax.scatter(row["hedge_cost"], row["es_red"], s=60, marker="D", color=color, edgecolors=INK, linewidths=0.8, zorder=6)
        ax.annotate("selected " + family, (row["hedge_cost"], row["es_red"]),
                    xytext=(8, -14) if family == "ratchet" else (-8, 10), ha="left" if family == "ratchet" else "right",
                    textcoords="offset points", color=INK, fontsize=7)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xlabel("Hedge cost: carry premium + trading (% of BTC notional p.a., median)")
    ax.set_ylabel("ES$_{97.5}$ reduction (%, median)")
    ax.set_title("Tail protection per unit of cost", color=INK)
    ax.legend(loc="upper right", handlelength=1.2)
    ax.set_xlim(left=0)
    fig.savefig(path)
    plt.close(fig)


def fig_forecasts(summary: pd.DataFrame, path: str):
    """Median QLIKE loss ratio vs HAR by model, one panel per horizon (dot plot)."""
    hs = sorted(summary["h"].unique())
    models = [m for m in summary["model"].unique()]
    fig, axes = plt.subplots(1, len(hs), figsize=(WIDTH, 1.75), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, h in zip(np.atleast_1d(axes), hs):
        d = summary[summary["h"] == h].set_index("model").reindex(models)
        y = np.arange(len(models))[::-1]
        ax.axvline(1.0, color=AXIS, lw=0.8)
        ax.hlines(y, 1.0, d["qlike_ratio"], color=GRID, lw=1.0)
        best = d["qlike_ratio"].idxmin()
        colors = [BLUE if m == best else MUTED for m in models]
        ax.scatter(d["qlike_ratio"], y, s=28, color=colors, edgecolors="#fcfcfb", linewidths=1.2, zorder=3)
        ax.set_title(f"{h}-day horizon", color=INK)
        ax.set_yticks(y, models)
        ax.set_xlabel("QLIKE / QLIKE(HAR)")
        ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


def pick_episode(sig: pd.DataFrame, after: pd.Timestamp, z_enter: float, quiet_days: float = 3.0,
                 before_days: float = 1.5, after_days: float = 2.5) -> slice:
    """Window around a *fresh* VWAP breakdown: the first trigger after ``quiet_days`` without one,
    choosing the onset followed by the deepest breakdown (so the ratchet's step-up and decay are visible)."""
    bars_per_day = int(pd.Timedelta("1D") / (sig.index[1] - sig.index[0]))
    trig = (sig["z"] < z_enter).astype(int)
    recent = trig.shift(1).rolling(int(quiet_days * bars_per_day), min_periods=1).max().fillna(0)
    onsets = sig.index[(trig == 1) & (recent == 0) & (sig.index > after + pd.Timedelta(days=quiet_days))]
    horizon = int(after_days * bars_per_day)
    depth = {t: sig["z"].loc[t:].iloc[:horizon].min() for t in onsets}
    t = min(depth, key=depth.get)
    return slice(t - pd.Timedelta(days=before_days), t + pd.Timedelta(days=after_days))
