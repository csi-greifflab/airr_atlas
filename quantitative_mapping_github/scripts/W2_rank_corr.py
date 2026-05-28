import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
CSV = """repetition,size,w2,w2_time,sinkhorn,sinkhorn_time
1,10000,0.0754,2.03,0.5901,897.39
1,30000,0.0555,2.87,0.5207,985.33
1,50000,0.0554,3.80,0.4990,1136.68
1,100000,0.0554,5.99,0.4708,1509.77
1,150000,0.0561,8.08,0.4575,1831.45
1,200000,0.0549,10.56,0.4498,2297.93
1,300000,0.0556,15.02,0.4419,3256.24
1,399900,0.0550,19.52,0.4374,4526.90
2,10000,0.0637,1.81,0.5835,897.01
2,30000,0.0572,2.67,0.5226,985.48
2,50000,0.0579,3.63,0.4993,1136.37
2,100000,0.0549,5.78,0.4704,1509.93
2,150000,0.0548,8.27,0.4570,1831.60
2,200000,0.0550,10.47,0.4498,2297.84
2,300000,0.0545,15.01,0.4413,3255.88
2,399900,0.0547,19.45,0.4372,4528.40
3,10000,0.0607,1.71,0.5833,897.52
3,30000,0.0578,2.56,0.5230,985.23
3,50000,0.0547,3.70,0.4976,1136.38
3,100000,0.0536,5.95,0.4703,1510.51
3,150000,0.0529,8.03,0.4569,1832.44
3,200000,0.0557,10.24,0.4500,2297.85
3,300000,0.0534,14.95,0.4412,3255.27
3,399900,0.0552,19.47,0.4373,4527.67
4,10000,0.0666,1.97,0.5863,897.08
4,30000,0.0569,2.73,0.5221,985.61
4,50000,0.0594,3.67,0.4996,1136.65
4,100000,0.0570,5.84,0.4708,1510.52
4,150000,0.0553,8.28,0.4578,1831.88
4,200000,0.0542,10.49,0.4492,2297.80
4,300000,0.0554,15.07,0.4418,3255.51
4,399900,0.0541,19.45,0.4370,4528.22
5,10000,0.0635,1.84,0.5836,896.98
5,30000,0.0542,2.81,0.5217,985.32
5,50000,0.0566,3.71,0.4976,1136.52
5,100000,0.0554,5.97,0.4708,1509.95
5,150000,0.0558,8.12,0.4581,1831.73
5,200000,0.0546,10.45,0.4504,2297.71
5,300000,0.0546,14.83,0.4410,3255.66
5,399900,0.0547,19.41,0.4368,4527.53
"""  # <-- paste your full CSV here, or load from file

df = pd.read_csv(StringIO(CSV))


sizes = sorted(df["size"].unique())
reps  = sorted(df["repetition"].unique())
size_labels = [f"{int(s/1000)}k" for s in sizes]

# ── Rank correlations ─────────────────────────────────────────────────────────

rho_global, p_rho = spearmanr(df["w2"], df["sinkhorn"])
tau_global, p_tau = kendalltau(df["w2"], df["sinkhorn"])

print("=== Global ===")
print(f"  Spearman ρ = {rho_global:.4f}  (p = {p_rho:.4e})")
print(f"  Kendall  τ = {tau_global:.4f}  (p = {p_tau:.4e})")

print("\n=== Within-repetition (across sizes) ===")
within = []
for rep, grp in df.groupby("repetition"):
    rho, p = spearmanr(grp["w2"], grp["sinkhorn"])
    within.append({"repetition": rep, "rho": rho, "p": p})
    print(f"  Rep {rep}: ρ = {rho:.4f}  (p = {p:.4e})")
within = pd.DataFrame(within)
print(f"  Mean ρ = {within['rho'].mean():.4f} ± {within['rho'].std():.4f}")

print("\n=== Across-repetition (at each fixed size) ===")
across = []
for size, grp in df.groupby("size"):
    rho, p = spearmanr(grp["w2"], grp["sinkhorn"])
    across.append({"size": size, "rho": rho, "p": p})
    print(f"  n={size:<7}: ρ = {rho:.4f}  (p = {p:.4e})")
across = pd.DataFrame(across)
print(f"  Mean ρ = {across['rho'].mean():.4f} ± {across['rho'].std():.4f}")

# ── Per-size summary stats ────────────────────────────────────────────────────

by_size = df.groupby("size")[["w2", "sinkhorn"]].agg(["mean", "std"])
by_size.columns = ["w2_mean", "w2_std", "sink_mean", "sink_std"]

# ── Plots ─────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

ax1  = fig.add_subplot(gs[0, :])       # convergence — full width
ax2  = fig.add_subplot(gs[1, 0])       # scatter
ax3  = fig.add_subplot(gs[1, 1])       # within-rep rho
ax4  = fig.add_subplot(gs[1, 2])       # across-rep rho

BLUE   = "#378ADD"
ORANGE = "#D85A30"
TEAL   = "#1D9E75"

# ── 1. Convergence ────────────────────────────────────────────────────────────
ax1b = ax1.twinx()

for rep, grp in df.groupby("repetition"):
    ax1.plot(grp["size"], grp["w2"],
             "o", color=BLUE, alpha=0.35, ms=5, zorder=2)
    ax1b.plot(grp["size"], grp["sinkhorn"],
              "o", color=ORANGE, alpha=0.35, ms=5, zorder=2)

ax1.plot(by_size.index, by_size["w2_mean"],
         color=BLUE, lw=2, label="W2 mean", zorder=3)
ax1b.plot(by_size.index, by_size["sink_mean"],
          color=ORANGE, lw=2, label="Sinkhorn mean", zorder=3)

ax1.fill_between(by_size.index,
                 by_size["w2_mean"] - by_size["w2_std"],
                 by_size["w2_mean"] + by_size["w2_std"],
                 color=BLUE, alpha=0.12)
ax1b.fill_between(by_size.index,
                  by_size["sink_mean"] - by_size["sink_std"],
                  by_size["sink_mean"] + by_size["sink_std"],
                  color=ORANGE, alpha=0.12)

ax1.set_xlabel("Sample size (n)", fontsize=11)
ax1.set_ylabel("W2 distance", color=BLUE, fontsize=11)
ax1b.set_ylabel("Sinkhorn distance", color=ORANGE, fontsize=11)
ax1.tick_params(axis="y", colors=BLUE)
ax1b.tick_params(axis="y", colors=ORANGE)
ax1.set_xticks(sizes)
ax1.set_xticklabels(size_labels, fontsize=9)
ax1.set_title("W2 and Sinkhorn vs sample size  (points = reps, line = mean ± 1 sd)",
              fontsize=11)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

# ── 2. Scatter ────────────────────────────────────────────────────────────────
sc = ax2.scatter(df["w2"], df["sinkhorn"],
                 c=df["size"], cmap="viridis", s=40, alpha=0.8, zorder=2)
plt.colorbar(sc, ax=ax2, label="Sample size", format="{x:.0f}")
ax2.set_xlabel("W2", fontsize=11)
ax2.set_ylabel("Sinkhorn", fontsize=11)
ax2.set_title(f"W2 vs Sinkhorn\nGlobal ρ = {rho_global:.4f}  (p = {p_rho:.2e})",
              fontsize=10)
ax2.grid(True, alpha=0.2)

# ── 3. Within-rep rho ────────────────────────────────────────────────────────
bar_colors = [BLUE if r >= 0 else ORANGE for r in within["rho"]]
bars = ax3.bar(within["repetition"].astype(str), within["rho"],
               color=bar_colors, alpha=0.75, edgecolor="white", linewidth=0.8)
ax3.axhline(0,   color="grey", lw=0.8, ls="--")
ax3.axhline(within["rho"].mean(), color=BLUE, lw=1.5, ls=":",
            label=f"mean = {within['rho'].mean():.3f}")
ax3.set_ylim(-1, 1)
ax3.set_xlabel("Repetition", fontsize=11)
ax3.set_ylabel("Spearman ρ", fontsize=11)
ax3.set_title("Within-rep ρ\n(across 8 sizes)", fontsize=10)
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.2)

for bar, val in zip(bars, within["rho"]):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             val + (0.03 if val >= 0 else -0.07),
             f"{val:.3f}", ha="center", va="bottom", fontsize=8)

# ── 4. Across-rep rho ────────────────────────────────────────────────────────
bar_colors2 = [TEAL if r >= 0 else ORANGE for r in across["rho"]]
bars2 = ax4.bar(size_labels, across["rho"],
                color=bar_colors2, alpha=0.75, edgecolor="white", linewidth=0.8)
ax4.axhline(0,   color="grey", lw=0.8, ls="--")
ax4.axhline(across["rho"].mean(), color=TEAL, lw=1.5, ls=":",
            label=f"mean = {across['rho'].mean():.3f}")
ax4.set_ylim(-1, 1)
ax4.set_xlabel("Sample size (n)", fontsize=11)
ax4.set_ylabel("Spearman ρ", fontsize=11)
ax4.set_title("Across-rep ρ\n(across 5 reps at each size)", fontsize=10)
ax4.tick_params(axis="x", rotation=45)
ax4.legend(fontsize=9)
ax4.grid(axis="y", alpha=0.2)

for bar, val in zip(bars2, across["rho"]):
    ax4.text(bar.get_x() + bar.get_width() / 2,
             val + (0.03 if val >= 0 else -0.07),
             f"{val:.3f}", ha="center", va="bottom", fontsize=8)

fig.suptitle("Rank correlation analysis: W2 vs Sinkhorn", fontsize=13, y=1.01)
plt.savefig("rank_correlation.png", dpi=150, bbox_inches="tight")
plt.show()



#save fig only ax1

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

ax1  = fig.add_subplot(gs[0, :])       # convergence — full width

BLUE   = "#378ADD"
ORANGE = "#D85A30"
TEAL   = "#1D9E75"

# ── 1. Convergence ────────────────────────────────────────────────────────────
ax1b = ax1.twinx()

for rep, grp in df.groupby("repetition"):
    ax1.plot(grp["size"], grp["w2"],
             "o", color=BLUE, alpha=0.35, ms=5, zorder=2)
    ax1b.plot(grp["size"], grp["sinkhorn"],
              "o", color=ORANGE, alpha=0.35, ms=5, zorder=2)

ax1.plot(by_size.index, by_size["w2_mean"],
         color=BLUE, lw=2, label="W2 mean", zorder=3)
ax1b.plot(by_size.index, by_size["sink_mean"],
          color=ORANGE, lw=2, label="Sinkhorn mean", zorder=3)

ax1.fill_between(by_size.index,
                 by_size["w2_mean"] - by_size["w2_std"],
                 by_size["w2_mean"] + by_size["w2_std"],
                 color=BLUE, alpha=0.12)
ax1b.fill_between(by_size.index,
                  by_size["sink_mean"] - by_size["sink_std"],
                  by_size["sink_mean"] + by_size["sink_std"],
                  color=ORANGE, alpha=0.12)

ax1.set_xlabel("Sample size (n)", fontsize=11)
ax1.set_ylabel("W2 distance", color=BLUE, fontsize=11)
ax1b.set_ylabel("Sinkhorn distance", color=ORANGE, fontsize=11)
ax1.tick_params(axis="y", colors=BLUE)
ax1b.tick_params(axis="y", colors=ORANGE)
ax1.set_xticks(sizes)
ax1.set_xticklabels(size_labels, fontsize=9)
ax1.set_title("W2 and Sinkhorn vs sample size  (points = reps, line = mean ± 1 sd)",
              fontsize=11)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

fig.suptitle("Rank correlation analysis: W2 vs Sinkhorn", fontsize=13, y=1.01)
plt.savefig("/doctorai/niccoloc/airr_atlas/quantitative_mapping_github/scripts/Convergence_plot.png", dpi=150, bbox_inches="tight")
plt.show()


#save fig only ax2

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

ax2  = fig.add_subplot(gs[0, :])       
BLUE   = "#378ADD"
ORANGE = "#D85A30"
TEAL   = "#1D9E75"

# ── 2. Scatter ────────────────────────────────────────────────────────────────
sc = ax2.scatter(df["w2"], df["sinkhorn"],
                 c=df["size"], cmap="viridis", s=40, alpha=0.8, zorder=2)
plt.colorbar(sc, ax=ax2, label="Sample size", format="{x:.0f}")
ax2.set_xlabel("W2", fontsize=11)
ax2.set_ylabel("Sinkhorn", fontsize=11)
ax2.set_title(f"W2 vs Sinkhorn\nGlobal ρ = {rho_global:.4f}  (p = {p_rho:.2e})",
              fontsize=10)
ax2.grid(True, alpha=0.2)

plt.savefig("/doctorai/niccoloc/airr_atlas/quantitative_mapping_github/scripts/Scatter_plot.png", dpi=150, bbox_inches="tight")

