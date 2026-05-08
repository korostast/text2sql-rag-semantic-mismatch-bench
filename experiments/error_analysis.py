import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Arial"]


path = "../artifacts/gpt-oss-120b_mini_dev_annotated.json"
with open(path, encoding="utf-8") as f:
    data = json.load(f)


groups = {
    "Semantic Mismatch": [
        "Evidence Misinterpretation",
        "Schema Linking",
        "Over/Under Filtering",
    ],
    "Data Operations": [
        "Output Format",
        "Aggregation & Ordering",
        "Values & Types",
    ],
    "Execution": [
        "Hallucinations",
        "Syntax",
    ],
    "Other": [
        "Incorrect Gold",
    ],
}

group_of = {sub: grp for grp, subs in groups.items() for sub in subs}

sub_counts = Counter()
missing = 0
for row in data:
    if row.get("metric_value") != 0:
        continue

    et = row.get("error_type")
    seen = set()
    for disp in et:
        if disp in group_of and disp not in seen:
            sub_counts[disp] += 1
            seen.add(disp)

group_counts = Counter()
for sub, cnt in sub_counts.items():
    group_counts[group_of[sub]] += cnt

total = sum(sub_counts.values())

group_order = ["Other", "Data Operations", "Execution", "Semantic Mismatch"]
sub_order = [s for g in group_order for s in groups[g]]

base_colors = {
    "Semantic Mismatch": "#e9847b",
    "Data Operations": "#eebe87",
    "Execution": "#7dc9dd",
    "Other": "#bf97ec",
}


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))


def blend(c1, c2=(1, 1, 1), a=0.15):
    return tuple((1 - a) * x + a * y for x, y in zip(hex_to_rgb(c1), c2))


outer_colors = []
for g in group_order:
    subs = groups[g]
    for i, _ in enumerate(subs):
        if len(subs) == 1:
            outer_colors.append(blend(base_colors[g], a=0.15))
        else:
            outer_colors.append(blend(base_colors[g], a=0.15))

inner_sizes = [group_counts[g] for g in group_order]
outer_sizes = [sub_counts[s] for s in sub_order]

fig, ax = plt.subplots(figsize=(16, 11), subplot_kw=dict(aspect="equal"), facecolor="white")

startangle = 45
outer_radius = 1.35
outer_width = 0.42
inner_radius = outer_radius - outer_width
inner_width = 0.42
center_radius = inner_radius - inner_width

outer_wedges, _ = ax.pie(
    outer_sizes,
    radius=outer_radius,
    startangle=startangle,
    counterclock=False,
    colors=outer_colors,
    wedgeprops=dict(width=outer_width, edgecolor="white", linewidth=1.6),
)

inner_wedges, _ = ax.pie(
    inner_sizes,
    radius=inner_radius,
    startangle=startangle,
    counterclock=False,
    colors=[base_colors[g] for g in group_order],
    wedgeprops=dict(width=inner_width, edgecolor="white", linewidth=1.6),
)

ax.add_artist(Circle((0, 0), center_radius, fc="white", ec="white"))


def mid_angle(w):
    return (w.theta1 + w.theta2) / 2.0


def polar_to_xy(r, ang_deg):
    ang = np.deg2rad(ang_deg)
    return r * np.cos(ang), r * np.sin(ang)


# Inner ring labels
for w, g in zip(inner_wedges, group_order):
    ang = mid_angle(w)
    x, y = polar_to_xy(inner_radius - inner_width / 2, ang)
    pct = group_counts[g] * 100 / total
    ax.text(
        x,
        y,
        f"{g}\n{pct:.0f}%",
        ha="center",
        va="center",
        weight="bold",
        fontsize=12.5,
        color="#2b2b2b",
    )

# Outer ring labels
for w, s in zip(outer_wedges, sub_order):
    pct = sub_counts[s] * 100 / total
    ang = mid_angle(w)

    if s == "Hallucinations":
        x0, y0 = polar_to_xy(outer_radius - outer_width * 0.5, ang)
        xt, yt = polar_to_xy(outer_radius + 0.34, ang)
        ha = "left" if np.cos(np.deg2rad(ang)) >= 0 else "right"
        xt += 0.08 if ha == "left" else -0.08

        angleA_val = 180 if ha == "left" else 0

        ax.annotate(
            f"Hallucinations\n{pct:.0f}%",
            xy=(x0, y0),
            xytext=(xt, yt),
            ha=ha,
            va="center",
            fontsize=12.5,
            arrowprops=dict(
                arrowstyle="-",
                connectionstyle=f"angle,angleA={angleA_val},angleB={ang}",
                lw=1.2,
                color="#555555",
                shrinkA=0,
                shrinkB=0,
            ),
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.85),
        )
        continue

    x, y = polar_to_xy(outer_radius - outer_width / 2, ang)
    label = s.replace(" ", "\n") if len(s) > 16 else s

    if s in {
        "Evidence Misinterpretation",
        "Over/Under Filtering",
        "Aggregation & Ordering",
        "Values & Types",
        "Output Format",
        "Incorrect Gold",
    }:
        label = label.replace(" ", "\n")
    elif s == "Schema Linking":
        label = "Schema\nLinking"
    elif s == "Data Operations":
        label = "Data\nOperations"
    elif s == "Syntax":
        label = "Syntax"

    ax.text(x, y, f"{label}\n{pct:.0f}%", ha="center", va="center", fontsize=11.5, color="#2b2b2b")

ax.text(0, 0.04, "Failures", ha="center", va="center", fontsize=16, color="#333333")
ax.text(0, -0.065, f"{total} in total", ha="center", va="center", fontsize=11, color="#666666")

ax.set_xlim(-1.75, 1.95)
ax.set_ylim(-1.55, 1.55)
ax.axis("off")

out = "../artifacts/error_analysis.png"
plt.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
