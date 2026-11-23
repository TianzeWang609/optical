"""Visualize continuous lightpaths and the occupied wavelength slots on each link."""

import colorsys
import math


# 用波长决定颜色；链路不重叠的不同请求可以复用同一波长。
WAVE_COLORS = ("#27679D", "#B75A1E", "#12796D", "#79569B", "#AA4252",
               "#887025", "#496F37", "#88536E")
INK = "#203345"
MUTED = "#526775"
RULE = "#CBD6DD"
FREE = "#E9EFF2"


def wavelength_color(wave):
    if wave <= len(WAVE_COLORS):
        return WAVE_COLORS[wave - 1]
    return colorsys.hls_to_rgb((wave * 0.61803398875) % 1, 0.36, 0.58)


def draw_ring(axis, nodes, edges, requests, count):
    from matplotlib.patches import Circle

    positions = {node: (math.sin(2 * math.pi * i / len(nodes)), math.cos(2 * math.pi * i / len(nodes)))
                 for i, node in enumerate(nodes)}
    peak = max(link["occupied_wavelengths"] for _, _, link in edges)
    busiest = []
    for u, v, link in edges:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        used = link["occupied_wavelengths"]
        if used == peak:
            busiest.append(f"{u}–{v}")
        if not used:
            axis.plot([x1, x2], [y1, y2], color=RULE, lw=2.1, linestyle=(0, (4, 3)), zorder=1)
        axis.text(0.61 * (x1 + x2), 0.61 * (y1 + y2), f"{used}/{count}",
                  ha="center", va="center", fontsize=10, color=INK if used else MUTED,
                  fontweight="bold" if used == peak else "normal")

    active = [r for r in requests if r["wavelength"] is not None]
    max_wave = max((r["wavelength"] for r in active), default=1)
    spacing = min(0.078, 0.18 / max(1, max_wave - 1))
    for request in active:
        wave = request["wavelength"]
        # 同心五边形为每个波长提供固定线位。一次绘制整条请求，转发节点两侧颜色一致。
        radius = 1 + (wave - (max_wave + 1) / 2) * spacing
        points = [(radius * positions[node][0], radius * positions[node][1]) for node in request["path"]]
        xs, ys = zip(*points)
        axis.plot(xs, ys, color=wavelength_color(wave), lw=min(3.7, 13 / max_wave),
                  solid_capstyle="round", solid_joinstyle="round", zorder=2)
    for node, (x, y) in positions.items():
        axis.add_patch(Circle((x, y), 0.115, facecolor="white", edgecolor=INK, lw=1.7, zorder=3))
        axis.text(x, y, node, ha="center", va="center", color=INK, fontsize=14, fontweight="bold", zorder=4)

    axis.text(0, 0.15, "Peak link occupancy", ha="center", va="center", color=MUTED, fontsize=10)
    axis.text(0, -0.035, f"{peak / count:.0%}", ha="center", va="center", color=INK,
              fontsize=29, fontweight="bold")
    label = busiest[0] if len(busiest) == 1 else f"{len(busiest)} links"
    axis.text(0, -0.21, f"{label}   ·   {peak}/{count} wavelengths", ha="center", va="center",
              fontsize=10.5, color=MUTED)
    axis.set(xlim=(-1.18, 1.18), ylim=(-1.07, 1.19), aspect="equal")
    axis.set_axis_off()
    axis.text(0.5, -0.045, "Colour follows wavelength. Dashed links are idle.",
              transform=axis.transAxes, ha="center", va="top", fontsize=10, color=MUTED)


def draw_slots(axis, edges, count):
    from matplotlib.colors import to_rgb

    colors = [to_rgb(wavelength_color(wave)) for wave in range(1, count + 1)]
    empty_color = to_rgb(FREE)
    pixels = [[colors[i] if occupant is not None else empty_color
               for i, occupant in enumerate(link["wavelength_slots"])] for _, _, link in edges]
    axis.imshow(pixels, aspect="auto", interpolation="nearest", origin="upper")
    ticks = list(range(0, count, max(1, math.ceil(count / 10))))
    if ticks[-1] != count - 1:
        ticks.append(count - 1)
    axis.set_xticks(ticks, [f"λ{i + 1}" for i in ticks])
    axis.xaxis.tick_top()
    axis.set_yticks(range(len(edges)), [f"{u}–{v}" for u, v, _ in edges])
    axis.tick_params(axis="both", length=0, labelsize=10.5, colors=INK, pad=6)
    axis.spines[:].set_visible(False)
    axis.hlines([i - 0.5 for i in range(1, len(edges))], -0.5, count - 0.5, color="white", lw=3.2)
    if count <= 24:
        axis.vlines([i - 0.5 for i in range(1, count)], -0.5, len(edges) - 0.5, color="white", lw=2.5)
    for row, (_, _, link) in enumerate(edges):
        if count <= 12:
            for col, occupant in enumerate(link["wavelength_slots"]):
                if occupant is not None:
                    axis.text(col, row, occupant, ha="center", va="center",
                              fontsize=10.5, fontweight="bold", color="white")
        axis.text(1.045, row, f"{link['utilization']:.0%}", transform=axis.get_yaxis_transform(),
                  va="center", fontsize=10.5, color=INK)
    axis.text(1.045, 1.04, "Load", transform=axis.transAxes, color=MUTED, fontsize=9)
    axis.set_title("Wavelengths on each link", loc="left", fontsize=14, fontweight="bold", color=INK, pad=31)


def draw_requests(axis, requests, shown):
    axis.set_xlim(0, 1)
    axis.set_ylim(shown + 0.65, -0.9)
    axis.set_axis_off()
    axis.set_title("Request routes", loc="left", fontsize=14, fontweight="bold", color=INK, pad=8)
    columns = (0.0, 0.16, 0.47, 0.65, 0.83)
    for x, header in zip(columns, ("ID", "Path", "Wavelength", "SNR", "Ideal capacity")):
        axis.text(x, -0.35, header, color=MUTED, fontsize=9.5, va="center")
    axis.axhline(0.05, color=RULE, lw=0.8)
    for i, r in enumerate(requests[:shown]):
        y = i + 0.67
        wave = r["wavelength"]
        axis.plot(0.009, y, marker="o", ms=6, markeredgewidth=0, color=wavelength_color(wave) if wave else RULE)
        axis.text(0.043, y, r["id"], color=INK, fontsize=10.5, va="center")
        values = (" → ".join(r["path"]), f"λ{wave}" if wave else "Blocked",
                  f"{r['snr_db']:.2f} dB", f"{r['allocated_capacity_gbps']:,.2f} Gbit/s" if wave else "—")
        for x, value in zip(columns[1:], values):
            axis.text(x, y, value, color=INK, fontsize=10.5, va="center")
    if len(requests) > shown:
        axis.text(0, shown + 0.55, f"First {shown} of {len(requests)} requests shown. Full list in ring_capacity.json.",
                  fontsize=9, color=MUTED)


def plot_route_occupancy(network, plan):
    """Return the ring, link-by-wavelength occupancy matrix and request key from a single allocation."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    nodes = list(network)
    usage = {frozenset(link["nodes"]): link for link in plan["links"]}
    edges = [(u, v, usage[frozenset((u, v))]) for u, v in zip(nodes, nodes[1:] + nodes[:1])]
    count = network.graph["wavelengths_per_link"]
    requests = plan["requests"]
    shown = min(20, len(requests))
    height = 6.8 + 0.30 * max(0, shown - 3)
    fig = plt.figure(figsize=(12.3, height), facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=(1, 1.12), height_ratios=(2.35, 0.85 + 0.25 * shown),
                            left=0.049, right=0.915, bottom=0.46 / height, top=1 - 1.86 / height,
                            wspace=0.22, hspace=0.43)
    fig.text(0.049, 1 - 0.54 / height, "Wavelength occupancy on the five-node ring",
             fontsize=23, fontweight="bold", color=INK)
    request_noun = "request" if len(requests) == 1 else "requests"
    wave_noun = "wavelength" if count == 1 else "wavelengths"
    fig.text(0.05, 1 - 0.87 / height,
             f"{plan['accepted_count']} of {len(requests)} {request_noun} carried   ·   "
             f"{count} {wave_noun} per link   ·   {network.graph['bandwidth_ghz_per_wavelength']:g} GHz per wavelength",
             fontsize=11.5, color=MUTED)
    divider = 1 - 1.09 / height
    fig.add_artist(Line2D([0.049, 0.966], [divider, divider], transform=fig.transFigure, color=RULE, lw=0.8))
    draw_ring(fig.add_subplot(grid[:, 0]), nodes, edges, requests, count)
    draw_slots(fig.add_subplot(grid[0, 1]), edges, count)
    draw_requests(fig.add_subplot(grid[1, 1]), requests, shown)
    return fig
