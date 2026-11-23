import argparse
import json
from pathlib import Path

import networkx as nx

NODES = ["WA", "C1", "C2", "UT", "CO", "NE", "IL", "MI", "PA", "NY", "NJ", "DC", "TX", "GA"]
LINKS_KM = [
    ("WA", "C1", 1520),
    ("WA", "C2", 2160),
    ("WA", "IL", 3520),
    ("C1", "C2", 1040),
    ("C1", "UT", 1440),
    ("C2", "TX", 2640),
    ("UT", "CO", 880),
    ("UT", "MI", 2960),
    ("CO", "NE", 1120),
    ("CO", "TX", 1840),
    ("NE", "IL", 1040),
    ("IL", "PA", 1040),
    ("MI", "NY", 880),
    ("MI", "NJ", 1200),
    ("PA", "NY", 560),
    ("PA", "NJ", 640),
    ("PA", "GA", 1280),
    ("NY", "DC", 560),
    ("NJ", "DC", 400),
    ("DC", "TX", 2480),
    ("TX", "GA", 1520),
]


SPAN_LENGTH_KM = 80
SOURCE = "C1"
DESTINATION = "NJ"


POSITIONS = {
    "WA": (1, 3), "C1": (0, 2), "C2": (1, 1),
    "UT": (2.6, 2), "CO": (4, 2), "NE": (5.4, 2),
    "IL": (7, 2), "MI": (8, 3), "PA": (9, 2),
    "NY": (11, 3), "NJ": (12, 2), "DC": (10, 1),
    "TX": (6, 0), "GA": (8, 0),
}


def build_nsf_network():
    """Build the undirected physical topology with lengths and span counts."""

    network = nx.Graph()
    network.add_nodes_from(NODES)
    for start, end, length_km in LINKS_KM:
        if start not in NODES or end not in NODES or start == end:
            raise ValueError(f"Invalid link endpoints: {start}, {end}")
        if network.has_edge(start, end):
            raise ValueError(f"Duplicate link: {start} - {end}")
        if length_km <= 0 or length_km % SPAN_LENGTH_KM != 0:
            raise ValueError("Link lengths must be positive multiples of the span length.")
        network.add_edge(
            start, end,
            length_km=length_km,
            spans=length_km // SPAN_LENGTH_KM,
        )
    if not nx.is_connected(network):
        raise ValueError("The reconstructed NSF network is not connected.")
    return network


def shortest_route(network, source, destination):
    """Return one minimum-distance route using Dijkstra's algorithm."""

    return nx.shortest_path(
        network, source=source, target=destination,
        weight="length_km", method="dijkstra",
    )


def route_length_km(network, route):
    """Sum the physical lengths of consecutive links in a route."""
    return sum(
        network[start][end]["length_km"]
        for start, end in zip(route[:-1], route[1:])
    )


def plot_network(network, route=None):
    """Return a figure of the topology, optionally highlighting a route."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    nx.draw_networkx_edges(network, POSITIONS, ax=ax, edge_color="#8c99a8", width=2)
    if route:
        nx.draw_networkx_edges(
            network, POSITIONS, ax=ax,
            edgelist=list(zip(route[:-1], route[1:])),
            edge_color="#d97706", width=4,
        )
    nx.draw_networkx_nodes(
        network, POSITIONS, ax=ax, node_size=950,
        node_color="#e2ecf9", edgecolors="#355d87", linewidths=1.7,
    )
    nx.draw_networkx_labels(network, POSITIONS, ax=ax, font_size=12, font_weight="bold")


    label_adjustments = {
        frozenset(("WA", "C2")): (0.35, 0.26, 0),
        frozenset(("WA", "IL")): (0.30, 0, 0.12),
        frozenset(("UT", "MI")): (0.70, 0, 0.12),
        frozenset(("MI", "NJ")): (0.22, 0, -0.13),
        frozenset(("PA", "NY")): (0.65, 0, 0.18),
        frozenset(("NY", "DC")): (0.23, 0.25, 0),
        frozenset(("PA", "GA")): (0.46, -0.24, 0),
        frozenset(("DC", "TX")): (0.30, 0, -0.14),
    }
    for start, end, data in network.edges(data=True):

        edge = next(item for item in LINKS_KM if set(item[:2]) == {start, end})
        first, second = edge[:2]
        x1, y1 = POSITIONS[first]
        x2, y2 = POSITIONS[second]
        t, dx, dy = label_adjustments.get(frozenset((start, end)), (0.5, 0, -0.13))
        ax.text(
            x1 + t * (x2 - x1) + dx, y1 + t * (y2 - y1) + dy,
            str(data["length_km"]), ha="center", va="center", fontsize=10,
            color="#263445", zorder=5,
            bbox={"facecolor": "#f8fafc", "edgecolor": "none", "pad": 1.4},
        )

    ax.set_title("NSF network | 14 nodes, 21 links", fontsize=18, fontweight="bold", pad=32)
    if route:
        length = route_length_km(network, route)
        subtitle = f"{' -> '.join(route)}  |  {length:,} km  |  {length // SPAN_LENGTH_KM} spans"
        ax.text(0.5, 1.035, subtitle, transform=ax.transAxes, ha="center", fontsize=11)
        ax.legend(
            handles=[Line2D([0], [0], color="#d97706", lw=4, label="Shortest route by distance")],
            loc="lower right", frameon=False,
        )
    fig.text(
        0.5, 0.035,
        "Source: Ives (2015), Fig. 3.4, p. 53. Labels: link length in km. Layout is not to scale.\n"
        "Line crossings without a labelled node are not junctions. Optical feasibility is not evaluated.",
        ha="center", fontsize=9, color="#526175",
    )
    ax.set_xlim(-0.8, 12.8)
    ax.set_ylim(-0.55, 3.55)
    ax.set_axis_off()
    fig.subplots_adjust(left=0.035, right=0.965, top=0.82, bottom=0.12)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=NODES, default=SOURCE)
    parser.add_argument("--target", choices=NODES, default=DESTINATION)
    parser.add_argument("--no-show", action="store_true", help="Save the figure without opening a window.")
    args = parser.parse_args()


    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    network = build_nsf_network()
    route = shortest_route(network, args.source, args.target)
    length = route_length_km(network, route)
    spans = sum(network[u][v]["spans"] for u, v in zip(route[:-1], route[1:]))

    print("--- NSF topology reconstructed from Ives (2015), Fig. 3.4 ---")
    print(f"Nodes = {network.number_of_nodes()}, Physical links = {network.number_of_edges()}")
    print(f"Span length = {SPAN_LENGTH_KM} km")
    print("\n--- Shortest route by physical distance ---")
    print(f"Route: {' -> '.join(route)}")
    for start, end in zip(route[:-1], route[1:]):
        data = network[start][end]
        print(f"  {start} -> {end}: {data['length_km']} km, {data['spans']} spans")
    print(f"Total distance = {length} km")
    print(f"Number of links = {len(route) - 1}")
    print(f"Number of spans = {spans}")
    print("This is a topological route; optical feasibility has not been evaluated.")


    output_dir = Path(__file__).resolve().parent / "results_nsf"
    output_dir.mkdir(exist_ok=True)
    prefix = f"nsf_{args.source}_to_{args.target}"
    figure_path = output_dir / f"{prefix}.png"
    results_path = output_dir / f"{prefix}.json"
    results = {
        "source_reference": "Ives (2015), Figure 3.4, p. 53; span assumption p. 52",
        "scope": "Topology and routing only; no optical feasibility or wavelength allocation",
        "span_length_km": SPAN_LENGTH_KM,
        "nodes": NODES,
        "links": [dict(source=u, target=v, length_km=km) for u, v, km in LINKS_KM],
        "source": args.source, "destination": args.target,
        "shortest_route": route, "route_length_km": length,
        "route_link_count": len(route) - 1, "route_span_count": spans,
    }
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    fig = plot_network(network, route)
    fig.savefig(figure_path, dpi=180, facecolor=fig.get_facecolor())
    print(f"Saved figure: {figure_path}")
    print(f"Saved results: {results_path}")
    if args.no_show:
        plt.close(fig)
    else:
        print("Close the figure window to finish.")
        plt.show()


if __name__ == "__main__":
    main()
