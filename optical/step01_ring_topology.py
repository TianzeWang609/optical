"""Five-node optical ring with end-to-end SNR, capacity and wavelength assignment."""

import argparse
import json
import math
import sys
from dataclasses import asdict
from functools import lru_cache
from itertools import combinations, pairwise, permutations
from pathlib import Path

import networkx as nx


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optical.ring_occupancy import plot_route_occupancy
from reference_physical_model import ReferenceParameters, ReferencePhysicalModel


NODES = ["A", "B", "C", "D", "E"]

LINKS_KM = [
    ("A", "B", 400.0),
    ("B", "C", 400.0),
    ("C", "D", 400.0),
    ("D", "E", 400.0),
    ("E", "A", 400.0),
]

REFERENCE = ReferenceParameters()
SPAN_LENGTH_KM = REFERENCE.span_length_km
REFERENCE_DISTANCE_KM = 400.0

DEFAULT_BANDWIDTH_GHZ = REFERENCE.symbol_rate_baud / 1e9
DEFAULT_WAVELENGTHS = 4
# First-fit depends on arrival order. Start with the traffic used in the progress report.
BASELINE_REQUESTS = (
    ("A", "B"), ("A", "C"), ("E", "B"), ("A", "E"), ("E", "C"), ("D", "B"),
    ("C", "E"), ("C", "B"), ("A", "D"), ("D", "C"), ("D", "E"),
)
DEFAULT_REQUESTS = BASELINE_REQUESTS + tuple(
    pair for pair in permutations(NODES, 2) if pair not in BASELINE_REQUESTS
)
SOURCE = "A"
DESTINATION = "C"
OUTPUT_DIR = Path(__file__).resolve().parent / "results_ring_step01"


@lru_cache(maxsize=1)
def reference_operating_point():
    """Return the 400 km central-channel SNR (dB) and launch power (dBm)."""
    physical = ReferencePhysicalModel(REFERENCE)
    spans = physical.spans_for_distance(REFERENCE_DISTANCE_KM)
    power_dbm = physical.optimal_power_dbm(spans)
    return physical.snr_db(power_dbm, spans), power_dbm


def dp_efficiency(snr_linear):
    """Ideal dual-polarization Shannon spectral efficiency in bit/s/Hz."""
    return 2 * math.log1p(snr_linear) / math.log(2)


def build_ring(snr_400_db=None, bandwidth_ghz=None, *, wavelengths=DEFAULT_WAVELENGTHS):
    """Build undirected links. Each wavelength gets an equal share of the usable band."""
    if not isinstance(wavelengths, int) or isinstance(wavelengths, bool) or wavelengths < 1:
        raise ValueError("The number of wavelengths per link must be a positive integer.")
    if bandwidth_ghz is None:
        bandwidth_ghz = wavelengths * DEFAULT_BANDWIDTH_GHZ
    if not math.isfinite(bandwidth_ghz) or bandwidth_ghz <= 0:
        raise ValueError("Link bandwidth in GHz must be finite and positive.")
    power_dbm = None
    if snr_400_db is None:
        snr_400_db, power_dbm = reference_operating_point()
    try:
        reference_snr = 10.0 ** (snr_400_db / 10)
    except OverflowError as error:
        raise ValueError("400 km SNR is outside the supported numeric range.") from error
    if not math.isfinite(reference_snr) or reference_snr <= 0:
        raise ValueError("400 km SNR must give a finite, positive linear ratio.")

    network = nx.Graph(
        reference_distance_km=REFERENCE_DISTANCE_KM, snr_400_db=snr_400_db,
        snr_source="reference_gn" if power_dbm is not None else "user_supplied",
        launch_power_dbm_per_channel=power_dbm, span_length_km=SPAN_LENGTH_KM,
        bandwidth_ghz_per_link=bandwidth_ghz, wavelengths_per_link=wavelengths,
        bandwidth_ghz_per_wavelength=bandwidth_ghz / wavelengths,
    )
    if power_dbm is not None:
        network.graph["gn_reference"] = {
            "operating_point": "Full-load central WDM channel at the 400 km optimal launch power",
            "noise_model": "ASE, coherent SPM within the reference link, linearly accumulated XPM",
            "parameters": asdict(REFERENCE),
        }
    network.add_nodes_from(NODES)
    for start, end, length_km in LINKS_KM:
        spans = length_km / SPAN_LENGTH_KM
        if length_km <= 0 or not spans.is_integer():
            raise ValueError("Each link must contain a positive integer number of spans.")
        snr = reference_snr * (REFERENCE_DISTANCE_KM / length_km)
        capacity = bandwidth_ghz * dp_efficiency(snr)
        if not math.isfinite(capacity) or capacity <= 0:
            raise ValueError("Full-band capacity is outside the supported numeric range.")
        network.add_edge(
            start, end, length_km=length_km, spans=int(spans),
            snr_linear=snr, snr_db=10 * math.log10(snr), bandwidth_ghz=bandwidth_ghz,
            capacity_gbps=capacity, wavelength_capacity_gbps=capacity / wavelengths,
        )
    return network


def route_length_km(network, route):
    """Sum physical link lengths along a route."""
    return sum(network[u][v]["length_km"] for u, v in pairwise(route))


def route_metrics(network, path):
    """Evaluate a transparent path with independently accumulated link noise."""
    path = list(path)
    if len(path) < 2 or not nx.is_simple_path(network, path):
        raise ValueError("A route must be a connected simple path with distinct endpoints.")
    links = [network[u][v] for u, v in pairwise(path)]
    # 在线性域累加噪声：两条相同链路串联时，端到端 SNR 减半。
    snr = 1 / math.fsum(1 / link["snr_linear"] for link in links)
    efficiency = dp_efficiency(snr)
    full_capacity = min(link["bandwidth_ghz"] for link in links) * efficiency
    return {
        "path": path, "length_km": route_length_km(network, path),
        "spans": sum(link["spans"] for link in links),
        "snr_linear": snr, "snr_db": 10 * math.log10(snr),
        "spectral_efficiency_bps_hz": efficiency,
        "full_band_capacity_gbps": full_capacity,
        "wavelength_capacity_gbps": full_capacity / network.graph["wavelengths_per_link"],
    }


def request_path(network, source, target):
    if source == target or source not in network or target not in network:
        raise ValueError(f"Invalid request {source}-{target}. Choose two different ring nodes.")
    return nx.shortest_path(network, source, target, weight="length_km")


def allocate_wavelengths(network, requests=DEFAULT_REQUESTS):
    """Reserve the first common free wavelength along each request's entire shortest path."""
    count = network.graph["wavelengths_per_link"]
    bandwidth = network.graph["bandwidth_ghz_per_wavelength"]
    links = {frozenset((u, v)): {"nodes": [u, v], **data, "wavelength_slots": [None] * count}
             for u, v, data in network.edges(data=True)}
    assignments = []
    for index, (source, target) in enumerate(requests, start=1):
        path = request_path(network, source, target)
        route = route_metrics(network, path)
        slots = [links[frozenset(edge)]["wavelength_slots"] for edge in pairwise(path)]
        # 先检查全路径，再一次性预留。正反方向共用同一组波长。
        slot = next((i for i in range(count) if all(row[i] is None for row in slots)), None)
        request_id = f"R{index}"
        admitted = slot is not None
        if admitted:
            for row in slots:
                row[slot] = request_id
        assignments.append({
            "id": request_id, "source": source, "target": target, **route,
            "status": "accepted" if admitted else "blocked",
            "wavelength": slot + 1 if admitted else None,
            "allocated_bandwidth_ghz": bandwidth if admitted else 0.0,
            "allocated_capacity_gbps": route["wavelength_capacity_gbps"] if admitted else 0.0,
            "blocking_reason": None if admitted else "No common free wavelength on the shortest path",
        })

    for link in links.values():
        used = count - link["wavelength_slots"].count(None)
        link["occupied_wavelengths"] = used
        link["utilization"] = used / count
        link["used_bandwidth_ghz"] = used * bandwidth
        link["free_bandwidth_ghz"] = (count - used) * bandwidth
    accepted = sum(request["wavelength"] is not None for request in assignments)
    return {"requests": assignments, "accepted_count": accepted, "blocked_count": len(assignments) - accepted,
            "links": list(links.values())}


def allocate_equal_capacity(network, pairs=None):
    """Maximize a common rate C. A path of efficiency eta uses C/eta GHz on every hop."""
    if pairs is None:
        pairs = combinations(network.nodes, 2)
    routes = {}
    links = {frozenset((u, v)): {"nodes": [u, v], **data, "pairs": []}
             for u, v, data in network.edges(data=True)}
    for source, target in pairs:
        path = request_path(network, source, target)
        name = "-".join(sorted((source, target)))
        if name in routes:
            raise ValueError(f"Duplicate undirected node pair {name}.")
        routes[name] = route_metrics(network, path)
        for u, v in pairwise(path):
            links[frozenset((u, v))]["pairs"].append(name)
    if not routes:
        raise ValueError("Choose at least one node pair.")

    # 每条链路需要的带宽为 sum(C / eta_i)。十个节点对同时走最短路时，每条边上
    # 恰好有一条单跳业务和两条双跳业务，即笔记中的 B1 + 2*B2 = B_total。
    limits = []
    for link in links.values():
        names = link["pairs"]
        if names:
            ghz_per_gbps = math.fsum(1 / routes[n]["spectral_efficiency_bps_hz"] for n in names)
            limits.append(link["bandwidth_ghz"] / ghz_per_gbps)
    common_rate = min(limits)
    for route in routes.values():
        route["allocated_bandwidth_ghz"] = common_rate / route["spectral_efficiency_bps_hz"]
        route["allocated_capacity_gbps"] = common_rate

    for link in links.values():
        used = math.fsum(routes[name]["allocated_bandwidth_ghz"] for name in link["pairs"])
        link["used_bandwidth_ghz"] = used
        link["free_bandwidth_ghz"] = max(0.0, link["bandwidth_ghz"] - used)
    return {"common_capacity_gbps": common_rate, "routes": routes, "links": list(links.values())}


def plot_shortest_route(network, shortest_route):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 6))
    positions = nx.circular_layout(network)
    nx.draw_networkx(network, pos=positions, ax=axis, node_size=1300,
                     node_color="#dbeafe", edge_color="#94a3b8", width=2, font_size=13)
    labels = {(a, b): f"{data['length_km']:.0f} km"
              for a, b, data in network.edges(data=True)}
    nx.draw_networkx_edges(network, positions, ax=axis,
                           edgelist=list(zip(shortest_route[:-1], shortest_route[1:])),
                           edge_color="#ea580c", width=4)
    label_artists = nx.draw_networkx_edge_labels(
        network, positions, edge_labels=labels, ax=axis)
    for label in label_artists.values():
        label.set_zorder(5)
    axis.set_title(f"Five-node ring: shortest {shortest_route[0]}-to-{shortest_route[-1]} route highlighted")
    axis.set_axis_off()
    axis.margins(0.18)
    figure.tight_layout()
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=NODES, default=SOURCE, help="Start of the route comparison.")
    parser.add_argument("--target", choices=NODES, default=DESTINATION, help="End of the route comparison.")
    parser.add_argument("--snr-400-db", type=float, help="Override the reference model's 400 km SNR in dB.")
    parser.add_argument("--wavelengths", type=int, default=DEFAULT_WAVELENGTHS,
                        help="Wavelengths available on each undirected link (default %(default)s).")
    parser.add_argument("--bandwidth-ghz", type=float,
                        help=f"Total usable GHz per link (default {DEFAULT_BANDWIDTH_GHZ:g} GHz per wavelength).")
    parser.add_argument("--requests", "--pairs", nargs="+", metavar="A-C",
                        help="Requests in arrival order, including repeats. Defaults to one request "
                             "from each node to every other node.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-show", action="store_true", help="Save the figures without opening windows.")
    args = parser.parse_args()
    if args.source == args.target:
        parser.error("--source and --target must differ.")
    requests = DEFAULT_REQUESTS
    if args.requests:
        requests = [name.upper().split("-") for name in args.requests]
        if any(len(pair) != 2 for pair in requests):
            parser.error("Write each request as A-C.")
    try:
        network = build_ring(args.snr_400_db, args.bandwidth_ghz, wavelengths=args.wavelengths)
        plan = allocate_wavelengths(network, requests)
    except ValueError as error:
        parser.error(str(error))

    reference = network.graph
    print(f"Five-node ring | {network.number_of_edges()} links | {SPAN_LENGTH_KM:g} km per span")
    print(f"400 km SNR = {reference['snr_400_db']:.2f} dB ({reference['snr_source']})")
    if reference["launch_power_dbm_per_channel"] is not None:
        print(f"Reference launch power = {reference['launch_power_dbm_per_channel']:.3f} dBm/channel")
    for u, v, data in network.edges(data=True):
        print(f"{u}-{v}: {data['length_km']:g} km, {data['spans']} spans, "
              f"{data['snr_db']:.2f} dB, full-band capacity {data['capacity_gbps']:.2f} Gbit/s")

    candidates = [route_metrics(network, path) for path in nx.shortest_simple_paths(
        network, args.source, args.target, weight="length_km")]
    print(f"\nRoutes from {args.source} to {args.target}, ordered by distance")
    for route in candidates:
        print(f"  {' -> '.join(route['path'])}: {route['length_km']:g} km, {route['spans']} spans, "
              f"SNR {route['snr_db']:.2f} dB, per-wavelength capacity {route['wavelength_capacity_gbps']:.2f} Gbit/s")
    shortest_route = candidates[0]["path"]
    print(f"Shortest route: {' -> '.join(shortest_route)}")
    print(f"\n{args.wavelengths} wavelengths per link, {reference['bandwidth_ghz_per_wavelength']:g} GHz each")
    for request in plan["requests"]:
        path = " -> ".join(request["path"])
        if request["wavelength"] is None:
            print(f"  {request['id']}  {path}  BLOCKED ({request['blocking_reason']})")
        else:
            print(f"  {request['id']}  {path}  lambda {request['wavelength']}, SNR {request['snr_db']:.2f} dB, "
                  f"lightpath capacity {request['allocated_capacity_gbps']:.2f} Gbit/s")
    print(f"Accepted {plan['accepted_count']}, blocked {plan['blocked_count']}")
    for link in plan["links"]:
        print(f"  {'-'.join(link['nodes'])}: {link['occupied_wavelengths']}/{args.wavelengths} wavelengths "
              f"({link['utilization']:.0%})")

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(exist_ok=True, parents=True)
    figures = {
        "ring_topology.png": plot_shortest_route(network, shortest_route),
        "ring_route_occupancy.png": plot_route_occupancy(network, plan),
    }
    for name, figure in figures.items():
        figure.savefig(args.output_dir / name, dpi=180)
    report = {
        "model": {
            **reference,
            "routing": "Fixed shortest-distance path for each request in arrival order",
            "snr_accumulation": "Independent link noise at equal signal power, inverse linear SNRs add",
            "wavelength_snr": "Each assignable wavelength uses the same reference SNR at fixed spectral efficiency",
            "capacity": "Ideal dual-polarization Shannon rate, 2 * B_GHz * log2(1 + SNR_linear) in Gbit/s",
            "allocation": "First-fit wavelength common to all links on a request's path. Each accepted request "
                          "uses one whole wavelength. Both directions share the same physical link slots.",
            "slot_key": "wavelength_slots is ordered from lambda 1, with the request ID or null for a free slot",
        },
        "route_comparison": candidates, **plan,
    }
    (args.output_dir / "ring_capacity.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"\nSaved ring_topology.png, ring_route_occupancy.png and ring_capacity.json in {args.output_dir}")
    if args.no_show:
        for figure in figures.values():
            plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
