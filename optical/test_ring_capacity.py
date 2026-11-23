"""Regression checks for the ring's optical capacity and wavelength reservations."""

import math
import unittest

if __package__:
    from .step01_ring_topology import allocate_equal_capacity, allocate_wavelengths, build_ring, route_metrics
else:
    from step01_ring_topology import allocate_equal_capacity, allocate_wavelengths, build_ring, route_metrics


class RingCapacityTests(unittest.TestCase):
    def setUp(self):
        self.network = build_ring(snr_400_db=20, bandwidth_ghz=32)

    def test_link_and_route_snr(self):
        link = self.network["A"]["B"]
        self.assertEqual(link["snr_linear"], 100)
        self.assertAlmostEqual(link["capacity_gbps"], 2 * 32 * math.log2(101))
        for path in (["A", "B", "C"], ["A", "E", "D", "C"]):
            hops = len(path) - 1
            route = route_metrics(self.network, path)
            self.assertEqual(route["length_km"], hops * 400)
            self.assertEqual(route["spans"], hops * 5)
            self.assertAlmostEqual(route["snr_linear"], 100 / hops)
            self.assertAlmostEqual(route["snr_db"], 20 - 10 * math.log10(hops))
            self.assertAlmostEqual(route["full_band_capacity_gbps"], 64 * math.log2(1 + 100 / hops))

    def test_all_pairs_match_the_bandwidth_equations(self):
        plan = allocate_equal_capacity(self.network)
        eta1, eta2 = 2 * math.log2(101), 2 * math.log2(51)
        expected_rate = 32 / (1 / eta1 + 2 / eta2)
        self.assertEqual(len(plan["routes"]), 10)
        self.assertAlmostEqual(plan["common_capacity_gbps"], expected_rate)
        for link in plan["links"]:
            routes = [plan["routes"][name] for name in link["pairs"]]
            self.assertEqual(sorted(r["length_km"] for r in routes), [400, 800, 800])
            self.assertAlmostEqual(sum(r["allocated_bandwidth_ghz"] for r in routes), 32)
            self.assertAlmostEqual(link["used_bandwidth_ghz"], 32)
            self.assertAlmostEqual(link["free_bandwidth_ghz"], 0)
            for route in routes:
                eta = eta1 if route["length_km"] == 400 else eta2
                self.assertAlmostEqual(route["allocated_bandwidth_ghz"] * eta, expected_rate)
                self.assertAlmostEqual(route["allocated_capacity_gbps"], expected_rate)

    def test_subset_counts_reverse_traversals_and_free_bandwidth(self):
        plan = allocate_equal_capacity(self.network, [("B", "A"), ("A", "C"), ("D", "E")])
        links = {frozenset(link["nodes"]): link for link in plan["links"]}
        shared = links[frozenset(("A", "B"))]
        self.assertEqual(set(shared["pairs"]), {"A-B", "A-C"})
        self.assertAlmostEqual(shared["used_bandwidth_ghz"], 32)
        self.assertEqual(plan["routes"]["A-C"]["path"], ["A", "B", "C"])
        bc = links[frozenset(("B", "C"))]
        self.assertAlmostEqual(bc["used_bandwidth_ghz"], plan["routes"]["A-C"]["allocated_bandwidth_ghz"])
        self.assertGreater(bc["free_bandwidth_ghz"], 0)
        self.assertEqual(links[frozenset(("C", "D"))]["free_bandwidth_ghz"], 32)
        solo = allocate_equal_capacity(self.network, [("A", "C")])
        self.assertAlmostEqual(solo["routes"]["A-C"]["allocated_bandwidth_ghz"], 32)
        self.assertAlmostEqual(solo["common_capacity_gbps"], 64 * math.log2(51))

    def test_reject_invalid_or_repeated_pairs(self):
        for pairs in ([], [("A", "A")], [("A", "F")], [("A", "C"), ("C", "A")]):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                allocate_equal_capacity(self.network, pairs)


class WavelengthAssignmentTests(unittest.TestCase):
    def test_default_traffic_has_one_request_per_ordered_pair(self):
        network = build_ring(snr_400_db=20)
        plan = allocate_wavelengths(network)
        self.assertCountEqual(
            [(r["source"], r["target"]) for r in plan["requests"]],
            [(source, target) for source in network for target in network if source != target],
        )

    def test_example_requests_keep_their_wave_on_every_link(self):
        requests = [("A", "B"), ("A", "C"), ("E", "B")]
        plan = allocate_wavelengths(build_ring(snr_400_db=20), requests)
        self.assertEqual([r["path"] for r in plan["requests"]],
                         [["A", "B"], ["A", "B", "C"], ["E", "A", "B"]])
        self.assertEqual([r["wavelength"] for r in plan["requests"]], [1, 2, 3])
        slots = {tuple(link["nodes"]): link["wavelength_slots"] for link in plan["links"]}
        self.assertEqual(slots[("A", "B")], ["R1", "R2", "R3", None])
        self.assertEqual(slots[("B", "C")], [None, "R2", None, None])
        self.assertEqual(slots[("A", "E")], [None, None, "R3", None])
        self.assertEqual(slots[("C", "D")], [None] * 4)
        self.assertEqual(slots[("D", "E")], [None] * 4)
        for r in plan["requests"]:
            self.assertAlmostEqual(r["allocated_capacity_gbps"], 64 * math.log2(1 + r["snr_linear"]))
        for link in plan["links"]:
            used = sum(slot is not None for slot in link["wavelength_slots"])
            self.assertEqual(link["occupied_wavelengths"], used)
            self.assertEqual(link["utilization"], used / 4)
            self.assertEqual(link["used_bandwidth_ghz"] + link["free_bandwidth_ghz"], 128)

    def test_reverse_and_repeated_requests_compete_but_disjoint_links_can_reuse(self):
        graph = build_ring(snr_400_db=20, bandwidth_ghz=40, wavelengths=2)
        plan = allocate_wavelengths(graph, [("A", "B"), ("C", "D"), ("B", "A"), ("A", "B")])
        self.assertEqual([r["wavelength"] for r in plan["requests"]], [1, 1, 2, None])
        self.assertEqual([r["id"] for r in plan["requests"]], ["R1", "R2", "R3", "R4"])
        self.assertEqual((plan["accepted_count"], plan["blocked_count"]), (3, 1))
        self.assertEqual(plan["requests"][2]["path"], ["B", "A"])
        self.assertEqual(plan["requests"][0]["allocated_bandwidth_ghz"], 20)
        self.assertEqual(plan["requests"][3]["allocated_capacity_gbps"], 0)
        fresh = allocate_wavelengths(graph, [("A", "B")])
        self.assertEqual(fresh["requests"][0]["wavelength"], 1)

    def test_disjoint_free_slots_block_a_path_without_partial_reservations(self):
        graph = build_ring(snr_400_db=20, wavelengths=2)
        requests = [("A", "B"), ("C", "D"), ("B", "D"), ("A", "C"), ("B", "C")]
        plan = allocate_wavelengths(graph, requests)
        # 第四条业务到达时，AB 仅有 λ2 空闲，BC 仅有 λ1 空闲。
        self.assertEqual([r["wavelength"] for r in plan["requests"]], [1, 1, 2, None, 1])
        blocked = plan["requests"][3]
        self.assertEqual(blocked["status"], "blocked")
        self.assertEqual(blocked["allocated_bandwidth_ghz"], 0)
        self.assertTrue(all("R4" not in link["wavelength_slots"] for link in plan["links"]))

    def test_wavelength_count_and_empty_batch(self):
        for count in (0, -1, 1.5, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                build_ring(wavelengths=count)
        empty = allocate_wavelengths(build_ring(snr_400_db=20, wavelengths=1), [])
        self.assertEqual(empty["requests"], [])
        self.assertTrue(all(link["wavelength_slots"] == [None] for link in empty["links"]))


if __name__ == "__main__":
    unittest.main()
