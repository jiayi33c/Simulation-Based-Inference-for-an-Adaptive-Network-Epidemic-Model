"""
Adaptive-network SIR epidemic simulator.

This module simulates an SIR epidemic on a network that evolves over time.
"""

import numpy as np


def simulate(beta, gamma, rho, N=200, p_edge=0.05, n_infected0=5, T=200, rng=None):
    """Run one replicate of the adaptive-network SIR model."""
    if rng is None:
        rng = np.random.default_rng()

    neighbors = [set() for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < p_edge:
                neighbors[i].add(j)
                neighbors[j].add(i)

    state = np.zeros(N, dtype=np.int8)
    initial_infected = rng.choice(N, size=n_infected0, replace=False)
    state[initial_infected] = 1

    infected_fraction = np.zeros(T + 1)
    rewire_counts = np.zeros(T + 1, dtype=np.int64)
    infected_fraction[0] = np.sum(state == 1) / N

    for t in range(1, T + 1):
        # Once no infected nodes remain, the epidemic and rewiring dynamics stop.
        if not np.any(state == 1):
            infected_fraction[t:] = 0.0
            rewire_counts[t:] = 0
            break

        new_infections = set()
        infected_nodes = np.where(state == 1)[0]

        for i in infected_nodes:
            for j in neighbors[i]:
                if state[j] == 0 and rng.random() < beta:
                    new_infections.add(j)

        for j in new_infections:
            state[j] = 1

        infected_nodes = np.where(state == 1)[0]
        for i in infected_nodes:
            if rng.random() < gamma:
                state[i] = 2

        rewire_count = 0
        if rho > 0.0 and np.any(state == 1):
            si_edges = []
            for i in range(N):
                if state[i] == 0:
                    for j in neighbors[i]:
                        if state[j] == 1:
                            si_edges.append((i, j))

            for s_node, i_node in si_edges:
                if rng.random() < rho:
                    if i_node not in neighbors[s_node]:
                        continue

                    neighbors[s_node].discard(i_node)
                    neighbors[i_node].discard(s_node)

                    candidates = []
                    for k in range(N):
                        if k != s_node and k not in neighbors[s_node]:
                            candidates.append(k)

                    if candidates:
                        new_partner = rng.choice(candidates)
                        neighbors[s_node].add(new_partner)
                        neighbors[new_partner].add(s_node)
                        rewire_count += 1

        infected_fraction[t] = np.sum(state == 1) / N
        rewire_counts[t] = rewire_count

    degree_histogram = np.zeros(31, dtype=np.int64)
    for i in range(N):
        deg = min(len(neighbors[i]), 30)
        degree_histogram[deg] += 1

    return infected_fraction, rewire_counts, degree_histogram
