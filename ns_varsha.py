# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 23:21:38 2025

@author: preth
"""

"""
Unified exam-ready script for Network Science lab (NetworkX).
Covers:
 - Graph I/O: GML, CSV edge list, edge list txt, adjlist
 - Generation: ER, BA, WS, Configuration (power-law)
 - Metrics: degree, in/out-degree, avg degree, avg k^2, critical threshold, clustering, avg path length, diameter
 - Degree distribution plotting + comparison with Normal & Uniform (matched to empirical stats)
 - Attack simulations (degree-based and clustering-based removals)
 - Sandpile (avalanche) model simulation
 - Exports: write_gml, write_edgelist, write_adjlist
 - Handles directed graphs (uses in/out-degree when directed)
 - Visualizations for each analysis
Notes: change parameters in the bottom block to run different tasks.
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, deque
import random
import os
import warnings

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -----------------------
# 1) FILE I/O & LOADING
# -----------------------
def load_graph_auto(path):
    """Load graph from path, auto-detect format by extension.
    Supports: .gml, .csv (edge list), .txt (edgelist), .edgelist, .adjlist
    If unknown, attempt to parse as edge list."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gml":
        G = nx.read_gml(path)
        print(f"Loaded GML: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
        return G
    elif ext in [".csv"]:
        df = pd.read_csv(path)
        # Try to guess the source and target columns
        if df.shape[1] >= 2:
            src, tgt = df.columns[0], df.columns[1]
            G = nx.from_pandas_edgelist(df, source=src, target=tgt, create_using=nx.DiGraph() if 'directed' in df.columns else nx.Graph())
            print(f"Loaded CSV edge list: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
            return G
        else:
            raise ValueError("CSV must have at least two columns (source, target).")
    elif ext in [".txt", ".edgelist"]:
        G = nx.read_edgelist(path, data=False)
        print(f"Loaded edge list: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
        return G
    elif ext == ".adjlist":
        G = nx.read_adjlist(path)
        print(f"Loaded adjlist: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
        return G
    else:
        # Try GML first, then fallback to edgelist
        try:
            G = nx.read_gml(path)
            print("Loaded as GML.")
            return G
        except Exception:
            try:
                G = nx.read_edgelist(path)
                print("Loaded as edgelist fallback.")
                return G
            except Exception as e:
                raise ValueError(f"Unsupported or unreadable file format: {e}")

def export_graph(G, name="exported_graph"):
    """Export graph in GML, edgelist, adjlist formats."""
    nx.write_gml(G, f"{name}.gml")
    nx.write_edgelist(G, f"{name}.edgelist")
    nx.write_adjlist(G, f"{name}.adjlist")
    print(f"Exported: {name}.gml, {name}.edgelist, {name}.adjlist")

# -----------------------
# 2) GRAPH GENERATORS
# -----------------------
def generate_er(n, p, seed=SEED):
    return nx.erdos_renyi_graph(n, p, seed=seed)

def generate_ba(n, m, seed=SEED):
    return nx.barabasi_albert_graph(n, m, seed=seed)

def generate_ws(n, k, p, seed=SEED):
    return nx.watts_strogatz_graph(n, k, p, seed=seed)

def generate_powerlaw_configuration(n, gamma, ensure_simple=True):
    # Use powerlaw_sequence to get floats, convert to integers (>=1), then make even sum
    seq = nx.utils.powerlaw_sequence(n, gamma)
    seq = [max(1, int(round(x))) for x in seq]
    if sum(seq) % 2 == 1:
        seq[np.random.randint(0, n)] += 1
    G_multi = nx.configuration_model(seq, create_using=None)
    if ensure_simple:
        G = nx.Graph(G_multi)  # removes parallel edges, keeps self-loops as edges (we remove them)
        G.remove_edges_from(nx.selfloop_edges(G))
        return G
    else:
        return G_multi

# -----------------------
# 3) METRICS & HELPERS
# -----------------------
def is_directed_graph(G):
    return G.is_directed()

def degree_stats(G):
    """Return degree list and summary stats depending on directedness."""
    if is_directed_graph(G):
        in_deg = [d for _, d in G.in_degree()]
        out_deg = [d for _, d in G.out_degree()]
        return {"in": in_deg, "out": out_deg}
    else:
        deg = [d for _, d in G.degree()]
        return {"deg": deg}

def average_degree(G):
    if is_directed_graph(G):
        # For directed graphs, average in-degree == average out-degree == m/n
        in_deg = [d for _, d in G.in_degree()]
        return np.mean(in_deg)
    else:
        deg = [d for _, d in G.degree()]
        return np.mean(deg)

def critical_threshold(G):
    """Compute <k^2> / <k>. For directed graphs, use total degree (in+out) per node."""
    if is_directed_graph(G):
        degs = [G.in_degree(n) + G.out_degree(n) for n in G.nodes()]
    else:
        degs = [d for _, d in G.degree()]
    k_avg = np.mean(degs) if len(degs) > 0 else 0
    k2_avg = np.mean(np.square(degs)) if len(degs) > 0 else 0
    thr = k2_avg / k_avg if k_avg > 0 else np.inf
    print(f"⟨k⟩ = {k_avg:.4f}, ⟨k²⟩ = {k2_avg:.4f}, Critical threshold ⟨k²⟩/⟨k⟩ = {thr:.4f}")
    return thr

def avg_clustering(G):
    # For directed graphs, networkx.clustering treats them as undirected by default
    avg = nx.average_clustering(G.to_undirected()) if is_directed_graph(G) else nx.average_clustering(G)
    print(f"Average clustering coefficient: {avg:.4f}")
    return avg

def avg_path_and_diameter(G):
    """Compute average shortest path length and diameter on largest connected/weakly connected component."""
    if is_directed_graph(G):
        # use largest weakly connected component
        if nx.is_weakly_connected(G):
            Gc = G
        else:
            comp = max(nx.weakly_connected_components(G), key=len)
            Gc = G.subgraph(comp).copy()
    else:
        if nx.is_connected(G):
            Gc = G
        else:
            comp = max(nx.connected_components(G), key=len)
            Gc = G.subgraph(comp).copy()

    if Gc.number_of_nodes() <= 1:
        print("Component too small to compute path stats.")
        return np.nan, np.nan

    try:
        if is_directed_graph(G):
            # shortest path length treating as undirected for distance measure
            Gc_undir = Gc.to_undirected()
            apl = nx.average_shortest_path_length(Gc_undir)
            diam = nx.diameter(Gc_undir)
        else:
            apl = nx.average_shortest_path_length(Gc)
            diam = nx.diameter(Gc)
        print(f"Average shortest path length (LCC): {apl:.4f}, Diameter (LCC): {diam}")
        return apl, diam
    except Exception as e:
        print("Error computing path/diameter:", e)
        return np.nan, np.nan

# -----------------------
# 4) DEGREE DISTRIBUTION & COMPARISONS
# -----------------------
def plot_degree_distribution(G, bins=30, title="Degree Distribution", show_cumulative=False):
    plt.figure(figsize=(8,5))
    if is_directed_graph(G):
        degs = [G.in_degree(n) + G.out_degree(n) for n in G.nodes()]
    else:
        degs = [d for _, d in G.degree()]

    counts, edges, patches = plt.hist(degs, bins=bins, density=True, alpha=0.7, label="Empirical deg")
    plt.xlabel("Degree")
    plt.ylabel("Probability density")
    plt.title(title)
    plt.legend()
    plt.show()

    if show_cumulative:
        # Complementary cumulative distribution
        degs_sorted = np.sort(degs)
        unique, counts = np.unique(degs_sorted, return_counts=True)
        probs = counts / counts.sum()
        ccdf = 1 - np.cumsum(probs) + probs  # P(K >= k)
        plt.figure(figsize=(6,4))
        plt.loglog(unique, ccdf, 'o-')
        plt.xlabel("Degree k")
        plt.ylabel("P(K ≥ k)")
        plt.title("Complementary Cumulative Degree Distribution (CCDF)")
        plt.show()

def compare_with_normal_uniform(G, bins=20):
    """Compare empirical degree distribution with fitted normal & uniform distributions (matched to empirical mean/std/range)."""
    if is_directed_graph(G):
        degs = np.array([G.in_degree(n) + G.out_degree(n) for n in G.nodes()])
    else:
        degs = np.array([d for _, d in G.degree()])

    mu, sigma = degs.mean(), degs.std()
    dmin, dmax = degs.min(), degs.max()
    n = len(degs)

    # Generate comparison samples with same sample size
    normal_sample = np.random.normal(loc=mu, scale=sigma if sigma>0 else 1, size=n)
    normal_sample = np.clip(normal_sample, a_min=0, a_max=None)
    uniform_sample = np.random.uniform(low=dmin, high=dmax, size=n)

    plt.figure(figsize=(8,6))
    plt.hist(degs, bins=bins, density=True, alpha=0.6, label="Empirical")
    plt.hist(normal_sample, bins=bins, density=True, alpha=0.5, label=f"Normal (μ={mu:.2f},σ={sigma:.2f})")
    plt.hist(uniform_sample, bins=bins, density=True, alpha=0.5, label=f"Uniform [{dmin},{dmax}]")
    plt.xlabel("Degree")
    plt.ylabel("Density")
    plt.title("Degree Distribution: Empirical vs Normal & Uniform")
    plt.legend()
    plt.show()

# -----------------------
# 5) ASSIGNMENT-SPECIFIC SIMULATIONS
# -----------------------
# 5.1 Attack simulations (Programming Assignment 3)
def simulate_attack(G, criterion='degree', fractions=np.linspace(0,0.5,11), directed=False):
    """Simulate removal of fraction f of nodes ranked by criterion (degree or clustering).
       Returns list of normalized giant component sizes for each fraction f.
    """
    G0 = G.copy()
    N = G0.number_of_nodes()
    sizes = []
    if criterion == 'degree':
        if directed:
            # use total degree for ranking
            nodes_sorted = sorted(G0.nodes(), key=lambda n: (G0.in_degree(n) + G0.out_degree(n)), reverse=True)
        else:
            nodes_sorted = sorted(G0.nodes(), key=lambda n: G0.degree(n), reverse=True)
    elif criterion == 'clustering':
        cl = nx.clustering(G0.to_undirected()) if directed else nx.clustering(G0)
        nodes_sorted = sorted(G0.nodes(), key=lambda n: cl.get(n, 0), reverse=True)
    else:
        raise ValueError("criterion must be 'degree' or 'clustering'")

    for f in fractions:
        k = int(np.round(f * N))
        Gtemp = G0.copy()
        remove_nodes = nodes_sorted[:k]
        Gtemp.remove_nodes_from(remove_nodes)
        if Gtemp.number_of_nodes() == 0:
            sizes.append(0.0)
        else:
            if is_directed_graph(Gtemp):
                comp = max(nx.weakly_connected_components(Gtemp), key=len) if nx.number_weakly_connected_components(Gtemp) > 0 else set()
            else:
                comp = max(nx.connected_components(Gtemp), key=len) if nx.number_connected_components(Gtemp) > 0 else set()
            sizes.append(len(comp) / N)
    return sizes

# 5.2 Sandpile simulation (Programming Assignment 4)
def simulate_sandpile(G, steps=2000):
    """Sandpile avalanche model: bucket capacity = node degree.
       Returns list of avalanche sizes (number of toppled nodes) per perturbation.
    """
    # Work with a simple graph structure for neighbors
    G_use = G.to_undirected() if is_directed_graph(G) else G
    buckets = {n: max(1, G_use.degree(n)) for n in G_use.nodes()}  # bucket size at least 1
    grains = {n: 0 for n in G_use.nodes()}
    avalanches = []

    nodes_list = list(G_use.nodes())
    for t in range(steps):
        i = random.choice(nodes_list)
        grains[i] += 1
        unstable = deque([i]) if grains[i] >= buckets[i] else deque()
        toppled = set()
        while unstable:
            u = unstable.popleft()
            if grains[u] >= buckets[u]:
                toppled.add(u)
                num = grains[u]
                grains[u] = 0
                neighbors = list(G_use.neighbors(u))
                if len(neighbors) == 0:
                    continue
                # distribute equally (integer division) and random remainder distribution
                per = num // len(neighbors)
                rem = num % len(neighbors)
                for nb in neighbors:
                    grains[nb] += per
                # distribute remainder to random neighbors
                for r in range(rem):
                    nb = random.choice(neighbors)
                    grains[nb] += 1
                for nb in neighbors:
                    if grains[nb] >= buckets[nb] and nb not in toppled:
                        unstable.append(nb)
        if len(toppled) > 0:
            avalanches.append(len(toppled))
    return avalanches

# -----------------------
# 6) VISUALIZATION HELPERS
# -----------------------
def visualize_graph(G, title="Graph", node_size=50, with_labels=False, layout_seed=SEED):
    plt.figure(figsize=(6,6))
    if G.number_of_nodes() > 500:
        node_size = 10
    try:
        pos = nx.spring_layout(G, seed=layout_seed)
    except:
        pos = nx.random_layout(G, seed=layout_seed)
    nx.draw(G, pos, node_size=node_size, with_labels=with_labels)
    plt.title(title)
    plt.show()

def plot_avalanche_sizes(avalanche_list, title="Avalanche size distribution"):
    if len(avalanche_list) == 0:
        print("No avalanches recorded.")
        return
    plt.figure(figsize=(7,5))
    plt.hist(avalanche_list, bins=30, density=True, log=True)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Avalanche size")
    plt.ylabel("Probability (log scale)")
    plt.title(title)
    plt.show()

def plot_attack_results(fractions, sizes_degree, sizes_clustering, title="Attack simulation"):
    plt.figure(figsize=(7,5))
    plt.plot(fractions, sizes_degree, 'o-', label='Remove by degree')
    plt.plot(fractions, sizes_clustering, 's-', label='Remove by clustering')
    plt.xlabel("Fraction nodes removed")
    plt.ylabel("Normalized giant component size")
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()

# -----------------------
# 7) UTILITY: summary printed report
# -----------------------
def full_report(G, name="Graph"):
    print("\n========== SUMMARY REPORT ==========")
    print(f"Name: {name}")
    print("Directed:", is_directed_graph(G))
    print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())
    # Degrees
    if is_directed_graph(G):
        indeg = [d for _, d in G.in_degree()]
        outdeg = [d for _, d in G.out_degree()]
        print(f"Average in-degree: {np.mean(indeg):.4f}, Average out-degree: {np.mean(outdeg):.4f}")
    else:
        degs = [d for _, d in G.degree()]
        print(f"Average degree: {np.mean(degs):.4f}")
    # Clustering
    avg_clustering(G)
    # Threshold
    critical_threshold(G)
    # path & diameter
    avg_path_and_diameter(G)
    # Visualize small
    if G.number_of_nodes() <= 1000:
        visualize_graph(G, title=f"{name} visualization")
    # Plot degree distribution and comparisons
    plot_degree_distribution(G, bins=30, title=f"{name} Degree Distribution", show_cumulative=True)
    compare_with_normal_uniform(G, bins=30)
    print("====================================\n")

# -----------------------
# 8) MAIN: examples & quick experiments
# -----------------------
if __name__ == "_main_":
    # ---------- SETTINGS ----------
    # Choose "mode": "example", "from_file", "assignment1", "assignment2", "assignment3", "assignment4"
    mode = "example"  # change below as needed for different runs

    # Common params
    N_small = 500
    N_large = 10000

    if mode == "example":
        # Simple demo: load sample GML if present, otherwise generate ER
        sample_path = "D:\\PSG\\sem9\\NS\\graph.gml"
        if os.path.exists(sample_path):
            try:
                G = load_graph_auto(sample_path)
            except Exception as e:
                print("Failed to load sample GML:", e)
                G = generate_er(200, 0.03)
        else:
            G = generate_er(200, 0.03)

        full_report(G, name="Example Graph")

    elif mode == "from_file":
        # If they give a file in exam, set path here and run
        path = "network.gml"  # change to actual provided file path in exam
        G = load_graph_auto(path)
        full_report(G, name=os.path.basename(path))
        export_graph(G, name=f"exported_{os.path.splitext(os.path.basename(path))[0]}")

    elif mode == "assignment1":
        # Programming Assignment 1: ER graphs with different avg degrees
        N = 500
        avg_degrees = [0.8, 1, 8]
        for k in avg_degrees:
            p = k / (N - 1)
            G = generate_er(N, p)
            print(f"\nER N={N}, <k>={k} (p={p:.6f}): nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
            full_report(G, name=f"ER_k_{k}")
        # You can add red/blue/purple community code from your assignment as needed.

    elif mode == "assignment2":
        # Programming Assignment 2: power-law seq and BA snapshots
        # 1) Percent multi-links & self-loops as function of N for gamma values
        for gamma in [2.2, 3.0]:
            for N in [1000, 10000, 100000]:
                G = generate_powerlaw_configuration(N, gamma)
                # Rough estimate: self-loops were removed in generation; but if using pure config model,
                # create multi-graph and count self-loops/multi-edges if needed.
                print(f"Generated configuration-based graph N={N}, gamma={gamma}, nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
                # Show degree distribution
                plot_degree_distribution(G, title=f"Config model: N={N}, γ={gamma}", bins=50, show_cumulative=True)
        # 2) BA network snapshots and clustering vs N
        N_full = 10000
        m = 4
        G_ba = generate_ba(N_full, m)
        for ns in [100, 1000, 10000]:
            # approximate snapshot by building a BA of that size
            Gb = generate_ba(ns, m)
            plot_degree_distribution(Gb, title=f"BA snapshot N={ns}", show_cumulative=True)
        Ns = [100, 200, 500, 1000, 2000, 5000, 10000]
        clust_values = []
        for n in Ns:
            Gb = generate_ba(n, m)
            clust_values.append(nx.average_clustering(Gb))
        plt.figure(); plt.plot(Ns, clust_values, 'o-'); plt.xscale('log'); plt.xlabel('N'); plt.ylabel('Avg clustering'); plt.title('BA clustering vs N'); plt.show()

    elif mode == "assignment3":
        # Programming Assignment 3: attack simulations
        N = 10000
        print("Generating config model (power-law, γ=2.5) ...")
        degrees = nx.utils.powerlaw_sequence(N, 2.5)
        seq = [max(1, int(round(x))) for x in degrees]
        if sum(seq) % 2 == 1: seq[np.random.randint(0, N)] += 1
        G_config = nx.configuration_model(seq)
        G_config = nx.Graph(G_config)
        G_config.remove_edges_from(nx.selfloop_edges(G_config))
        fractions = np.linspace(0, 0.5, 11)
        sizes_deg = simulate_attack(G_config, 'degree', fractions)
        sizes_clu = simulate_attack(G_config, 'clustering', fractions)
        print("Config model attack results (degree vs clustering):", sizes_deg, sizes_clu)
        plot_attack_results(fractions, sizes_deg, sizes_clu, title="Config model attack")

        # Hierarchical (powerlaw_cluster_graph)
        print("Generating hierarchical model ...")
        G_hier = nx.powerlaw_cluster_graph(N, m=4, p=0.1)
        sizes_deg_h = simulate_attack(G_hier, 'degree', fractions)
        sizes_clu_h = simulate_attack(G_hier, 'clustering', fractions)
        plot_attack_results(fractions, sizes_deg_h, sizes_clu_h, title="Hierarchical model attack")

    elif mode == "assignment4":
        # Programming Assignment 4: Sandpile on ER vs Scale-free
        N = 1000  # smaller for speed in exam environment
        avg_k = 2
        p = avg_k / (N - 1)
        G_er = generate_er(N, p)
        aval_er = simulate_sandpile(G_er, steps=2000)
        print(f"ER mean avalanche size: {np.mean(aval_er) if len(aval_er)>0 else 0:.3f}")
        plot_avalanche_sizes(aval_er, title=f"ER avalanches N={N}, <k>={avg_k}")

        # Scale-free via configuration
        seq = generate_powerlaw_configuration(N, gamma=2.5, ensure_simple=True)
        G_sf = seq  # function returned a graph
        aval_sf = simulate_sandpile(G_sf, steps=2000)
        print(f"SF mean avalanche size: {np.mean(aval_sf) if len(aval_sf)>0 else 0:.3f}")
        plot_avalanche_sizes(aval_sf, title=f"SF avalanches N={N}, <k>={avg_k}")

    else:
        print("Unknown mode. Set mode to one of: example, from_file, assignment1, assignment2, assignment3, assignment4")

    # End of main