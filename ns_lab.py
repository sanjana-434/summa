import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import powerlaw
from scipy.stats import norm, shapiro, kstest
import numpy as np


def load_graph(path):
    """
    Load a graph from txt/csv (edge list) or gml.
    """
    if path.endswith(".gml"):
        return nx.read_gml(path)
    else:
        # auto-detect delimiter
        try:
            df = pd.read_csv(path, comment="#", header=None, delim_whitespace=True)
        except Exception:
            df = pd.read_csv(path, comment="#", header=None)
        edges = df.iloc[:, :2].values.tolist()
        return nx.from_edgelist(edges)

def check_power_law(degree_sequence):
    """
    Fit power law to the degree distribution and return results.
    """
    fit = powerlaw.Fit(degree_sequence, discrete=True, verbose=False)

    alpha = fit.power_law.alpha   # exponent
    xmin = fit.power_law.xmin     # minimum value from which power law holds
    R, p = fit.distribution_compare('power_law', 'exponential')

    print("\n--- Power Law Fitting ---")
    print(f"Exponent (alpha): {alpha:.4f}")
    print(f"x_min: {xmin}")
    print(f"Loglikelihood ratio (R, powerlaw vs exponential): {R:.4f}")
    print(f"p-value: {p:.4f}")

    # Plot
    fig = fit.plot_ccdf(color='b', label='Empirical CCDF')
    fit.power_law.plot_ccdf(ax=fig, color='r', linestyle='--', label='Power Law fit')
    plt.legend()
    plt.xlabel("Degree (k)")
    plt.ylabel("P(X ≥ k)")
    plt.title("Power Law Fit of Degree Distribution")
    plt.show()

    return alpha, xmin, (R, p)

def check_normal_distribution(degree_sequence):
    """
    Test if the degree distribution follows a normal distribution.
    """
    degrees = np.array(degree_sequence)
    
    # Fit normal distribution
    mu, std = norm.fit(degrees)

    print("\n--- Normal Distribution Fitting ---")
    print(f"Mean (mu): {mu:.4f}")
    print(f"Std dev (sigma): {std:.4f}")

    # Shapiro-Wilk test (small-medium sample)
    stat, p_shapiro = shapiro(degrees)
    print(f"Shapiro-Wilk test: stat={stat:.4f}, p-value={p_shapiro:.4f}")
    if p_shapiro > 0.05:
        print("Degrees *likely* follow a normal distribution (fail to reject H0)")
    else:
        print("Degrees *do NOT* follow a normal distribution (reject H0)")

    # Kolmogorov-Smirnov test against fitted normal
    stat, p_ks = kstest(degrees, 'norm', args=(mu, std))
    print(f"KS test: stat={stat:.4f}, p-value={p_ks:.4f}")
    if p_ks > 0.05:
        print("KS test: Degrees *likely* follow a normal distribution")
    else:
        print("KS test: Degrees *do NOT* follow a normal distribution")

    # Plot histogram with fitted normal curve
    plt.figure()
    count, bins, ignored = plt.hist(degrees, bins=20, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    plt.plot(bins, norm.pdf(bins, mu, std), 'r--', linewidth=2, label='Fitted Normal')
    plt.xlabel("Degree (k)")
    plt.ylabel("Probability Density")
    plt.title("Degree Distribution with Normal Fit")
    plt.legend()
    plt.show()


def generate_graph(model="er", n=100, p=0.05, k=4, beta=0.1, m=2, seed=42):
    """
    Generate a graph (ER, WS, BA).x
    """
    if model == "er":   # Erdős–Rényi random graph
        return nx.erdos_renyi_graph(n, p, seed=seed)
    elif model == "ws": # Watts–Strogatz small-world
        return nx.watts_strogatz_graph(n, k, beta, seed=seed)
    elif model == "ba": # Barabási–Albert scale-free
        return nx.barabasi_albert_graph(n, m, seed=seed)
    else:
        raise ValueError("Choose model from: er, ws, ba")

def check_six_degrees(avg_path_len):
    if avg_path_len <= 6:
        print(f"\nThis network obeys the 6 degrees of separation (avg path length = {avg_path_len:.2f})")
    else:
        print(f"\nThis network does NOT obey the 6 degrees of separation (avg path length = {avg_path_len:.2f})")

def phase_transition_er(n=500, p_values=None, seed=42):
    if p_values is None:
        p_values = [i/100 for i in range(1, 51)]  # 0.01 to 0.5

    giant_component_fraction = []

    for p in p_values:
        G = nx.erdos_renyi_graph(n, p, seed=seed)
        largest_cc = max(nx.connected_components(G), key=len)
        giant_component_fraction.append(len(largest_cc)/n)

    plt.figure()
    plt.plot(p_values, giant_component_fraction, marker='o')
    plt.xlabel("Edge Probability (p)")
    plt.ylabel("Fraction of Nodes in Largest Component")
    plt.title("Phase Transition in ER Random Graph")
    plt.grid(True)
    plt.show()

def phase_transition_er_by_k(n=500, k_values=None, seed=42):
    """
    Phase transition diagram for ER graph using average degree (k) on x-axis.
    """
    if k_values is None:
        k_values = list(range(1, 21))  # average degree from 1 to 20

    giant_component_fraction = []

    for k in k_values:
        p = k / (n - 1)  # Convert average degree to edge probability
        G = nx.erdos_renyi_graph(n, p, seed=seed)
        largest_cc = max(nx.connected_components(G), key=len)
        giant_component_fraction.append(len(largest_cc) / n)

    plt.figure()
    plt.plot(k_values, giant_component_fraction, marker='o')
    plt.xlabel("Average Degree (k)")
    plt.ylabel("Fraction of Nodes in Largest Component")
    plt.title("Phase Transition in ER Graph vs Average Degree")
    plt.grid(True)
    plt.show()


def graph_metrics(G):
    """
    Compute required metrics: degree distribution, clustering, avg path length, thresholds.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / n

    # Degree distribution
    degree_counts = pd.Series(degrees).value_counts(normalize=True).sort_index()

    # Clustering
    avg_clustering = nx.average_clustering(G)
    transitivity = nx.transitivity(G)

    # Largest Connected Component for path length
    if nx.is_connected(G):
        LCC = G
    else:
        LCC = G.subgraph(max(nx.connected_components(G), key=len))
    avg_path_len = nx.average_shortest_path_length(LCC)

    # Critical thresholds
    k1 = sum(degrees) / n
    k2 = sum(d**2 for d in degrees) / n
    pc = k1 / (k2 - k1) if k2 > k1 else float("nan")
    tau = k1 / k2 if k2 > 0 else float("nan")

    return {
        "Nodes": n,
        "Edges": m,
        "Average degree": avg_degree,
        "Average clustering": avg_clustering,
        "Transitivity": transitivity,
        "Avg path length (LCC)": avg_path_len,
        "Percolation threshold": pc,
        "SIS epidemic threshold": tau,
        "Degree distribution": degree_counts
    }

def plot_degree_distribution(degree_counts, title="Degree Distribution"):
    plt.figure()
    degree_counts.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.xlabel("Degree (k)")
    plt.ylabel("P(k)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_graph(G, title="Graph Visualization"):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)  # spring layout for nice spacing
    nx.draw(
        G, pos,
        with_labels=True,
        node_color='skyblue',
        node_size=300,
        edge_color='gray',
        linewidths=0.5,
        font_size=8
    )
    plt.title(title)
    plt.show()

# ---------------- Example Usage ----------------
if __name__ == "__main__":
    # Option 1: Load from file
    # G = load_graph("graph.gml")

    # Option 2: Generate synthetic graph
    G = generate_graph(model="ba", n=500, m=3)

    # Visualize the graph
    plot_graph(G, title="Graph Visualization")

    results = graph_metrics(G)

    for k, v in results.items():
        if k != "Degree distribution":
            print(f"{k}: {v}")

    # Save degree distribution to CSV
    results["Degree distribution"].to_csv("degree_distribution.csv")

    # Plot degree distribution
    plot_degree_distribution(results["Degree distribution"])

    # Check power law
    degrees = [d for _, d in G.degree()]
    # Power-law check
    check_power_law(degrees)
    # Normal distribution check
    check_normal_distribution(degrees)

    # Check six degrees of separation
    avg_path_len = results["Avg path length (LCC)"]
    check_six_degrees(avg_path_len)

    # Phase transition diagram for ER graph
    phase_transition_er(n=500)
    phase_transition_er_by_k(n=500, k_values=range(1, 21))

