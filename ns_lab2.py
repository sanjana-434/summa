import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson
import warnings

# ====== STEP 1: Load Graph from File ======
def load_graph(file_path, file_type='edgelist'):
    """
    Load graph from file. Supports edgelist, csv, and gml formats.
    For CSV, assumes columns named 'source' and 'target' by default.
    """
    try:
        if file_type == 'edgelist':
            G = nx.read_edgelist(file_path)
        elif file_type == 'csv':
            df = pd.read_csv(file_path)
            # Check if required columns exist
            if 'source' not in df.columns or 'target' not in df.columns:
                print(f"Warning: Expected 'source' and 'target' columns. Found: {df.columns.tolist()}")
                # Try to use first two columns
                df.columns = ['source', 'target'] + list(df.columns[2:])
            G = nx.from_pandas_edgelist(df, source='source', target='target')
        elif file_type == 'gml':
            G = nx.read_gml(file_path)
        else:
            raise ValueError("Unsupported file type. Use 'edgelist', 'csv', or 'gml'")
        
        # Convert to undirected if needed (for most network analyses)
        if G.is_directed():
            print("Converting directed graph to undirected for analysis")
            G = G.to_undirected()
            
        return G
    except FileNotFoundError:
        print(f"File {file_path} not found. Creating a sample graph for demonstration.")
        return nx.karate_club_graph()  # Return sample graph if file not found

# ====== STEP 2: Compute Network Metrics ======
def compute_metrics(G):
    """Compute comprehensive network metrics."""
    N = G.number_of_nodes()
    E = G.number_of_edges()
    
    if N <= 1:
        return {"Error": "Graph must have more than 1 node"}, [], [], {}
    
    # Edge probability (for simple graphs)
    max_edges = N * (N - 1) / 2
    p = E / max_edges if max_edges > 0 else 0
    
    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / N if N > 0 else 0
    degree_values = list(degrees.values())

    # Degree distribution
    if degree_values:
        max_degree = max(degree_values)
        degree_hist = np.bincount(degree_values, minlength=max_degree + 1)
    else:
        degree_hist = np.array([])

    # Clustering Coefficient
    try:
        clustering = nx.clustering(G)
        avg_clustering = nx.average_clustering(G)
    except:
        clustering = {}
        avg_clustering = 0

    # Average Path Length and Diameter (only for connected graphs)
    if nx.is_connected(G):
        try:
            avg_path_length = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
        except:
            avg_path_length = "Could not compute"
            diameter = "Could not compute"
    else:
        # For disconnected graphs, analyze largest component
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        try:
            avg_path_length = f"Disconnected (Largest component: {nx.average_shortest_path_length(subgraph):.3f})"
            diameter = f"Disconnected (Largest component: {nx.diameter(subgraph)})"
        except:
            avg_path_length = "Disconnected"
            diameter = "Disconnected"

    # Critical Threshold: <k^2>/<k> (for epidemic models)
    if degree_values and avg_degree > 0:
        k_sq = np.mean([k**2 for k in degree_values])
        critical_threshold = k_sq / avg_degree
    else:
        critical_threshold = 0

    # Additional useful metrics
    density = nx.density(G)
    num_components = nx.number_connected_components(G)
    
    metrics = {
        "Nodes": N,
        "Edges": E,
        "Density": round(density, 4),
        "Edge Probability (p)": round(p, 4),
        "Average Degree": round(avg_degree, 4),
        "Average Clustering Coefficient": round(avg_clustering, 4),
        "Average Path Length": avg_path_length,
        "Diameter": diameter,
        "Connected Components": num_components,
        "Critical Threshold (<k²>/<k>)": round(critical_threshold, 4)
    }

    return metrics, degree_values, degree_hist, clustering

# ====== STEP 3: Plot Degree Distribution ======
def plot_degree_distribution(degree_values, title="Degree Distribution", fit_poisson=False):
    """Plot degree distribution with optional Poisson fit."""
    if not degree_values:
        print("No degree values to plot")
        return
        
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    counts, bins, patches = plt.hist(degree_values, bins=min(20, max(degree_values)), 
                                   color='skyblue', alpha=0.7, edgecolor='black', 
                                   label="Observed Distribution")
    
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(alpha=0.3)

    if fit_poisson and degree_values:
        # Fit Poisson distribution
        lambda_param = np.mean(degree_values)
        k_values = np.arange(0, max(degree_values) + 1)
        poisson_probs = poisson.pmf(k_values, mu=lambda_param)
        
        # Scale to match histogram
        poisson_expected = poisson_probs * len(degree_values)
        
        plt.plot(k_values, poisson_expected, 'ro-', label=f"Poisson Fit (λ={lambda_param:.2f})", 
                markersize=4, linewidth=2)
        
        # Add goodness of fit information
        observed_freq = np.bincount(degree_values)
        chi_sq = np.sum((observed_freq[:len(poisson_expected)] - poisson_expected[:len(observed_freq)])**2 / 
                       (poisson_expected[:len(observed_freq)] + 1e-8))
        plt.text(0.6, 0.9, f'χ² = {chi_sq:.2f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.legend()
    plt.tight_layout()
    plt.show()

# ====== STEP 4: Generate Comparison Graphs ======
def generate_er_graph(n, p):
    """Generate Erdős-Rényi random graph."""
    return nx.erdos_renyi_graph(n, p)

def generate_ws_graph(n, k, p=0.1):
    """Generate Watts-Strogatz small-world graph."""
    if k % 2 != 0:
        k += 1  # k must be even for ring lattice
    k = max(2, min(k, n-1))  # Ensure valid k
    return nx.watts_strogatz_graph(n, k, p)

def generate_ba_graph(n, m):
    """Generate Barabási-Albert preferential attachment graph."""
    m = max(1, min(m, n-1))  # Ensure valid m
    return nx.barabasi_albert_graph(n, m)

# ====== STEP 5: Visualize Graph ======
def visualize_graph(G, title="Graph Visualization", max_nodes=100):
    """Visualize graph with smart layout selection."""
    if G.number_of_nodes() > max_nodes:
        print(f"Graph too large ({G.number_of_nodes()} nodes). Showing subgraph of {max_nodes} nodes.")
        nodes = list(G.nodes())[:max_nodes]
        G = G.subgraph(nodes)
    
    plt.figure(figsize=(10, 8))
    
    # Choose layout based on graph size and structure
    if G.number_of_nodes() < 20:
        pos = nx.spring_layout(G, k=1, iterations=50)
    elif G.number_of_nodes() < 50:
        pos = nx.spring_layout(G, k=0.5, iterations=30)
    else:
        pos = nx.spring_layout(G, k=0.3, iterations=20)
    
    # Draw graph
    nx.draw(G, pos, node_size=max(10, 200/np.sqrt(G.number_of_nodes())), 
           node_color='lightblue', edge_color='gray', with_labels=G.number_of_nodes() <= 20,
           font_size=8, alpha=0.8)
    
    plt.title(f"{title}\nNodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ====== STEP 6: Critical Point Analysis ======
def analyze_critical_point(n=1000):
    """Analyze phase transition in ER graphs."""
    print("Analyzing critical point in ER graphs...")
    
    avg_degrees = np.linspace(0.1, 3, 30)
    largest_component_sizes = []
    
    for avg_k in avg_degrees:
        p = avg_k / (n - 1)
        G = nx.erdos_renyi_graph(n, p)
        
        if G.number_of_nodes() > 0:
            largest_component = max(nx.connected_components(G), key=len)
            largest_component_sizes.append(len(largest_component))
        else:
            largest_component_sizes.append(0)

    # Plot phase transition
    plt.figure(figsize=(10, 6))
    plt.plot(avg_degrees, np.array(largest_component_sizes) / n, 'b.-', 
            label="Giant Component Size", linewidth=2, markersize=6)
    plt.axvline(x=1, color='r', linestyle='--', linewidth=2, 
                label='Critical Point (⟨k⟩=1)')
    
    plt.xlabel("Average Degree ⟨k⟩", fontsize=12)
    plt.ylabel("Largest Component Size / N", fontsize=12)
    plt.title("Phase Transition in Erdős-Rényi Graph", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.xlim(0, 3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

# ====== STEP 7: Compare Graph Models ======
def compare_graph_models(original_G):
    """Compare original graph with ER, WS, and BA models."""
    n = original_G.number_of_nodes()
    e = original_G.number_of_edges()
    
    if n <= 1:
        print("Graph too small for comparison")
        return
    
    # Calculate parameters for model graphs
    p_er = (2 * e) / (n * (n - 1)) if n > 1 else 0
    avg_degree = 2 * e / n if n > 0 else 0
    k_ws = max(2, int(avg_degree))
    if k_ws % 2 != 0:
        k_ws += 1
    m_ba = max(1, int(avg_degree / 2))
    
    # Generate comparison graphs
    graphs = {
        "Original": original_G,
        "Erdős-Rényi": generate_er_graph(n, p_er),
        "Watts-Strogatz": generate_ws_graph(n, k_ws, p=0.1),
        "Barabási-Albert": generate_ba_graph(n, m_ba)
    }
    
    print("\n" + "="*60)
    print("GRAPH MODEL COMPARISON")
    print("="*60)
    
    # Compare metrics
    for name, G in graphs.items():
        metrics, degree_values, _, _ = compute_metrics(G)
        print(f"\n{name} Graph:")
        print("-" * 40)
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if isinstance(v, float):
                    print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")
            else:
                print(f"{k}: {v}")
    
    # Plot degree distributions for comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, G) in enumerate(graphs.items()):
        degree_values = [d for _, d in G.degree()]
        if degree_values:
            axes[i].hist(degree_values, bins=min(20, max(degree_values)), 
                        color='skyblue', alpha=0.7, edgecolor='black')
            axes[i].set_title(f"{name}\nMean degree: {np.mean(degree_values):.2f}")
            axes[i].set_xlabel("Degree")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ====== MAIN EXECUTION ======
def main():
    """Main execution function."""
    # Try to load graph, fall back to sample if file not found
    graph_file = "graph.gml"  # Change this to your file
    G = load_graph(graph_file, file_type='gml')
    
    print("="*60)
    print("NETWORK ANALYSIS RESULTS")
    print("="*60)
    
    # Compute and display metrics
    metrics, degree_values, degree_hist, clustering = compute_metrics(G)
    
    print("\nOriginal Graph Metrics:")
    print("-" * 30)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    
    # Analysis and visualization
    if G.number_of_nodes() > 0:
        print(f"\nAnalyzing graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges...")
        
        # Plot degree distribution
        plot_degree_distribution(degree_values, 
                               title="Degree Distribution of Original Graph", 
                               fit_poisson=True)
        
        # Visualize graph
        visualize_graph(G, title="Original Graph")
        
        # Compare with model graphs
        compare_graph_models(G)
        
        # Analyze critical point (for reasonably sized graphs)
        if G.number_of_nodes() >= 10:
            analyze_critical_point(min(1000, G.number_of_nodes()))
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()