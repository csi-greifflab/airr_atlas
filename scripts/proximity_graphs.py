from pathlib import Path
import os
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from pqdm.processes import pqdm
import networkx as nx

output_dir = "sample_files/proximity_graphs"
root_dir = "/doctorai/niccoloc/Vicinity_results_100k_Density2"
ld_edge_file = (
    "sample_files/tz_heavy_chain_100k_edges.csv"
)
fasta_file = "sample_files/proximity_graphs/tz_heavy_chain_100k.fa"
deduplicated_file = "sample_files/proximity_graphs/tz_heavy_chains_airr_dedup_final.tsv"


def read_fasta(fasta_file):
    sequences = {}
    current_id = None
    current_seq = []
    with open(fasta_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                if current_id:
                    sequences[current_id] = "".join(current_seq)
                current_id = line.strip()[1:]
                current_seq = []
            else:
                current_seq.append(line.strip())
        if current_id:
            sequences[current_id] = "".join(current_seq)
    return sequences


def normalize_edges(edge_set):
    return set((min(a, b), max(a, b)) for a, b, *_ in edge_set)


def read_edge_file(file_path):
    edges = set()
    with open(file_path, "r") as f:
        weighted = False
        for i, line in enumerate(f):
            if i == 0:
                if len(line.split(",")) == 3:
                    weighted = True
                continue
            line = line.strip().split(",")
            if weighted:
                edges.add((line[0], line[1], int(line[2])))
            else:
                edges.add((line[0], line[1]))
    edges = normalize_edges(edges)
    return edges


def get_degrees(edges):
    degrees = {}
    for edge in edges:
        src, tgt = edge[:2]
        degrees[src] = degrees.get(src, 0) + 1
        degrees[tgt] = degrees.get(tgt, 0) + 1
    average_degree = sum(degrees.values()) / len(degrees)
    return (
        dict(sorted(degrees.items(), key=lambda item: item[1], reverse=True)),
        average_degree,
    )


def compare_graphs(set1, set2):
    only_in_set1 = set1 - set2
    only_in_set2 = set2 - set1
    in_both = set1 & set2
    len1 = len(set1)
    len2 = len(set2)
    len_both = len(in_both)
    len_union = len1 + len2 - len_both
    return {
        "precision": len_both / len1 if len1 else 0,
        "recall": len_both / len2 if len2 else 0,
        "jaccard": len_both / len_union if len_union else 0,
        "only_in_set1": len(only_in_set1),
        "only_in_set2": len(only_in_set2),
        "in_both": len_both,
    }


def compare_degrees(ld_degrees, graph_degrees, shared_only=True):
    if shared_only:
        nodes = set(ld_degrees) & set(graph_degrees)
    else:
        nodes = set(ld_degrees) | set(graph_degrees)
    x = [ld_degrees.get(n, 0) for n in nodes]
    y = [graph_degrees.get(n, 0) for n in nodes]
    return {
        "spearman": float(spearmanr(x, y).correlation),
        "pearson": float(pearsonr(x, y)[0]),
    }


def get_edge_labels(edges, labels):
    label_map = dict(zip(labels["sequence_id"], labels["binding_label"]))
    for node1, node2 in edges:
        yield (
            node1,
            node2,
            label_map.get(node1),
            label_map.get(node2),
        )


def get_label_counts(edges, labels):
    different_labels = set()
    hb_hb = set()
    lb_lb = set()
    for node1, node2, label1, label2 in set(get_edge_labels(edges, labels)):
        if label1 is None or label2 is None:
            continue
        if label1 != label2:
            different_labels.add((node1, node2))
        elif label1 == "hb":
            hb_hb.add((node1, node2))
        elif label1 == "lb":
            lb_lb.add((node1, node2))
    same_labels = hb_hb | lb_lb
    return {
        "different_labels": len(different_labels),
        "same_labels": len(same_labels),
        "hb_hb": len(hb_hb),
        "lb_lb": len(lb_lb),
        "same_labels_ratio": len(same_labels) / len(edges) if edges else 0,
        "total_edges": len(same_labels) + len(different_labels),
    }


def get_isolated_components(edges):
    G = nx.Graph(edges)
    return len(list(nx.connected_components(G)))


# Global LD graph and labels to be shared via args
ld_edges = read_edge_file(ld_edge_file)
ld_degrees, ld_average_degree = get_degrees(ld_edges)

df = pd.read_csv(deduplicated_file, sep="\t")
labels = df[["sequence_id", "binding_label"]]


def process_file(file):
    try:
        name = Path(file).stem
        if "Unpooled_esm2_t33_650M_UR50D_heavy_chain_layer_15" in file:
            return None  # Skip heavy chain attention matrices
        model = "ab2" if "antiberta" in name else "esm2"
        if "heavy" in name:
            chain_type = "heavy"
        elif "CDRH" in name:
            chain_type = "all_cdrh"
        elif "cdr3" in name.lower():
            chain_type = "cdr3"
        else:
            raise ValueError(f"Unknown chain type: {name}")
        if "attention" in name.lower():
            output_type = "attention_matrix"
            pooling = "average_layer"
        elif "unpooled" in name.lower():
            output_type = "embeddings"
            pooling = "unpooled"
        elif "pooled" in name.lower():
            output_type = "embeddings"
            pooling = "pooled"
        else:
            raise ValueError(f"Unknown output type: {name}")

        try:
            threshold = int(name.split("_")[2][-1])
            layer = int(name.split("_")[-1]) + 1
        except Exception:
            threshold = -1
            layer = -1

        edges = read_edge_file(file)
        if len(edges) == 0:
            print(f"Empty edge file: {file}")
            return {
                "threshold": threshold,
                "pooling": pooling,
                "model": model,
                "chain_type": chain_type,
                "output_type": output_type,
                "layer": layer,
            }
        degrees, average_degree = get_degrees(edges)
        deg_shared = compare_degrees(ld_degrees, degrees, shared_only=True)
        deg_all = compare_degrees(ld_degrees, degrees, shared_only=False)

        return {
            "threshold": threshold,
            "pooling": pooling,
            "model": model,
            "chain_type": chain_type,
            "output_type": output_type,
            "layer": layer,
            "isolated_components": get_isolated_components(edges),
            "average_degree": average_degree,
            "deg_shared_spearman": deg_shared["spearman"],
            "deg_shared_pearson": deg_shared["pearson"],
            "deg_all_spearman": deg_all["spearman"],
            "deg_all_pearson": deg_all["pearson"],
            **compare_graphs(ld_edges, edges),
            **get_label_counts(edges, labels),
        }
    except Exception as e:
        print(f"Failed on file {file}: {e}")
        return None


# Find edge files
edge_files = []
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.startswith("edges") and filename.endswith(".csv"):
            edge_files.append(os.path.join(dirpath, filename))

# Run in parallel with progress bar
results = pqdm(edge_files, process_file, n_jobs=98, argument_type="single")

# Filter out None
graphs_metrics = [r for r in results if r is not None]

# Export
df_metrics = pd.DataFrame(graphs_metrics)
df_metrics.to_csv(
    os.path.join(output_dir, "graph_comparison_metrics.csv"), index=False
)
