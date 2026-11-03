import argparse
import ast
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import networkx as nx


def load_decker_relations(project_root):
    graph_source = project_root / "data_preprocess" / "data_preprocess-graph.py"
    tree = ast.parse(graph_source.read_text(encoding="utf-8"))

    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "merged_relations"
            for target in statement.targets
        ):
            relations = ast.literal_eval(statement.value)
            if isinstance(relations, list):
                return relations

    raise RuntimeError(f"Could not find merged_relations in {graph_source}")


def load_decker_concept_capacity(project_root):
    model_source = project_root / "model" / "modeling_v3.py"
    tree = ast.parse(model_source.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "CustomizedEmbedding":
            capacity = ast.literal_eval(node.args[0])
            if isinstance(capacity, int):
                return capacity

    raise RuntimeError(f"Could not find the concept embedding size in {model_source}")


def install_lightweight_helper():
    # GreaseLM's ConceptNet module imports this helper, but the selected functions
    # do not use it. Avoid importing GreaseLM's general utils module and PyTorch.
    helper = types.ModuleType("preprocess_utils.utils")
    helper.__package__ = "preprocess_utils"
    helper.check_file = lambda path: Path(path).is_file()
    sys.modules[helper.__name__] = helper


def validate_outputs(
    vocabulary_path,
    graph_path,
    patterns_path,
    expected_relations,
    concept_capacity,
):
    concepts = vocabulary_path.read_text(encoding="utf-8").splitlines()
    if len(concepts) > concept_capacity:
        raise ValueError(
            f"Generated {len(concepts):,} concepts, exceeding DECKER's "
            f"{concept_capacity:,}-row concept embedding table."
        )
    if len(set(concepts)) != len(concepts):
        raise ValueError("The generated concept vocabulary contains duplicate entries.")

    graph = nx.read_gpickle(graph_path)
    if not isinstance(graph, nx.MultiDiGraph):
        raise TypeError(f"Expected a NetworkX MultiDiGraph; got {type(graph).__name__}.")

    for node in graph.nodes:
        if not isinstance(node, int) or not 0 <= node < len(concepts):
            raise ValueError(f"Graph contains an invalid concept ID: {node!r}")

    relation_limit = 2 * len(expected_relations)
    for _, _, attributes in graph.edges(data=True):
        relation_id = attributes.get("rel")
        if not isinstance(relation_id, int) or not 0 <= relation_id < relation_limit:
            raise ValueError(f"Graph contains an invalid relation ID: {relation_id!r}")

    with patterns_path.open(encoding="utf-8") as file:
        patterns = json.load(file)

    valid_pattern_keys = {concept.replace(" ", "_") for concept in concepts}
    unexpected_keys = set(patterns) - valid_pattern_keys
    if unexpected_keys:
        examples = sorted(unexpected_keys)[:5]
        raise ValueError(f"Matcher patterns contain unknown concepts: {examples}")

    return (
        len(concepts),
        graph.number_of_nodes(),
        graph.number_of_edges(),
        len(patterns),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate DECKER-compatible ConceptNet resources using GreaseLM."
    )
    parser.add_argument(
        "--greaselm-root",
        required=True,
        type=Path,
        help="Path to a local checkout of snap-stanford/GreaseLM.",
    )
    parser.add_argument(
        "--conceptnet-csv",
        type=Path,
        default=Path("data/cpnet/conceptnet-assertions-5.6.0.csv"),
        help="Path to the decompressed ConceptNet 5.6.0 CSV.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    greaselm_root = args.greaselm_root.resolve()
    conceptnet_csv = args.conceptnet_csv.resolve()
    output_dir = project_root / "data" / "cpnet"

    if not (greaselm_root / "preprocess_utils" / "conceptnet.py").is_file():
        raise FileNotFoundError(
            f"GreaseLM ConceptNet source not found under {greaselm_root}"
        )
    if not (greaselm_root / "preprocess_utils" / "grounding.py").is_file():
        raise FileNotFoundError(
            f"GreaseLM grounding source not found under {greaselm_root}"
        )
    if not conceptnet_csv.is_file():
        raise FileNotFoundError(f"ConceptNet CSV not found: {conceptnet_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    greaselm_path = str(greaselm_root)
    if greaselm_path not in sys.path:
        sys.path.insert(0, greaselm_path)
    install_lightweight_helper()

    from preprocess_utils.conceptnet import (
        construct_graph,
        extract_english,
        merged_relations as greaselm_relations,
    )
    from preprocess_utils.grounding import create_matcher_patterns

    decker_relations = load_decker_relations(project_root)
    if list(greaselm_relations) != decker_relations:
        raise ValueError(
            "GreaseLM and DECKER relation orders differ; refusing to generate "
            "incompatible relation IDs."
        )

    concept_capacity = load_decker_concept_capacity(project_root)

    with tempfile.TemporaryDirectory(prefix=".cpnet-build-", dir=output_dir) as temporary_dir:
        staging_dir = Path(temporary_dir)
        english_csv = staging_dir / "conceptnet.en.csv"
        vocabulary = staging_dir / "concept.txt"
        graph = staging_dir / "conceptnet.en.pruned.graph"
        patterns = staging_dir / "matcher_patterns.json"

        extract_english(
            str(conceptnet_csv),
            str(english_csv),
            str(vocabulary),
        )
        construct_graph(
            str(english_csv),
            str(vocabulary),
            str(graph),
            prune=True,
        )
        create_matcher_patterns(
            str(vocabulary),
            str(patterns),
        )

        counts = validate_outputs(
            vocabulary,
            graph,
            patterns,
            decker_relations,
            concept_capacity,
        )

        for filename in (
            "concept.txt",
            "conceptnet.en.pruned.graph",
            "matcher_patterns.json",
        ):
            os.replace(staging_dir / filename, output_dir / filename)

    print(f"Generated files in {output_dir}")
    print(
        f"Concepts: {counts[0]:,}; graph nodes: {counts[1]:,}; "
        f"edges: {counts[2]:,}"
    )
    print(f"Matcher patterns: {counts[3]:,}")


if __name__ == "__main__":
    main()