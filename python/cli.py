#!/usr/bin/env python3
"""Interactive REPL CLI for the graphdb engine.

Run with::

    python cli.py [graph.json]

Type ``help`` inside the REPL for the list of commands.
"""

from __future__ import annotations

import cmd
import hashlib
import math
import shlex
import sys
from typing import Dict, List, Optional

from graphdb import Edge, GraphStore, Node
from graphdb.exceptions import GraphDBError
from graphdb.query import bfs, find_path

EMBED_DIM = 16


def text_to_embedding(text: str, dim: int = EMBED_DIM) -> List[float]:
    """Deterministically map a text string to a pseudo-embedding vector.

    This is a *dummy* embedder used for demos and the CLI ``search`` command.
    It hashes the lower-cased tokens into a fixed-dimension bag-of-hashes vector
    so that identical / similar text yields similar vectors.
    """
    vec = [0.0] * dim
    tokens = text.lower().split()
    if not tokens:
        tokens = [text.lower()]
    for tok in tokens:
        h = hashlib.md5(tok.encode("utf-8")).hexdigest()
        for i in range(0, len(h), 4):
            bucket = int(h[i : i + 4], 16) % dim
            vec[bucket] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def parse_kv(tokens: List[str]) -> Dict[str, object]:
    """Parse ``key=value`` tokens, coercing int/float/bool where possible."""
    props: Dict[str, object] = {}
    for tok in tokens:
        if "=" not in tok:
            continue
        key, _, raw = tok.partition("=")
        props[key] = _coerce(raw)
    return props


def _coerce(raw: str) -> object:
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def short(text: str, n: int = 8) -> str:
    return text[:n]


class GraphShell(cmd.Cmd):
    intro = (
        "graphdb interactive shell. Type 'help' for commands, 'exit' to quit.\n"
    )
    prompt = "graphdb> "

    def __init__(self, path: Optional[str] = None) -> None:
        super().__init__()
        self.store = GraphStore(path=path)
        self.path = path

    # Treat '-' as part of a command word so 'add-node' parses as one token.
    identchars = cmd.Cmd.identchars + "-"

    def parseline(self, line):
        """Allow hyphenated commands (e.g. ``add-node``) to map to ``do_add_node``."""
        cmd_name, arg, line = super().parseline(line)
        if cmd_name:
            cmd_name = cmd_name.replace("-", "_")
        return cmd_name, arg, line

    # ------------------------------------------------------------- helpers
    def _err(self, msg: str) -> None:
        print(f"  [error] {msg}")

    def _print_node(self, node: Node) -> None:
        emb = "yes" if node.embedding is not None else "no"
        print(f"  id={node.id}")
        print(f"  label={node.label}")
        print(f"  properties={node.properties}")
        print(f"  embedding={emb}")

    def _print_edge(self, edge: Edge) -> None:
        print(f"  id={edge.id}")
        print(f"  {edge.src_id} --[{edge.label}]--> {edge.dst_id}")
        print(f"  weight={edge.weight} properties={edge.properties}")

    # -------------------------------------------------------------- commands
    def do_add_node(self, arg: str) -> None:
        """add-node <label> [key=value ...]   Add a node."""
        tokens = shlex.split(arg)
        if not tokens:
            self._err("usage: add-node <label> [key=value ...]")
            return
        label = tokens[0]
        props = parse_kv(tokens[1:])
        # Derive a dummy embedding from label + a 'name'/'text' prop if present.
        text_src = str(props.get("name") or props.get("text") or label)
        node = Node(
            label=label, properties=props, embedding=text_to_embedding(text_src)
        )
        try:
            self.store.add_node(node)
        except GraphDBError as exc:
            self._err(str(exc))
            return
        print(f"  added node {node.id} (label={label})")

    def do_add_edge(self, arg: str) -> None:
        """add-edge <src_id> <dst_id> <label> [key=value ...]   Add an edge."""
        tokens = shlex.split(arg)
        if len(tokens) < 3:
            self._err("usage: add-edge <src_id> <dst_id> <label> [key=value ...]")
            return
        src_id, dst_id, label = tokens[0], tokens[1], tokens[2]
        props = parse_kv(tokens[3:])
        weight = float(props.pop("weight", 1.0)) if "weight" in props else 1.0
        edge = Edge(
            src_id=src_id,
            dst_id=dst_id,
            label=label,
            properties=props,
            weight=weight,
        )
        try:
            result = self.store.add_edge(edge)
        except GraphDBError as exc:
            self._err(str(exc))
            return
        print(f"  added edge {edge.id} ({src_id} --[{label}]--> {dst_id})")
        if result.similar_edges:
            print(
                f"  [similarity] found {len(result.similar_edges)} similar "
                "existing edge(s):"
            )
            for m in result.similar_edges:
                print(
                    f"    - edge {short(m.existing_edge_id)} "
                    f"combined={m.combined_score:.3f} "
                    f"(src={m.src_similarity:.3f}, dst={m.dst_similarity:.3f})"
                )

    def do_get_node(self, arg: str) -> None:
        """get-node <id>   Show node details."""
        node_id = arg.strip()
        try:
            self._print_node(self.store.get_node(node_id))
        except GraphDBError as exc:
            self._err(str(exc))

    def do_get_edge(self, arg: str) -> None:
        """get-edge <id>   Show edge details."""
        edge_id = arg.strip()
        try:
            self._print_edge(self.store.get_edge(edge_id))
        except GraphDBError as exc:
            self._err(str(exc))

    def do_list_nodes(self, arg: str) -> None:
        """list-nodes [label]   List nodes, optionally filtered by label."""
        label = arg.strip() or None
        nodes = (
            self.store.nodes_by_label(label) if label else self.store.all_nodes()
        )
        if not nodes:
            print("  (no nodes)")
            return
        print(f"  {'ID':<38} {'LABEL':<14} PROPERTIES")
        print(f"  {'-' * 38} {'-' * 14} {'-' * 20}")
        for n in nodes:
            print(f"  {n.id:<38} {n.label:<14} {n.properties}")

    def do_list_edges(self, arg: str) -> None:
        """list-edges   List all edges."""
        edges = self.store.all_edges()
        if not edges:
            print("  (no edges)")
            return
        print(f"  {'ID':<38} {'LABEL':<12} {'SRC':<10} {'DST':<10} WEIGHT")
        print(f"  {'-' * 38} {'-' * 12} {'-' * 10} {'-' * 10} ------")
        for e in edges:
            print(
                f"  {e.id:<38} {e.label:<12} {short(e.src_id):<10} "
                f"{short(e.dst_id):<10} {e.weight}"
            )

    def do_search(self, arg: str) -> None:
        """search <label> <query_text>   Similarity search within a label."""
        tokens = shlex.split(arg)
        if len(tokens) < 2:
            self._err("usage: search <label> <query_text>")
            return
        label = tokens[0]
        query_text = " ".join(tokens[1:])
        query_vec = text_to_embedding(query_text)
        results = self.store.search_similar_nodes(query_vec, label=label, k=5)
        if not results:
            print("  (no matches)")
            return
        print(f"  top {len(results)} matches for '{query_text}' in label '{label}':")
        for node, score in results:
            print(f"    {score:.4f}  {short(node.id)}  {node.properties}")

    def do_traverse(self, arg: str) -> None:
        """traverse <node_id> [depth]   BFS traverse from a node."""
        tokens = shlex.split(arg)
        if not tokens:
            self._err("usage: traverse <node_id> [depth]")
            return
        node_id = tokens[0]
        depth = int(tokens[1]) if len(tokens) > 1 else 2
        if self.store.get_node_or_none(node_id) is None:
            self._err(f"no such node: {node_id}")
            return
        nodes = bfs(self.store, node_id, max_depth=depth, direction="out")
        if not nodes:
            print("  (no reachable nodes)")
            return
        print(f"  reachable from {short(node_id)} (depth<={depth}):")
        for n in nodes:
            print(f"    {short(n.id)}  {n.label}  {n.properties}")

    def do_path(self, arg: str) -> None:
        """path <src_id> <dst_id>   Find shortest path."""
        tokens = shlex.split(arg)
        if len(tokens) < 2:
            self._err("usage: path <src_id> <dst_id>")
            return
        path = find_path(self.store, tokens[0], tokens[1])
        if not path:
            print("  (no path found)")
            return
        print("  " + " -> ".join(f"{short(n.id)}({n.label})" for n in path))

    def do_save(self, arg: str) -> None:
        """save   Save the graph to disk."""
        target = arg.strip() or self.path
        if not target:
            self._err("no path configured; usage: save <path>")
            return
        try:
            self.store.save(target)
        except GraphDBError as exc:
            self._err(str(exc))
            return
        print(f"  saved to {target}")

    def do_load(self, arg: str) -> None:
        """load   Load the graph from disk."""
        target = arg.strip() or self.path
        if not target:
            self._err("no path configured; usage: load <path>")
            return
        try:
            self.store.load(target)
        except GraphDBError as exc:
            self._err(str(exc))
            return
        print(f"  loaded from {target}")

    def do_stats(self, arg: str) -> None:
        """stats   Show graph statistics."""
        s = self.store.stats()
        print(f"  nodes: {s['node_count']}")
        print(f"  edges: {s['edge_count']}")
        print(f"  labels: {', '.join(s['labels']) or '(none)'}")
        print(f"  path: {s['path']}")

    def do_exit(self, arg: str) -> bool:
        """exit   Exit the shell."""
        print("  bye")
        return True

    def do_EOF(self, arg: str) -> bool:  # Ctrl-D
        print()
        return self.do_exit(arg)

    def emptyline(self) -> None:
        pass


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    path = argv[0] if argv else None
    shell = GraphShell(path=path)
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\n  bye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
