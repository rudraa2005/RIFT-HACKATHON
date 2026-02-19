"""
Shell Chain Detection Module.

Identifies shell accounts (low-activity intermediaries) and finds chains.

Time Complexity: O(V × 3^D) bounded by shell degree ≤ 3, D = max depth (capped at 8)
Memory: O(V + chains × chain_length)
"""

from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import pandas as pd

from app.config import (
    SHELL_HOLDING_TIME_HOURS,
    SHELL_MAX_DEGREE,
    SHELL_MAX_TRANSACTIONS,
    SHELL_MIN_CHAIN_LENGTH,
)


def _identify_shell_accounts(G: nx.MultiDiGraph, df: pd.DataFrame) -> Set[str]:
    """Identify shell accounts using vectorized grouping for O(N) speed."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Pre-calculate node-level stats
    # 1. Transaction counts
    in_counts = df.groupby("receiver_id").size()
    out_counts = df.groupby("sender_id").size()
    
    # 2. Holding times (min timestamp per account as sender vs receiver)
    first_in = df.groupby("receiver_id")["timestamp"].min()
    first_out = df.groupby("sender_id")["timestamp"].min()

    shell_accounts: Set[str] = set()

    for node in G.nodes():
        node_str = str(node)
        
        # Static graph check
        total_degree = G.in_degree(node) + G.out_degree(node)
        if total_degree > SHELL_MAX_DEGREE:
            continue

        # Must have BOTH incoming and outgoing edges (pass-through behavior)
        if G.in_degree(node) == 0 or G.out_degree(node) == 0:
            continue

        # Transaction count check
        n_in = in_counts.get(node_str, 0)
        n_out = out_counts.get(node_str, 0)
        if (n_in + n_out) > SHELL_MAX_TRANSACTIONS:
            continue

        # Must have both incoming and outgoing transactions
        if n_in == 0 or n_out == 0:
            continue

        # Holding time check
        if n_in > 0 and n_out > 0:
            t_in = first_in.get(node_str)
            t_out = first_out.get(node_str)
            if t_in and t_out:
                holding_hours = (t_out - t_in).total_seconds() / 3600
                if holding_hours > SHELL_HOLDING_TIME_HOURS:
                    continue

        shell_accounts.add(node_str)

    return shell_accounts


def _find_shell_chains(
    G: nx.MultiDiGraph, shell_accounts: Set[str]
) -> List[List[str]]:
    """DFS to find chains ≥ SHELL_MIN_CHAIN_LENGTH where all intermediates are shell."""
    simple_G = nx.DiGraph(G)
    chains: List[List[str]] = []
    visited_chains: Set[tuple] = set()

    for start_node in simple_G.nodes():
        stack = [(start_node, [start_node])]

        while stack:
            current, path = stack.pop()

            for neighbor in simple_G.successors(current):
                if neighbor in path:
                    continue

                new_path = path + [neighbor]
                intermediates = new_path[1:-1]

                if intermediates and all(n in shell_accounts for n in intermediates):
                    if len(new_path) >= SHELL_MIN_CHAIN_LENGTH:
                        chain_key = tuple(new_path)
                        if chain_key not in visited_chains:
                            visited_chains.add(chain_key)
                            chains.append(new_path)

                if neighbor in shell_accounts and len(new_path) < 8:
                    stack.append((neighbor, new_path))

    return chains


def detect_shell_chains(
    G: nx.MultiDiGraph, df: pd.DataFrame, exclude_nodes: Set[str] | None = None
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    Detect shell chain patterns.

    Returns:
        (rings_list, shell_account_ids_set)
    """
    shell_accounts = _identify_shell_accounts(G, df)
    if exclude_nodes:
        shell_accounts = shell_accounts - exclude_nodes
    chains = _find_shell_chains(G, shell_accounts)

    rings: List[Dict[str, Any]] = []
    for i, chain in enumerate(chains, 1):
        rings.append(
            {
                "ring_id": f"RING_SHELL_{i:03d}",
                "members": chain,
                "pattern_type": "shell_chain",
                "risk_score": round(min(100, 50 + len(chain) * 10), 2),
            }
        )

    return rings, shell_accounts
