"""
Shell Chain Detection Module.

Identifies shell accounts (low-activity intermediaries) and finds chains
of ≥3 hops where all intermediate nodes are shell accounts.

Shell Account Criteria:
  - Total degree ≤ 3
  - Total transactions ≤ 3
  - Holding time (first-in → first-out) < 24 hours

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
    """
    Identify shell accounts by degree, transaction count, and holding time.

    Time Complexity: O(V + E)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    shell_accounts: Set[str] = set()

    for node in G.nodes():
        total_degree = G.in_degree(node) + G.out_degree(node)
        if total_degree > SHELL_MAX_DEGREE:
            continue

        in_txns = df[df["receiver_id"] == node]
        out_txns = df[df["sender_id"] == node]
        total_txns = len(in_txns) + len(out_txns)
        if total_txns > SHELL_MAX_TRANSACTIONS:
            continue

        # Holding time: gap between first inbound and first outbound
        if not in_txns.empty and not out_txns.empty:
            first_in = pd.to_datetime(in_txns["timestamp"]).min()
            first_out = pd.to_datetime(out_txns["timestamp"]).min()
            holding_hours = (first_out - first_in).total_seconds() / 3600
            if holding_hours > SHELL_HOLDING_TIME_HOURS:
                continue

        shell_accounts.add(str(node))

    return shell_accounts


def _find_shell_chains(
    G: nx.MultiDiGraph, shell_accounts: Set[str]
) -> List[List[str]]:
    """
    DFS to find chains ≥ SHELL_MIN_CHAIN_LENGTH where all intermediates are shell.

    Time Complexity: O(V × 3^D), bounded because shell degree ≤ 3
    """
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

                # Keep extending through shell nodes, cap depth at 8
                if neighbor in shell_accounts and len(new_path) < 8:
                    stack.append((neighbor, new_path))

    return chains


def detect_shell_chains(
    G: nx.MultiDiGraph, df: pd.DataFrame
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    Detect shell chain patterns.

    Returns:
        (rings_list, shell_account_ids_set)
    """
    shell_accounts = _identify_shell_accounts(G, df)
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
