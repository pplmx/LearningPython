from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

# Type variable for vertices
V = TypeVar("V")


@dataclass(frozen=True, eq=True, slots=True)
class Edge(Generic[V]):
    """Immutable directed edge (u -> v) with an optional weight."""

    u: V
    v: V
    weight: Any = 1


class Graph(ABC, Generic[V]):
    """Abstract Base Class for graphs (directed or undirected)."""

    __slots__ = ("_vertices",)

    def __init__(self) -> None:
        self._vertices: set[V] = set()

    # --- Vertex Management ---
    def add_vertex(self, vertex: V) -> bool:
        """Add a vertex if not exists. Returns True if added."""
        if vertex in self._vertices:
            return False
        self._vertices.add(vertex)
        self._add_vertex_internal(vertex)
        return True

    def remove_vertex(self, vertex: V) -> bool:
        """Remove a vertex and its incident edges."""
        if vertex not in self._vertices:
            return False
        for edge in self.get_incident_edges(vertex):
            self.remove_edge(edge.u, edge.v)
        self._vertices.remove(vertex)
        self._remove_vertex_internal(vertex)
        return True

    # --- Edge Management ---
    def add_edge(self, u: V, v: V, weight: Any = 1) -> bool:
        """Add an edge (u -> v). Returns False if already exists."""
        self.add_vertex(u)
        self.add_vertex(v)
        if self.has_edge(u, v):
            return False
        self._add_edge_internal(u, v, weight)
        return True

    def remove_edge(self, u: V, v: V) -> bool:
        """Remove an edge (u -> v). Returns False if not exists."""
        if not self.has_edge(u, v):
            return False
        self._remove_edge_internal(u, v)
        return True

    # --- Properties ---
    @property
    def vertices(self) -> set[V]:
        return self._vertices.copy()

    @property
    def vertex_count(self) -> int:
        return len(self._vertices)

    @property
    @abstractmethod
    def edge_count(self) -> int: ...

    # --- Abstract methods for subclasses ---
    @abstractmethod
    def _add_vertex_internal(self, vertex: V) -> None: ...

    @abstractmethod
    def _remove_vertex_internal(self, vertex: V) -> None: ...

    @abstractmethod
    def _add_edge_internal(self, u: V, v: V, weight: Any) -> None: ...

    @abstractmethod
    def _remove_edge_internal(self, u: V, v: V) -> None: ...

    @abstractmethod
    def has_edge(self, u: V, v: V) -> bool: ...

    @abstractmethod
    def neighbors(self, vertex: V) -> Iterator[V]: ...

    @abstractmethod
    def get_incident_edges(self, vertex: V) -> Iterator[Edge[V]]: ...

    @abstractmethod
    def get_edge(self, u: V, v: V) -> Edge[V] | None: ...

    @abstractmethod
    def is_directed(self) -> bool: ...

    # --- Traversal Algorithms ---
    def dfs(self, start: V) -> list[V]:
        """Depth-First Search traversal order starting from start."""
        if start not in self._vertices:
            return []
        visited, result = set(), []

        def _dfs(v: V) -> None:
            visited.add(v)
            result.append(v)
            for neighbor in self.neighbors(v):
                if neighbor not in visited:
                    _dfs(neighbor)

        _dfs(start)
        return result

    def bfs(self, start: V) -> list[V]:
        """Breadth-First Search traversal order starting from start."""
        if start not in self._vertices:
            return []
        visited, queue, result = {start}, deque([start]), []
        while queue:
            v = queue.popleft()
            result.append(v)
            for neighbor in self.neighbors(v):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return result

    # --- Magic methods ---
    def __contains__(self, vertex: V) -> bool:
        return vertex in self._vertices

    def __len__(self) -> int:
        return self.vertex_count

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vertices={self.vertex_count}, edges={self.edge_count})"


class UndirectedGraph(Graph[V]):
    """Undirected graph implemented with adjacency lists."""

    __slots__ = ("_adj", "_edges")

    def __init__(self) -> None:
        super().__init__()
        self._adj: dict[V, set[V]] = defaultdict(set)
        self._edges: dict[frozenset[V], Edge[V]] = {}

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def _add_vertex_internal(self, vertex: V) -> None:
        self._adj.setdefault(vertex, set())

    def _remove_vertex_internal(self, vertex: V) -> None:
        self._adj.pop(vertex, None)

    def _add_edge_internal(self, u: V, v: V, weight: Any) -> None:
        self._adj[u].add(v)
        self._adj[v].add(u)
        self._edges[frozenset({u, v})] = Edge(u, v, weight)

    def _remove_edge_internal(self, u: V, v: V) -> None:
        self._adj[u].discard(v)
        self._adj[v].discard(u)
        self._edges.pop(frozenset({u, v}), None)

    def has_edge(self, u: V, v: V) -> bool:
        return frozenset({u, v}) in self._edges

    def neighbors(self, vertex: V) -> Iterator[V]:
        return iter(self._adj.get(vertex, ()))

    def get_incident_edges(self, vertex: V) -> Iterator[Edge[V]]:
        for neighbor in self._adj.get(vertex, ()):  # direct lookup
            if edge := self.get_edge(vertex, neighbor):
                yield edge

    def get_edge(self, u: V, v: V) -> Edge[V] | None:
        return self._edges.get(frozenset({u, v}))

    def degree(self, vertex: V) -> int:
        return len(self._adj.get(vertex, ()))

    def is_directed(self) -> bool:
        return False

    def __getitem__(self, vertex: V) -> frozenset[V]:
        return frozenset(self._adj.get(vertex, ()))


class DirectedGraph(Graph[V]):
    """Directed graph with separate in/out adjacency lists."""

    __slots__ = ("_out_adj", "_in_adj", "_edges")

    def __init__(self) -> None:
        super().__init__()
        self._out_adj: dict[V, set[V]] = defaultdict(set)
        self._in_adj: dict[V, set[V]] = defaultdict(set)
        self._edges: dict[tuple[V, V], Edge[V]] = {}

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def _add_vertex_internal(self, vertex: V) -> None:
        self._out_adj.setdefault(vertex, set())
        self._in_adj.setdefault(vertex, set())

    def _remove_vertex_internal(self, vertex: V) -> None:
        self._out_adj.pop(vertex, None)
        self._in_adj.pop(vertex, None)

    def _add_edge_internal(self, u: V, v: V, weight: Any) -> None:
        self._out_adj[u].add(v)
        self._in_adj[v].add(u)
        self._edges[(u, v)] = Edge(u, v, weight)

    def _remove_edge_internal(self, u: V, v: V) -> None:
        self._out_adj[u].discard(v)
        self._in_adj[v].discard(u)
        self._edges.pop((u, v), None)

    def has_edge(self, u: V, v: V) -> bool:
        return (u, v) in self._edges

    def neighbors(self, vertex: V) -> Iterator[V]:
        return iter(self._out_adj.get(vertex, ()))

    def predecessors(self, vertex: V) -> Iterator[V]:
        return iter(self._in_adj.get(vertex, ()))

    def get_incident_edges(self, vertex: V) -> Iterator[Edge[V]]:
        for successor in self._out_adj.get(vertex, ()):  # outgoing
            if edge := self.get_edge(vertex, successor):
                yield edge
        for predecessor in self._in_adj.get(vertex, ()):  # incoming
            if edge := self.get_edge(predecessor, vertex):
                yield edge

    def get_edge(self, u: V, v: V) -> Edge[V] | None:
        return self._edges.get((u, v))

    def out_degree(self, vertex: V) -> int:
        return len(self._out_adj.get(vertex, ()))

    def in_degree(self, vertex: V) -> int:
        return len(self._in_adj.get(vertex, ()))

    def degree(self, vertex: V) -> int:
        return self.in_degree(vertex) + self.out_degree(vertex)

    def is_directed(self) -> bool:
        return True

    def topological_sort(self) -> list[V] | None:
        """Return topological order if DAG, else None."""
        in_degree = {v: self.in_degree(v) for v in self._vertices}
        queue = deque(v for v in self._vertices if in_degree[v] == 0)
        result = []
        while queue:
            v = queue.popleft()
            result.append(v)
            for successor in self.neighbors(v):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        return result if len(result) == len(self._vertices) else None

    def __getitem__(self, vertex: V) -> dict[str, frozenset[V]]:
        return {
            "out": frozenset(self._out_adj.get(vertex, ())),
            "in": frozenset(self._in_adj.get(vertex, ())),
        }


# ==================== Demo ====================

def demo() -> None:
    print("=== Graph Implementation Demo ===\n")

    print("--- Undirected Graph ---")
    ug = UndirectedGraph[str]()
    ug.add_edge("A", "B", weight=5)
    ug.add_edge("A", "C", weight=3)
    ug.add_edge("B", "D", weight=7)
    ug.add_edge("C", "D", weight=2)

    print(f"Graph: {ug}")
    print(f"Neighbors of 'A': {list(ug.neighbors('A'))}")
    print(f"Degree of 'A': {ug.degree('A')}")
    if edge_ab := ug.get_edge("A", "B"):
        print(f"Weight of edge (A,B): {edge_ab.weight}")

    print("\n--- Directed Graph ---")
    dg = DirectedGraph[str]()
    for u, v in [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]:
        dg.add_edge(u, v)

    print(f"Graph: {dg}")
    print(f"Successors of 'A': {list(dg.neighbors('A'))}")
    print(f"Predecessors of 'D': {list(dg.predecessors('D'))}")
    print(f"B: Out-degree={dg.out_degree('B')}, In-degree={dg.in_degree('B')}")
    print(f"Topological sort: {dg.topological_sort()}")

    print(f"\nDFS from 'A' in UG: {ug.dfs('A')}")
    print(f"BFS from 'A' in DG: {dg.bfs('A')}")


if __name__ == "__main__":
    demo()
