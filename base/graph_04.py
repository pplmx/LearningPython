from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Generic, Self, TypeVar

# --- Type variable definitions ---
V = TypeVar("V")  # Type for vertices
W = TypeVar("W")  # Type for edge weights


@dataclass(frozen=True, eq=True, slots=True)
class Edge(Generic[V, W]):
    """Immutable edge representation."""

    u: V
    v: V
    weight: W = 1

    def __str__(self) -> str:
        return f"{self.u} --({self.weight})--> {self.v}"

    def reverse(self) -> Self:
        """Return a new edge with reversed direction."""
        return Edge(self.v, self.u, self.weight)


# --- Custom exceptions ---
class GraphError(Exception): ...


class VertexNotFoundError(GraphError): ...


class EdgeNotFoundError(GraphError): ...


# --- Base class ---
class Graph(ABC, Generic[V, W]):
    """Abstract base class for graphs (directed or undirected)."""

    __slots__ = ("_vertices",)

    def __init__(self) -> None:
        self._vertices: set[V] = set()

    # --- Vertex Management ---
    def add_vertex(self, vertex: V) -> bool:
        if vertex in self._vertices:
            return False
        self._vertices.add(vertex)
        self._on_vertex_added(vertex)
        return True

    def remove_vertex(self, vertex: V) -> bool:
        if vertex not in self._vertices:
            return False
        for edge in list(self.get_incident_edges(vertex)):
            self.remove_edge(edge.u, edge.v)
        self._vertices.remove(vertex)
        self._on_vertex_removed(vertex)
        return True

    def add_vertices(self, vertices: Iterable[V]) -> int:
        return sum(1 for v in vertices if self.add_vertex(v))

    # --- Edge Management ---
    def add_edge(self, u: V, v: V, weight: W = 1) -> bool:
        self.add_vertex(u)
        self.add_vertex(v)
        if self.has_edge(u, v):
            return False
        self._on_edge_added(u, v, weight)
        return True

    def add_edges(self, edges: Iterable[tuple[V, V] | tuple[V, V, W]]) -> int:
        count = 0
        for data in edges:
            u, v, *rest = data
            w = rest[0] if rest else 1
            if self.add_edge(u, v, w):
                count += 1
        return count

    def remove_edge(self, u: V, v: V) -> bool:
        if not self.has_edge(u, v):
            return False
        self._on_edge_removed(u, v)
        return True

    # --- Properties ---
    @property
    def vertices(self) -> frozenset[V]:
        return frozenset(self._vertices)

    @property
    def vertex_count(self) -> int:
        return len(self._vertices)

    @property
    @abstractmethod
    def edge_count(self) -> int: ...

    def is_empty(self) -> bool:
        return not self._vertices

    def has_vertex(self, vertex: V) -> bool:
        return vertex in self._vertices

    def get_edge_weight(self, u: V, v: V) -> W:
        edge = self.get_edge(u, v)
        if edge is None:
            raise EdgeNotFoundError(f"Edge ({u}, {v}) not found")
        return edge.weight

    # --- Abstract API ---
    @abstractmethod
    def _on_vertex_added(self, vertex: V) -> None: ...
    @abstractmethod
    def _on_vertex_removed(self, vertex: V) -> None: ...
    @abstractmethod
    def _on_edge_added(self, u: V, v: V, weight: W) -> None: ...
    @abstractmethod
    def _on_edge_removed(self, u: V, v: V) -> None: ...
    @abstractmethod
    def has_edge(self, u: V, v: V) -> bool: ...
    @abstractmethod
    def neighbors(self, vertex: V) -> Iterator[V]: ...
    @abstractmethod
    def get_incident_edges(self, vertex: V) -> Iterator[Edge[V, W]]: ...
    @abstractmethod
    def get_edge(self, u: V, v: V) -> Edge[V, W] | None: ...
    @abstractmethod
    def is_directed(self) -> bool: ...

    # --- Algorithms ---
    def dfs(self, start: V, visited: set[V] | None = None) -> Iterator[V]:
        if start not in self._vertices:
            return
        if visited is None:
            visited = set()
        stack = [start]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            yield v
            for n in reversed(list(self.neighbors(v))):
                if n not in visited:
                    stack.append(n)

    def bfs(self, start: V, visited: set[V] | None = None) -> Iterator[V]:
        if start not in self._vertices:
            return
        if visited is None:
            visited = set()
        if start in visited:
            return
        visited.add(start)
        q = deque([start])
        while q:
            v = q.popleft()
            yield v
            for n in self.neighbors(v):
                if n not in visited:
                    visited.add(n)
                    q.append(n)

    def connected_components(self) -> Iterator[list[V]]:
        visited: set[V] = set()
        for v in self._vertices:
            if v not in visited:
                yield list(self.dfs(v, visited))

    def is_connected(self) -> bool:
        return self.is_empty() or len(list(self.connected_components())) == 1

    def has_path(self, start: V, end: V) -> bool:
        if start not in self._vertices or end not in self._vertices:
            return False
        if start == end:
            return True
        visited = {start}
        q = deque([start])
        while q:
            v = q.popleft()
            for n in self.neighbors(v):
                if n == end:
                    return True
                if n not in visited:
                    visited.add(n)
                    q.append(n)
        return False

    # --- Utilities ---
    def edges(self) -> Iterator[Edge[V, W]]:
        """
        Yield each edge exactly once.
        - For directed graphs the key is (u, v).
        - For undirected graphs the key is frozenset({u, v}) to avoid duplicates.
        """
        seen = set()
        for v in self._vertices:
            for e in self.get_incident_edges(v):
                key = (e.u, e.v) if self.is_directed() else frozenset({e.u, e.v})
                if key in seen:
                    continue
                seen.add(key)
                yield e

    def clear(self) -> None:
        """Remove all vertices and edges."""
        for v in list(self._vertices):
            self.remove_vertex(v)

    def copy(self) -> Self:
        """Return a shallow copy of the graph."""
        new = self.__class__()  # type: ignore
        new.add_vertices(self._vertices)
        for e in self.edges():
            new.add_edge(e.u, e.v, e.weight)
        return new

    # --- Dunder ---
    def __contains__(self, vertex: V) -> bool:
        return self.has_vertex(vertex)

    def __len__(self) -> int:
        return self.vertex_count

    def __iter__(self) -> Iterator[V]:
        return iter(self._vertices)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(V={self.vertex_count}, E={self.edge_count})"


# --- Undirected ---
class UndirectedGraph(Graph[V, W]):
    __slots__ = ("_adj", "_edges")

    def __init__(self) -> None:
        super().__init__()
        self._adj: dict[V, set[V]] = defaultdict(set)
        self._edges: dict[frozenset[V], Edge[V, W]] = {}

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def _on_vertex_added(self, v: V) -> None:
        self._adj.setdefault(v, set())

    def _on_vertex_removed(self, v: V) -> None:
        self._adj.pop(v, None)

    def _on_edge_added(self, u: V, v: V, w: W) -> None:
        self._adj[u].add(v)
        self._adj[v].add(u)
        self._edges[frozenset({u, v})] = Edge(u, v, w)

    def _on_edge_removed(self, u: V, v: V) -> None:
        self._adj[u].discard(v)
        self._adj[v].discard(u)
        self._edges.pop(frozenset({u, v}), None)

    def has_edge(self, u: V, v: V) -> bool:
        return frozenset({u, v}) in self._edges

    def neighbors(self, v: V) -> Iterator[V]:
        return iter(self._adj.get(v, set()))

    def get_incident_edges(self, v: V) -> Iterator[Edge[V, W]]:
        for n in self._adj.get(v, set()):
            e = self.get_edge(v, n)
            if e:
                yield e

    def get_edge(self, u: V, v: V) -> Edge[V, W] | None:
        return self._edges.get(frozenset({u, v}))

    def degree(self, v: V) -> int:
        return len(self._adj.get(v, set()))

    def is_directed(self) -> bool:
        return False


# --- Directed ---
class DirectedGraph(Graph[V, W]):
    __slots__ = ("_out", "_in", "_edges")

    def __init__(self) -> None:
        super().__init__()
        self._out: dict[V, set[V]] = defaultdict(set)
        self._in: dict[V, set[V]] = defaultdict(set)
        self._edges: dict[tuple[V, V], Edge[V, W]] = {}

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def _on_vertex_added(self, v: V) -> None:
        self._out.setdefault(v, set())
        self._in.setdefault(v, set())

    def _on_vertex_removed(self, v: V) -> None:
        self._out.pop(v, None)
        self._in.pop(v, None)

    def _on_edge_added(self, u: V, v: V, w: W) -> None:
        self._out[u].add(v)
        self._in[v].add(u)
        self._edges[(u, v)] = Edge(u, v, w)

    def _on_edge_removed(self, u: V, v: V) -> None:
        self._out[u].discard(v)
        self._in[v].discard(u)
        self._edges.pop((u, v), None)

    def has_edge(self, u: V, v: V) -> bool:
        return (u, v) in self._edges

    def neighbors(self, v: V) -> Iterator[V]:
        return iter(self._out.get(v, set()))

    def predecessors(self, v: V) -> Iterator[V]:
        return iter(self._in.get(v, set()))

    def get_incident_edges(self, v: V) -> Iterator[Edge[V, W]]:
        for n in self._out.get(v, set()):
            e = self.get_edge(v, n)
            if e:
                yield e
        for p in self._in.get(v, set()):
            e = self.get_edge(p, v)
            if e:
                yield e

    def get_edge(self, u: V, v: V) -> Edge[V, W] | None:
        return self._edges.get((u, v))

    def out_degree(self, v: V) -> int:
        return len(self._out.get(v, set()))

    def in_degree(self, v: V) -> int:
        return len(self._in.get(v, set()))

    def degree(self, v: V) -> int:
        return self.in_degree(v) + self.out_degree(v)

    def is_directed(self) -> bool:
        return True

    def topological_sort(self) -> list[V] | None:
        indeg = {v: self.in_degree(v) for v in self._vertices}
        q = deque(v for v, d in indeg.items() if d == 0)
        res: list[V] = []
        while q:
            v = q.popleft()
            res.append(v)
            for n in self.neighbors(v):
                indeg[n] -= 1
                if indeg[n] == 0:
                    q.append(n)
        return res if len(res) == len(self._vertices) else None

    def is_acyclic(self) -> bool:
        return self.topological_sort() is not None

    def reverse(self) -> Self:
        g = self.__class__()  # type: ignore
        g.add_vertices(self._vertices)
        for e in self._edges.values():
            g.add_edge(e.v, e.u, e.weight)
        return g


if __name__ == "__main__":
    # --- Undirected graph demo ---
    ug = UndirectedGraph[str, int]()
    ug.add_edges([("A", "B", 5), ("A", "C", 3), ("B", "D", 7), ("C", "D", 2)])

    print("--- UndirectedGraph ---")
    print("repr:", repr(ug))
    print("Vertices:", list(ug))
    print("Edges (u, v, w):", [(e.u, e.v, e.weight) for e in ug.edges()])
    print("Neighbors of A:", list(ug.neighbors("A")))
    print("Degree of A:", ug.degree("A"))
    print("Is connected:", ug.is_connected())
    print("Connected components:", list(ug.connected_components()))
    print("DFS from A:", list(ug.dfs("A")))
    print("BFS from A:", list(ug.bfs("A")))
    print("Has path A -> D:", ug.has_path("A", "D"))
    try:
        print("Weight A-B:", ug.get_edge_weight("A", "B"))
    except EdgeNotFoundError as ex:
        print("Edge error:", ex)

    # --- Directed graph demo ---
    dg = DirectedGraph[str, int]()
    # edges may be (u, v) or (u, v, w); weight defaults to 1
    dg.add_edges([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")])

    print("\n--- DirectedGraph ---")
    print("repr:", repr(dg))
    print("Vertices:", list(dg))
    print("Edges (u, v, w):", [(e.u, e.v, e.weight) for e in dg.edges()])
    print("Successors of A:", list(dg.neighbors("A")))
    print("Predecessors of D:", list(dg.predecessors("D")))
    print("Out/In degree of B:", dg.out_degree("B"), dg.in_degree("B"))
    print("Is DAG:", dg.is_acyclic())
    topo = dg.topological_sort()
    print("Topological order:", topo if topo is not None else "Graph has cycle")
    print("Reverse graph edges:", [(e.u, e.v, e.weight) for e in dg.reverse().edges()])
