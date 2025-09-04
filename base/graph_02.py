from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

# Define a TypeVar for generic vertex types. This allows for type-safe graphs.
# For example, Graph[str] or Graph[int].
V = TypeVar("V")


@dataclass(frozen=True, eq=True)
class Edge(Generic[V]):
    """
    Represents a directed edge from u to v.
    'frozen=True' makes instances immutable, allowing them to be hashable.
    'eq=True' is the default, ensuring two Edge objects are equal if their fields are equal.
    """

    u: V  # The source vertex of the edge
    v: V  # The destination vertex of the edge
    weight: Any = 1  # The weight associated with the edge


class Graph(ABC, Generic[V]):
    """
    Abstract Base Class for a graph data structure.
    It is generic, meaning it can be instantiated with a specific vertex type.
    """

    def __init__(self):
        self._vertices: set[V] = set()

    def add_vertex(self, vertex: V) -> bool:
        """Adds a vertex to the graph."""
        if vertex in self._vertices:
            return False
        self._vertices.add(vertex)
        self._add_vertex_internal(vertex)
        return True

    def remove_vertex(self, vertex: V) -> bool:
        """Removes a vertex and all its incident edges from the graph."""
        if vertex not in self._vertices:
            return False

        # This approach favors encapsulation by calling the public `remove_edge` API.
        # For performance-critical applications with high-degree vertices,
        # a direct internal manipulation of edge data structures might be faster.
        incident_edges = list(self.get_incident_edges(vertex))
        for edge in incident_edges:
            self.remove_edge(edge.u, edge.v)

        self._vertices.remove(vertex)
        self._remove_vertex_internal(vertex)
        return True

    def add_edge(self, u: V, v: V, weight: Any = 1) -> bool:
        """Adds an edge between two vertices."""
        self.add_vertex(u)
        self.add_vertex(v)
        if self.has_edge(u, v):
            return False
        self._add_edge_internal(u, v, weight)
        return True

    def remove_edge(self, u: V, v: V) -> bool:
        """Removes an edge between two vertices."""
        if not self.has_edge(u, v):
            return False
        self._remove_edge_internal(u, v)
        return True

    @property
    def vertices(self) -> set[V]:
        return self._vertices.copy()

    @property
    def vertex_count(self) -> int:
        return len(self._vertices)

    @property
    @abstractmethod
    def edge_count(self) -> int:
        pass

    @abstractmethod
    def _add_vertex_internal(self, vertex: V) -> None:
        pass

    @abstractmethod
    def _remove_vertex_internal(self, vertex: V) -> None:
        pass

    @abstractmethod
    def _add_edge_internal(self, u: V, v: V, weight: Any) -> None:
        pass

    @abstractmethod
    def _remove_edge_internal(self, u: V, v: V) -> None:
        pass

    @abstractmethod
    def has_edge(self, u: V, v: V) -> bool:
        pass

    @abstractmethod
    def neighbors(self, vertex: V) -> Iterator[V]:
        pass

    @abstractmethod
    def get_incident_edges(self, vertex: V) -> Iterator[Edge[V]]:
        pass

    @abstractmethod
    def get_edge(self, u: V, v: V) -> Edge[V] | None:
        pass

    @abstractmethod
    def is_directed(self) -> bool:
        pass

    # ... DFS and BFS methods remain the same ...
    def dfs(self, start: V) -> list[V]:
        """Performs a Depth-First Search starting from a given vertex."""
        if start not in self._vertices:
            return []
        visited, result = set(), []

        def _dfs_recursive(v):
            visited.add(v)
            result.append(v)
            for neighbor in self.neighbors(v):
                if neighbor not in visited:
                    _dfs_recursive(neighbor)

        _dfs_recursive(start)
        return result

    def bfs(self, start: V) -> list[V]:
        """Performs a Breadth-First Search starting from a given vertex."""
        if start not in self._vertices:
            return []
        visited, queue, result = {start}, deque([start]), [start]
        while queue:
            v = queue.popleft()
            for neighbor in self.neighbors(v):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    result.append(neighbor)
        return result

    def __contains__(self, vertex: V) -> bool:
        return vertex in self._vertices

    def __len__(self) -> int:
        return self.vertex_count

    def __repr__(self) -> str:
        # Optimized: Directly use the O(1) edge_count property.
        return (
            f"{self.__class__.__name__}(vertices={self.vertex_count}, "
            f"edges={self.edge_count})"
        )


class UndirectedGraph(Graph[V]):
    """An undirected graph implementation using an adjacency list representation."""

    def __init__(self):
        super().__init__()
        self._adj: dict[V, set[V]] = defaultdict(set)
        # The key is a frozenset of the vertices, robustly handling non-comparable objects.
        self._edges: dict[frozenset[V], Edge[V]] = {}

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def _add_vertex_internal(self, vertex: V) -> None:
        if vertex not in self._adj:
            self._adj[vertex] = set()

    def _remove_vertex_internal(self, vertex: V) -> None:
        if vertex in self._adj:
            del self._adj[vertex]

    def _add_edge_internal(self, u: V, v: V, weight: Any) -> None:
        self._adj[u].add(v)
        self._adj[v].add(u)
        # Use a frozenset as the key for the edge for order-insensitivity.
        self._edges[frozenset({u, v})] = Edge(u, v, weight)

    def _remove_edge_internal(self, u: V, v: V) -> None:
        self._adj[u].discard(v)
        self._adj[v].discard(u)
        self._edges.pop(frozenset({u, v}), None)

    def has_edge(self, u: V, v: V) -> bool:
        return frozenset({u, v}) in self._edges

    def neighbors(self, vertex: V) -> Iterator[V]:
        return iter(self._adj.get(vertex, set()))

    def get_incident_edges(self, vertex: V) -> Iterator[Edge[V]]:
        """Yields all Edge objects connected to the vertex."""
        for neighbor in self._adj.get(vertex, set()):
            edge = self.get_edge(vertex, neighbor)
            if edge:
                yield edge

    def get_edge(self, u: V, v: V) -> Edge[V] | None:
        return self._edges.get(frozenset({u, v}))

    def degree(self, vertex: V) -> int:
        return len(self._adj.get(vertex, set()))

    def is_directed(self) -> bool:
        return False

    def __getitem__(self, vertex: V) -> set[V]:
        return self._adj.get(vertex, set()).copy()


class DirectedGraph(Graph[V]):
    """A directed graph implementation using separate in/out adjacency lists."""

    def __init__(self):
        super().__init__()
        self._out_adj: dict[V, set[V]] = defaultdict(set)
        self._in_adj: dict[V, set[V]] = defaultdict(set)
        self._edges: dict[tuple[V, V], Edge[V]] = {}

    # ... other properties and methods like add/remove_vertex_internal ...
    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def _add_vertex_internal(self, vertex: V) -> None:
        if vertex not in self._out_adj:
            self._out_adj[vertex] = set()
        if vertex not in self._in_adj:
            self._in_adj[vertex] = set()

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
        return iter(self._out_adj.get(vertex, set()))

    def predecessors(self, vertex: V) -> Iterator[V]:
        return iter(self._in_adj.get(vertex, set()))

    def get_incident_edges(self, vertex: V) -> Iterator[Edge[V]]:
        """
        Yields all incident Edge objects for a vertex (both incoming and outgoing).
        For an edge e, if e.u == vertex it's an outgoing edge.
        If e.v == vertex it's an incoming edge.
        """
        # Outgoing edges
        for successor in self._out_adj.get(vertex, set()):
            edge = self.get_edge(vertex, successor)
            if edge:
                yield edge
        # Incoming edges
        for predecessor in self._in_adj.get(vertex, set()):
            edge = self.get_edge(predecessor, vertex)
            if edge:
                yield edge

    def get_edge(self, u: V, v: V) -> Edge[V] | None:
        return self._edges.get((u, v))

    # ... degree methods and topological_sort remain the same ...
    def out_degree(self, vertex: V) -> int:
        return len(self._out_adj.get(vertex, set()))

    def in_degree(self, vertex: V) -> int:
        return len(self._in_adj.get(vertex, set()))

    def degree(self, vertex: V) -> int:
        return self.in_degree(vertex) + self.out_degree(vertex)

    def is_directed(self) -> bool:
        return True

    def topological_sort(self) -> list[V] | None:
        in_degree = {v: self.in_degree(v) for v in self._vertices}
        queue = deque([v for v in self._vertices if in_degree[v] == 0])
        result = []
        while queue:
            vertex = queue.popleft()
            result.append(vertex)
            for successor in self.neighbors(vertex):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        return result if len(result) == len(self._vertices) else None

    def __getitem__(self, vertex: V) -> tuple[set[V], set[V]]:
        out_neighbors = self._out_adj.get(vertex, set()).copy()
        in_neighbors = self._in_adj.get(vertex, set()).copy()
        return (out_neighbors, in_neighbors)


# ==================== Usage Example ====================


def demo():
    print("=== Enhanced Graph Implementation Demo ===\n")

    # --- Undirected Graph Demo ---
    print("--- Undirected Graph (ug: UndirectedGraph[str]) ---")
    ug = UndirectedGraph[str]()  # Type-safe graph for strings

    ug.add_edge("A", "B", weight=5)
    ug.add_edge("A", "C", weight=3)
    ug.add_edge("B", "D", weight=7)
    ug.add_edge("C", "D", weight=2)

    print(f"Graph: {ug}")
    print(f"Neighbors of 'A': {list(ug.neighbors('A'))}")
    print(f"Degree of 'A': {ug.degree('A')}")

    edge_ab = ug.get_edge("A", "B")
    if edge_ab:
        print(f"Weight of edge (A,B): {edge_ab.weight}")

    print(f"Is 'A' in graph? {'A' in ug}")
    print(f"Neighbors of 'A' (getitem): {ug['A']}")

    # --- Directed Graph Demo ---
    print("\n--- Directed Graph (dg: DirectedGraph[str]) ---")
    dg = DirectedGraph[str]()

    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]
    for u, v in edges:
        dg.add_edge(u, v)

    print(f"Graph: {dg}")
    print(f"Successors of 'A': {list(dg.neighbors('A'))}")
    print(f"Predecessors of 'D': {list(dg.predecessors('D'))}")
    print(f"B: Out-degree={dg.out_degree('B')}, In-degree={dg.in_degree('B')}")

    # Topological Sort
    topo_order = dg.topological_sort()
    print(f"Topological sort: {topo_order}")

    # --- Traversal Algorithms ---
    print(f"\nDFS from 'A' in undirected graph: {ug.dfs('A')}")
    print(f"BFS from 'A' in directed graph: {dg.bfs('A')}")


if __name__ == "__main__":
    demo()
