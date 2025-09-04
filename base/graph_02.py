from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

# Define a TypeVar for generic vertex types. This allows for type-safe graphs.
# For example, Graph[str] or Graph[int].
V = TypeVar("V")


@dataclass(frozen=True)
class Edge(Generic[V]):
    """
    A dataclass to represent an edge in the graph.
    It is immutable (frozen=True) to prevent accidental modification.
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
        # A set to store all unique vertices in the graph.
        # Using a set provides O(1) average time complexity for add, remove, and contains operations.
        self._vertices: set[V] = set()

    def add_vertex(self, vertex: V) -> bool:
        """Adds a vertex to the graph."""
        if vertex in self._vertices:
            return False  # Vertex already exists
        self._vertices.add(vertex)
        self._add_vertex_internal(vertex)
        return True

    def remove_vertex(self, vertex: V) -> bool:
        """Removes a vertex and all its incident edges from the graph."""
        if vertex not in self._vertices:
            return False  # Vertex does not exist

        # Must remove all edges connected to this vertex first.
        # We convert to a list to avoid modification issues while iterating.
        incident_edges = list(self.get_incident_edges(vertex))
        for u, v in incident_edges:
            self.remove_edge(u, v)

        self._vertices.remove(vertex)
        self._remove_vertex_internal(vertex)
        return True

    def add_edge(self, u: V, v: V, weight: Any = 1) -> bool:
        """Adds an edge between two vertices."""
        # Automatically add vertices if they don't exist yet.
        self.add_vertex(u)
        self.add_vertex(v)

        if self.has_edge(u, v):
            return False  # Edge already exists

        self._add_edge_internal(u, v, weight)
        return True

    def remove_edge(self, u: V, v: V) -> bool:
        """Removes an edge between two vertices."""
        if not self.has_edge(u, v):
            return False  # Edge does not exist

        self._remove_edge_internal(u, v)
        return True

    # --- Properties ---
    @property
    def vertices(self) -> set[V]:
        """Returns a copy of the set of vertices."""
        return self._vertices.copy()

    @property
    def vertex_count(self) -> int:
        """Returns the number of vertices."""
        return len(self._vertices)

    @property
    @abstractmethod
    def edge_count(self) -> int:
        """Returns the number of edges."""
        pass

    # --- Abstract Methods (to be implemented by subclasses) ---
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
    def get_incident_edges(self, vertex: V) -> Iterator[tuple[V, V]]:
        pass

    @abstractmethod
    def get_edge(self, u: V, v: V) -> Edge[V] | None:
        pass

    @abstractmethod
    def is_directed(self) -> bool:
        pass

    # --- Standard Graph Algorithms ---
    def dfs(self, start: V) -> list[V]:
        """Performs a Depth-First Search starting from a given vertex."""
        if start not in self._vertices:
            return []

        visited = set()
        result = []

        def _dfs_recursive(vertex):
            visited.add(vertex)
            result.append(vertex)
            for neighbor in self.neighbors(vertex):
                if neighbor not in visited:
                    _dfs_recursive(neighbor)

        _dfs_recursive(start)
        return result

    def bfs(self, start: V) -> list[V]:
        """Performs a Breadth-First Search starting from a given vertex."""
        if start not in self._vertices:
            return []

        visited = {start}
        queue = deque([start])
        result = [start]

        while queue:
            vertex = queue.popleft()
            for neighbor in self.neighbors(vertex):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    result.append(neighbor)

        return result

    # --- Pythonic "Magic" Methods ---
    def __contains__(self, vertex: V) -> bool:
        """Enables the 'in' operator, e.g., `if vertex in graph:`."""
        return vertex in self._vertices

    def __len__(self) -> int:
        """Enables the `len()` function, e.g., `len(graph)`."""
        return self.vertex_count

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(vertices={self.vertex_count}, "
            f"edges={self.edge_count})"
        )


class UndirectedGraph(Graph[V]):
    """An undirected graph implementation using an adjacency list representation."""

    def __init__(self):
        super().__init__()
        # Adjacency list: maps a vertex to a set of its neighbors.
        self._adj: dict[V, set[V]] = defaultdict(set)
        # Stores Edge objects. The key is a canonical representation of the edge (min, max).
        self._edges: dict[tuple[V, V], Edge[V]] = {}

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def _add_vertex_internal(self, vertex: V) -> None:
        # defaultdict handles creation automatically, but this ensures the key exists.
        if vertex not in self._adj:
            self._adj[vertex] = set()

    def _remove_vertex_internal(self, vertex: V) -> None:
        # No need to remove edges here; `remove_vertex` in the base class handles it.
        if vertex in self._adj:
            del self._adj[vertex]

    def _add_edge_internal(self, u: V, v: V, weight: Any) -> None:
        # In an undirected graph, the edge goes both ways.
        self._adj[u].add(v)
        self._adj[v].add(u)

        # Use a canonical key (min, max) to represent the undirected edge.
        edge_key = (min(u, v), max(u, v))
        self._edges[edge_key] = Edge(u, v, weight)

    def _remove_edge_internal(self, u: V, v: V) -> None:
        self._adj[u].discard(v)
        self._adj[v].discard(u)

        edge_key = (min(u, v), max(u, v))
        self._edges.pop(edge_key, None)

    def has_edge(self, u: V, v: V) -> bool:
        return u in self._adj and v in self._adj[u]

    def neighbors(self, vertex: V) -> Iterator[V]:
        """Returns an iterator over the neighbors of a vertex."""
        return iter(self._adj.get(vertex, set()))

    def get_incident_edges(self, vertex: V) -> Iterator[tuple[V, V]]:
        """Returns an iterator over the pairs of vertices for edges incident to the given vertex."""
        for neighbor in self._adj.get(vertex, set()):
            yield vertex, neighbor

    def get_edge(self, u: V, v: V) -> Edge[V] | None:
        """Retrieves the Edge object between two vertices, if it exists."""
        edge_key = (min(u, v), max(u, v))
        return self._edges.get(edge_key)

    def degree(self, vertex: V) -> int:
        """Returns the degree of a vertex."""
        return len(self._adj.get(vertex, set()))

    def is_directed(self) -> bool:
        return False

    def __getitem__(self, vertex: V) -> set[V]:
        """Allows dictionary-style access to a vertex's neighbors, e.g., `graph[vertex]`."""
        return self._adj.get(vertex, set()).copy()


class DirectedGraph(Graph[V]):
    """A directed graph implementation using separate in/out adjacency lists."""

    def __init__(self):
        super().__init__()
        # Out-adjacency list: maps a vertex to the set of vertices it points to (successors).
        self._out_adj: dict[V, set[V]] = defaultdict(set)
        # In-adjacency list: maps a vertex to the set of vertices that point to it (predecessors).
        self._in_adj: dict[V, set[V]] = defaultdict(set)
        # Stores Edge objects. The key is the directed tuple (u, v).
        self._edges: dict[tuple[V, V], Edge[V]] = {}

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def _add_vertex_internal(self, vertex: V) -> None:
        if vertex not in self._out_adj:
            self._out_adj[vertex] = set()
        if vertex not in self._in_adj:
            self._in_adj[vertex] = set()

    def _remove_vertex_internal(self, vertex: V) -> None:
        # No need to remove edges here; `remove_vertex` in the base class handles it.
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
        return u in self._out_adj and v in self._out_adj[u]

    def neighbors(self, vertex: V) -> Iterator[V]:
        """Returns an iterator over the successors of a vertex (out-neighbors)."""
        return iter(self._out_adj.get(vertex, set()))

    def predecessors(self, vertex: V) -> Iterator[V]:
        """Returns an iterator over the predecessors of a vertex (in-neighbors)."""
        return iter(self._in_adj.get(vertex, set()))

    def get_incident_edges(self, vertex: V) -> Iterator[tuple[V, V]]:
        """Returns an iterator over all edges connected to a vertex (both in and out)."""
        # Outgoing edges
        for successor in self._out_adj.get(vertex, set()):
            yield vertex, successor
        # Incoming edges
        for predecessor in self._in_adj.get(vertex, set()):
            yield predecessor, vertex

    def get_edge(self, u: V, v: V) -> Edge[V] | None:
        """Retrieves the directed Edge object from u to v, if it exists."""
        return self._edges.get((u, v))

    def out_degree(self, vertex: V) -> int:
        """Returns the out-degree of a vertex."""
        return len(self._out_adj.get(vertex, set()))

    def in_degree(self, vertex: V) -> int:
        """Returns the in-degree of a vertex."""
        return len(self._in_adj.get(vertex, set()))

    def degree(self, vertex: V) -> int:
        """Returns the total degree (in-degree + out-degree) of a vertex."""
        return self.in_degree(vertex) + self.out_degree(vertex)

    def is_directed(self) -> bool:
        return True

    def topological_sort(self) -> list[V] | None:
        """
        Performs a topological sort of the graph using Kahn's algorithm.
        Returns the sorted list of vertices or None if the graph has a cycle.
        """
        in_degree = {v: self.in_degree(v) for v in self._vertices}
        # Initialize the queue with all vertices having an in-degree of 0.
        queue = deque([v for v in self._vertices if in_degree[v] == 0])
        result = []

        while queue:
            vertex = queue.popleft()
            result.append(vertex)

            # For each neighbor, "remove" the edge by decrementing the in-degree.
            for successor in self.neighbors(vertex):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        # If the result count matches the vertex count, the sort is valid.
        # Otherwise, a cycle was detected.
        return result if len(result) == len(self._vertices) else None

    def __getitem__(self, vertex: V) -> tuple[set[V], set[V]]:
        """Returns a tuple of (out-neighbors, in-neighbors) for a vertex."""
        out_neighbors = self._out_adj.get(vertex, set()).copy()
        in_neighbors = self._in_adj.get(vertex, set()).copy()
        return out_neighbors, in_neighbors


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
