from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Iterator, Iterable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, Optional, Union

# --- Type variable definitions ---
V = TypeVar("V")  # Type for vertices
W = TypeVar("W")  # Type for edge weights


@dataclass(frozen=True, eq=True, slots=True)
class Edge(Generic[V, W]):
    """
    Represents an immutable directed edge (u -> v) with an optional weight.

    Args:
        u: The source vertex.
        v: The destination vertex.
        weight: The weight of the edge, defaults to 1.
    """
    u: V
    v: V
    weight: W = 1

    def __str__(self) -> str:
        return f"{self.u} --({self.weight})--> {self.v}"

    def reverse(self) -> Edge[V, W]:
        """Returns the reversed edge."""
        return Edge(self.v, self.u, self.weight)


class GraphError(Exception):
    """Base class for exceptions related to graph operations."""
    pass


class VertexNotFoundError(GraphError):
    """Raised when a vertex is not found in the graph."""
    pass


class EdgeNotFoundError(GraphError):
    """Raised when an edge is not found in the graph."""
    pass


class Graph(ABC, Generic[V, W]):
    """
    Abstract base class for a graph (directed or undirected).
    Provides the core interface for graph operations and common algorithm implementations.
    """

    __slots__ = ("_vertices",)

    def __init__(self) -> None:
        self._vertices: set[V] = set()

    # --- Vertex Management ---

    def add_vertex(self, vertex: V) -> bool:
        """
        Adds a vertex to the graph if it does not already exist.

        Args:
            vertex: The vertex to add.

        Returns:
            True if the vertex was added, False if it already existed.
        """
        if vertex in self._vertices:
            return False
        self._vertices.add(vertex)
        self._on_vertex_added(vertex)
        return True

    def remove_vertex(self, vertex: V) -> bool:
        """
        Removes a vertex and all its incident edges from the graph.

        Args:
            vertex: The vertex to remove.

        Returns:
            True if the vertex was removed, False if it did not exist.
        """
        if vertex not in self._vertices:
            return False

        # Collect incident edges first to avoid modification during iteration.
        incident_edges = list(self.get_incident_edges(vertex))
        for edge in incident_edges:
            self.remove_edge(edge.u, edge.v)

        self._vertices.remove(vertex)
        self._on_vertex_removed(vertex)
        return True

    def add_vertices(self, vertices: Iterable[V]) -> int:
        """
        Adds multiple vertices to the graph.

        Args:
            vertices: An iterable of vertices to add.

        Returns:
            The number of vertices that were actually added.
        """
        count = 0
        for vertex in vertices:
            if self.add_vertex(vertex):
                count += 1
        return count

    # --- Edge Management ---

    def add_edge(self, u: V, v: V, weight: W = 1) -> bool:
        """
        Adds an edge (u -> v) to the graph.

        Args:
            u: The source vertex.
            v: The destination vertex.
            weight: The weight of the edge.

        Returns:
            True if the edge was added, False if it already existed.
        """
        # Automatically add vertices if they don't exist
        self.add_vertex(u)
        self.add_vertex(v)

        if self.has_edge(u, v):
            return False

        self._on_edge_added(u, v, weight)
        return True

    def remove_edge(self, u: V, v: V) -> bool:
        """
        Removes the edge (u -> v) from the graph.

        Args:
            u: The source vertex.
            v: The destination vertex.

        Returns:
            True if the edge was removed, False if it did not exist.
        """
        if not self.has_edge(u, v):
            return False

        self._on_edge_removed(u, v)
        return True

    def add_edges(self, edges: Iterable[tuple[V, V] | tuple[V, V, W]]) -> int:
        """
        Adds multiple edges to the graph.

        Args:
            edges: An iterable of edges, where each element is (u, v) or (u, v, weight).

        Returns:
            The number of edges that were actually added.
        """
        count = 0
        for edge_data in edges:
            u, v, *rest = edge_data
            weight = rest[0] if rest else 1

            if self.add_edge(u, v, weight):
                count += 1
        return count

    # --- Property Queries ---

    @property
    def vertices(self) -> frozenset[V]:
        """Returns an immutable view of the set of vertices."""
        return frozenset(self._vertices)

    @property
    def vertex_count(self) -> int:
        """Returns the number of vertices in the graph."""
        return len(self._vertices)

    @property
    @abstractmethod
    def edge_count(self) -> int:
        """Returns the number of edges in the graph."""
        ...

    def is_empty(self) -> bool:
        """Checks if the graph is empty (has no vertices)."""
        return len(self._vertices) == 0

    def has_vertex(self, vertex: V) -> bool:
        """Checks if a vertex exists in the graph."""
        return vertex in self._vertices

    def get_edge_weight(self, u: V, v: V) -> W:
        """
        Gets the weight of an edge.

        Args:
            u: The source vertex.
            v: The destination vertex.

        Returns:
            The weight of the edge.

        Raises:
            EdgeNotFoundError: If the edge does not exist.
        """
        edge = self.get_edge(u, v)
        if edge is None:
            raise EdgeNotFoundError(f"Edge ({u}, {v}) not found")
        return edge.weight

    # --- Abstract Methods (for subclass implementation) ---

    @abstractmethod
    def _on_vertex_added(self, vertex: V) -> None:
        """Callback executed when a vertex is added."""
        ...

    @abstractmethod
    def _on_vertex_removed(self, vertex: V) -> None:
        """Callback executed when a vertex is removed."""
        ...

    @abstractmethod
    def _on_edge_added(self, u: V, v: V, weight: W) -> None:
        """Callback executed when an edge is added."""
        ...

    @abstractmethod
    def _on_edge_removed(self, u: V, v: V) -> None:
        """Callback executed when an edge is removed."""
        ...

    @abstractmethod
    def has_edge(self, u: V, v: V) -> bool:
        """Checks if an edge exists between two vertices."""
        ...

    @abstractmethod
    def neighbors(self, vertex: V) -> Iterator[V]:
        """Returns an iterator over the neighbors of a vertex."""
        ...

    @abstractmethod
    def get_incident_edges(self, vertex: V) -> Iterator[Edge[V, W]]:
        """
        Returns an iterator over all edges incident to a vertex.

        Warning:
            If you need to modify the graph during iteration, convert to a list() first.
        """
        ...

    @abstractmethod
    def get_edge(self, u: V, v: V) -> Optional[Edge[V, W]]:
        """Returns the Edge object between two vertices, or None if it doesn't exist."""
        ...

    @abstractmethod
    def is_directed(self) -> bool:
        """Returns True if the graph is directed, False otherwise."""
        ...

    # --- Graph Algorithms ---

    def dfs(self, start: V, visited: Optional[set[V]] = None) -> list[V]:
        """
        Performs a Depth-First Search traversal.

        Args:
            start: The starting vertex for the traversal.
            visited: A set of already visited vertices (for multi-component traversals).

        Returns:
            A list of vertices in DFS traversal order.
        """
        if start not in self._vertices:
            return []

        if visited is None:
            visited = set()
        result = []

        def _dfs_recursive(vertex: V) -> None:
            visited.add(vertex)
            result.append(vertex)
            for neighbor in self.neighbors(vertex):
                if neighbor not in visited:
                    _dfs_recursive(neighbor)

        if start not in visited:
            _dfs_recursive(start)
        return result

    def bfs(self, start: V) -> list[V]:
        """
        Performs a Breadth-First Search traversal.

        Args:
            start: The starting vertex for the traversal.

        Returns:
            A list of vertices in BFS traversal order.
        """
        if start not in self._vertices:
            return []

        visited = {start}
        queue = deque([start])
        result = []

        while queue:
            vertex = queue.popleft()
            result.append(vertex)

            for neighbor in self.neighbors(vertex):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return result

    def connected_components(self) -> list[list[V]]:
        """
        Finds all connected components in the graph.

        Returns:
            A list of lists, where each inner list contains the vertices of a component.
        """
        visited = set()
        components = []

        for vertex in self._vertices:
            if vertex not in visited:
                component = self.dfs(vertex, visited)
                if component:
                    components.append(component)

        return components

    def is_connected(self) -> bool:
        """Checks if the graph is connected."""
        if self.is_empty():
            return True
        return len(self.connected_components()) <= 1

    def has_path(self, start: V, end: V) -> bool:
        """
        Checks if a path exists between two vertices.

        Args:
            start: The starting vertex.
            end: The ending vertex.

        Returns:
            True if a path exists, False otherwise.
        """
        if start not in self._vertices or end not in self._vertices:
            return False
        if start == end:
            return True

        visited = {start}
        queue = deque([start])

        while queue:
            vertex = queue.popleft()
            for neighbor in self.neighbors(vertex):
                if neighbor == end:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    # --- Dunder Methods ---

    def __contains__(self, vertex: V) -> bool:
        """Supports 'vertex in graph' syntax."""
        return self.has_vertex(vertex)

    def __len__(self) -> int:
        """Returns the number of vertices."""
        return self.vertex_count

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        directed = "directed" if self.is_directed() else "undirected"
        return f"{class_name}({directed}, vertices={self.vertex_count}, edges={self.edge_count})"

    def __str__(self) -> str:
        return self.__repr__()


class UndirectedGraph(Graph[V, W]):
    """An undirected graph implementation using an adjacency list."""

    __slots__ = ("_adjacency", "_edges")

    def __init__(self) -> None:
        super().__init__()
        self._adjacency: dict[V, set[V]] = defaultdict(set)
        self._edges: dict[frozenset[V], Edge[V, W]] = {}

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def _on_vertex_added(self, vertex: V) -> None:
        self._adjacency.setdefault(vertex, set())

    def _on_vertex_removed(self, vertex: V) -> None:
        self._adjacency.pop(vertex, None)

    def _on_edge_added(self, u: V, v: V, weight: W) -> None:
        self._adjacency[u].add(v)
        self._adjacency[v].add(u)
        edge_key = frozenset({u, v})
        self._edges[edge_key] = Edge(u, v, weight)

    def _on_edge_removed(self, u: V, v: V) -> None:
        self._adjacency[u].discard(v)
        self._adjacency[v].discard(u)
        edge_key = frozenset({u, v})
        self._edges.pop(edge_key, None)

    def has_edge(self, u: V, v: V) -> bool:
        return frozenset({u, v}) in self._edges

    def neighbors(self, vertex: V) -> Iterator[V]:
        """Returns an iterator over the neighbors of a vertex."""
        return iter(self._adjacency.get(vertex, set()))

    def get_incident_edges(self, vertex: V) -> Iterator[Edge[V, W]]:
        """Returns an iterator over all edges connected to the vertex."""
        for neighbor in self._adjacency.get(vertex, set()):
            edge = self.get_edge(vertex, neighbor)
            if edge is not None:
                yield edge

    def get_edge(self, u: V, v: V) -> Optional[Edge[V, W]]:
        edge_key = frozenset({u, v})
        return self._edges.get(edge_key)

    def degree(self, vertex: V) -> int:
        """Returns the degree of a vertex."""
        return len(self._adjacency.get(vertex, set()))

    def is_directed(self) -> bool:
        return False

    def __getitem__(self, vertex: V) -> frozenset[V]:
        """Returns an immutable set of neighbors for a vertex."""
        return frozenset(self._adjacency.get(vertex, set()))


class DirectedGraph(Graph[V, W]):
    """A directed graph implementation using separate adjacency lists for successors and predecessors."""

    __slots__ = ("_out_adjacency", "_in_adjacency", "_edges")

    def __init__(self) -> None:
        super().__init__()
        self._out_adjacency: dict[V, set[V]] = defaultdict(set)
        self._in_adjacency: dict[V, set[V]] = defaultdict(set)
        self._edges: dict[tuple[V, V], Edge[V, W]] = {}

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def _on_vertex_added(self, vertex: V) -> None:
        self._out_adjacency.setdefault(vertex, set())
        self._in_adjacency.setdefault(vertex, set())

    def _on_vertex_removed(self, vertex: V) -> None:
        self._out_adjacency.pop(vertex, None)
        self._in_adjacency.pop(vertex, None)

    def _on_edge_added(self, u: V, v: V, weight: W) -> None:
        self._out_adjacency[u].add(v)
        self._in_adjacency[v].add(u)
        self._edges[(u, v)] = Edge(u, v, weight)

    def _on_edge_removed(self, u: V, v: V) -> None:
        self._out_adjacency[u].discard(v)
        self._in_adjacency[v].discard(u)
        self._edges.pop((u, v), None)

    def has_edge(self, u: V, v: V) -> bool:
        return (u, v) in self._edges

    def neighbors(self, vertex: V) -> Iterator[V]:
        """Returns an iterator over the successors (out-neighbors) of a vertex."""
        return iter(self._out_adjacency.get(vertex, set()))

    def predecessors(self, vertex: V) -> Iterator[V]:
        """Returns an iterator over the predecessors (in-neighbors) of a vertex."""
        return iter(self._in_adjacency.get(vertex, set()))

    def get_incident_edges(self, vertex: V) -> Iterator[Edge[V, W]]:
        """Returns an iterator over all incoming and outgoing edges for a vertex."""
        # Outgoing edges
        for successor in self._out_adjacency.get(vertex, set()):
            edge = self.get_edge(vertex, successor)
            if edge is not None:
                yield edge
        # Incoming edges
        for predecessor in self._in_adjacency.get(vertex, set()):
            edge = self.get_edge(predecessor, vertex)
            if edge is not None:
                yield edge

    def get_edge(self, u: V, v: V) -> Optional[Edge[V, W]]:
        return self._edges.get((u, v))

    def out_degree(self, vertex: V) -> int:
        """Returns the out-degree of a vertex."""
        return len(self._out_adjacency.get(vertex, set()))

    def in_degree(self, vertex: V) -> int:
        """Returns the in-degree of a vertex."""
        return len(self._in_adjacency.get(vertex, set()))

    def degree(self, vertex: V) -> int:
        """Returns the total degree (in-degree + out-degree) of a vertex."""
        return self.in_degree(vertex) + self.out_degree(vertex)

    def is_directed(self) -> bool:
        return True

    def topological_sort(self) -> Optional[list[V]]:
        """
        Performs a topological sort of the graph.

        Returns:
            A list of vertices in topological order, or None if the graph has a cycle.
        """
        in_degree = {v: self.in_degree(v) for v in self._vertices}
        queue = deque(v for v, deg in in_degree.items() if deg == 0)
        result = []

        while queue:
            vertex = queue.popleft()
            result.append(vertex)
            for successor in self.neighbors(vertex):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        return result if len(result) == len(self._vertices) else None

    def is_acyclic(self) -> bool:
        """Checks if the directed graph is acyclic (is a DAG)."""
        return self.topological_sort() is not None

    def reverse(self) -> DirectedGraph[V, W]:
        """Returns a new graph with all edge directions reversed."""
        reversed_graph = DirectedGraph[V, W]()
        reversed_graph.add_vertices(self._vertices)
        for edge in self._edges.values():
            reversed_graph.add_edge(edge.v, edge.u, edge.weight)
        return reversed_graph

    def __getitem__(self, vertex: V) -> dict[str, frozenset[V]]:
        """Returns the sets of in-neighbors and out-neighbors for a vertex."""
        return {
            "out": frozenset(self._out_adjacency.get(vertex, set())),
            "in": frozenset(self._in_adjacency.get(vertex, set())),
        }


# --- Factory Function ---

def create_graph(directed: bool = False) -> Union[DirectedGraph, UndirectedGraph]:
    """
    Factory function to create a graph.
    Note: Type information is lost. Prefer direct instantiation like `UndirectedGraph[str, int]()`.
    """
    return DirectedGraph() if directed else UndirectedGraph()


# --- Demo ---

def demo() -> None:
    """Demonstrates the functionality of the graph data structures."""
    print("=== Optimized Graph Data Structure Demo ===\n")

    # --- Undirected Graph Demo ---
    print("--- Undirected Graph ---")
    ug = UndirectedGraph[str, int]()

    edges = [("A", "B", 5), ("A", "C", 3), ("B", "D", 7), ("C", "D", 2)]
    added_count = ug.add_edges(edges)
    print(f"Added {added_count} edges.")

    print(f"Graph info: {ug}")
    print(f"Vertices: {ug.vertices}")
    print(f"Neighbors of 'A': {list(ug.neighbors('A'))}")
    print(f"Degree of 'A': {ug.degree('A')}")
    print(f"Is connected: {ug.is_connected()}")

    try:
        weight = ug.get_edge_weight("A", "B")
        print(f"Weight of edge (A,B): {weight}")
    except EdgeNotFoundError as e:
        print(f"Error: {e}")

    # --- Directed Graph Demo ---
    print("\n--- Directed Graph ---")
    dg = DirectedGraph[str, int]()

    dag_edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]
    dg.add_edges(dag_edges)

    print(f"Graph info: {dg}")
    print(f"Successors of 'A': {list(dg.neighbors('A'))}")
    print(f"Predecessors of 'D': {list(dg.predecessors('D'))}")
    print(f"Degrees of 'B': out={dg.out_degree('B')}, in={dg.in_degree('B')}")
    print(f"Is DAG: {dg.is_acyclic()}")

    topo_order = dg.topological_sort()
    if topo_order:
        print(f"Topological order: {topo_order}")

    # --- Traversal Algorithms ---
    print(f"\n--- Graph Traversal Algorithms ---")
    print(f"DFS from 'A' in undirected graph: {ug.dfs('A')}")
    print(f"BFS from 'A' in directed graph: {dg.bfs('A')}")

    # --- Connectivity ---
    print(f"\n--- Connectivity ---")
    print(f"Connected components in undirected graph: {ug.connected_components()}")
    print(f"Path exists from 'A' to 'D': {ug.has_path('A', 'D')}")


if __name__ == "__main__":
    demo()
