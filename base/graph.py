from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

# Type variables
V = TypeVar("V")  # Vertex type
W = TypeVar("W")  # Weight type


@dataclass(frozen=True, eq=True, slots=True)
class Edge(Generic[V, W]):
    """
    An immutable directed edge (u -> v) with an optional weight.

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
        """Return edge with reversed direction."""
        return Edge(self.v, self.u, self.weight)


# Exception hierarchy
class GraphError(Exception): ...


class VertexNotFoundError(GraphError): ...


class EdgeNotFoundError(GraphError): ...


class Graph(ABC, Generic[V, W]):
    """
    Abstract graph base class supporting both directed and undirected graphs.

    Provides efficient vertex/edge management and common graph algorithms.
    Uses generic types for vertices (V) and edge weights (W).
    """

    __slots__ = ("_vertices",)

    def __init__(self) -> None:
        self._vertices: set[V] = set()

    # Core Properties
    @property
    def vertices(self) -> frozenset[V]:
        """Immutable view of all vertices."""
        return frozenset(self._vertices)

    @property
    def vertex_count(self) -> int:
        """Number of vertices in the graph."""
        return len(self._vertices)

    @property
    @abstractmethod
    def edge_count(self) -> int:
        """Number of edges in the graph."""

    @property
    def is_empty(self) -> bool:
        """True if graph has no vertices."""
        return len(self._vertices) == 0

    @property
    def density(self) -> float:
        """Graph density (ratio of actual edges to maximum possible edges)."""
        n = self.vertex_count
        if n < 2:
            return 0.0

        max_edges = n * (n - 1)
        if not self.is_directed():
            max_edges //= 2

        return self.edge_count / max_edges

    # Vertex Operations
    def add_vertex(self, vertex: V) -> bool:
        """Add vertex if not exists. Returns True if added."""
        if vertex in self._vertices:
            return False
        self._vertices.add(vertex)
        self._on_vertex_added(vertex)
        return True

    def remove_vertex(self, vertex: V) -> bool:
        """Remove vertex and all incident edges. Returns True if removed."""
        if vertex not in self._vertices:
            return False

        # Remove all incident edges before removing vertex
        for edge in list(self.incident_edges(vertex)):
            self.remove_edge(edge.u, edge.v)

        self._vertices.remove(vertex)
        self._on_vertex_removed(vertex)
        return True

    def add_vertices(self, vertices: Iterable[V]) -> int:
        """Add multiple vertices. Returns count of vertices actually added."""
        return sum(self.add_vertex(v) for v in vertices)

    def has_vertex(self, vertex: V) -> bool:
        """Check if vertex exists."""
        return vertex in self._vertices

    # Edge Operations
    def add_edge(self, u: V, v: V, weight: W = 1) -> bool:
        """Add edge with automatic vertex creation. Returns True if added."""
        self.add_vertex(u)
        self.add_vertex(v)

        if self.has_edge(u, v):
            return False

        self._on_edge_added(u, v, weight)
        return True

    def remove_edge(self, u: V, v: V) -> bool:
        """Remove edge. Returns True if removed."""
        if not self.has_edge(u, v):
            return False
        self._on_edge_removed(u, v)
        return True

    def add_edges(self, edges: Iterable[tuple[V, V] | tuple[V, V, W]]) -> int:
        """Add multiple edges. Returns count of edges actually added."""
        count = 0
        for edge_data in edges:
            u, v, *rest = edge_data
            weight = rest[0] if rest else 1
            if self.add_edge(u, v, weight):
                count += 1
        return count

    def get_edge_weight(self, u: V, v: V) -> W:
        """Get edge weight. Raises EdgeNotFoundError if edge not found."""
        edge = self.get_edge(u, v)
        if edge is None:
            raise EdgeNotFoundError(f"Edge ({u}, {v}) not found")
        return edge.weight

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph to dictionary format."""
        edges_data = []
        processed_edges = set()

        for vertex in self._vertices:
            for edge in self.incident_edges(vertex):
                edge_key = (edge.u, edge.v) if self.is_directed() else tuple(sorted([edge.u, edge.v]))
                if edge_key not in processed_edges:
                    edges_data.append({"u": edge.u, "v": edge.v, "weight": edge.weight})
                    processed_edges.add(edge_key)

        return {
            "type": "directed" if self.is_directed() else "undirected",
            "vertices": list(self._vertices),
            "edges": edges_data,
        }

    def to_json(self, **kwargs) -> str:
        """Serialize graph to JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Graph[V, W]:
        """Deserialize graph from dictionary."""
        graph = cls()
        graph.add_vertices(data["vertices"])

        for edge_data in data["edges"]:
            graph.add_edge(edge_data["u"], edge_data["v"], edge_data.get("weight"))

        return graph

    @classmethod
    def from_json(cls, json_str: str) -> Graph[V, W]:
        """Deserialize graph from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # Abstract methods for subclasses
    @abstractmethod
    def _on_vertex_added(self, vertex: V) -> None:
        """Handle vertex addition in subclass."""

    @abstractmethod
    def _on_vertex_removed(self, vertex: V) -> None:
        """Handle vertex removal in subclass."""

    @abstractmethod
    def _on_edge_added(self, u: V, v: V, weight: W) -> None:
        """Handle edge addition in subclass."""

    @abstractmethod
    def _on_edge_removed(self, u: V, v: V) -> None:
        """Handle edge removal in subclass."""

    @abstractmethod
    def has_edge(self, u: V, v: V) -> bool:
        """Check if edge exists."""

    @abstractmethod
    def neighbors(self, vertex: V) -> Iterator[V]:
        """Iterate over vertex neighbors."""

    @abstractmethod
    def incident_edges(self, vertex: V) -> Iterator[Edge[V, W]]:
        """Iterate over all edges incident to vertex."""

    @abstractmethod
    def get_edge(self, u: V, v: V) -> Edge[V, W] | None:
        """Get edge object or None if not found."""

    @abstractmethod
    def is_directed(self) -> bool:
        """True if graph is directed."""

    # Graph Algorithms
    def dfs(self, start: V, visited: set[V] | None = None) -> Iterator[V]:
        """
        Depth-first traversal yielding vertices in visit order.

        Args:
            start: Starting vertex
            visited: Pre-existing visited set (for multi-component traversal)
        """
        if start not in self._vertices:
            return

        if visited is None:
            visited = set()

        if start in visited:
            return

        stack = [start]
        while stack:
            vertex = stack.pop()
            if vertex in visited:
                continue

            visited.add(vertex)
            yield vertex

            # Add neighbors in reverse order for consistent traversal
            for neighbor in reversed(list(self.neighbors(vertex))):
                if neighbor not in visited:
                    stack.append(neighbor)

    def bfs(self, start: V, visited: set[V] | None = None) -> Iterator[V]:
        """
        Breadth-first traversal yielding vertices in visit order.

        Args:
            start: Starting vertex
            visited: Pre-existing visited set (for multi-component traversal)
        """
        if start not in self._vertices:
            return

        if visited is None:
            visited = set()

        if start in visited:
            return

        visited.add(start)
        queue = deque([start])

        while queue:
            vertex = queue.popleft()
            yield vertex

            for neighbor in self.neighbors(vertex):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    def connected_components(self) -> Iterator[list[V]]:
        """
        Yield connected components as lists.
        Memory-efficient for large graphs.
        """
        visited = set()
        for vertex in self._vertices:
            if vertex not in visited:
                yield list(self.dfs(vertex, visited))

    def is_connected(self) -> bool:
        """True if graph is connected (single component)."""
        if self.is_empty:
            return True
        return len(list(self.connected_components())) <= 1

    def has_path(self, start: V, end: V) -> bool:
        """True if path exists between vertices."""
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

    # Python protocols
    def __contains__(self, vertex: V) -> bool:
        """Support 'vertex in graph' syntax."""
        return self.has_vertex(vertex)

    def __len__(self) -> int:
        """Return vertex count."""
        return self.vertex_count

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vertices={self.vertex_count}, edges={self.edge_count}, density={self.density:.3f})"


class UndirectedGraph(Graph[V, W]):
    """
    Undirected graph using adjacency lists.

    Edges are bidirectional and stored once using frozenset keys.
    Provides O(1) edge operations and O(degree) neighbor iteration.
    """

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
        self._edges[frozenset({u, v})] = Edge(u, v, weight)

    def _on_edge_removed(self, u: V, v: V) -> None:
        self._adjacency[u].discard(v)
        self._adjacency[v].discard(u)
        self._edges.pop(frozenset({u, v}), None)

    def has_edge(self, u: V, v: V) -> bool:
        return frozenset({u, v}) in self._edges

    def neighbors(self, vertex: V) -> Iterator[V]:
        return iter(self._adjacency.get(vertex, set()))

    def incident_edges(self, vertex: V) -> Iterator[Edge[V, W]]:
        """Iterate over edges connected to vertex."""
        vertex_set = {vertex}
        for edge_key, edge in self._edges.items():
            if edge_key & vertex_set:  # Intersection check
                yield edge

    def get_edge(self, u: V, v: V) -> Edge[V, W] | None:
        return self._edges.get(frozenset({u, v}))

    def degree(self, vertex: V) -> int:
        """Vertex degree (number of incident edges)."""
        neighbors = self._adjacency.get(vertex, set())
        # Count self-loops twice
        return len(neighbors) + (1 if vertex in neighbors else 0)

    def is_directed(self) -> bool:
        return False

    def __getitem__(self, vertex: V) -> frozenset[V]:
        """Get neighbors as immutable set."""
        return frozenset(self._adjacency.get(vertex, set()))


class DirectedGraph(Graph[V, W]):
    """
    Directed graph using separate in/out adjacency lists.

    Maintains both successors and predecessors for efficient queries.
    Provides O(1) edge operations and supports DAG algorithms.
    """

    __slots__ = ("_successors", "_predecessors", "_edges")

    def __init__(self) -> None:
        super().__init__()
        self._successors: dict[V, set[V]] = defaultdict(set)
        self._predecessors: dict[V, set[V]] = defaultdict(set)
        self._edges: dict[tuple[V, V], Edge[V, W]] = {}

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def _on_vertex_added(self, vertex: V) -> None:
        self._successors.setdefault(vertex, set())
        self._predecessors.setdefault(vertex, set())

    def _on_vertex_removed(self, vertex: V) -> None:
        self._successors.pop(vertex, None)
        self._predecessors.pop(vertex, None)

    def _on_edge_added(self, u: V, v: V, weight: W) -> None:
        self._successors[u].add(v)
        self._predecessors[v].add(u)
        self._edges[(u, v)] = Edge(u, v, weight)

    def _on_edge_removed(self, u: V, v: V) -> None:
        self._successors[u].discard(v)
        self._predecessors[v].discard(u)
        self._edges.pop((u, v), None)

    def has_edge(self, u: V, v: V) -> bool:
        return (u, v) in self._edges

    def neighbors(self, vertex: V) -> Iterator[V]:
        """Iterate over successors (outgoing neighbors)."""
        return iter(self._successors.get(vertex, set()))

    def predecessors(self, vertex: V) -> Iterator[V]:
        """Iterate over predecessors (incoming neighbors)."""
        return iter(self._predecessors.get(vertex, set()))

    def incident_edges(self, vertex: V) -> Iterator[Edge[V, W]]:
        """Iterate over all incident edges (incoming + outgoing)."""
        # Outgoing edges
        for v in self._successors.get(vertex, set()):
            yield self._edges[(vertex, v)]
        # Incoming edges
        for u in self._predecessors.get(vertex, set()):
            yield self._edges[(u, vertex)]

    def get_edge(self, u: V, v: V) -> Edge[V, W] | None:
        return self._edges.get((u, v))

    def out_degree(self, vertex: V) -> int:
        """Number of outgoing edges."""
        return len(self._successors.get(vertex, set()))

    def in_degree(self, vertex: V) -> int:
        """Number of incoming edges."""
        return len(self._predecessors.get(vertex, set()))

    def degree(self, vertex: V) -> int:
        """Total degree (in + out)."""
        return self.in_degree(vertex) + self.out_degree(vertex)

    def is_directed(self) -> bool:
        return True

    def topological_sort(self) -> list[V] | None:
        """
        Return topological ordering or None if graph has cycles.
        Uses Kahn's algorithm for O(V + E) complexity.
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
        """True if graph is a DAG (Directed Acyclic Graph)."""
        return self.topological_sort() is not None

    def strongly_connected_components(self) -> Iterator[list[V]]:
        """
        Find strongly connected components using an iterative Tarjan's algorithm.

        Yields each component as a list of vertices, making it memory-efficient
        for graphs with many components. Complexity: O(V + E).
        """
        index_counter = 0
        scc_stack: list[V] = []
        lowlinks: dict[V, int] = {}
        index: dict[V, int] = {}
        on_stack: dict[V, bool] = defaultdict(bool)

        # To manage the DFS traversal iteratively, we need to store the state
        # of neighbor iteration for each vertex on the traversal path.
        neighbor_iters: dict[V, Iterator[V]] = {}

        for vertex in self._vertices:
            if vertex in index:
                continue  # Skip already visited vertices

            # Start a new traversal from this unvisited vertex
            traversal_stack: list[V] = [vertex]

            while traversal_stack:
                v = traversal_stack[-1]  # Peek at the top of the stack

                # This is the first time we are seeing `v` in the current DFS path.
                # This corresponds to the pre-order actions in a recursive DFS.
                if v not in neighbor_iters:
                    index[v] = index_counter
                    lowlinks[v] = index_counter
                    index_counter += 1
                    scc_stack.append(v)
                    on_stack[v] = True
                    neighbor_iters[v] = self.neighbors(v)

                # Explore the next neighbor of `v`
                try:
                    neighbor = next(neighbor_iters[v])

                    if neighbor not in index:
                        # If neighbor is unvisited, push it to the stack to visit next
                        traversal_stack.append(neighbor)
                    elif on_stack[neighbor]:
                        # If neighbor is visited and on the scc_stack, update lowlink
                        lowlinks[v] = min(lowlinks[v], index[neighbor])

                except StopIteration:
                    # All neighbors of `v` have been visited (post-order actions).
                    traversal_stack.pop()

                    if lowlinks[v] == index[v]:
                        # `v` is the root of a strongly connected component
                        component = []
                        while True:
                            w = scc_stack.pop()
                            on_stack[w] = False
                            component.append(w)
                            if w == v:
                                break
                        yield component

                    # After processing `v` and its children, if there's a parent on the
                    # traversal stack, update its lowlink with `v`'s finalized lowlink.
                    if traversal_stack:
                        parent = traversal_stack[-1]
                        lowlinks[parent] = min(lowlinks[parent], lowlinks[v])

    def reverse(self) -> DirectedGraph[V, W]:
        """Return new graph with all edges reversed."""
        reversed_graph = DirectedGraph[V, W]()
        reversed_graph.add_vertices(self._vertices)
        for edge in self._edges.values():
            reversed_graph.add_edge(edge.v, edge.u, edge.weight)
        return reversed_graph

    def __getitem__(self, vertex: V) -> dict[str, frozenset[V]]:
        """Get predecessors and successors as immutable sets."""
        return {
            "in": frozenset(self._predecessors.get(vertex, set())),
            "out": frozenset(self._successors.get(vertex, set())),
        }


def create_graph(directed: bool = False) -> DirectedGraph | UndirectedGraph:
    """
    Factory function for graph creation.

    Note: Direct instantiation (e.g., UndirectedGraph[str, int]())
    preserves type information and is preferred.
    """
    return DirectedGraph() if directed else UndirectedGraph()


def demo() -> None:
    """Demonstrate graph functionality with comprehensive examples."""
    print("=== Graph Data Structure Demo ===\n")

    # Undirected graph example
    print("--- Undirected Graph ---")
    ug = UndirectedGraph[str, int]()

    # Add edges with weights
    social_network = [("Alice", "Bob", 5), ("Alice", "Charlie", 3), ("Bob", "David", 7), ("Charlie", "David", 2)]
    added = ug.add_edges(social_network)
    print(f"Created social network: {added} edges added")
    print(f"Network info: {ug}")
    print(f"Alice's friends: {list(ug.neighbors('Alice'))}")
    print(f"Alice's degree: {ug.degree('Alice')}")
    print(f"Is connected: {ug.is_connected()}")

    # Directed graph example
    print("\n--- Directed Graph (Task Dependencies) ---")
    dg = DirectedGraph[str, int]()

    tasks = [
        ("Design", "Code"),
        ("Code", "Test"),
        ("Test", "Deploy"),
        ("Design", "Documentation"),
        ("Documentation", "Deploy"),
    ]
    dg.add_edges(tasks)

    print(f"Task graph: {dg}")
    print(f"Design leads to: {list(dg.neighbors('Design'))}")
    print(f"Deploy depends on: {list(dg.predecessors('Deploy'))}")

    schedule = dg.topological_sort()
    print(f"Task execution order: {schedule}")
    print(f"Is valid workflow (no cycles): {dg.is_acyclic()}")

    # Strongly Connected Components example
    print("\n--- Strongly Connected Components ---")
    scc_graph = DirectedGraph[int, int]()
    scc_edges = [(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 6), (6, 4), (6, 7)]
    scc_graph.add_edges(scc_edges)
    scc_graph.add_vertex(8)  # Add a disconnected vertex
    print(f"SCC Graph: {scc_graph}")

    # Using the new iterative method
    components = list(scc_graph.strongly_connected_components())
    print(f"Strongly connected components (iterative): {components}")

    # Traversal algorithms
    print("\n--- Graph Traversal ---")
    print(f"DFS from Alice: {list(ug.dfs('Alice'))}")
    print(f"BFS from Design: {list(dg.bfs('Design'))}")

    # Connectivity analysis
    print(f"\nSocial network components: {list(ug.connected_components())}")
    print(f"Path Alice→David: {ug.has_path('Alice', 'David')}")


if __name__ == "__main__":
    demo()
