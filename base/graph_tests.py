from dataclasses import dataclass

import pytest
from graph import (
    DirectedGraph,
    Edge,
    EdgeNotFoundError,
    UndirectedGraph,
    create_graph,
)


# Test data classes for generic testing
@dataclass(frozen=True)
class Person:
    name: str
    age: int


@dataclass(frozen=True)
class Distance:
    value: float
    unit: str = "km"

    def __post_init__(self):
        if self.value < 0:
            raise ValueError("Distance cannot be negative")


class TestEdge:
    """Test Edge class functionality."""

    def test_edge_creation(self):
        """Test basic edge creation."""
        edge = Edge("A", "B", 5)
        assert edge.u == "A"
        assert edge.v == "B"
        assert edge.weight == 5

    def test_edge_default_weight(self):
        """Test edge with default weight."""
        edge = Edge("A", "B")
        assert edge.weight == 1

    def test_edge_generic_types(self):
        """Test edge with different generic types."""
        # String vertices, float weight
        edge1 = Edge[str, float]("A", "B", 3.14)
        assert isinstance(edge1.weight, float)

        # Integer vertices, custom weight type
        distance = Distance(10.5, "miles")
        edge2 = Edge[int, Distance](1, 2, distance)
        assert edge2.weight.value == 10.5
        assert edge2.weight.unit == "miles"

    def test_edge_immutability(self):
        """Test that edges are immutable."""
        edge = Edge("A", "B", 5)
        with pytest.raises(AttributeError):
            edge.u = "C"  # type: ignore
        with pytest.raises(AttributeError):
            edge.weight = 10  # type: ignore

    def test_edge_equality(self):
        """Test edge equality."""
        edge1 = Edge("A", "B", 5)
        edge2 = Edge("A", "B", 5)
        edge3 = Edge("A", "B", 10)
        edge4 = Edge("B", "A", 5)

        assert edge1 == edge2
        assert edge1 != edge3
        assert edge1 != edge4

    def test_edge_str_representation(self):
        """Test edge string representation."""
        edge = Edge("A", "B", 5)
        assert str(edge) == "A --(5)--> B"

    def test_edge_reverse(self):
        """Test edge reversal."""
        edge = Edge("A", "B", 5)
        reversed_edge = edge.reverse()
        assert reversed_edge.u == "B"
        assert reversed_edge.v == "A"
        assert reversed_edge.weight == 5


class TestUndirectedGraph:
    """Test UndirectedGraph functionality."""

    def test_graph_creation(self):
        """Test basic graph creation."""
        graph = UndirectedGraph[str, int]()
        assert graph.vertex_count == 0
        assert graph.edge_count == 0
        assert graph.is_empty
        assert not graph.is_directed()

    def test_vertex_operations(self):
        """Test vertex addition and removal."""
        graph = UndirectedGraph[str, int]()

        # Add vertices
        assert graph.add_vertex("A") == True
        assert graph.add_vertex("B") == True
        assert graph.add_vertex("A") == False  # Already exists

        assert graph.vertex_count == 2
        assert "A" in graph
        assert "C" not in graph

        # Remove vertex
        assert graph.remove_vertex("A") == True
        assert graph.remove_vertex("A") == False  # Already removed
        assert graph.vertex_count == 1
        assert "A" not in graph

    def test_batch_vertex_operations(self):
        """Test batch vertex operations."""
        graph = UndirectedGraph[str, int]()
        vertices = ["A", "B", "C", "A"]  # Duplicate "A"

        added_count = graph.add_vertices(vertices)
        assert added_count == 3  # Only unique vertices counted
        assert graph.vertex_count == 3

    def test_edge_operations(self):
        """Test edge addition and removal."""
        graph = UndirectedGraph[str, int]()

        # Add edges
        assert graph.add_edge("A", "B", 5) == True
        assert graph.add_edge("B", "C", 10) == True
        assert graph.add_edge("A", "B", 15) == False  # Already exists

        assert graph.edge_count == 2
        assert graph.vertex_count == 3  # Auto-added vertices

        # Check edge existence
        assert graph.has_edge("A", "B") == True
        assert graph.has_edge("B", "A") == True  # Undirected
        assert graph.has_edge("A", "C") == False

        # Remove edge
        assert graph.remove_edge("A", "B") == True
        assert graph.remove_edge("A", "B") == False  # Already removed
        assert graph.edge_count == 1

    def test_batch_edge_operations(self):
        """Test batch edge operations."""
        graph = UndirectedGraph[str, int]()
        edges = [("A", "B"), ("B", "C", 10), ("A", "B")]  # Duplicate edge

        added_count = graph.add_edges(edges)
        assert added_count == 2  # Only unique edges counted
        assert graph.edge_count == 2

    def test_neighbors_and_degree(self):
        """Test neighbor iteration and degree calculation."""
        graph = UndirectedGraph[str, int]()
        graph.add_edge("A", "B", 5)
        graph.add_edge("A", "C", 10)
        graph.add_edge("B", "C", 15)

        neighbors_a = set(graph.neighbors("A"))
        assert neighbors_a == {"B", "C"}

        assert graph.degree("A") == 2
        assert graph.degree("B") == 2
        assert graph.degree("C") == 2
        assert graph.degree("D") == 0  # Non-existent vertex

    def test_incident_edges(self):
        """Test incident edge iteration."""
        graph = UndirectedGraph[str, int]()
        graph.add_edge("A", "B", 5)
        graph.add_edge("A", "C", 10)

        incident_edges = list(graph.incident_edges("A"))
        assert len(incident_edges) == 2

        edge_weights = {edge.weight for edge in incident_edges}
        assert edge_weights == {5, 10}

    def test_edge_retrieval(self):
        """Test edge retrieval and weight access."""
        graph = UndirectedGraph[str, int]()
        graph.add_edge("A", "B", 42)

        edge = graph.get_edge("A", "B")
        assert edge is not None
        assert edge.weight == 42

        # Undirected graph - both directions work
        edge_reverse = graph.get_edge("B", "A")
        assert edge_reverse is not None
        assert edge_reverse.weight == 42

        # Non-existent edge
        assert graph.get_edge("A", "C") is None

        # Test weight getter
        assert graph.get_edge_weight("A", "B") == 42

        with pytest.raises(EdgeNotFoundError):
            graph.get_edge_weight("A", "C")

    def test_vertex_removal_with_edges(self):
        """Test that removing vertex removes incident edges."""
        graph = UndirectedGraph[str, int]()
        graph.add_edge("A", "B", 5)
        graph.add_edge("A", "C", 10)
        graph.add_edge("B", "C", 15)

        initial_edge_count = graph.edge_count
        graph.remove_vertex("A")

        assert graph.edge_count == 1  # Only B-C remains
        assert not graph.has_edge("A", "B")
        assert not graph.has_edge("A", "C")
        assert graph.has_edge("B", "C")

    def test_graph_indexing(self):
        """Test graph indexing operations."""
        graph = UndirectedGraph[str, int]()
        graph.add_edge("A", "B", 5)
        graph.add_edge("A", "C", 10)

        neighbors = graph["A"]
        assert isinstance(neighbors, frozenset)
        assert neighbors == {"B", "C"}


class TestDirectedGraph:
    """Test DirectedGraph functionality."""

    def test_directed_graph_creation(self):
        """Test directed graph creation."""
        graph = DirectedGraph[str, int]()
        assert graph.is_directed()
        assert graph.vertex_count == 0
        assert graph.edge_count == 0

    def test_directed_edge_operations(self):
        """Test directed edge behavior."""
        graph = DirectedGraph[str, int]()
        graph.add_edge("A", "B", 5)

        assert graph.has_edge("A", "B") == True
        assert graph.has_edge("B", "A") == False  # Directed!

        # Test directional neighbors
        assert list(graph.neighbors("A")) == ["B"]
        assert list(graph.neighbors("B")) == []

        assert list(graph.predecessors("A")) == []
        assert list(graph.predecessors("B")) == ["A"]

    def test_degree_calculations(self):
        """Test degree calculations for directed graph."""
        graph = DirectedGraph[str, int]()
        graph.add_edge("A", "B", 5)
        graph.add_edge("A", "C", 10)
        graph.add_edge("D", "A", 15)

        assert graph.out_degree("A") == 2
        assert graph.in_degree("A") == 1
        assert graph.degree("A") == 3  # in + out

        assert graph.out_degree("B") == 0
        assert graph.in_degree("B") == 1

    def test_incident_edges_directed(self):
        """Test incident edges for directed graph."""
        graph = DirectedGraph[str, int]()
        graph.add_edge("A", "B", 5)
        graph.add_edge("C", "A", 10)

        incident_edges = list(graph.incident_edges("A"))
        assert len(incident_edges) == 2  # One outgoing, one incoming

        # Check that we get both directions
        edge_descriptions = [(e.u, e.v) for e in incident_edges]
        assert ("A", "B") in edge_descriptions
        assert ("C", "A") in edge_descriptions

    def test_topological_sort(self):
        """Test topological sorting."""
        graph = DirectedGraph[str, int]()

        # Create a DAG
        edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]
        graph.add_edges(edges)

        topo_order = graph.topological_sort()
        assert topo_order is not None
        assert len(topo_order) == 5

        # Verify topological properties
        topo_positions = {vertex: i for i, vertex in enumerate(topo_order)}
        for edge in edges:
            u, v = edge[:2]
            assert topo_positions[u] < topo_positions[v]

    def test_topological_sort_with_cycle(self):
        """Test topological sort with cyclic graph."""
        graph = DirectedGraph[str, int]()

        # Create a cycle
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "A")  # Creates cycle

        assert graph.topological_sort() is None
        assert not graph.is_acyclic()

    def test_is_acyclic(self):
        """Test cycle detection."""
        # DAG
        dag = DirectedGraph[str, int]()
        dag.add_edges([("A", "B"), ("B", "C"), ("A", "C")])
        assert dag.is_acyclic()

        # Cyclic graph
        cyclic = DirectedGraph[str, int]()
        cyclic.add_edges([("A", "B"), ("B", "C"), ("C", "A")])
        assert not cyclic.is_acyclic()

    def test_graph_reversal(self):
        """Test graph reversal."""
        graph = DirectedGraph[str, int]()
        graph.add_edge("A", "B", 5)
        graph.add_edge("B", "C", 10)

        reversed_graph = graph.reverse()

        assert reversed_graph.vertex_count == graph.vertex_count
        assert reversed_graph.edge_count == graph.edge_count

        # Check reversed edges
        assert reversed_graph.has_edge("B", "A")
        assert reversed_graph.has_edge("C", "B")
        assert not reversed_graph.has_edge("A", "B")
        assert not reversed_graph.has_edge("B", "C")

        # Check weights are preserved
        assert reversed_graph.get_edge_weight("B", "A") == 5
        assert reversed_graph.get_edge_weight("C", "B") == 10

    def test_directed_graph_indexing(self):
        """Test directed graph indexing."""
        graph = DirectedGraph[str, int]()
        graph.add_edge("A", "B", 5)
        graph.add_edge("C", "A", 10)

        neighbors = graph["A"]
        assert isinstance(neighbors, dict)
        assert neighbors["out"] == {"B"}
        assert neighbors["in"] == {"C"}


class TestGraphTraversal:
    """Test graph traversal algorithms."""

    def test_dfs_undirected(self):
        """Test DFS on undirected graph."""
        graph = UndirectedGraph[str, int]()
        edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "E")]
        graph.add_edges(edges)

        dfs_result = list(graph.dfs("A"))
        assert len(dfs_result) == 5
        assert dfs_result[0] == "A"  # Starting vertex
        assert set(dfs_result) == {"A", "B", "C", "D", "E"}

    def test_dfs_with_visited_set(self):
        """Test DFS with pre-existing visited set."""
        graph = UndirectedGraph[str, int]()
        edges = [("A", "B"), ("C", "D")]
        graph.add_edges(edges)

        visited = {"A"}
        dfs_result = list(graph.dfs("B", visited))

        assert "A" not in dfs_result  # Was already visited
        assert "B" in dfs_result

    def test_bfs_directed(self):
        """Test BFS on directed graph."""
        graph = DirectedGraph[str, int]()
        edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "E")]
        graph.add_edges(edges)

        bfs_result = list(graph.bfs("A"))
        assert bfs_result[0] == "A"

        # BFS should visit level by level
        a_index = bfs_result.index("A")
        b_index = bfs_result.index("B")
        c_index = bfs_result.index("C")
        d_index = bfs_result.index("D")
        e_index = bfs_result.index("E")

        assert a_index < b_index and a_index < c_index
        assert b_index < d_index and c_index < e_index

    def test_traversal_nonexistent_vertex(self):
        """Test traversal from non-existent vertex."""
        graph = UndirectedGraph[str, int]()
        graph.add_edge("A", "B")

        assert list(graph.dfs("X")) == []
        assert list(graph.bfs("X")) == []


class TestGraphConnectivity:
    """Test graph connectivity algorithms."""

    def test_connected_components_undirected(self):
        """Test connected components in undirected graph."""
        graph = UndirectedGraph[str, int]()

        # Create two separate components
        graph.add_edges([("A", "B"), ("B", "C")])  # Component 1
        graph.add_edges([("D", "E")])  # Component 2
        graph.add_vertex("F")  # Component 3 (isolated)

        components = list(graph.connected_components())
        assert len(components) == 3

        component_sets = [set(comp) for comp in components]
        assert {"A", "B", "C"} in component_sets
        assert {"D", "E"} in component_sets
        assert {"F"} in component_sets

    def test_is_connected(self):
        """Test connectivity check."""
        # Connected graph
        connected = UndirectedGraph[str, int]()
        connected.add_edges([("A", "B"), ("B", "C"), ("C", "D")])
        assert connected.is_connected()

        # Disconnected graph
        disconnected = UndirectedGraph[str, int]()
        disconnected.add_edges([("A", "B"), ("C", "D")])
        assert not disconnected.is_connected()

        # Empty graph
        empty = UndirectedGraph[str, int]()
        assert empty.is_connected()

    def test_has_path(self):
        """Test path existence check."""
        graph = UndirectedGraph[str, int]()
        graph.add_edges([("A", "B"), ("B", "C")])
        graph.add_vertex("D")  # Isolated vertex

        assert graph.has_path("A", "C")
        assert graph.has_path("A", "A")  # Self path
        assert not graph.has_path("A", "D")
        assert not graph.has_path("A", "X")  # Non-existent vertex


class TestGraphWithCustomTypes:
    """Test graphs with custom vertex and weight types."""

    def test_person_vertices(self):
        """Test graph with custom vertex type."""
        alice = Person("Alice", 30)
        bob = Person("Bob", 25)
        charlie = Person("Charlie", 35)

        graph = UndirectedGraph[Person, float]()
        graph.add_edge(alice, bob, 0.8)  # Friendship strength
        graph.add_edge(bob, charlie, 0.6)

        assert graph.vertex_count == 3
        assert alice in graph

        neighbors = list(graph.neighbors(alice))
        assert bob in neighbors

        friendship_strength = graph.get_edge_weight(alice, bob)
        assert friendship_strength == 0.8

    def test_distance_weights(self):
        """Test graph with custom weight type."""
        graph = DirectedGraph[str, Distance]()

        distance1 = Distance(10.5, "km")
        distance2 = Distance(5.0, "miles")

        graph.add_edge("CityA", "CityB", distance1)
        graph.add_edge("CityB", "CityC", distance2)

        edge_weight = graph.get_edge_weight("CityA", "CityB")
        assert isinstance(edge_weight, Distance)
        assert edge_weight.value == 10.5
        assert edge_weight.unit == "km"

    def test_mixed_generic_operations(self):
        """Test operations with mixed generic types."""
        # Graph with integer vertices and string weights
        graph = UndirectedGraph[int, str]()
        graph.add_edge(1, 2, "fast")
        graph.add_edge(2, 3, "slow")

        assert graph.get_edge_weight(1, 2) == "fast"

        # Ensure type safety
        edge = graph.get_edge(1, 2)
        assert edge is not None
        assert isinstance(edge.weight, str)


class TestFactoryFunction:
    """Test factory function."""

    def test_create_undirected_graph(self):
        """Test creating undirected graph via factory."""
        graph = create_graph(directed=False)
        assert isinstance(graph, UndirectedGraph)
        assert not graph.is_directed()

    def test_create_directed_graph(self):
        """Test creating directed graph via factory."""
        graph = create_graph(directed=True)
        assert isinstance(graph, DirectedGraph)
        assert graph.is_directed()


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_edge_not_found_error(self):
        """Test EdgeNotFoundError."""
        graph = UndirectedGraph[str, int]()
        graph.add_vertex("A")

        with pytest.raises(EdgeNotFoundError):
            graph.get_edge_weight("A", "B")

    def test_custom_weight_validation(self):
        """Test custom weight type validation."""
        graph = UndirectedGraph[str, Distance]()

        # Valid distance
        valid_distance = Distance(10.0)
        graph.add_edge("A", "B", valid_distance)

        # Invalid distance should raise error during creation
        with pytest.raises(ValueError):
            Distance(-5.0)  # Negative distance

    def test_graph_representation(self):
        """Test string representation."""
        graph = UndirectedGraph[str, int]()
        graph.add_edge("A", "B", 5)

        repr_str = repr(graph)
        assert "UndirectedGraph" in repr_str
        assert "vertices=2" in repr_str
        assert "edges=1" in repr_str

    def test_empty_operations(self):
        """Test operations on empty graph."""
        graph = UndirectedGraph[str, int]()

        assert list(graph.neighbors("A")) == []
        assert graph.degree("A") == 0
        assert list(graph.incident_edges("A")) == []
        assert graph["A"] == frozenset()


class TestPerformanceAndEdgeCases:
    """Test performance considerations and edge cases."""

    def test_large_graph_operations(self):
        """Test operations on moderately large graph."""
        graph = UndirectedGraph[int, int]()

        # Add many vertices and edges
        n = 1000
        for i in range(n):
            graph.add_vertex(i)

        # Add edges in a ring
        for i in range(n):
            graph.add_edge(i, (i + 1) % n, i)

        assert graph.vertex_count == n
        assert graph.edge_count == n
        assert graph.is_connected()

    def test_self_loops(self):
        """Test self-loops."""
        # Undirected graph with self-loop
        ug = UndirectedGraph[str, int]()
        ug.add_edge("A", "A", 5)

        assert ug.has_edge("A", "A")
        assert ug.degree("A") == 2  # Self-loop counts twice

        # Directed graph with self-loop
        dg = DirectedGraph[str, int]()
        dg.add_edge("A", "A", 5)

        assert dg.has_edge("A", "A")
        assert dg.in_degree("A") == 1
        assert dg.out_degree("A") == 1

    def test_graph_modification_during_iteration(self):
        """Test safe modification during iteration."""
        graph = UndirectedGraph[str, int]()
        graph.add_edges([("A", "B"), ("A", "C"), ("A", "D")])

        # This should not raise an error due to list conversion
        incident_edges = list(graph.incident_edges("A"))
        for edge in incident_edges:
            graph.remove_edge(edge.u, edge.v)

        assert graph.degree("A") == 0


# Test runner configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
