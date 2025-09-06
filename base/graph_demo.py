"""
Enhanced Graph Data Structure Demo

Demonstrates comprehensive usage of the graph library with real-world examples,
visual output, and performance benchmarks.
"""

import random
import time
from typing import Any

from graph import DirectedGraph, UndirectedGraph


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print("=" * 60)


def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def visualize_graph(graph: Any, title: str) -> None:
    """Simple text visualization of a graph."""
    print(f"\n{title}:")
    print(f"  Vertices: {graph.vertex_count}, Edges: {graph.edge_count}")

    if graph.vertex_count > 10:
        print("  (Graph too large to display)")
        return

    vertices = sorted(graph.vertices) if graph.vertices else []
    for vertex in vertices:
        neighbors = list(graph.neighbors(vertex))
        if neighbors:
            if graph.is_directed():
                predecessors = list(graph.predecessors(vertex)) if hasattr(graph, "predecessors") else []
                print(f"  {vertex}: out→{neighbors}, in←{predecessors}")
            else:
                degree = graph.degree(vertex)
                print(f"  {vertex}: {neighbors} (degree: {degree})")
        else:
            print(f"  {vertex}: isolated")


def demo_social_network() -> None:
    """Demonstrate undirected graph with a social network."""
    print_section("Social Network Analysis")

    # Create a social network graph
    network = UndirectedGraph[str, int]()

    # Define friendships with connection strength (1-10)
    friendships = [
        ("Alice", "Bob", 9),
        ("Alice", "Charlie", 7),
        ("Alice", "Diana", 8),
        ("Bob", "Charlie", 6),
        ("Bob", "Eve", 5),
        ("Charlie", "Frank", 8),
        ("Diana", "Eve", 7),
        ("Eve", "Frank", 9),
        ("Frank", "Grace", 6),
        ("Grace", "Henry", 8),
        ("Henry", "Ian", 7),
        ("Grace", "Ian", 5),
        ("Jack", "Kate", 9),  # Isolated component
    ]

    added = network.add_edges(friendships)
    print(f"Created social network with {added} friendships")
    visualize_graph(network, "Social Network Structure")

    # Analyze network properties
    print("\nNetwork Analysis:")
    components = list(network.connected_components())
    print(f"  Connected components: {len(components)}")
    for i, component in enumerate(components, 1):
        print(f"    Group {i}: {component} ({len(component)} people)")

    # Find most connected people
    degrees = [(person, network.degree(person)) for person in network.vertices]
    most_connected = max(degrees, key=lambda x: x[1])
    print(f"  Most connected person: {most_connected[0]} ({most_connected[1]} connections)")

    # Analyze strongest friendships
    strong_friendships = []
    for person in network.vertices:
        for edge in network.incident_edges(person):
            if edge.u < edge.v:  # Avoid duplicates in undirected graph
                strong_friendships.append((edge.u, edge.v, edge.weight))

    strongest = max(strong_friendships, key=lambda x: x[2])
    print(f"  Strongest friendship: {strongest[0]} ↔ {strongest[1]} (strength: {strongest[2]})")

    # Path analysis
    if network.has_path("Alice", "Ian"):
        print("  Alice and Ian are connected through the network!")

    if not network.has_path("Alice", "Jack"):
        print("  Alice and Jack are in different social circles")


def demo_task_management() -> None:
    """Demonstrate directed graph with project task dependencies."""
    print_section("Project Task Management")

    # Create a project workflow
    project = DirectedGraph[str, int]()

    # Define task dependencies with estimated hours
    tasks = [
        ("Requirements", "Design", 2),
        ("Requirements", "Research", 1),
        ("Design", "Frontend", 3),
        ("Design", "Backend", 2),
        ("Research", "Backend", 1),
        ("Frontend", "Integration", 2),
        ("Backend", "Integration", 1),
        ("Backend", "Database", 2),
        ("Database", "Integration", 1),
        ("Integration", "Testing", 3),
        ("Testing", "Documentation", 2),
        ("Testing", "Deployment", 1),
        ("Documentation", "Deployment", 1),
        ("Deployment", "Release", 1),
    ]

    project.add_edges(tasks)
    print(f"Created project workflow with {project.edge_count} dependencies")
    visualize_graph(project, "Task Dependency Graph")

    # Analyze project structure
    print("\nProject Analysis:")

    # Critical path analysis (simplified)
    schedule = project.topological_sort()
    if schedule:
        print(f"  Task execution order: {' → '.join(schedule)}")

        # Calculate earliest start times
        earliest_start = {"Requirements": 0}
        for task in schedule[1:]:
            max_predecessor_time = 0
            for pred in project.predecessors(task):
                pred_finish = earliest_start.get(pred, 0)
                # Add the edge weight (task duration)
                edge = project.get_edge(pred, task)
                if edge:
                    pred_finish += edge.weight
                max_predecessor_time = max(max_predecessor_time, pred_finish)
            earliest_start[task] = max_predecessor_time

        total_time = max(earliest_start.values())
        print(f"  Minimum project duration: {total_time} hours")

        # Find critical tasks (those that would delay the project if delayed)
        critical_tasks = [
            task for task, time in earliest_start.items() if time >= total_time - 2
        ]  # Within 2 hours of end
        print(f"  Critical tasks: {critical_tasks}")

    # Task complexity analysis
    task_complexity = []
    for task in project.vertices:
        in_deg = project.in_degree(task)
        out_deg = project.out_degree(task)
        complexity = in_deg + out_deg
        task_complexity.append((task, complexity, in_deg, out_deg))

    most_complex = max(task_complexity, key=lambda x: x[1])
    print(f"  Most complex task: {most_complex[0]} ({most_complex[2]} dependencies, {most_complex[3]} dependents)")


def demo_graph_algorithms() -> None:
    """Demonstrate various graph algorithms."""
    print_section("Graph Algorithm Demonstrations")

    # Create a more complex graph for algorithm testing
    graph = UndirectedGraph[str, int]()

    # Create a graph with interesting structure
    connections = [
        # Central hub
        ("Hub", "A"),
        ("Hub", "B"),
        ("Hub", "C"),
        # Linear chain
        ("A", "A1"),
        ("A1", "A2"),
        ("A2", "A3"),
        # Triangle
        ("B", "B1"),
        ("B1", "B2"),
        ("B2", "B"),
        # Star pattern
        ("C", "C1"),
        ("C", "C2"),
        ("C", "C3"),
        ("C", "C4"),
        # Isolated component
        ("X", "Y"),
        ("Y", "Z"),
        ("Z", "X"),
    ]

    graph.add_edges(connections)
    visualize_graph(graph, "Algorithm Test Graph")

    print_subsection("Traversal Algorithms")

    # Compare DFS and BFS from the hub
    print("Starting from 'Hub':")
    dfs_order = list(graph.dfs("Hub"))
    bfs_order = list(graph.bfs("Hub"))

    print(f"  DFS order: {' → '.join(dfs_order[:8])}{'...' if len(dfs_order) > 8 else ''}")
    print(f"  BFS order: {' → '.join(bfs_order[:8])}{'...' if len(bfs_order) > 8 else ''}")
    print(f"  Vertices reached: {len(dfs_order)} (DFS), {len(bfs_order)} (BFS)")

    print_subsection("Connectivity Analysis")

    components = list(graph.connected_components())
    print(f"Connected components: {len(components)}")
    for i, component in enumerate(components, 1):
        size = len(component)
        if size > 1:
            diameter = estimate_diameter(graph, component)
            print(f"  Component {i}: {size} vertices, diameter ≈ {diameter}")
        else:
            print(f"  Component {i}: {component[0]} (isolated)")

    # Path testing
    print("\nPath Analysis:")
    test_paths = [("Hub", "A3"), ("B1", "C4"), ("A", "X")]
    for start, end in test_paths:
        has_path = graph.has_path(start, end)
        status = "✓ Connected" if has_path else "✗ Disconnected"
        print(f"  {start} to {end}: {status}")


def estimate_diameter(graph: Any, vertices: list) -> int:
    """Estimate graph diameter using BFS from a sample of vertices."""
    max_distance = 0
    sample_size = min(3, len(vertices))  # Sample a few vertices

    for start in vertices[:sample_size]:
        distances = {start: 0}
        queue = [start]

        while queue:
            current = queue.pop(0)
            current_dist = distances[current]

            for neighbor in graph.neighbors(current):
                if neighbor not in distances and neighbor in vertices:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
                    max_distance = max(max_distance, current_dist + 1)

    return max_distance


def demo_performance_benchmark() -> None:
    """Benchmark graph performance with different sizes."""
    print_section("Performance Benchmark")

    sizes = [100, 500, 1000]
    results = []

    for n in sizes:
        print(f"\nTesting graph with {n} vertices...")

        # Create random graph
        graph = UndirectedGraph[int, int]()

        # Measure vertex addition
        start_time = time.time()
        for i in range(n):
            graph.add_vertex(i)
        vertex_time = time.time() - start_time

        # Measure edge addition (create random edges)
        edges_to_add = min(n * 2, n * (n - 1) // 4)  # Sparse graph
        random.seed(42)  # Reproducible results

        start_time = time.time()
        for _ in range(edges_to_add):
            u, v = random.randint(0, n - 1), random.randint(0, n - 1)
            if u != v:
                graph.add_edge(u, v, random.randint(1, 10))
        edge_time = time.time() - start_time

        # Measure traversal
        start_vertex = random.randint(0, n - 1)
        start_time = time.time()
        dfs_result = list(graph.dfs(start_vertex))
        traversal_time = time.time() - start_time

        # Measure connectivity
        start_time = time.time()
        components = list(graph.connected_components())
        connectivity_time = time.time() - start_time

        results.append(
            {
                "n": n,
                "vertices": graph.vertex_count,
                "edges": graph.edge_count,
                "vertex_time": vertex_time,
                "edge_time": edge_time,
                "traversal_time": traversal_time,
                "connectivity_time": connectivity_time,
                "components": len(components),
                "largest_component": max(len(comp) for comp in components) if components else 0,
            }
        )

        print(f"  Added {graph.vertex_count} vertices in {vertex_time:.4f}s")
        print(f"  Added {graph.edge_count} edges in {edge_time:.4f}s")
        print(f"  DFS traversed {len(dfs_result)} vertices in {traversal_time:.4f}s")
        print(f"  Found {len(components)} components in {connectivity_time:.4f}s")

    # Summary table
    print("\nPerformance Summary:")
    print("Size  | Vertices | Edges | V.Time | E.Time | DFS.Time | Comp.Time | Components")
    print("-" * 80)
    for r in results:
        print(
            f"{r['n']:4d} | {r['vertices']:8d} | {r['edges']:5d} | "
            f"{r['vertex_time']:6.3f} | {r['edge_time']:6.3f} | "
            f"{r['traversal_time']:8.4f} | {r['connectivity_time']:9.4f} | "
            f"{r['components']:10d}"
        )


def demo_advanced_features() -> None:
    """Demonstrate advanced graph features and edge cases."""
    print_section("Advanced Features & Edge Cases")

    print_subsection("Custom Types")

    # Graph with custom vertex and edge types
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class City:
        name: str
        population: int

        def __str__(self):
            return f"{self.name}({self.population // 1000}k)"

    @dataclass(frozen=True)
    class Distance:
        km: float
        highway: bool = False

        def __str__(self):
            road_type = "highway" if self.highway else "road"
            return f"{self.km}km via {road_type}"

    # Create a transportation network
    transport = UndirectedGraph[City, Distance]()

    cities = [
        City("New York", 8500000),
        City("Boston", 680000),
        City("Philadelphia", 1580000),
        City("Washington DC", 700000),
    ]

    routes = [
        (cities[0], cities[1], Distance(306, True)),  # NYC-Boston
        (cities[0], cities[2], Distance(153, True)),  # NYC-Philadelphia
        (cities[0], cities[3], Distance(361, True)),  # NYC-Washington
        (cities[1], cities[2], Distance(436, False)),  # Boston-Philadelphia
        (cities[2], cities[3], Distance(199, True)),  # Philadelphia-Washington
    ]

    transport.add_edges(routes)

    print("Transportation Network:")
    for city in cities:
        connections = []
        for edge in transport.incident_edges(city):
            other_city = edge.v if edge.u == city else edge.u
            connections.append(f"{other_city} ({edge.weight})")
        print(f"  {city}: {', '.join(connections)}")

    print_subsection("Graph Modification Safety")

    # Demonstrate safe modification during iteration
    graph = UndirectedGraph[str, int]()
    graph.add_edges([("A", "B"), ("A", "C"), ("A", "D"), ("B", "C")])

    print("Original graph edges:")
    all_edges = [
        (edge.u, edge.v)
        for edge in [
            graph.get_edge(u, v) for u in graph.vertices for v in graph.vertices if graph.has_edge(u, v) and u <= v
        ]
    ]
    for u, v in all_edges:
        print(f"  {u} - {v}")

    # Safe removal during iteration
    print("\nRemoving edges incident to 'A':")
    edges_to_remove = list(graph.incident_edges("A"))
    for edge in edges_to_remove:
        print(f"  Removing {edge.u} - {edge.v}")
        graph.remove_edge(edge.u, edge.v)

    print(f"Remaining edges: {graph.edge_count}")

    print_subsection("Directed Graph Advanced Operations")

    # DAG with topological analysis
    dag = DirectedGraph[str, int]()

    # Create a complex DAG (compiler dependency graph)
    dependencies = [
        ("Source", "Lexer"),
        ("Source", "Parser"),
        ("Lexer", "Parser"),
        ("Parser", "AST"),
        ("AST", "Optimizer"),
        ("AST", "CodeGen"),
        ("Optimizer", "CodeGen"),
        ("CodeGen", "Linker"),
        ("Linker", "Executable"),
    ]

    dag.add_edges(dependencies)

    print("Compiler Pipeline DAG:")
    topo_order = dag.topological_sort()
    if topo_order:
        print(f"  Build order: {' → '.join(topo_order)}")

        # Identify parallel opportunities
        levels = {}
        for task in topo_order:
            max_pred_level = -1
            for pred in dag.predecessors(task):
                max_pred_level = max(max_pred_level, levels.get(pred, -1))
            levels[task] = max_pred_level + 1

        level_groups = {}
        for task, level in levels.items():
            level_groups.setdefault(level, []).append(task)

        print("  Parallel execution levels:")
        for level in sorted(level_groups.keys()):
            print(f"    Level {level}: {level_groups[level]}")


def main() -> None:
    """Main demo function."""
    print("🔗 Graph Data Structure - Comprehensive Demo")
    print("=" * 60)

    try:
        demo_social_network()
        demo_task_management()
        demo_graph_algorithms()
        demo_advanced_features()
        demo_performance_benchmark()

        print_section("Demo Complete")
        print("The graph library successfully demonstrates:")
        print("  ✓ Efficient vertex and edge operations")
        print("  ✓ Multiple graph algorithms (DFS, BFS, connectivity)")
        print("  ✓ Support for custom vertex and edge types")
        print("  ✓ Both directed and undirected graph variants")
        print("  ✓ Safe concurrent modification")
        print("  ✓ Good performance characteristics")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
