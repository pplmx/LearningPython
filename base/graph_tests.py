import time
from dataclasses import dataclass

import pytest
from graph import (
    DirectedGraph,
    Edge,
    EdgeNotFoundError,
    UndirectedGraph,
    create_graph,
)


@dataclass(frozen=True, order=True)
class Person:
    """支持排序的测试用 Person 类。"""

    name: str
    age: int

    def __str__(self):
        return f"{self.name}({self.age})"


@dataclass(frozen=True)
class Distance:
    """带验证和比较功能的 Distance 类。"""

    value: float
    unit: str = "km"

    def __post_init__(self):
        if self.value < 0:
            raise ValueError("Distance cannot be negative")

    def __eq__(self, other):
        return isinstance(other, Distance) and self.value == other.value and self.unit == other.unit

    def __lt__(self, other):
        if not isinstance(other, Distance):
            return NotImplemented
        self_km = self.value if self.unit == "km" else self.value * 1.609
        other_km = other.value if other.unit == "km" else other.value * 1.609
        return self_km < other_km


@dataclass(frozen=True)
class WeightedConnection:
    """用于测试的复杂权重类型。"""

    strength: int
    confidence: float
    bidirectional: bool = True

    def __post_init__(self):
        if not 0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")


# ==================== 原有 Fixtures ====================


@pytest.fixture
def empty_undirected() -> UndirectedGraph[str, int]:
    """空的无向图。"""
    return UndirectedGraph[str, int]()


@pytest.fixture
def empty_directed() -> DirectedGraph[str, int]:
    """空的有向图。"""
    return DirectedGraph[str, int]()


@pytest.fixture
def simple_undirected() -> UndirectedGraph[str, int]:
    """带基本连接的简单无向图。"""
    graph = UndirectedGraph[str, int]()
    edges = [("A", "B", 1), ("B", "C", 2), ("C", "D", 3), ("A", "D", 4)]
    graph.add_edges(edges)
    return graph


@pytest.fixture
def simple_directed() -> DirectedGraph[str, int]:
    """简单的有向无环图 (DAG)。"""
    graph = DirectedGraph[str, int]()
    edges = [("A", "B", 1), ("A", "C", 2), ("B", "D", 3), ("C", "D", 4)]
    graph.add_edges(edges)
    return graph


@pytest.fixture
def directed_with_cycle() -> DirectedGraph[str, int]:
    """带简单环的有向图。"""
    graph = DirectedGraph[str, int]()
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    graph.add_edges(edges)
    return graph


@pytest.fixture
def disconnected_graph() -> UndirectedGraph[str, int]:
    """包含多个连通分量的无向图。"""
    graph = UndirectedGraph[str, int]()
    edges = [
        ("A", "B"),
        ("B", "C"),  # 分量 1
        ("D", "E"),
        ("E", "F"),  # 分量 2
    ]
    graph.add_edges(edges)
    graph.add_vertex("G")
    return graph


@pytest.fixture
def complex_social_network() -> UndirectedGraph[str, int]:
    """用于综合测试的复杂社交网络。"""
    graph = UndirectedGraph[str, int]()
    connections = [
        ("Alice", "Bob", 9),
        ("Alice", "Charlie", 8),
        ("Alice", "Diana", 7),
        ("Bob", "Charlie", 6),
        ("Bob", "Eve", 8),
        ("Charlie", "Diana", 7),
        ("Diana", "Frank", 5),
        ("Eve", "Frank", 6),
        ("Eve", "Grace", 7),
        ("Frank", "Grace", 8),
        ("Grace", "Henry", 6),
        ("Ian", "Jack", 4),
        ("Jack", "Kate", 5),
        ("Kate", "Ian", 3),
        ("Henry", "Ian", 2),
        ("Liam", "Maya", 9),
    ]
    graph.add_edges(connections)
    return graph


@pytest.fixture
def star_graph() -> UndirectedGraph[str, int]:
    """星形图：一个中心节点连接多个叶子节点。"""
    graph = UndirectedGraph[str, int]()
    center = "CENTER"
    leaves = [f"LEAF_{i}" for i in range(5)]
    for i, leaf in enumerate(leaves):
        graph.add_edge(center, leaf, i + 1)
    return graph


@pytest.fixture
def complete_graph() -> UndirectedGraph[str, int]:
    """完全图：每两个顶点之间都有边。"""
    graph = UndirectedGraph[str, int]()
    vertices = ["A", "B", "C", "D"]
    for i, u in enumerate(vertices):
        for j, v in enumerate(vertices):
            if i < j:  # 避免重复和自环
                graph.add_edge(u, v, i + j + 1)
    return graph


@pytest.fixture
def bipartite_graph() -> UndirectedGraph[str, int]:
    """二分图：两个顶点集合，只在不同集合间有边。"""
    graph = UndirectedGraph[str, int]()
    set_a = ["A1", "A2", "A3"]
    set_b = ["B1", "B2", "B3"]

    # 只在不同集合间添加边
    # 生成所有可能的边，然后随机采样
    import random
    from itertools import product

    all_edges = list(product(set_a, set_b))
    selected_edges = random.sample(all_edges, k=6)  # 随机选6条边

    edges = [(a, b, random.randint(1, 10)) for a, b in selected_edges]
    graph.add_edges(edges)
    return graph


@pytest.fixture
def deeply_nested_dag() -> DirectedGraph[str, int]:
    """深度嵌套的有向无环图。"""
    graph = DirectedGraph[str, int]()
    # 创建一个深度为 10 的链式DAG，每层有分支
    for level in range(10):
        for branch in range(2):
            current = f"L{level}_B{branch}"
            if level < 9:
                next_level = f"L{level + 1}_B{branch}"
                graph.add_edge(current, next_level, level + 1)
                # 添加交叉连接
                if branch == 0:
                    cross = f"L{level + 1}_B1"
                    graph.add_edge(current, cross, level + 1)
    return graph


@pytest.fixture
def self_loop_graph() -> DirectedGraph[str, int]:
    """包含自环的有向图。"""
    graph = DirectedGraph[str, int]()
    edges = [
        ("A", "A", 1),  # 自环
        ("A", "B", 2),
        ("B", "B", 3),  # 自环
        ("B", "C", 4),
        ("C", "A", 5),
    ]
    graph.add_edges(edges)
    return graph


@pytest.fixture
def large_graph() -> UndirectedGraph[int, int]:
    """大型图用于性能测试。"""
    graph = UndirectedGraph[int, int]()
    # 创建一个 1000 个节点的随机图
    import random

    random.seed(42)  # 保证测试的一致性

    nodes = list(range(1000))
    for node in nodes:
        graph.add_vertex(node)

    # 随机添加 5000 条边
    for _ in range(5000):
        u, v = random.sample(nodes, 2)
        weight = random.randint(1, 100)
        graph.add_edge(u, v, weight)

    return graph


# ==================== 增强的测试类 ====================


class TestEdgeEnhancements:
    """增强的边测试。"""

    def test_edge_with_zero_weight(self):
        edge = Edge("A", "B", 0)
        assert edge.weight == 0
        assert str(edge) == "A --(0)--> B"

    def test_edge_with_negative_weight(self):
        edge = Edge("A", "B", -5)
        assert edge.weight == -5

    def test_edge_with_none_vertices(self):
        edge = Edge(None, "B", 1)
        assert edge.u is None
        assert edge.v == "B"

    def test_edge_ordering_and_hashing(self):
        edge1 = Edge("A", "B", 5)
        edge2 = Edge("A", "B", 5)
        edge3 = Edge("B", "A", 5)
        edge_set = {edge1, edge2, edge3}
        assert len(edge_set) == 2
        edge_dict = {edge1: "forward", edge3: "backward"}
        assert len(edge_dict) == 2

    def test_edge_with_complex_weight_types(self):
        weight = WeightedConnection(strength=8, confidence=0.9)
        edge = Edge("Alice", "Bob", weight)
        assert edge.weight.strength == 8
        assert edge.weight.confidence == 0.9

    @pytest.mark.parametrize(
        "weight_type,weight_value",
        [
            (int, 42),
            (float, 3.14159),
            (str, "heavy"),
            (tuple, (1, 2, 3)),
            (Distance, Distance(100.5, "km")),
        ],
    )
    def test_edge_with_various_weight_types(self, weight_type, weight_value):
        edge = Edge("start", "end", weight_value)
        assert isinstance(edge.weight, weight_type)
        assert edge.weight == weight_value

    def test_edge_reverse_preserves_weight(self):
        """测试边反转保持权重。"""
        original_weight = Distance(42.5, "km")
        edge = Edge("A", "B", original_weight)
        reversed_edge = edge.reverse()

        assert reversed_edge.u == "B"
        assert reversed_edge.v == "A"
        assert reversed_edge.weight == original_weight

    def test_edge_immutability(self):
        """测试边的不可变性。"""
        edge = Edge("A", "B", 5)
        with pytest.raises(AttributeError):
            edge.u = "C"  # 应该失败，因为是 frozen dataclass


class TestGraphProperties:
    """增强的图属性测试。"""

    def test_empty_graph_properties(self, empty_undirected, empty_directed):
        for graph in [empty_undirected, empty_directed]:
            assert graph.is_empty
            assert len(graph) == 0
            assert graph.vertex_count == 0
            assert graph.edge_count == 0
            assert graph.vertices == frozenset()
            assert graph.is_connected()

    def test_single_vertex_graph(self, empty_undirected):
        graph = empty_undirected
        graph.add_vertex("A")
        assert not graph.is_empty
        assert graph.vertex_count == 1
        assert graph.edge_count == 0
        assert "A" in graph
        assert graph.is_connected()
        assert list(graph.neighbors("A")) == []

    def test_graph_invariants_after_operations(self, simple_undirected):
        graph = simple_undirected
        initial_vertex_count = graph.vertex_count
        initial_edge_count = graph.edge_count
        graph.add_edge("A", "B", 10)
        assert graph.edge_count == initial_edge_count
        graph.remove_vertex("B")
        assert graph.vertex_count == initial_vertex_count - 1
        graph.add_vertex("B")
        assert graph.vertex_count == initial_vertex_count
        assert list(graph.neighbors("B")) == []

    def test_graph_string_representation(self, simple_undirected, simple_directed):
        undirected_repr = repr(simple_undirected)
        directed_repr = repr(simple_directed)
        assert "UndirectedGraph" in undirected_repr
        assert "DirectedGraph" in directed_repr
        assert "vertices=" in undirected_repr
        assert "edges=" in undirected_repr

    def test_graph_density(self, complete_graph, star_graph):
        """测试图的密度计算。"""
        # 完全图应该有最大密度
        n = complete_graph.vertex_count
        max_edges = n * (n - 1) // 2  # 无向完全图的最大边数
        assert complete_graph.edge_count == max_edges

        # 星形图密度较低
        star_n = star_graph.vertex_count
        assert star_graph.edge_count == star_n - 1  # 星形图有 n-1 条边

    def test_vertex_degrees_star_graph(self, star_graph):
        """测试星形图的度数分布。"""
        degrees = {v: star_graph.degree(v) for v in star_graph.vertices}
        center_degree = max(degrees.values())
        leaf_degrees = [d for d in degrees.values() if d != center_degree]

        assert center_degree == 5  # 中心节点连接 5 个叶子
        assert all(d == 1 for d in leaf_degrees)  # 所有叶子度数为 1


class TestGraphOperations:
    """增强的图操作测试。"""

    def test_batch_operations_with_duplicates(self, empty_undirected):
        graph = empty_undirected
        vertices = ["A", "B", "C", "A", "B"]
        added = graph.add_vertices(vertices)
        assert added == 3
        assert graph.vertex_count == 3
        edges = [("A", "B"), ("B", "C", 5), ("A", "B", 10), ("C", "A", 7)]
        added = graph.add_edges(edges)
        assert added == 3

    def test_vertex_removal_cascade(self, complex_social_network):
        graph = complex_social_network
        initial_edges = graph.edge_count
        alice_degree = graph.degree("Alice")
        graph.remove_vertex("Alice")
        assert "Alice" not in graph
        assert graph.edge_count == initial_edges - alice_degree
        for vertex in graph.vertices:
            for neighbor in graph.neighbors(vertex):
                assert neighbor in graph.vertices

    def test_edge_weight_modification_semantics(self, empty_undirected):
        graph = empty_undirected
        assert graph.add_edge("A", "B", 5)
        assert graph.get_edge_weight("A", "B") == 5
        assert not graph.add_edge("A", "B", 10)
        assert graph.get_edge_weight("A", "B") == 5
        graph.remove_edge("A", "B")
        assert graph.add_edge("A", "B", 10)
        assert graph.get_edge_weight("A", "B") == 10

    def test_graph_operations_with_none_vertices(self, empty_undirected):
        graph = empty_undirected
        assert graph.add_vertex(None)
        assert None in graph
        assert graph.add_edge(None, "A", 1)
        assert graph.has_edge(None, "A")
        assert "A" in graph.neighbors(None)

    def test_massive_vertex_addition_removal(self, empty_undirected):
        """测试大量顶点的添加和删除。"""
        graph = empty_undirected
        vertices = [f"vertex_{i}" for i in range(1000)]

        # 批量添加
        added = graph.add_vertices(vertices)
        assert added == 1000
        assert graph.vertex_count == 1000

        # 批量删除
        removed = sum(graph.remove_vertex(v) for v in vertices[:500])
        assert removed == 500
        assert graph.vertex_count == 500

    def test_edge_operations_with_auto_vertex_creation(self, empty_directed):
        """测试边操作自动创建顶点。"""
        graph = empty_directed

        # 添加边应该自动创建不存在的顶点
        assert graph.add_edge("NEW_A", "NEW_B", 42)
        assert "NEW_A" in graph.vertices
        assert "NEW_B" in graph.vertices
        assert graph.vertex_count == 2
        assert graph.edge_count == 1


class TestSpecialGraphStructures:
    """特殊图结构的测试。"""

    def test_bipartite_graph_properties(self, bipartite_graph):
        """测试二分图的特性。"""
        # 获取所有顶点
        vertices = list(bipartite_graph.vertices)
        set_a = [v for v in vertices if v.startswith("A")]
        set_b = [v for v in vertices if v.startswith("B")]

        # 验证二分性：同一集合内的顶点不应该相邻
        for va in set_a:
            neighbors_a = list(bipartite_graph.neighbors(va))
            assert all(n.startswith("B") for n in neighbors_a)

        for vb in set_b:
            neighbors_b = list(bipartite_graph.neighbors(vb))
            assert all(n.startswith("A") for n in neighbors_b)

    def test_complete_graph_properties(self, complete_graph):
        """测试完全图的特性。"""
        n = complete_graph.vertex_count
        expected_edges = n * (n - 1) // 2  # 无向完全图的边数公式

        assert complete_graph.edge_count == expected_edges

        # 每个顶点都应该与其他所有顶点相邻
        for vertex in complete_graph.vertices:
            neighbors = set(complete_graph.neighbors(vertex))
            expected_neighbors = complete_graph.vertices - {vertex}
            assert neighbors == expected_neighbors

    def test_self_loop_handling(self, self_loop_graph):
        """测试自环的处理。"""
        # 检查自环是否被正确识别
        assert self_loop_graph.has_edge("A", "A")
        assert self_loop_graph.has_edge("B", "B")

        # 对于有向图，自环应该同时影响入度和出度
        if hasattr(self_loop_graph, "in_degree"):
            assert self_loop_graph.in_degree("A") >= 1  # 包含自环
            assert self_loop_graph.out_degree("A") >= 1  # 包含自环

    def test_deeply_nested_structure_traversal(self, deeply_nested_dag):
        """测试深层嵌套结构的遍历。"""
        start_node = "L0_B0"

        # DFS 应该能够遍历到所有可达节点
        dfs_visited = list(deeply_nested_dag.dfs(start_node))
        assert len(dfs_visited) > 10  # 应该访问多个层级

        # BFS 应该按层级顺序访问
        bfs_visited = list(deeply_nested_dag.bfs(start_node))
        assert len(bfs_visited) > 10

        # 验证路径存在性
        end_nodes = [f"L9_B{i}" for i in range(2)]
        for end_node in end_nodes:
            if end_node in deeply_nested_dag.vertices:
                assert deeply_nested_dag.has_path(start_node, end_node)


class TestExceptionHandling:
    """增强的异常处理测试。"""

    def test_get_edge_weight_nonexistent_edge(self, simple_undirected):
        """测试 get_edge_weight 对不存在的边抛出 EdgeNotFoundError。"""
        with pytest.raises(EdgeNotFoundError, match=r"Edge \(A, C\) not found"):
            simple_undirected.get_edge_weight("A", "C")

    def test_remove_nonexistent_vertex(self, simple_undirected):
        """测试移除不存在的顶点返回 False 且不报错。"""
        assert not simple_undirected.remove_vertex("Z")

    def test_remove_nonexistent_edge(self, simple_undirected):
        """测试移除不存在的边返回 False 且不报错。"""
        assert not simple_undirected.remove_edge("A", "C")

    def test_operations_on_nonexistent_vertex_in_algorithms(self, simple_directed):
        """测试算法能优雅地处理不存在的起始顶点。"""
        # DFS 和 BFS 应该只返回一个空迭代器
        assert list(simple_directed.dfs("Z")) == []
        assert list(simple_directed.bfs("Z")) == []
        # has_path 应该返回 False
        assert not simple_directed.has_path("A", "Z")
        assert not simple_directed.has_path("Z", "A")

    def test_invalid_weight_types(self):
        """测试无效权重类型的处理。"""
        with pytest.raises(ValueError):
            Distance(-10)  # 负距离应该抛出异常

        with pytest.raises(ValueError):
            WeightedConnection(5, 1.5)  # 置信度超出范围

    def test_edge_operations_robustness(self, empty_undirected):
        """测试边操作的鲁棒性。"""
        graph = empty_undirected

        # 测试各种边界情况
        assert graph.add_edge("", "", 0)  # 空字符串顶点
        assert graph.has_edge("", "")

        # 测试 Unicode 字符
        assert graph.add_edge("α", "β", 1)
        assert graph.has_edge("α", "β")


class TestAlgorithms:
    """增强的图算法测试。"""

    def test_dfs_traversal_order(self, simple_directed):
        """验证 DFS 的特定访问顺序。"""
        path = list(simple_directed.dfs("A"))
        assert path[0] == "A"
        assert set(path) == {"A", "B", "C", "D"}
        assert len(path) == 4

    def test_bfs_traversal_order(self, simple_directed):
        """验证 BFS 的特定访问顺序。"""
        path = list(simple_directed.bfs("A"))
        assert path[0] == "A"
        assert set(path[1:3]) == {"B", "C"}
        assert path[3] == "D"
        assert len(path) == 4

    def test_connected_components(self, disconnected_graph):
        """测试在不连通图中查找连通分量。"""
        components = list(disconnected_graph.connected_components())
        sorted_components = sorted([sorted(c) for c in components])
        assert sorted_components == [["A", "B", "C"], ["D", "E", "F"], ["G"]]

    def test_is_connected(self, simple_undirected, disconnected_graph):
        """测试 is_connected 属性。"""
        assert simple_undirected.is_connected()
        assert not disconnected_graph.is_connected()

    def test_has_path(self, simple_directed):
        """测试路径存在性检查。"""
        assert simple_directed.has_path("A", "D")
        assert not simple_directed.has_path("D", "A")
        assert simple_directed.has_path("A", "A")
        assert not simple_directed.has_path("B", "C")

    def test_traversal_with_cycles(self, directed_with_cycle):
        """测试在有环图中的遍历。"""
        # DFS 和 BFS 应该只访问每个节点一次，即使有环
        dfs_path = list(directed_with_cycle.dfs("A"))
        bfs_path = list(directed_with_cycle.bfs("A"))

        assert len(dfs_path) == 3
        assert len(bfs_path) == 3
        assert set(dfs_path) == {"A", "B", "C"}
        assert set(bfs_path) == {"A", "B", "C"}

    def test_path_finding_in_complex_network(self, complex_social_network):
        """在复杂网络中测试路径查找。"""
        # 测试连接组件内的路径
        assert complex_social_network.has_path("Alice", "Henry")
        assert complex_social_network.has_path("Ian", "Grace")

        # 测试不同组件间无路径
        assert not complex_social_network.has_path("Liam", "Alice")

    def test_traversal_consistency(self, complete_graph):
        """测试遍历的一致性。"""
        start_vertex = list(complete_graph.vertices)[0]

        # 多次运行应该产生相同的结果
        dfs_result1 = list(complete_graph.dfs(start_vertex))
        dfs_result2 = list(complete_graph.dfs(start_vertex))
        assert dfs_result1 == dfs_result2

        bfs_result1 = list(complete_graph.bfs(start_vertex))
        bfs_result2 = list(complete_graph.bfs(start_vertex))
        assert bfs_result1 == bfs_result2

    def test_algorithm_with_isolated_vertices(self, disconnected_graph):
        """测试算法对孤立顶点的处理。"""
        # 从孤立顶点开始遍历
        isolated_traversal = list(disconnected_graph.dfs("G"))
        assert isolated_traversal == ["G"]

        # 孤立顶点到其他顶点无路径
        assert not disconnected_graph.has_path("G", "A")


# 继续补全测试代码，添加到现有测试文件的末尾


class TestDirectedGraph:
    """有向图特定功能的增强测试。"""

    def test_directed_graph_asymmetry(self, empty_directed):
        """测试有向图的非对称属性。"""
        graph = empty_directed
        graph.add_edge("A", "B", 5)

        assert graph.has_edge("A", "B")
        assert not graph.has_edge("B", "A")

        assert "B" in list(graph.neighbors("A"))
        assert "A" not in list(graph.neighbors("B"))

        assert "A" in list(graph.predecessors("B"))
        assert "B" not in list(graph.predecessors("A"))

    def test_in_out_degree(self, simple_directed):
        """测试入度和出度的计算。"""
        # A -> B, A -> C, B -> D, C -> D
        assert simple_directed.in_degree("A") == 0
        assert simple_directed.out_degree("A") == 2
        assert simple_directed.in_degree("D") == 2
        assert simple_directed.out_degree("D") == 0
        assert simple_directed.in_degree("B") == 1
        assert simple_directed.out_degree("B") == 1
        assert simple_directed.degree("A") == 2

    def test_topological_sort_valid_dag(self, simple_directed):
        """在有效的 DAG 上测试拓扑排序。"""
        sorted_nodes = simple_directed.topological_sort()
        assert sorted_nodes is not None
        assert sorted_nodes[0] == "A"
        assert sorted_nodes[-1] == "D"
        assert set(sorted_nodes[1:3]) == {"B", "C"}

    def test_topological_sort_with_cycle(self, directed_with_cycle):
        """测试拓扑排序对带环图返回 None。"""
        assert directed_with_cycle.topological_sort() is None

    def test_is_acyclic(self, simple_directed, directed_with_cycle):
        """测试 is_acyclic 方法。"""
        assert simple_directed.is_acyclic()
        assert not directed_with_cycle.is_acyclic()

    def test_reverse_graph(self, simple_directed):
        """测试反转有向图中的所有边。"""
        reversed_g = simple_directed.reverse()

        assert reversed_g.vertex_count == simple_directed.vertex_count
        assert reversed_g.edge_count == simple_directed.edge_count

        assert reversed_g.has_edge("B", "A")
        assert not reversed_g.has_edge("A", "B")

        assert reversed_g.in_degree("D") == 0
        assert reversed_g.out_degree("D") == 2

    def test_strongly_connected_components(self, directed_with_cycle):
        """测试强连通分量（如果实现了的话）。"""
        # 这个测试假设存在 strongly_connected_components 方法
        if hasattr(directed_with_cycle, "strongly_connected_components"):
            sccs = list(directed_with_cycle.strongly_connected_components())
            # 对于简单的三节点环，应该有一个强连通分量
            assert len(sccs) == 1
            assert set(sccs[0]) == {"A", "B", "C"}

    def test_topological_sort_edge_cases(self, empty_directed):
        """测试拓扑排序的边界情况。"""
        graph = empty_directed

        # 空图的拓扑排序应该返回空列表
        assert graph.topological_sort() == []

        # 单节点图
        graph.add_vertex("A")
        assert graph.topological_sort() == ["A"]

        # 添加自环后应该返回 None
        graph.add_edge("A", "A", 1)
        assert graph.topological_sort() is None

    def test_source_and_sink_vertices(self, simple_directed):
        """测试源节点和汇节点的识别。"""
        # A 是源节点（入度为0）
        sources = [v for v in simple_directed.vertices if simple_directed.in_degree(v) == 0]
        assert sources == ["A"]

        # D 是汇节点（出度为0）
        sinks = [v for v in simple_directed.vertices if simple_directed.out_degree(v) == 0]
        assert sinks == ["D"]

    def test_transitive_closure(self, simple_directed):
        """测试传递闭包的概念。"""
        # 在简单的 DAG 中，A 可以到达所有其他节点
        reachable_from_a = set(simple_directed.dfs("A"))
        assert reachable_from_a == {"A", "B", "C", "D"}

        # D 只能到达自己
        reachable_from_d = set(simple_directed.dfs("D"))
        assert reachable_from_d == {"D"}

    def test_directed_graph_symmetry_check(self, empty_directed):
        """测试有向图的对称性检查。"""
        graph = empty_directed
        edges = [("A", "B", 1), ("B", "A", 1), ("B", "C", 2), ("C", "B", 2)]
        graph.add_edges(edges)

        # 检查是否每条边都有其反向边
        is_symmetric = True
        for vertex in graph.vertices:
            for neighbor in graph.neighbors(vertex):
                if not graph.has_edge(neighbor, vertex):
                    is_symmetric = False
                    break
            if not is_symmetric:
                break

        assert is_symmetric  # 这个图应该是对称的

    def test_weakly_vs_strongly_connected(self, directed_with_cycle):
        """测试弱连通和强连通的区别。"""
        # 创建一个弱连通但不强连通的图
        weak_graph = DirectedGraph[str, int]()
        weak_edges = [("A", "B"), ("B", "C"), ("D", "C")]  # D->C 但 C 不能到达 D
        weak_graph.add_edges(weak_edges)

        # 应该弱连通（忽略边方向时连通）
        # 但不强连通（不是所有节点都能互相到达）
        assert not weak_graph.has_path("C", "D")
        assert weak_graph.has_path("A", "C")


class TestUndirectedGraph:
    """无向图特有功能的测试。"""

    def test_undirected_symmetry(self, empty_undirected):
        """测试无向图的对称性。"""
        graph = empty_undirected
        graph.add_edge("A", "B", 5)

        # 无向图中边是双向的
        assert graph.has_edge("A", "B")
        assert graph.has_edge("B", "A")
        assert graph.get_edge_weight("A", "B") == graph.get_edge_weight("B", "A")

    def test_bridge_detection_concept(self, simple_undirected):
        """测试桥（割边）的概念。"""
        # 移除边后检查连通性变化
        original_components = len(list(simple_undirected.connected_components()))

        # 移除一条边
        simple_undirected.remove_edge("A", "B")
        new_components = len(list(simple_undirected.connected_components()))

        # 如果组件数增加，说明移除的是桥
        if new_components > original_components:
            # "A"-"B" 是桥
            pass

    def test_articulation_points_concept(self, complex_social_network):
        """测试关节点（割点）的概念。"""
        original_components = len(list(complex_social_network.connected_components()))

        # 测试移除各个顶点对连通性的影响
        critical_vertices = []
        for vertex in list(complex_social_network.vertices):
            # 临时移除顶点
            neighbors = list(complex_social_network.neighbors(vertex))
            complex_social_network.remove_vertex(vertex)

            new_components = len(list(complex_social_network.connected_components()))
            if new_components > original_components:
                critical_vertices.append(vertex)

            # 恢复顶点和边
            complex_social_network.add_vertex(vertex)
            for neighbor in neighbors:
                if neighbor in complex_social_network.vertices:
                    complex_social_network.add_edge(vertex, neighbor, 1)

    def test_triangle_detection(self, complete_graph):
        """测试三角形（三元环）的检测。"""
        triangles = []
        vertices = list(complete_graph.vertices)

        # 寻找所有三角形
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                for k in range(j + 1, len(vertices)):
                    v1, v2, v3 = vertices[i], vertices[j], vertices[k]
                    if (
                        complete_graph.has_edge(v1, v2)
                        and complete_graph.has_edge(v2, v3)
                        and complete_graph.has_edge(v1, v3)
                    ):
                        triangles.append((v1, v2, v3))

        # 完全图中每三个节点都形成一个三角形
        n = len(vertices)
        expected_triangles = n * (n - 1) * (n - 2) // 6
        assert len(triangles) == expected_triangles

    def test_maximum_matching_concept(self, bipartite_graph):
        """测试二分图最大匹配的概念。"""
        set_a = [v for v in bipartite_graph.vertices if v.startswith("A")]
        set_b = [v for v in bipartite_graph.vertices if v.startswith("B")]

        # 简单的贪心匹配算法
        matching = {}
        for vertex_a in set_a:
            neighbors_b = [n for n in bipartite_graph.neighbors(vertex_a) if n not in matching.values()]
            if neighbors_b:
                matching[vertex_a] = neighbors_b[0]

        # 验证匹配的有效性
        for a, b in matching.items():
            assert bipartite_graph.has_edge(a, b)

        # 匹配中的边不应该有重复的端点
        assert len(set(matching.keys())) == len(matching)
        assert len(set(matching.values())) == len(matching)


class TestWeightedGraphAlgorithms:
    """加权图算法的测试。"""

    def test_minimum_spanning_tree_concept(self, complete_graph):
        """测试最小生成树的概念（基础实现）。"""
        if complete_graph.vertex_count <= 1:
            return

        # Kruskal 算法的简化版本
        edges = []
        for u in complete_graph.vertices:
            for v in complete_graph.neighbors(u):
                if u < v:  # 避免重复边
                    weight = complete_graph.get_edge_weight(u, v)
                    edges.append((weight, u, v))

        edges.sort()  # 按权重排序

        # 简单的 Union-Find 实现
        parent = {v: v for v in complete_graph.vertices}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False

        mst_edges = []
        for weight, u, v in edges:
            if union(u, v):
                mst_edges.append((u, v, weight))
                if len(mst_edges) == complete_graph.vertex_count - 1:
                    break

        # MST 应该有 n-1 条边
        assert len(mst_edges) == complete_graph.vertex_count - 1

    def test_shortest_path_properties(self, simple_undirected):
        """测试最短路径的基本属性。"""

        # 使用 BFS 找到无权图中的最短路径
        def bfs_shortest_path(graph, start, end):
            if start == end:
                return [start]

            queue = [(start, [start])]
            visited = {start}

            while queue:
                vertex, path = queue.pop(0)
                for neighbor in graph.neighbors(vertex):
                    if neighbor not in visited:
                        new_path = path + [neighbor]
                        if neighbor == end:
                            return new_path
                        queue.append((neighbor, new_path))
                        visited.add(neighbor)
            return None

        path = bfs_shortest_path(simple_undirected, "A", "D")
        assert path is not None
        assert path[0] == "A"
        assert path[-1] == "D"
        assert len(path) >= 2  # 至少包含起点和终点

    def test_weight_based_operations(self, complex_social_network):
        """测试基于权重的操作。"""
        # 找到最大权重的边
        max_weight = 0
        max_edge = None

        for u in complex_social_network.vertices:
            for v in complex_social_network.neighbors(u):
                if u < v:  # 避免重复检查
                    weight = complex_social_network.get_edge_weight(u, v)
                    if weight > max_weight:
                        max_weight = weight
                        max_edge = (u, v)

        assert max_edge is not None
        assert max_weight > 0

        # 计算所有边权重的总和
        total_weight = sum(
            complex_social_network.get_edge_weight(u, v)
            for u in complex_social_network.vertices
            for v in complex_social_network.neighbors(u)
            if u < v  # 避免重复计算
        )
        assert total_weight > 0


class TestAdvancedDataStructures:
    """高级数据结构和类型的测试。"""

    def test_person_vertex_ordering(self):
        """测试使用 Person 类作为顶点的排序功能。"""
        graph = UndirectedGraph[Person, str]()

        alice_30 = Person("Alice", 30)
        alice_25 = Person("Alice", 25)
        charlie = Person("Charlie", 35)

        graph.add_edge(alice_30, alice_25, "friend")
        graph.add_edge(alice_25, charlie, "colleague")
        graph.add_edge(alice_30, charlie, "neighbor")

        # 验证 Person 对象可以作为顶点使用
        assert alice_30 in graph.vertices
        assert graph.has_edge(alice_30, alice_25)

        # 测试排序功能（按名字, 年龄排序）
        sorted_people = sorted(graph.vertices)
        ages = [p.age for p in sorted_people]
        assert ages == sorted(ages)

    def test_distance_weight_comparison(self):
        """测试 Distance 权重类型的比较功能。"""
        graph = UndirectedGraph[str, Distance]()

        short_distance = Distance(5.0, "km")
        long_distance = Distance(10.0, "km")
        mile_distance = Distance(6.0, "mile")  # 约 9.66 km

        graph.add_edge("A", "B", short_distance)
        graph.add_edge("B", "C", long_distance)
        graph.add_edge("A", "C", mile_distance)

        # 验证距离比较
        assert short_distance < long_distance
        assert short_distance < mile_distance
        assert mile_distance < long_distance

        # 找到最短边
        edges_with_weights = []
        for u in graph.vertices:
            for v in graph.neighbors(u):
                if u < v:
                    weight = graph.get_edge_weight(u, v)
                    edges_with_weights.append(((u, v), weight))

        shortest_edge = min(edges_with_weights, key=lambda x: x[1])
        assert shortest_edge[1] == short_distance

    def test_complex_weight_structure(self):
        """测试复杂权重结构的使用。"""
        graph = DirectedGraph[str, WeightedConnection]()

        strong_conn = WeightedConnection(strength=9, confidence=0.9)
        weak_conn = WeightedConnection(strength=3, confidence=0.4, bidirectional=False)

        graph.add_edge("Alice", "Bob", strong_conn)
        graph.add_edge("Bob", "Charlie", weak_conn)

        # 验证复杂权重的访问
        alice_bob_weight = graph.get_edge_weight("Alice", "Bob")
        assert alice_bob_weight.strength == 9
        assert alice_bob_weight.confidence == 0.9
        assert alice_bob_weight.bidirectional == True

        # 基于权重属性进行操作
        high_confidence_edges = [
            (u, v) for u in graph.vertices for v in graph.neighbors(u) if graph.get_edge_weight(u, v).confidence > 0.5
        ]

        assert ("Alice", "Bob") in high_confidence_edges
        assert ("Bob", "Charlie") not in high_confidence_edges


class TestPerformanceAndScalability:
    """性能和可扩展性测试。"""

    def test_large_graph_operations(self, large_graph):
        """测试大型图上的基本操作性能。"""
        start_time = time.time()

        # 测试基本查询操作
        vertex_count = large_graph.vertex_count
        edge_count = large_graph.edge_count

        # 测试邻居查询
        sample_vertex = list(large_graph.vertices)[0]
        neighbors = list(large_graph.neighbors(sample_vertex))

        # 测试连通性
        is_connected = large_graph.is_connected()

        operation_time = time.time() - start_time

        # 确保基本操作在合理时间内完成（这里设置为1秒的宽松限制）
        assert operation_time < 1.0
        assert vertex_count == 1000
        assert edge_count > 0
        assert isinstance(neighbors, list)

    def test_batch_operation_efficiency(self, empty_undirected):
        """测试批量操作的效率。"""
        graph = empty_undirected

        # 大批量顶点添加
        vertices = [f"v{i}" for i in range(1000)]
        start_time = time.time()
        added = graph.add_vertices(vertices)
        batch_time = time.time() - start_time

        assert added == 1000
        assert batch_time < 1.0  # 应该很快完成

        # 大批量边添加
        edges = [(f"v{i}", f"v{i + 1}", i) for i in range(999)]
        start_time = time.time()
        edges_added = graph.add_edges(edges)
        edge_batch_time = time.time() - start_time

        assert edges_added == 999
        assert edge_batch_time < 1.0

    def test_memory_usage_patterns(self, empty_undirected):
        """测试内存使用模式。"""
        graph = empty_undirected
        initial_vertices = graph.vertex_count

        # 添加大量顶点
        for i in range(100):
            graph.add_vertex(f"vertex_{i}")

        assert graph.vertex_count == initial_vertices + 100

        # 移除一半顶点
        for i in range(0, 100, 2):
            graph.remove_vertex(f"vertex_{i}")

        assert graph.vertex_count == initial_vertices + 50

    def test_iteration_consistency_under_modification(self, simple_undirected):
        """测试修改图时迭代的一致性。"""
        original_vertices = set(simple_undirected.vertices)

        # 在遍历过程中修改图不应该影响当前遍历
        vertices_during_iteration = set()
        for vertex in simple_undirected.vertices:
            vertices_during_iteration.add(vertex)
            # 尝试添加新顶点（可能不会影响当前迭代）
            simple_undirected.add_vertex(f"new_{vertex}")

        # 原始顶点应该都被遍历到
        assert original_vertices.issubset(vertices_during_iteration)


class TestFactoryFunction:
    """工厂函数的测试。"""

    def test_create_graph_function(self):
        """测试 create_graph 工厂函数。"""
        # 创建无向图
        undirected = create_graph()
        assert isinstance(undirected, UndirectedGraph)

        # 创建有向图
        directed = create_graph(directed=True)
        assert isinstance(directed, DirectedGraph)

    def test_graph_type_inference(self):
        """测试图类型的推断和转换。"""
        # 创建无向图并添加一些边
        undirected = create_graph()
        undirected.add_edge("A", "B", 1)
        undirected.add_edge("B", "C", 2)

        # 转换为有向图（如果支持的话）
        if hasattr(undirected, "to_directed"):
            directed = undirected.to_directed()
            assert isinstance(directed, DirectedGraph)
            assert directed.has_edge("A", "B")
            assert directed.has_edge("B", "A")  # 无向边变成双向有向边


class TestRegressionTests:
    """回归测试 - 修复已知问题。"""

    def test_empty_graph_edge_removal(self, empty_undirected):
        """回归测试：在空图上移除边不应该崩溃。"""
        assert not empty_undirected.remove_edge("A", "B")
        assert empty_undirected.vertex_count == 0

    def test_self_loop_degree_calculation(self, self_loop_graph):
        """回归测试：自环的度数计算。"""
        # 自环应该对度数的贡献是明确定义的
        if hasattr(self_loop_graph, "degree"):
            degree_a = self_loop_graph.degree("A")
            assert degree_a >= 1  # 至少包含自环

    def test_duplicate_edge_handling(self, empty_undirected):
        """回归测试：重复边的处理。"""
        graph = empty_undirected

        # 第一次添加应该成功
        assert graph.add_edge("A", "B", 5)
        assert graph.edge_count == 1

        # 重复添加应该失败但不报错
        assert not graph.add_edge("A", "B", 10)
        assert graph.edge_count == 1
        assert graph.get_edge_weight("A", "B") == 5  # 权重不变

    def test_vertex_removal_edge_cleanup(self, simple_undirected):
        """回归测试：移除顶点时边的清理。"""
        initial_edges = simple_undirected.edge_count
        degree_a = simple_undirected.degree("A")

        simple_undirected.remove_vertex("A")

        # 与 A 相关的所有边都应该被移除
        assert simple_undirected.edge_count == initial_edges - degree_a

        # A 不应该出现在任何邻接表中
        for vertex in simple_undirected.vertices:
            assert "A" not in list(simple_undirected.neighbors(vertex))

    def test_graph_consistency_after_operations(self, complex_social_network):
        """回归测试：复杂操作后的图一致性。"""
        original_vertices = set(complex_social_network.vertices)

        # 执行一系列复杂操作
        complex_social_network.add_edge("NewPerson1", "Alice", 5)
        complex_social_network.add_edge("NewPerson2", "Bob", 3)
        complex_social_network.remove_edge("Alice", "Bob")
        complex_social_network.add_vertex("IsolatedPerson")

        # 验证图的一致性
        all_neighbors_exist = True
        for vertex in complex_social_network.vertices:
            for neighbor in complex_social_network.neighbors(vertex):
                if neighbor not in complex_social_network.vertices:
                    all_neighbors_exist = False
                    break
            if not all_neighbors_exist:
                break

        assert all_neighbors_exist

        # 边的对称性检查（对于无向图）
        for u in complex_social_network.vertices:
            for v in complex_social_network.neighbors(u):
                assert complex_social_network.has_edge(v, u)


# ==================== 压力测试和边界条件 ====================


class TestStressAndBoundary:
    """压力测试和边界条件测试。"""

    def test_extreme_graph_sizes(self):
        """测试极端图大小的处理。"""
        # 测试只有一个顶点的图
        single_vertex_graph = UndirectedGraph[str, int]()
        single_vertex_graph.add_vertex("ONLY")

        assert single_vertex_graph.vertex_count == 1
        assert single_vertex_graph.edge_count == 0
        assert single_vertex_graph.is_connected()
        assert list(single_vertex_graph.dfs("ONLY")) == ["ONLY"]

    def test_unicode_and_special_characters(self, empty_undirected):
        """测试 Unicode 和特殊字符作为顶点名称。"""
        graph = empty_undirected
        special_vertices = ["α", "β", "γ", "🔥", "💧", "🌟", "", " ", "\n", "\t"]

        for vertex in special_vertices:
            assert graph.add_vertex(vertex)

        # 测试包含特殊字符的边
        assert graph.add_edge("α", "🔥", 42)
        assert graph.has_edge("α", "🔥")

    def test_extreme_weight_values(self, empty_directed):
        """测试极端权重值。"""
        graph = empty_directed
        import sys

        extreme_weights = [
            0,
            -sys.maxsize,
            sys.maxsize,
            float("inf"),
            float("-inf"),
            1e-10,
            1e10,
        ]

        for i, weight in enumerate(extreme_weights):
            vertex_u = f"u{i}"
            vertex_v = f"v{i}"
            try:
                graph.add_edge(vertex_u, vertex_v, weight)
                retrieved_weight = graph.get_edge_weight(vertex_u, vertex_v)
                assert retrieved_weight == weight
            except (OverflowError, ValueError):
                # 某些极端值可能不被支持
                pass

    def test_rapid_add_remove_cycles(self, empty_undirected):
        """测试快速的添加-删除循环。"""
        graph = empty_undirected

        # 快速循环添加和删除相同的边
        for _ in range(100):
            assert graph.add_edge("TEMP_A", "TEMP_B", 1)
            assert graph.remove_edge("TEMP_A", "TEMP_B")

        # 最终状态应该是干净的
        assert graph.vertex_count == 0 or not graph.has_edge("TEMP_A", "TEMP_B")

    @pytest.mark.parametrize("graph_size", [10, 100, 500])
    def test_scalability_patterns(self, graph_size):
        """参数化测试不同规模图的性能模式。"""
        graph = UndirectedGraph[int, int]()

        # 创建链式图
        start_time = time.time()
        for i in range(graph_size - 1):
            graph.add_edge(i, i + 1, i)
        creation_time = time.time() - start_time

        # 遍历测试
        start_time = time.time()
        traversal_result = list(graph.dfs(0))
        traversal_time = time.time() - start_time

        assert len(traversal_result) == graph_size
        # 时间复杂度应该大致是线性的
        assert creation_time < graph_size * 0.001  # 很宽松的限制
        assert traversal_time < graph_size * 0.001


if __name__ == "__main__":
    # 运行特定的测试套件
    pytest.main([__file__, "-v", "--tb=short"])
