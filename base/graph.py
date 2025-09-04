from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Iterator
from typing import Any


class Graph(ABC):
    """图的抽象基类 - Pythonic版本"""

    def __init__(self):
        self._vertices: set[Any] = set()
        self._edge_count = 0

    def add_vertex(self, vertex: Any) -> bool:
        """添加顶点"""
        if vertex in self._vertices:
            return False
        self._vertices.add(vertex)
        self._add_vertex_internal(vertex)
        return True

    def remove_vertex(self, vertex: Any) -> bool:
        """删除顶点"""
        if vertex not in self._vertices:
            return False

        # 获取所有相关边并删除
        incident_edges = list(self.get_incident_edges(vertex))
        for u, v in incident_edges:
            self.remove_edge(u, v)

        self._vertices.remove(vertex)
        self._remove_vertex_internal(vertex)
        return True

    def add_edge(self, u: Any, v: Any, weight: Any = 1) -> bool:
        """添加边"""
        # 自动添加不存在的顶点
        self.add_vertex(u)
        self.add_vertex(v)

        if self.has_edge(u, v):
            return False

        self._add_edge_internal(u, v, weight)
        self._edge_count += 1
        return True

    def remove_edge(self, u: Any, v: Any) -> bool:
        """删除边"""
        if not self.has_edge(u, v):
            return False

        self._remove_edge_internal(u, v)
        self._edge_count -= 1
        return True

    # 属性访问器
    @property
    def vertices(self) -> set[Any]:
        return self._vertices.copy()

    @property
    def vertex_count(self) -> int:
        return len(self._vertices)

    @property
    def edge_count(self) -> int:
        return self._edge_count

    # 抽象方法
    @abstractmethod
    def _add_vertex_internal(self, vertex: Any) -> None:
        pass

    @abstractmethod
    def _remove_vertex_internal(self, vertex: Any) -> None:
        pass

    @abstractmethod
    def _add_edge_internal(self, u: Any, v: Any, weight: Any) -> None:
        pass

    @abstractmethod
    def _remove_edge_internal(self, u: Any, v: Any) -> None:
        pass

    @abstractmethod
    def has_edge(self, u: Any, v: Any) -> bool:
        pass

    @abstractmethod
    def neighbors(self, vertex: Any) -> Iterator[Any]:
        pass

    @abstractmethod
    def get_incident_edges(self, vertex: Any) -> Iterator[tuple[Any, Any]]:
        pass

    @abstractmethod
    def is_directed(self) -> bool:
        pass

    # 通用算法
    def dfs(self, start: Any) -> list[Any]:
        """深度优先搜索"""
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

    def bfs(self, start: Any) -> list[Any]:
        """广度优先搜索"""
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

    def __contains__(self, vertex: Any) -> bool:
        """支持 'vertex in graph' 语法"""
        return vertex in self._vertices

    def __len__(self) -> int:
        """支持 len(graph) 语法"""
        return len(self._vertices)

    def __repr__(self) -> str:
        edge_list = list(self.get_incident_edges(v) for v in self._vertices)
        all_edges = set()
        for edges in edge_list:
            all_edges.update(edges)
        return f"{self.__class__.__name__}(vertices={len(self._vertices)}, edges={len(all_edges)})"


class UndirectedGraph(Graph):
    """无向图 - 使用邻接多重表思想但Pythonic实现"""

    def __init__(self):
        super().__init__()
        self._adj: dict[Any, set[Any]] = defaultdict(set)
        self._edges: dict[tuple[Any, Any], Any] = {}  # 存储边权重

    def _add_vertex_internal(self, vertex: Any) -> None:
        if vertex not in self._adj:
            self._adj[vertex] = set()

    def _remove_vertex_internal(self, vertex: Any) -> None:
        if vertex in self._adj:
            del self._adj[vertex]

    def _add_edge_internal(self, u: Any, v: Any, weight: Any) -> None:
        self._adj[u].add(v)
        self._adj[v].add(u)
        # 统一边的存储顺序
        edge_key = (min(u, v), max(u, v))
        self._edges[edge_key] = weight

    def _remove_edge_internal(self, u: Any, v: Any) -> None:
        self._adj[u].discard(v)
        self._adj[v].discard(u)
        edge_key = (min(u, v), max(u, v))
        self._edges.pop(edge_key, None)

    def has_edge(self, u: Any, v: Any) -> bool:
        return u in self._adj and v in self._adj[u]

    def neighbors(self, vertex: Any) -> Iterator[Any]:
        """获取邻接顶点"""
        return iter(self._adj.get(vertex, set()))

    def get_incident_edges(self, vertex: Any) -> Iterator[tuple[Any, Any]]:
        """获取关联的边"""
        for neighbor in self._adj.get(vertex, set()):
            yield vertex, neighbor

    def degree(self, vertex: Any) -> int:
        """获取度数"""
        return len(self._adj.get(vertex, set()))

    def get_edge_weight(self, u: Any, v: Any) -> Any:
        """获取边权重"""
        edge_key = (min(u, v), max(u, v))
        return self._edges.get(edge_key)

    def is_directed(self) -> bool:
        return False

    def __getitem__(self, vertex: Any) -> set[Any]:
        """支持 graph[vertex] 语法获取邻接点"""
        return self._adj.get(vertex, set()).copy()


class DirectedGraph(Graph):
    """有向图 - 使用十字链表思想但Pythonic实现"""

    def __init__(self):
        super().__init__()
        self._out_adj: dict[Any, set[Any]] = defaultdict(set)  # 出邻接表
        self._in_adj: dict[Any, set[Any]] = defaultdict(set)  # 入邻接表
        self._edges: dict[tuple[Any, Any], Any] = {}  # 边权重

    def _add_vertex_internal(self, vertex: Any) -> None:
        if vertex not in self._out_adj:
            self._out_adj[vertex] = set()
        if vertex not in self._in_adj:
            self._in_adj[vertex] = set()

    def _remove_vertex_internal(self, vertex: Any) -> None:
        self._out_adj.pop(vertex, None)
        self._in_adj.pop(vertex, None)

    def _add_edge_internal(self, u: Any, v: Any, weight: Any) -> None:
        self._out_adj[u].add(v)
        self._in_adj[v].add(u)
        self._edges[(u, v)] = weight

    def _remove_edge_internal(self, u: Any, v: Any) -> None:
        self._out_adj[u].discard(v)
        self._in_adj[v].discard(u)
        self._edges.pop((u, v), None)

    def has_edge(self, u: Any, v: Any) -> bool:
        return u in self._out_adj and v in self._out_adj[u]

    def neighbors(self, vertex: Any) -> Iterator[Any]:
        """获取后继顶点（出邻接）"""
        return iter(self._out_adj.get(vertex, set()))

    def predecessors(self, vertex: Any) -> Iterator[Any]:
        """获取前驱顶点（入邻接）"""
        return iter(self._in_adj.get(vertex, set()))

    def get_incident_edges(self, vertex: Any) -> Iterator[tuple[Any, Any]]:
        """获取关联的边（出边+入边）"""
        # 出边
        for successor in self._out_adj.get(vertex, set()):
            yield vertex, successor
        # 入边
        for predecessor in self._in_adj.get(vertex, set()):
            yield predecessor, vertex

    def out_degree(self, vertex: Any) -> int:
        """出度"""
        return len(self._out_adj.get(vertex, set()))

    def in_degree(self, vertex: Any) -> int:
        """入度"""
        return len(self._in_adj.get(vertex, set()))

    def degree(self, vertex: Any) -> int:
        """总度数（入度+出度）"""
        return self.in_degree(vertex) + self.out_degree(vertex)

    def get_edge_weight(self, u: Any, v: Any) -> Any:
        """获取边权重"""
        return self._edges.get((u, v))

    def is_directed(self) -> bool:
        return True

    def topological_sort(self) -> list[Any] | None:
        """拓扑排序"""
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

    def __getitem__(self, vertex: Any) -> tuple[set[Any], set[Any]]:
        """返回 (出邻接点, 入邻接点)"""
        out_neighbors = self._out_adj.get(vertex, set()).copy()
        in_neighbors = self._in_adj.get(vertex, set()).copy()
        return out_neighbors, in_neighbors


# ==================== 使用示例 ====================


def demo():
    print("=== Pythonic 图实现演示 ===\n")

    # 无向图演示
    print("无向图:")
    ug = UndirectedGraph()

    # 简洁的API
    ug.add_edge("A", "B", weight=5)
    ug.add_edge("A", "C", weight=3)
    ug.add_edge("B", "D", weight=7)
    ug.add_edge("C", "D", weight=2)

    print(f"图: {ug}")
    print(f"A的邻接点: {list(ug.neighbors('A'))}")
    print(f"A的度数: {ug.degree('A')}")
    print(f"边(A,B)权重: {ug.get_edge_weight('A', 'B')}")
    print(f"包含顶点A: {'A' in ug}")
    print(f"A的邻接点: {ug['A']}")  # 字典式访问

    # 有向图演示
    print("\n有向图:")
    dg = DirectedGraph()

    edges = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")]
    for u, v in edges:
        dg.add_edge(u, v)

    print(f"图: {dg}")
    print(f"A的后继: {list(dg.neighbors('A'))}")
    print(f"D的前驱: {list(dg.predecessors('D'))}")
    print(f"B的出度: {dg.out_degree('B')}, 入度: {dg.in_degree('B')}")

    # 拓扑排序
    topo_order = dg.topological_sort()
    print(f"拓扑排序: {topo_order}")

    # 遍历算法
    print(f"从A开始DFS: {ug.dfs('A')}")
    print(f"从A开始BFS: {dg.bfs('A')}")


if __name__ == "__main__":
    demo()
