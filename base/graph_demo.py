"""
图数据结构综合演示脚本

本脚本通过一系列真实世界的示例，全面展示了图数据结构库的功能。
涵盖了无向图、有向图、核心图算法、性能基准测试以及高级用例。
"""

import random
import time
from typing import TypeVar

# 导入图类
from graph import DirectedGraph, Graph, UndirectedGraph

# 定义类型变量以便于类型提示
V = TypeVar("V")
W = TypeVar("W")

# ==================== 辅助函数 ====================


def print_section(title: str) -> None:
    """打印格式化的章节标题。"""
    print(f"\n\n{'=' * 70}")
    print(f"▶ {title.upper()}")
    print(f"{'=' * 70}")


def print_subsection(title: str) -> None:
    """打印格式化的子章节标题。"""
    print(f"\n--- {title} ---")


def visualize_graph(graph: Graph[V, W], title: str) -> None:
    """以文本形式简单可视化图的结构。"""
    print(f"\n📊 {title}:")
    print(f"   - 顶点数 (Vertices): {graph.vertex_count}")
    print(f"   - 边数 (Edges): {graph.edge_count}")
    print(f"   - 图密度 (Density): {graph.density:.3f}")

    if graph.is_empty:
        print("   (图为空)")
        return
    if graph.vertex_count > 15:
        print("   (图太大，不在此完整显示)")
        return

    print("   结构详情:")
    # 对顶点进行排序以保证输出的一致性
    try:
        vertices = sorted(list(graph.vertices))
    except TypeError:
        # 如果顶点类型不支持排序，则不排序
        vertices = list(graph.vertices)

    # 用于在无向图中避免重复打印边
    printed_edges = set()

    for u in vertices:
        # 收集所有与 u 相关的边
        edges_str_list = []
        incident_edges = list(graph.incident_edges(u))

        if not incident_edges:
            print(f"     - {u} (孤立节点)")
            continue

        for edge in incident_edges:
            weight_str = f"({edge.weight})"

            if graph.is_directed():
                if edge.u == u:  # 出边
                    edges_str_list.append(f"--{weight_str}--> {edge.v}")
            else:  # 无向图
                v = edge.v if edge.u == u else edge.u
                # 通过排序确保边的唯一表示，避免重复
                edge_key = tuple(sorted((str(u), str(v))))
                if edge_key not in printed_edges:
                    edges_str_list.append(f"<--{weight_str}--> {v}")
                    printed_edges.add(edge_key)

        if edges_str_list:
            # 打印顶点的出边
            print(f"     - {u}:")
            for s in edges_str_list:
                print(f"         {s}")


# ==================== 演示模块 ====================


def demo_social_network() -> None:
    """使用无向图模拟和分析社交网络。"""
    print_section("场景一：社交网络分析 (Undirected Graph)")

    print("我们将创建一个无向图来表示一个社交网络，其中边权重代表关系的亲密度。")
    network = UndirectedGraph[str, int]()

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
        ("Jack", "Kate", 9),  # 另一个社交圈
    ]
    network.add_edges(friendships)

    visualize_graph(network, "社交网络图谱")

    print_subsection("网络洞察")

    # 1. 识别社交圈 (连通分量)
    components = list(network.connected_components())
    print(f"💡 发现 {len(components)} 个独立的社交圈:")
    for i, component in enumerate(components, 1):
        print(f"   - 社交圈 {i}: {sorted(component)} (共 {len(component)} 人)")

    # 2. 找到社交达人 (度最高的顶点)
    degrees = {person: network.degree(person) for person in network.vertices}
    most_connected = max(degrees, key=degrees.get)
    print(f"💡 社交达人是 {most_connected}，拥有 {degrees[most_connected]} 个好友。")

    # 3. 寻找最亲密的关系 (权重最大的边)
    strongest_edge = max(network._edges.values(), key=lambda e: e.weight)
    print(f"💡 最亲密的关系是 {strongest_edge.u} 和 {strongest_edge.v} (亲密度: {strongest_edge.weight})。")

    # 4. 探索连接路径
    print(f"💡 Alice 和 Ian 是否在同一个网络中? {'是' if network.has_path('Alice', 'Ian') else '否'}")
    print(f"💡 Alice 和 Jack 是否在同一个网络中? {'是' if network.has_path('Alice', 'Jack') else '否'}")


def demo_project_workflow() -> None:
    """使用有向图 (DAG) 规划和分析项目任务流。"""
    print_section("场景二：项目工作流管理 (Directed Acyclic Graph)")

    print("我们将创建一个有向无环图 (DAG) 来表示项目任务的依赖关系，权重为任务所需小时数。")
    project = DirectedGraph[str, int]()

    tasks = [
        ("需求分析", "UI设计", 8),
        ("需求分析", "后端架构", 12),
        ("UI设计", "前端开发", 24),
        ("后端架构", "数据库设计", 8),
        ("后端架构", "API开发", 30),
        ("数据库设计", "API开发", 16),
        ("前端开发", "集成测试", 16),
        ("API开发", "集成测试", 10),
        ("集成测试", "部署", 8),
        ("部署", "发布", 4),
    ]
    project.add_edges(tasks)

    visualize_graph(project, "项目任务依赖图")

    print_subsection("工作流分析")

    # 1. 确定任务执行顺序 (拓扑排序)
    if not project.is_acyclic():
        print("❌ 错误：项目任务流中存在循环依赖，无法进行拓扑排序！")
        return

    schedule = project.topological_sort()
    print("💡 推荐的任务执行顺序 (拓扑排序):")
    print(f"   {' → '.join(schedule)}")

    # 2. 识别关键节点
    degrees = {task: (project.in_degree(task), project.out_degree(task)) for task in project.vertices}
    # 入度为0的是起始任务，出度为0的是最终任务
    source_tasks = [t for t, (ind, _) in degrees.items() if ind == 0]
    sink_tasks = [t for t, (_, outd) in degrees.items() if outd == 0]
    print(f"💡 项目起始任务: {source_tasks}")
    print(f"💡 项目最终任务: {sink_tasks}")

    # 3. 识别最复杂的任务（连接数最多的节点）
    most_complex_task = max(degrees, key=lambda t: degrees[t][0] + degrees[t][1])
    in_d, out_d = degrees[most_complex_task]
    print(f"💡 最复杂的任务是 '{most_complex_task}' (依赖 {in_d} 个前置任务, 影响 {out_d} 个后续任务)。")


def demo_scc_and_cycles() -> None:
    """演示强连通分量 (SCC) 算法识别图中的循环依赖。"""
    print_section("场景三：识别复杂系统中的循环模块 (Strongly Connected Components)")

    print("我们将创建一个复杂的有向图，模拟系统模块间的调用关系，并使用 SCC 算法找出紧密耦合的模块组。")
    scc_graph = DirectedGraph[str, int]()
    edges = [
        # SCC 1: {A, B, C}
        ("A", "B"),
        ("B", "C"),
        ("C", "A"),
        # SCC 2: {D, E, F}
        ("D", "E"),
        ("E", "F"),
        ("F", "D"),
        # SCC 3: {G} (自循环)
        ("G", "G"),
        # 连接不同 SCC 的边
        ("C", "D"),
        ("B", "E"),  # 从 SCC1 到 SCC2
        ("F", "G"),  # 从 SCC2 到 SCC3
        # 其他无环路径
        ("H", "A"),
        ("H", "I"),
        ("I", "J"),
    ]
    scc_graph.add_edges(edges)
    scc_graph.add_vertex("K")  # 孤立模块

    visualize_graph(scc_graph, "系统模块调用图")

    print_subsection("强连通分量 (SCC) 分析")
    print("强连通分量 (SCC) 是图中的一个子集，其中任何一个节点都可以到达其他所有节点。")
    print("在软件工程中，一个 SCC 通常代表一组高度耦合、存在循环依赖的模块，它们需要被当作一个单元来对待。")

    sccs = list(scc_graph.strongly_connected_components())

    print(f"\n💡 在图中发现了 {len(sccs)} 个强连通分量:")
    for i, component in enumerate(sorted(sccs, key=len, reverse=True), 1):
        if len(component) > 1:
            print(f"   - 耦合模块组 {i}: {sorted(component)} (这是一个紧密耦合的循环)")
        else:
            print(f"   - 独立模块 {i}: {component}")


def demo_performance_benchmark() -> None:
    """对不同规模的图进行性能基准测试。"""
    print_section("性能基准测试")

    sizes = [100, 1000, 5000, 100000]
    edges_per_vertex = 3

    print("将对不同规模的随机图进行操作计时，以评估性能。")

    # 打印表头
    print("\n" + "-" * 85)
    print(
        f"{'规模 (V)':<10} | {'顶点数':>8} | {'边数':>8} | {'添加顶点':>10} | {'添加边':>10} | {'DFS遍历':>10} | {'BFS遍历':>10} | {'连通分量':>12}"
    )
    print("-" * 85)

    for n in sizes:
        graph = UndirectedGraph[int, int]()
        random.seed(42)

        # 1. 测量顶点添加时间
        start = time.perf_counter()
        graph.add_vertices(range(n))
        vertex_time = time.perf_counter() - start

        # 2. 测量边添加时间
        edges_to_add = n * edges_per_vertex
        start = time.perf_counter()
        for _ in range(edges_to_add):
            u, v = random.randint(0, n - 1), random.randint(0, n - 1)
            if u != v:
                graph.add_edge(u, v)
        edge_time = time.perf_counter() - start

        # 3. 测量 DFS 遍历时间
        start_vertex = random.randint(0, n - 1)
        start = time.perf_counter()
        _ = list(graph.dfs(start_vertex))
        dfs_time = time.perf_counter() - start

        # 4. 测量 BFS 遍历时间
        start_vertex = random.randint(0, n - 1)
        start = time.perf_counter()
        _ = list(graph.bfs(start_vertex))
        bfs_time = time.perf_counter() - start

        # 5. 测量连通分量计算时间
        start = time.perf_counter()
        _ = list(graph.connected_components())
        conn_time = time.perf_counter() - start

        # 打印结果行
        print(
            f"{n:<10} | {graph.vertex_count:>8} | {graph.edge_count:>8} | {f'{vertex_time:.4f}s':>10} | "
            f"{f'{edge_time:.4f}s':>10} | {f'{dfs_time:.4f}s':>10} | {f'{bfs_time:.4f}s':>10} | {f'{conn_time:.4f}s':>12}"
        )
    print("-" * 85)


def main() -> None:
    """主函数，按顺序运行所有演示。"""
    print("🔗 图数据结构 - 综合功能演示")

    try:
        demo_social_network()
        demo_project_workflow()
        demo_scc_and_cycles()
        demo_performance_benchmark()

        print_section("演示完成")
        print("🎉 图数据结构库成功演示了以下核心功能:")
        print("  ✓ 高效的顶点和边操作")
        print("  ✓ 对无向图和有向图的良好支持")
        print("  ✓ 核心图算法 (DFS, BFS, 连通分量)")
        print("  ✓ 高级有向图算法 (拓扑排序, 强连通分量)")
        print("  ✓ 在不同规模下保持良好性能")

    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        # 在调试时可以取消下面的注释以获得完整的堆栈跟踪
        # raise


if __name__ == "__main__":
    main()
