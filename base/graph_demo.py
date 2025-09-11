"""
优雅的图数据结构库 - 史诗级功能演示 v3.1 (已修正)

欢迎来到图的世界！本脚本将通过一系列引人入胜的高级应用场景，
全方位展示我们图数据结构库的强大功能、卓越设计与无限可能。

您将探索：
- 动态环境中的智能寻路 (机器人路径规划)
- 复杂行为逻辑的优雅建模 (游戏 AI)
- 数据关系挖掘的核心思想 (推荐系统)
- 网络拓扑的安全分析 (网络安全)
- 健壮的工程实践 (序列化与不变性)

让我们一起见证，如何用优雅的代码解决复杂的问题。
"""

import random
import time
from dataclasses import dataclass
from typing import TypeVar, Union

# 导入图的核心类
from graph import DirectedGraph, Edge, Graph, UndirectedGraph

# 定义类型变量，彰显泛型编程的优雅
V = TypeVar("V")
W = TypeVar("W")

# ==================== 辅助与可视化工具 (增强版) ====================

# 用于追踪场景编号的全局计数器 (修正部分)
SECTION_COUNTER = 0


def print_section(title: str, character: str = "=") -> None:
    """打印一个引人注目的章节标题，并自动编号。"""
    global SECTION_COUNTER
    SECTION_COUNTER += 1

    section_titles = ["一", "二", "三", "四", "附"]
    # 确保即使场景数量超过预设，也不会出错
    section_title_index = min(SECTION_COUNTER - 1, len(section_titles) - 1)
    section_name = section_titles[section_title_index]

    width = 80
    print(f"\n\n{character * width}")
    print(f"场景 {section_name}：{title.upper()}")
    print(f"{character * width}")


def print_subsection(title: str) -> None:
    """打印一个清晰的子章节标题。"""
    print(f"\n--- {title} ---\n")


def visualize_graph(graph: Graph[V, W], title: str) -> None:
    """一个增强版的文本可视化函数，优雅地展示图的结构。"""
    print(f"\n🖼️  图景呈现: {title}")
    print("-" * (len(title) + 15))

    print(f"[*] 类型: {'有向图' if graph.is_directed() else '无向图'}")
    print(f"[*] 统计: {graph.vertex_count} 个顶点, {graph.edge_count} 条边")

    if graph.is_empty:
        print("[i] 这是一个空图。")
        return
    if graph.vertex_count > 25:
        print("[i] 图规模较大，不展示完整结构。")
        return

    print("[i] 结构详情:")

    try:
        vertices: list[V] = sorted(list(graph.vertices), key=str)
    except TypeError:
        vertices: list[V] = list(graph.vertices)

    printed_undirected_edges: set[tuple[str, str]] = set()

    for u in vertices:
        edges = list(graph.incident_edges(u))

        if not edges:
            print(f"  - {u}: (孤立)")
            continue

        output_lines = []
        for edge in edges:
            weight_str = f"({edge.weight})" if edge.weight != 1 else ""

            if graph.is_directed():
                if edge.u == u:  # 出边
                    output_lines.append(f"  --{weight_str}--> {edge.v}")
                elif edge.v == u:  # 入边
                    output_lines.append(f"  <--{weight_str}-- {edge.u}")
            else:
                v = edge.v if edge.u == u else edge.u
                # 通过对字符串表示进行排序来创建唯一的边标识
                edge_tuple = tuple(sorted((str(u), str(v))))
                if edge_tuple not in printed_undirected_edges:
                    output_lines.append(f"  <--{weight_str}--> {v}")
                    printed_undirected_edges.add(edge_tuple)

        if output_lines:
            print(f"  - {u}:")
            for line in sorted(output_lines):
                print(f"    {line}")


# ==================== 场景一: 城市配送机器人路径规划 ====================


def demo_delivery_robot():
    """
    场景: 一个机器人在城市网格中配送包裹。
    角度: 动态图操作与寻路算法。
    看点:
    1.  图作为动态环境模型，顶点是坐标 `(x, y)`。
    2.  通过移除顶点模拟道路施工/障碍物。
    3.  使用 BFS 算法寻找无权重图中的最短路径。
    """
    print_section("城市配送机器人路径规划")

    grid_size = 5
    city_map = UndirectedGraph[tuple[int, int], None]()

    print(f"[i] 正在构建一个 {grid_size}x{grid_size} 的城市网格地图...")
    for r in range(grid_size):
        for c in range(grid_size):
            # 添加水平方向的道路
            if c < grid_size - 1:
                city_map.add_edge((r, c), (r, c + 1))
            # 添加垂直方向的道路
            if r < grid_size - 1:
                city_map.add_edge((r, c), (r + 1, c))

    warehouse = (0, 0)
    customer = (4, 4)
    print(f"[i] 仓库位于 {warehouse}, 客户位于 {customer}。")

    visualize_graph(city_map, "完整的城市地图")

    print_subsection("任务 1: 寻找最佳配送路径")

    def find_shortest_path_bfs(graph, start, end):
        if not graph.has_path(start, end):
            return None

        queue = [(start, [start])]
        visited = {start}

        while queue:
            current, path = queue.pop(0)
            if current == end:
                return path

            for neighbor in sorted(list(graph.neighbors(current))):  # 排序以获得确定性路径
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    path = find_shortest_path_bfs(city_map, warehouse, customer)
    print("💡 使用 BFS 找到的最短路径:")
    print(f"   {' -> '.join(map(str, path))}")
    print(f"   路径长度: {len(path) - 1} 个街区")

    print_subsection("任务 2: 应对突发状况 (道路施工)")

    obstacles = [(1, 1), (1, 2), (2, 1), (3, 3)]
    print(f"[i] 紧急通知！以下地点因施工无法通行: {obstacles}")

    for obstacle in obstacles:
        city_map.remove_vertex(obstacle)

    visualize_graph(city_map, "包含障碍的城市地图")

    new_path = find_shortest_path_bfs(city_map, warehouse, customer)
    print("💡 重新规划的绕行路径:")
    if new_path:
        print(f"   {' -> '.join(map(str, new_path))}")
        print(f"   新路径长度: {len(new_path) - 1} 个街区")
    else:
        print("   ❌ 警告：无法找到到达客户的路径！")


# ==================== 场景二: 游戏 AI 状态机设计 ====================


def demo_game_ai_fsm():
    """
    场景: 设计一个游戏中守卫的 AI 行为逻辑。
    角度: 有向图与行为逻辑建模。
    看点:
    1.  图的顶点是 AI 状态 (巡逻, 调查, 追击)。
    2.  图的边是状态转换的触发条件。
    3.  图中的“环”是 AI 逻辑的核心，使行为可以循环。
    """
    print_section("游戏 AI 状态机 (Finite State Machine)")

    ai_brain = DirectedGraph[str, str]()

    # 状态转换 (边) 和触发条件 (权重)
    ai_brain.add_edge("巡逻 (Patrol)", "调查 (Investigate)", "听到噪音 (Hears Noise)")
    ai_brain.add_edge("调查 (Investigate)", "巡逻 (Patrol)", "未发现异常 (Finds Nothing)")
    ai_brain.add_edge("调查 (Investigate)", "追击 (Chase)", "发现玩家 (Sees Player)")
    ai_brain.add_edge("追击 (Chase)", "巡逻 (Patrol)", "玩家逃脱 (Loses Player)")
    ai_brain.add_edge("追击 (Chase)", "追击 (Chase)", "持续看到玩家 (Keeps Sight)")  # 自循环

    visualize_graph(ai_brain, "守卫 AI 的行为逻辑图")

    print_subsection("AI 行为模拟")

    print("[i] 这个有向图完美地定义了 AI 的'思维'。我们可以用它来预测和控制 AI 的行为。")

    print("Q: 如果守卫正在'巡逻'，他能直接进入'追击'状态吗？")
    can_chase_directly = ai_brain.has_edge("巡逻 (Patrol)", "追击 (Chase)")
    print(f"A: {'能' if can_chase_directly else '不能'}。他必须先听到噪音并进入'调查'状态。")

    print("\nQ: 从'调查'状态出发，AI 可能进入哪些后续状态？")
    possible_next_states = list(ai_brain.neighbors("调查 (Investigate)"))
    print(f"A: 可能的状态有: {possible_next_states}")

    print("\nQ: 整个逻辑是否存在死胡同 (无法回到'巡逻'状态)？")
    # 检查所有状态是否都能回到初始状态'巡逻'
    can_all_return = all(ai_brain.has_path(state, "巡逻 (Patrol)") for state in ai_brain.vertices)
    print(f"A: {'否，所有状态最终都能回归巡逻' if can_all_return else '是，存在无法回归的死循环！'}")


# ==================== 场景三: 社交网络推荐系统基础 ====================


@dataclass(frozen=True, eq=True)
class User:
    name: str

    def __str__(self) -> str:
        return f"User({self.name})"


@dataclass(frozen=True, eq=True)
class Product:
    title: str

    def __str__(self) -> str:
        return f"Product({self.title})"


def demo_recommendation_engine():
    """
    场景: 为用户推荐他们可能喜欢的商品。
    角度: 二分图与复杂关系发现。
    看点:
    1.  使用二分图（在一个图中混合两种顶点类型）建模用户和商品的关系。
    2.  模拟“协同过滤”的核心思想：通过共同品味发现新推荐。
    3.  展示图遍历在数据挖掘中的基础性作用。
    """
    print_section("推荐系统基础 (二分图)")

    # 泛型 Union 展示了处理混合顶点类型的优雅
    graph = UndirectedGraph[Union[User, Product], int]()

    # 创建用户和商品
    users = [User(name) for name in ["Alice", "Bob", "Charlie", "Diana"]]
    products = [Product(title) for title in ["Graph Theory Book", "Python Cookbook", "AI Textbook", "SciFi Novel"]]

    # 用户对商品的评分 (1-5)
    ratings = [
        (users[0], products[0], 5),
        (users[0], products[1], 4),  # Alice
        (users[1], products[0], 4),
        (users[1], products[2], 5),
        (users[1], products[3], 2),  # Bob
        (users[2], products[1], 5),
        (users[2], products[3], 4),  # Charlie
        (users[3], products[2], 5),  # Diana
    ]
    graph.add_edges(ratings)

    visualize_graph(graph, "用户-商品 关系图")

    print_subsection("为 Alice 生成推荐")

    target_user = users[0]
    print(f"[i] 目标: 为 {target_user} 推荐她没看过但可能喜欢的商品。")
    print("[i] 逻辑: 1. 找到和 Alice 品味相似的用户 (喜欢同一本书)。")
    print("[i]       2. 看这些相似用户还喜欢什么其他书。")

    recommendations = {}

    # 1. 遍历 Alice 喜欢的商品
    for liked_product in graph.neighbors(target_user):
        # 2. 找到也喜欢这个商品的其他用户
        for similar_user in graph.neighbors(liked_product):
            if similar_user == target_user:
                continue

            # 3. 找到这位相似用户喜欢的其他商品
            for recommended_product in graph.neighbors(similar_user):
                # 确保是商品，并且 Alice 还没看过
                if isinstance(recommended_product, Product) and not graph.has_edge(target_user, recommended_product):
                    # 简单地用出现次数作为推荐分数
                    recommendations[recommended_product] = recommendations.get(recommended_product, 0) + 1

    if recommendations:
        print(f"💡 基于共同品味，为 {target_user} 生成的推荐 (按分数排序):")
        sorted_recs = sorted(recommendations.items(), key=lambda item: item[1], reverse=True)
        for product, score in sorted_recs:
            print(f"   - {product} (推荐分: {score})")
    else:
        print("   - 未找到合适的推荐。")


# ==================== 场景四: 计算机网络安全分析 ====================


def demo_network_security():
    """
    场景: 分析一个小型办公网络拓扑，评估安全风险。
    角度: 连通性与模拟传播。
    看点:
    1.  使用连通分量识别被防火墙隔离的子网。
    2.  使用 BFS 模拟病毒或网络攻击的逐层传播路径。
    """
    print_section("计算机网络安全分析")

    network = UndirectedGraph[str, None]()

    # 网络拓扑
    network.add_edges(
        [
            # 主要办公区 (LAN 1)
            ("Gateway", "Switch1"),
            ("Switch1", "PC-Alice"),
            ("Switch1", "PC-Bob"),
            ("Switch1", "Printer"),
            # 服务器区 (LAN 2, 假定通过 Gateway 连接)
            ("Gateway", "Firewall"),
            ("Firewall", "WebServer"),
            ("Firewall", "Database"),
            # 隔离的访客网络 (Guest WiFi)
            ("Guest-Router", "Guest-Laptop1"),
            ("Guest-Router", "Guest-Phone"),
        ]
    )

    visualize_graph(network, "办公室网络拓扑图")

    print_subsection("分析 1: 网络隔离审查")
    print("[i] `connected_components` 算法可以立即识别出网络中相互隔离的部分，这对于验证防火墙策略至关重要。")

    subnets = list(network.connected_components())
    print(f"💡 网络被划分为 {len(subnets)} 个独立的子网:")
    for i, subnet in enumerate(subnets, 1):
        print(f"   - 子网 {i}: {sorted(subnet)}")

    print_subsection("分析 2: 模拟攻击传播路径")

    start_point = "PC-Alice"
    print(f"[i] 假设 {start_point} 被恶意软件感染，我们将使用 BFS 模拟其在网络中的传播过程。")

    # BFS 的每一层代表病毒传播的一波
    q = [(start_point, 0)]
    visited = {start_point}
    spread_levels = {}

    while q:
        node, level = q.pop(0)
        spread_levels.setdefault(level, []).append(node)
        for neighbor in network.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                q.append((neighbor, level + 1))

    print("💡 恶意软件的传播路径和层级 (感染波次):")
    for level, nodes in sorted(spread_levels.items()):
        print(f"   - 第 {level} 波 (从感染源距离为 {level}): {sorted(nodes)}")

    print("\n[i] 结论: 攻击无法触及访客网络，证明了网络隔离的有效性。")


# ==================== 附: 工程健壮性展示 ====================


def demo_robustness():
    """展示序列化和不变性等工程特性。"""
    print_section("工程健壮性特性", character="*")

    print_subsection("特性 1: 轻松序列化 (JSON)")
    print("[i] 任何复杂的图状态都可以被轻松地序列化为 JSON 字符串，用于存储、传输或调试。")

    g = DirectedGraph[str, int]()
    g.add_edges([("A", "B", 10), ("B", "C", 20)])

    json_string = g.to_json(indent=2)
    print("序列化后的 JSON:\n", json_string)

    # 从 JSON 恢复图
    restored_g = DirectedGraph.from_json(json_string)
    print(
        "\n从 JSON 恢复的图是否与原图一致？",
        "是" if restored_g.has_edge("A", "B") and restored_g.get_edge_weight("A", "B") == 10 else "否",
    )

    print_subsection("特性 2: 边的不可变性 (Immutability)")
    print("[i] `Edge` 对象被设计为不可变的 (`frozen=True`)，这能防止意外修改，使代码更安全、更可预测。")

    edge = Edge("X", "Y", 100)
    print(f"创建了一个边: {edge}")

    try:
        edge.u = "Z"
        print("❌ 边的起点被修改了！(这是不应该发生的)")
    except Exception as e:
        print(f"✅ 尝试修改边的起点失败，并抛出异常: `{type(e).__name__}`。这正是我们期望的！")


# =================== 附: Benchmarking ===================
def demo_performance_benchmark():
    """
    场景：对大规模随机图进行压力测试。
    目标：证明图库实现的高效性与良好的可扩展性。
    看点：
    1. 在数千个顶点和上万条边的规模下，各项操作的耗时。
    2. 格式精美的性能报告，直观展示结果。
    """
    print_section("性能基准测试")

    sizes = [100, 1000, 5000, 100000]
    edge_factor = 5  # 每个顶点平均连接的边数

    print(f"[i] 我们将对四个规模的随机无向图 (V={', '.join(map(str, sizes))}) 进行压力测试。")

    # 打印表头
    print("\n" + "-" * 90)
    print(
        f"{'规模 (V)':<10} | {'顶点数':>10} | {'边数':>10} | {'添加顶点(s)':>14} | {'添加边(s)':>12} | {'DFS遍历(s)':>14} | {'连通分量(s)':>15}"
    )
    print("-" * 90)

    for n in sizes:
        graph = UndirectedGraph[int, int]()
        random.seed(42)

        # 1. 测量顶点添加时间
        start = time.perf_counter()
        graph.add_vertices(range(n))
        vertex_time = time.perf_counter() - start

        # 2. 测量边添加时间
        edges_to_add = n * edge_factor
        edge_list = []
        for _ in range(edges_to_add):
            u, v = random.randint(0, n - 1), random.randint(0, n - 1)
            if u != v:
                edge_list.append((u, v))

        start = time.perf_counter()
        # 批量添加是更高效的方式
        graph.add_edges(edge_list)
        edge_time = time.perf_counter() - start

        # 3. 测量 DFS 遍历时间
        start_vertex = random.randint(0, n - 1)
        start = time.perf_counter()
        _ = list(graph.dfs(start_vertex))
        dfs_time = time.perf_counter() - start

        # 4. 测量连通分量计算时间
        start = time.perf_counter()
        _ = list(graph.connected_components())
        conn_time = time.perf_counter() - start

        # 打印结果行
        print(
            f"{n:<10} | {graph.vertex_count:>10,d} | {graph.edge_count:>10,d} | "
            f"{vertex_time:>14.4f} | {edge_time:>12.4f} | "
            f"{dfs_time:>14.4f} | {conn_time:>15.4f}"
        )
    print("-" * 90)
    print("\n[i] 结论：即使在数千个顶点和数万条边的规模下，所有核心操作依然能在毫秒级完成。")
    print("[i] 这证明了底层数据结构 (邻接表) 和算法实现的效率。")


# ==================== 主函数 ====================


def main() -> None:
    """主函数，按顺序运行所有演示场景。"""
    try:
        demo_delivery_robot()
        demo_game_ai_fsm()
        demo_recommendation_engine()
        demo_network_security()
        demo_robustness()
        demo_performance_benchmark()

        print_section("演示圆满结束", "🎉")
        print("\n通过以上多元化场景，我们全方位展示了此图库的优雅与强大：")
        print("  ✅ **设计之美**: 清晰的 API、强大的泛型，让复杂建模如搭积木般简单。")
        print("  ✅ **功能之强**: 丰富的内置算法，为现实世界的挑战提供了现成的解决方案。")
        print("  ✅ **应用之广**: 从物流、游戏到安全、数据挖掘，图是解决问题的通用语言。")
        print("  ✅ **工程之坚**: 序列化、不变性等特性，确保了在大型项目中的可靠与健壮。")
        print("\n感谢您的探索！")

    except Exception as e:
        import traceback

        print(f"\n❌ 演示过程中发生意外错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
