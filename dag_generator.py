# environment/dag_generator.py

import networkx as nx
import numpy as np
import random
from typing import List, Tuple

from config import WORKLOAD_RANGE, PROCESSOR_SPEED_RANGE


def generate_dag(m_range: Tuple[int, int], k_range: Tuple[int, int], ccr_range: Tuple[float, float]):
    """
    根据给定的参数范围生成一个随机的DAG任务图、处理器数量及异构处理器速度。

    Args:
        m_range (Tuple[int, int]): 任务数量范围 [min, max]。
        k_range (Tuple[int, int]): 处理器数量范围 [min, max]。
        ccr_range (Tuple[float, float]): CCR范围 [min, max]。

    Returns:
        Tuple[nx.DiGraph, int, np.ndarray]: 返回生成的DAG图、处理器数量K以及处理器速度数组。
    """
    m = random.randint(*m_range)
    k = random.randint(*k_range)
    ccr = random.uniform(*ccr_range)

    # 新增：生成异构处理器速度
    proc_speeds = np.random.uniform(*PROCESSOR_SPEED_RANGE, size=k)

    dag = nx.DiGraph()

    # 1. 添加节点并分配工作量
    total_workload = 0
    for i in range(m):
        workload = random.randint(*WORKLOAD_RANGE)
        dag.add_node(i, w=workload)
        total_workload += workload

    # 2. 添加边以确保是DAG
    # 通过确保边的方向总是从小编号节点指向大编号节点来保证无环
    for i in range(m):
        for j in range(i + 1, m):
            # 以一定概率添加边
            if random.random() < 0.3:  # 可调概率
                dag.add_edge(i, j)

    # 确保图是弱连通的 (作为一个整体)
    if not nx.is_weakly_connected(dag):
        components = list(nx.weakly_connected_components(dag))
        for i in range(len(components) - 1):
            u = random.choice(list(components[i]))
            v = random.choice(list(components[i + 1]))
            # 确保边的方向
            if u > v: u, v = v, u
            if not dag.has_edge(u, v):
                dag.add_edge(u, v)

    # 3. 根据CCR分配通信成本
    avg_workload = total_workload / m
    avg_comm_cost = ccr * avg_workload

    for u, v in dag.edges():
        # 通信成本在平均值附近随机波动
        comm_cost = max(0, int(np.random.normal(loc=avg_comm_cost, scale=avg_comm_cost / 4)))
        dag.edges[u, v]['c'] = comm_cost

    # 4. 移除传递性边 (可选，使图更稀疏)
    # transitive_reduction 比较耗时，对于大图可以跳过
    # dag = nx.transitive_reduction(dag)

    return dag, k, proc_speeds