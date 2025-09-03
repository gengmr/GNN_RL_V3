# utils/heuristics.py

import networkx as nx
import numpy as np


def calculate_rank_u(dag: nx.DiGraph, proc_speeds: np.ndarray):
    """
    计算图中每个任务的向上秩 (Upward Rank)，遵循经典的HEFT定义，适配异构环境。
    rank_u(i) = mean_exec_time(i) + max_{j in succ(i)} (c_ij + rank_u(j))
    这个值代表了从任务i开始到图出口的最长路径长度，是任务优先级的关键指标。

    Args:
        dag (nx.DiGraph): 任务图，节点属性'w'为工作量，边属性'c'为通信成本。
        proc_speeds (np.ndarray): 处理器速度数组。
    """
    # 使用动态规划，从出口任务开始反向计算
    memo = {}
    mean_proc_speed = np.mean(proc_speeds)

    for node in reversed(list(nx.topological_sort(dag))):
        # 任务的平均计算成本
        avg_exec_time = dag.nodes[node]['w'] / mean_proc_speed

        successors = list(dag.successors(node))
        if not successors:
            # 出口任务，其rank_u就是其自身的平均计算成本
            rank = avg_exec_time
        else:
            # 递归计算后继任务的最大 (c_ij + rank_u(j))
            max_succ_val = 0
            for succ in successors:
                val = dag.edges[node, succ]['c'] + memo[succ]
                if val > max_succ_val:
                    max_succ_val = val

            rank = avg_exec_time + max_succ_val
        memo[node] = rank

    # 将计算出的rank_u存入节点属性
    nx.set_node_attributes(dag, memo, 'rank_u')


def heft_scheduler(dag: nx.DiGraph, k: int, proc_speeds: np.ndarray):
    """
    实现经典的HEFT (Heterogeneous Earliest Finish Time) 算法。
    此版本精确地计算异构环境下的EFT，并选择最优的（任务，处理器）配对。

    Args:
        dag (nx.DiGraph): 带有 'w' 和 'rank_u' 属性的任务图。
        k (int): 处理器数量。
        proc_speeds (np.ndarray): 处理器速度数组。

    Returns:
        float: 调度完成时间 (Makespan)。
    """
    # 确保dag中包含rank_u属性，如果没有则计算
    if 'rank_u' not in list(dag.nodes(data=True))[0][1]:
        calculate_rank_u(dag, proc_speeds)

    proc_avail_time = np.zeros(k)
    task_finish_times = {}
    task_assignments = {}

    # 根据向上秩对任务进行降序排序，得到调度顺序
    schedule_order = sorted(dag.nodes(), key=lambda n: dag.nodes[n]['rank_u'], reverse=True)

    for task_id in schedule_order:
        best_proc = -1
        earliest_finish_time = float('inf')

        # 为当前任务在每个处理器上计算EFT
        for proc_idx in range(k):
            # 计算数据到达时间 (Data Arrival Time, DAT)
            # DAT是所有前驱任务的数据传输完成时间的最大值
            data_ready_time = 0
            for pred in dag.predecessors(task_id):
                pred_proc = task_assignments[pred]
                # 同处理器通信成本为0
                comm_cost = 0 if pred_proc == proc_idx else dag.edges[pred, task_id]['c']
                dat = task_finish_times[pred] + comm_cost
                if dat > data_ready_time:
                    data_ready_time = dat

            # 任务可以在 max(处理器可用时间, 数据到达时间) 后开始
            start_time = max(proc_avail_time[proc_idx], data_ready_time)
            # 异构处理器：执行时间 = 工作量 / 处理器速度
            exec_time = dag.nodes[task_id]['w'] / proc_speeds[proc_idx]
            finish_time = start_time + exec_time

            if finish_time < earliest_finish_time:
                earliest_finish_time = finish_time
                best_proc = proc_idx

        # 分配任务到最佳处理器并更新状态
        proc_avail_time[best_proc] = earliest_finish_time
        task_finish_times[task_id] = earliest_finish_time
        task_assignments[task_id] = best_proc

    makespan = np.max(proc_avail_time) if k > 0 else 0
    return makespan