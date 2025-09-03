# batch_evaluator.py (in your project this is run_test.py)
# 一个用于批量生成和评估调度问题的独立脚本。
# v4.7: 增加DAG图可视化功能，并为所有三种方案生成甘特图。

import os
import time
import torch
import numpy as np
import networkx as nx
import json

# --- 导入绘图和ILP求解所需的库 ---
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError:
    print("\n[错误] Matplotlib 库未找到。请使用 'pip install matplotlib' 命令安装后重试。")
    exit()

try:
    import gurobipy as gp
except ImportError:
    print("\n[错误] Gurobi 库未找到。请确保已正确安装并配置 Gurobi。")
    exit()

# --- 确保可以导入项目模块 ---
try:
    import config
    from model import GraphTransformer
    from dag_generator import generate_dag
    from scheduling_env import SchedulingEnv
    from mcts import MCTS
    from normalization import Normalizer
    from config import MCTS_SIMULATIONS
except ImportError as e:
    print("=" * 80, f"\n错误: 无法导入项目模块。请确保此脚本位于您项目的根目录下。\n详细信息: {e}\n", "=" * 80)
    exit()


# ==============================================================================
#                      可视化函数
# ==============================================================================

def save_dag_visualization(dag, k, proc_speeds, output_path, title):
    """
    根据给定的DAG对象和处理器信息，生成并保存一张结构图。
    """
    plt.figure(figsize=(16, 9))
    ax = plt.gca()

    # 使用 spring_layout 布局算法，让图更美观
    pos = nx.spring_layout(dag, seed=42)

    # 准备节点标签（任务ID和工作量w）
    node_labels = {node: f"T{node}\nw={data['w']}" for node, data in dag.nodes(data=True)}

    # 绘制节点
    nx.draw_networkx_nodes(dag, pos, node_size=1500, node_color='skyblue', ax=ax)
    # 绘制节点标签
    nx.draw_networkx_labels(dag, pos, labels=node_labels, font_size=10, font_weight='bold', ax=ax)

    # 绘制边
    nx.draw_networkx_edges(dag, pos, node_size=1500, arrowstyle='->', arrowsize=20, edge_color='gray', ax=ax)

    # 准备并绘制边标签（通信成本c）
    edge_labels = nx.get_edge_attributes(dag, 'c')
    nx.draw_networkx_edge_labels(dag, pos, edge_labels=edge_labels, font_color='red', font_size=9, ax=ax)

    # 在标题中包含处理器信息
    proc_speeds_str = ", ".join([f"{s:.2f}" for s in proc_speeds])
    full_title = f"{title}\n{k} Processors with Speeds: [{proc_speeds_str}]"
    ax.set_title(full_title, fontsize=16, weight='bold')

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()


def save_gantt_chart(schedule_details, makespan, num_processors, output_path, title):
    if not schedule_details:
        print(f"  [警告] 无法为 '{title}' 生成甘特图，调度方案为空。")
        return
    fig, ax = plt.subplots(figsize=(20, 8))
    all_task_ids = sorted(list(set(t['task_id'] for t in schedule_details)))
    num_tasks = len(all_task_ids)
    task_to_color_idx = {task_id: i for i, task_id in enumerate(all_task_ids)}
    colors = cm.get_cmap('viridis', num_tasks) if num_tasks > 0 else cm.get_cmap('viridis')
    for task in schedule_details:
        task_id, proc_id, start = task['task_id'], task['proc_id'], task['start_time']
        duration = task['finish_time'] - start
        color_idx = task_to_color_idx.get(task_id, 0)
        ax.barh(f"Processor {proc_id}", duration, left=start, height=0.6, align='center',
                color=colors(color_idx / max(1, num_tasks - 1)), edgecolor='black', alpha=0.8)
        ax.text(start + duration / 2, f"Processor {proc_id}", f'T{task_id}', ha='center', va='center', color='white',
                weight='bold', fontsize=10)
    ax.set_yticks(range(num_processors))
    ax.set_yticklabels([f"Processor {i}" for i in range(num_processors)])
    ax.invert_yaxis()
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Processors", fontsize=12)
    ax.set_title(title, fontsize=16, weight='bold')
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5)
    ax.axvline(x=makespan, color='red', linestyle='--', linewidth=2, label=f'Makespan: {makespan:.2f}')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


# ==============================================================================
#                      本地实现的求解器函数
# ==============================================================================

def solve_dag_scheduling_ilp_local(dag: nx.DiGraph, k: int, proc_speeds):
    tasks = list(dag.nodes())
    procs = list(range(k))
    num_tasks = len(tasks)
    try:
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model("DAG_Scheduling_Local", env=env) as model:
                exec_times_data = {(i, p): dag.nodes[i]['w'] / proc_speeds[p] for i in tasks for p in procs}
                big_m = sum(dag.nodes[i]['w'] for i in tasks) / np.min(proc_speeds) + sum(
                    e['c'] for _, _, e in dag.edges(data=True))
                start_times, finish_times = model.addVars(tasks, name="S", vtype=gp.GRB.CONTINUOUS,
                                                          lb=0), model.addVars(tasks, name="F", vtype=gp.GRB.CONTINUOUS,
                                                                               lb=0)
                makespan = model.addVar(name="Makespan", vtype=gp.GRB.CONTINUOUS, lb=0)
                assign = model.addVars(tasks, procs, name="X", vtype=gp.GRB.BINARY)
                order = model.addVars(tasks, tasks, name="Y", vtype=gp.GRB.BINARY)
                model.setObjective(makespan, gp.GRB.MINIMIZE)
                for i in tasks:
                    model.addConstr(gp.quicksum(assign[i, p] for p in procs) == 1)
                    model.addConstr(finish_times[i] == start_times[i] + gp.quicksum(
                        assign[i, p] * exec_times_data[i, p] for p in procs))
                    model.addConstr(makespan >= finish_times[i])
                for i, j in dag.edges():
                    for p1 in procs:
                        for p2 in procs:
                            model.addConstr(start_times[j] >= finish_times[i] + (
                                0 if p1 == p2 else dag.edges[i, j]['c']) - big_m * (2 - assign[i, p1] - assign[j, p2]))
                tc = nx.transitive_closure(dag, reflexive=False)
                for p in procs:
                    for i in range(num_tasks):
                        for j in range(i + 1, num_tasks):
                            ti, tj = tasks[i], tasks[j]
                            if (ti, tj) not in tc.edges and (tj, ti) not in tc.edges:
                                model.addConstr(order[ti, tj] + order[tj, ti] >= assign[ti, p] + assign[tj, p] - 1)
                                model.addConstr(
                                    start_times[tj] >= finish_times[ti] - big_m * (1 - order[ti, tj]) - big_m * (
                                                2 - assign[ti, p] - assign[tj, p]))
                                model.addConstr(
                                    start_times[ti] >= finish_times[tj] - big_m * (1 - order[tj, ti]) - big_m * (
                                                2 - assign[ti, p] - assign[tj, p]))
                model.setParam('TimeLimit', config.ILP_TIME_LIMIT)
                model.optimize()
                if model.Status == gp.GRB.OPTIMAL or (model.Status == gp.GRB.TIME_LIMIT and model.SolCount > 0):
                    schedule = [
                        {'task_id': i, 'proc_id': p, 'start_time': start_times[i].X, 'finish_time': finish_times[i].X}
                        for i in tasks for p in procs if assign[i, p].X > 0.5]
                    schedule.sort(key=lambda x: x['finish_time'])
                    return makespan.X, schedule
                return np.nan, []
    except gp.GurobiError as e:
        return np.nan, []
    except Exception:
        return np.nan, []


def calculate_rank_u_local(dag: nx.DiGraph, proc_speeds: np.ndarray):
    memo = {}
    avg_exec_times = {i: np.mean([data['w'] / s for s in proc_speeds]) for i, data in dag.nodes(data=True)}
    for node in reversed(list(nx.topological_sort(dag))):
        avg_exec_time = avg_exec_times[node]
        successors = list(dag.successors(node))
        if not successors:
            rank = avg_exec_time
        else:
            rank = avg_exec_time + max(dag.edges[node, succ]['c'] + memo[succ] for succ in successors)
        memo[node] = rank
    nx.set_node_attributes(dag, memo, 'rank_u')


def heft_solver_local(dag: nx.DiGraph, k: int, proc_speeds: np.ndarray):
    proc_avail_time = np.zeros(k)
    task_finish_times, task_assignments, schedule_details = {}, {}, []
    schedule_order = sorted(dag.nodes(), key=lambda n: dag.nodes[n]['rank_u'], reverse=True)
    for task_id in schedule_order:
        best_proc, earliest_finish_time, best_start_time = -1, float('inf'), -1.0
        for proc_idx in range(k):
            dat = max([task_finish_times.get(p, 0) + (
                0 if task_assignments.get(p) == proc_idx else dag.edges[p, task_id]['c']) for p in
                       dag.predecessors(task_id)], default=0)
            start_time = max(proc_avail_time[proc_idx], dat)
            finish_time = start_time + dag.nodes[task_id]['w'] / proc_speeds[proc_idx]
            if finish_time < earliest_finish_time: earliest_finish_time, best_proc, best_start_time = finish_time, proc_idx, start_time
        proc_avail_time[best_proc] = earliest_finish_time
        task_finish_times[task_id], task_assignments[task_id] = earliest_finish_time, best_proc
        schedule_details.append({'task_id': task_id, 'proc_id': best_proc, 'start_time': best_start_time,
                                 'finish_time': earliest_finish_time})
    makespan = np.max(proc_avail_time) if k > 0 else 0
    schedule_details.sort(key=lambda x: x['finish_time'])
    return makespan, schedule_details


def agent_solver_local(model, device, dag: nx.DiGraph, k: int, proc_speeds: np.ndarray):
    m_heft_for_norm, _ = heft_solver_local(dag.copy(), k, proc_speeds)
    normalizer = Normalizer(dag, m_heft_for_norm, proc_speeds)
    env = SchedulingEnv(dag, k, proc_speeds, normalizer=normalizer)
    _ = env.reset()
    mcts_evaluator = MCTS(model, device)
    while env.scheduled_tasks_count < env.num_tasks:
        if not np.any(env.get_action_mask()): break
        pi, _ = mcts_evaluator.search(env, MCTS_SIMULATIONS, env.heft_makespan, dirichlet_epsilon=0.0)
        if not pi: break
        best_action = max(pi, key=pi.get)
        _, _, done = env.step(best_action)
        if done: break
    makespan = env.get_makespan()
    schedule_details = [{'task_id': tid, 'proc_id': pid,
                         'start_time': env.task_finish_time[tid] - (dag.nodes[tid]['w'] / proc_speeds[pid]),
                         'finish_time': env.task_finish_time[tid]} for tid, pid in env.task_assignments.items()]
    schedule_details.sort(key=lambda x: x['finish_time'])
    return makespan, schedule_details


# ==============================================================================
#                            主测试流程
# ==============================================================================

if __name__ == '__main__':
    NUM_TEST_CASES = 20
    RESULTS_FILENAME = os.path.join(config.RESULT_DIR, "final_schedules_and_graphs.json")
    GANTT_CHART_DIR = os.path.join(config.RESULT_DIR, "gantt_charts/")
    DAG_VIS_DIR = os.path.join(config.RESULT_DIR, "dag_visualizations/")

    os.makedirs(GANTT_CHART_DIR, exist_ok=True)
    os.makedirs(DAG_VIS_DIR, exist_ok=True)

    print("=" * 60, "\n       批量调度性能评估脚本 (v4.7 - 全方案与DAG可视化)\n", "=" * 60)

    try:
        print(f"[*] 步骤 1: 正在加载已训练的模型...")
        device = config.DEVICE
        dummy_dag, _, _ = generate_dag((5, 5), (2, 2), (1, 1))
        calculate_rank_u_local(dummy_dag, np.array([1.0, 1.0]))
        s = SchedulingEnv(dummy_dag, 2, np.array([1.0, 1.0])).reset()
        model_params = {"task_feature_dim": s['task'].x.shape[1], "proc_feature_dim": s['proc'].x.shape[1],
                        "edge_feature_dim": s['task', 'depends_on', 'task'].edge_attr.shape[1],
                        "embedding_dim": config.EMBEDDING_DIM, "num_heads": config.NUM_ATTENTION_HEADS,
                        "num_layers": config.NUM_ENCODER_LAYERS}
        model = GraphTransformer(**model_params).to(device)
        checkpoint = torch.load(config.CHECKPOINT_FILE, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"    ✅ 模型加载成功 (来自 step {checkpoint.get('global_step', 'N/A')}).")
    except Exception as e:
        print(f"\n[错误] 加载模型时出错: {e}. 请确保已成功训练并保存模型。")
        exit()

    print(f"\n[*] 步骤 2: 开始执行 {NUM_TEST_CASES} 个测试用例...")
    all_results = []
    heft_makespans, ilp_makespans, agent_makespans = [], [], []

    for i in range(NUM_TEST_CASES):
        print(f"\n{'=' * 25} 测试用例 {i + 1}/{NUM_TEST_CASES} {'=' * 25}")
        master_dag, k, proc_speeds = generate_dag(config.M_RANGE, config.K_RANGE, config.CCR_RANGE)
        print(f"  问题参数: 任务数(M)={len(master_dag.nodes())}, 处理器数(K)={k}")
        calculate_rank_u_local(master_dag, proc_speeds)

        # 新增：保存DAG可视化图
        dag_img_path = os.path.join(DAG_VIS_DIR, f"case_{i + 1:02d}_dag.png")
        save_dag_visualization(master_dag, k, proc_speeds, dag_img_path, f"Test Case {i + 1} DAG Structure")
        print(f"  [Plot]  DAG 结构图已保存到 '{dag_img_path}'")

        # 求解
        m_heft, heft_schedule = heft_solver_local(master_dag.copy(), k, proc_speeds)
        m_agent, agent_schedule = agent_solver_local(model, device, master_dag.copy(), k, proc_speeds)
        m_ilp, ilp_schedule = solve_dag_scheduling_ilp_local(master_dag.copy(), k, proc_speeds)

        heft_makespans.append(m_heft)
        ilp_makespans.append(m_ilp)
        agent_makespans.append(m_agent)

        print(f"  [HEFT]  Makespan: {m_heft:<10.2f}")
        print(f"  [Agent] Makespan: {m_agent:<10.2f}")
        if not np.isnan(m_ilp):
            print(f"  [ILP]   Makespan: {m_ilp:<10.2f}")
        else:
            print("  [ILP]   求解失败或超时")

        # 保存甘特图
        save_gantt_chart(heft_schedule, m_heft, k, os.path.join(GANTT_CHART_DIR, f"case_{i + 1:02d}_heft.png"),
                         f"Test Case {i + 1}: HEFT Schedule (Makespan: {m_heft:.2f})")
        save_gantt_chart(agent_schedule, m_agent, k, os.path.join(GANTT_CHART_DIR, f"case_{i + 1:02d}_agent.png"),
                         f"Test Case {i + 1}: Agent Schedule (Makespan: {m_agent:.2f})")
        if ilp_schedule: save_gantt_chart(ilp_schedule, m_ilp, k,
                                          os.path.join(GANTT_CHART_DIR, f"case_{i + 1:02d}_ilp.png"),
                                          f"Test Case {i + 1}: ILP Schedule (Makespan: {m_ilp:.2f})")
        print(f"  [Plot]  所有甘特图已保存到 '{GANTT_CHART_DIR}'")

        # 记录结果
        all_results.append({
            "test_case": i + 1, "num_tasks": len(master_dag.nodes()), "num_processors": k,
            "makespans": {"heft": m_heft, "ilp": m_ilp if not np.isnan(m_ilp) else None, "agent": m_agent},
            "schedules": {"heft": heft_schedule, "ilp": ilp_schedule, "agent": agent_schedule}
        })

    print(f"\n[*] 步骤 3: 正在将详细调度结果保存到 '{RESULTS_FILENAME}'...")
    with open(RESULTS_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"    ✅ 结果已成功保存。")

    print(f"\n\n{'=' * 28} 最终总结报告 {'=' * 28}")
    avg_heft, avg_ilp, avg_agent = np.nanmean(heft_makespans), np.nanmean(ilp_makespans), np.nanmean(agent_makespans)
    ilp_success_rate = 1 - np.isnan(ilp_makespans).sum() / max(1, len(ilp_makespans))
    print(f"在 {NUM_TEST_CASES} 个随机测试用例上评估完毕。")
    print("\n----------------- 平均Makespan对比 -----------------")
    print(f"  - 平均 HEFT Makespan : {avg_heft:.2f}")
    print(f"  - 平均 ILP Makespan  : {avg_ilp:.2f} (成功率: {ilp_success_rate:.0%})")
    print(f"  - 平均 Agent Makespan: {avg_agent:.2f}")
    print("----------------------------------------------------")
    if not np.isnan(avg_heft) and avg_heft > 0: print(
        f"\nAgent vs HEFT 平均调度长度比 (SLR): {avg_agent / avg_heft:.4f}")
    if not np.isnan(avg_ilp) and avg_ilp > 0: print(f"Agent vs ILP 平均最优间隙 (Gap): {avg_agent / avg_ilp:.4f}")
    print("\n评估完成。")