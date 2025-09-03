# validation/ilp_solver.py

import gurobipy as gp
import networkx as nx
import numpy as np
from config import ILP_TIME_LIMIT


def solve_dag_scheduling_ilp(dag: nx.DiGraph, k: int, proc_speeds):
    """
    使用 Gurobi 求解DAG调度问题的最优makespan。
    此版本经过简化，如果在设定的时限内未找到最优或可行的解，
    将直接抛出异常，不再进行回退尝试。
    """
    tasks = list(dag.nodes())
    procs = list(range(k))
    num_tasks = len(tasks)

    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model("DAG_Scheduling_Heterogeneous", env=env) as model:
            # --- 辅助数据 ---
            exec_times_data = {(i, p): dag.nodes[i]['w'] / proc_speeds[p] for i in tasks for p in procs}
            big_m = sum(dag.nodes[i]['w'] for i in tasks) / np.min(proc_speeds) + sum(
                e['c'] for _, _, e in dag.edges(data=True))

            # --- 决策变量 ---
            start_times = model.addVars(tasks, name="S", vtype=gp.GRB.CONTINUOUS, lb=0)
            finish_times = model.addVars(tasks, name="F", vtype=gp.GRB.CONTINUOUS, lb=0)
            makespan = model.addVar(name="Makespan", vtype=gp.GRB.CONTINUOUS, lb=0)
            assign = model.addVars(tasks, procs, name="X", vtype=gp.GRB.BINARY)
            order = model.addVars(tasks, tasks, name="Y", vtype=gp.GRB.BINARY)

            # --- 目标函数 ---
            model.setObjective(makespan, gp.GRB.MINIMIZE)

            # --- 约束 ---
            for i in tasks:
                model.addConstr(gp.quicksum(assign[i, p] for p in procs) == 1, name=f"assign_{i}")
                exec_time = gp.quicksum(assign[i, p] * exec_times_data[i, p] for p in procs)
                model.addConstr(finish_times[i] == start_times[i] + exec_time, name=f"finish_time_{i}")
                model.addConstr(makespan >= finish_times[i], name=f"makespan_{i}")

            for i, j in dag.edges():
                comm_cost = dag.edges[i, j]['c']
                for p1 in procs:
                    for p2 in procs:
                        effective_comm_cost = 0 if p1 == p2 else comm_cost
                        model.addConstr(start_times[j] >= finish_times[i] + effective_comm_cost
                                        - big_m * (2 - assign[i, p1] - assign[j, p2]),
                                        name=f"prec_{i}_{j}_{p1}_{p2}")

            transitive_closure = nx.transitive_closure(dag, reflexive=False)
            tc_edges = set(transitive_closure.edges())

            for p in procs:
                for i in range(num_tasks):
                    for j in range(i + 1, num_tasks):
                        task_i, task_j = tasks[i], tasks[j]
                        if (task_i, task_j) not in tc_edges and (task_j, task_i) not in tc_edges:
                            model.addConstr(
                                order[task_i, task_j] + order[task_j, task_i] >= assign[task_i, p] + assign[
                                    task_j, p] - 1,
                                name=f"order_excl_{task_i}_{task_j}_{p}")
                            model.addConstr(start_times[task_j] >= finish_times[task_i]
                                            - big_m * (1 - order[task_i, task_j])
                                            - big_m * (2 - assign[task_i, p] - assign[task_j, p]),
                                            name=f"overlap_{task_i}_{task_j}_{p}")
                            model.addConstr(start_times[task_i] >= finish_times[task_j]
                                            - big_m * (1 - order[task_j, task_i])
                                            - big_m * (2 - assign[task_i, p] - assign[task_j, p]),
                                            name=f"overlap_{task_j}_{task_i}_{p}")

            # --- 求解 ---
            model.setParam('TimeLimit', ILP_TIME_LIMIT)
            model.optimize()

            # 检查求解状态
            # 只有当找到最优解，或在时限内找到了至少一个可行解时，才认为成功
            if model.Status == gp.GRB.OPTIMAL or (model.Status == gp.GRB.TIME_LIMIT and model.SolCount > 0):
                return makespan.X
            else:
                # 在任何其他情况下 (如：不可行, 无界, 或时限内无解)，都抛出异常
                raise RuntimeError(
                    f"ILP solver failed to find a feasible solution within the time limit for a DAG with {num_tasks} tasks. "
                    f"Gurobi model status code: {model.Status}"
                )