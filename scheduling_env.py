# environment/scheduling_env.py

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData
from typing import Tuple, Dict

from heuristics import calculate_rank_u, heft_scheduler
from normalization import Normalizer


class SchedulingEnv:
    """
    任务调度强化学习环境。
    管理调度过程的状态、动作执行和奖励计算。
    此版本为异构处理器环境进行了适配。
    """

    def __init__(self, dag: nx.DiGraph, k: int, proc_speeds: np.ndarray, normalizer: Normalizer = None):
        self.initial_dag = dag
        self.k = k
        self.proc_speeds = proc_speeds

        if normalizer is None:
            if 'rank_u' not in list(self.initial_dag.nodes(data=True))[0][1]:
                calculate_rank_u(self.initial_dag, self.proc_speeds)
            heft_makespan = heft_scheduler(self.initial_dag.copy(), self.k, self.proc_speeds)
            self.normalizer = Normalizer(self.initial_dag, heft_makespan, self.proc_speeds)
            self.heft_makespan = heft_makespan
        else:
            self.normalizer = normalizer
            self.heft_makespan = normalizer.heft_makespan

        self.reset()

    def reset(self):
        """
        重置环境到初始状态。
        """
        self.dag = self.initial_dag.copy()
        self.num_tasks = len(self.dag.nodes)

        # 状态变量
        self.task_status: Dict[int, str] = {i: 'uncompleted' for i in self.dag.nodes()}
        self.proc_available_time: np.ndarray = np.zeros(self.k)
        self.task_finish_time: Dict[int, float] = {}
        self.task_assignments: Dict[int, int] = {}
        self.earliest_finish_times: Dict[int, float] = {i: 0.0 for i in self.dag.nodes()}
        self.scheduled_tasks_count: int = 0

        self._update_ready_tasks()
        self._update_earliest_finish_times()

        return self._get_state()

    def clone(self):
        """
        创建一个环境的深拷贝副本，用于MCTS模拟。
        """
        cloned_env = SchedulingEnv(self.initial_dag, self.k, self.proc_speeds, self.normalizer)

        cloned_env.task_status = self.task_status.copy()
        cloned_env.proc_available_time = self.proc_available_time.copy()
        cloned_env.task_finish_time = self.task_finish_time.copy()
        cloned_env.task_assignments = self.task_assignments.copy()
        cloned_env.earliest_finish_times = self.earliest_finish_times.copy()
        cloned_env.scheduled_tasks_count = self.scheduled_tasks_count

        return cloned_env

    def _get_ready_tasks(self) -> set:
        """从 task_status 动态计算就绪任务集合。"""
        return {task_id for task_id, status in self.task_status.items() if status == 'ready'}

    def _update_ready_tasks(self):
        """
        根据已完成任务，更新处于“就绪”状态的任务列表。
        """
        for task_id in self.dag.nodes():
            if self.task_status[task_id] == 'uncompleted':
                if all(self.task_status.get(p) == 'completed' for p in self.dag.predecessors(task_id)):
                    self.task_status[task_id] = 'ready'

    def _update_earliest_finish_times(self):
        """
        为所有就绪任务计算并更新其理论最早完成时间 (EFT)。
        """
        ready_tasks = self._get_ready_tasks()
        for task_id in ready_tasks:
            min_eft = float('inf')
            for proc_id in range(self.k):
                data_arrival_time = 0
                for pred in self.dag.predecessors(task_id):
                    pred_proc = self.task_assignments.get(pred)
                    comm_cost = self.dag.edges[pred, task_id]['c'] if pred_proc != proc_id else 0
                    dat = self.task_finish_time.get(pred, 0) + comm_cost
                    data_arrival_time = max(data_arrival_time, dat)

                start_time = max(self.proc_available_time[proc_id], data_arrival_time)
                # 异构处理器：执行时间 = 工作量 / 处理器速度
                exec_time = self.dag.nodes[task_id]['w'] / self.proc_speeds[proc_id]
                finish_time = start_time + exec_time
                min_eft = min(min_eft, finish_time)
            self.earliest_finish_times[task_id] = min_eft

    def _get_state(self) -> HeteroData:
        """
        构建并返回当前状态的异构图表示。
        """
        data = HeteroData()

        task_features = []
        status_map = {
            'uncompleted': [1, 0, 0],
            'ready': [0, 1, 0],
            'completed': [0, 0, 1]
        }

        for i in range(self.num_tasks):
            status = self.task_status[i]
            status_one_hot = status_map[status]

            features = [
                self.normalizer.z_score_normalize(self.dag.nodes[i]['w'], 'w'),
                *status_one_hot,
                self.normalizer.z_score_normalize(self.dag.nodes[i]['rank_u'], 'rank_u'),
                self.normalizer.time_normalize(self.earliest_finish_times.get(i, 0.0))
            ]
            task_features.append(features)

        data['task'].x = torch.tensor(task_features, dtype=torch.float)

        proc_features = []
        for i in range(self.k):
            features = [
                self.normalizer.time_normalize(self.proc_available_time[i]),
                self.normalizer.z_score_normalize(self.proc_speeds[i], 'speed')
            ]
            proc_features.append(features)
        data['proc'].x = torch.tensor(proc_features, dtype=torch.float)

        edge_index_tt = []
        edge_attr_tt = []
        for u, v, edge_data in self.dag.edges(data=True):
            edge_index_tt.append([u, v])
            edge_attr_tt.append([self.normalizer.z_score_normalize(edge_data['c'], 'c')])

        if edge_index_tt:
            data['task', 'depends_on', 'task'].edge_index = torch.tensor(edge_index_tt,
                                                                         dtype=torch.long).t().contiguous()
            data['task', 'depends_on', 'task'].edge_attr = torch.tensor(edge_attr_tt, dtype=torch.float)
        else:
            data['task', 'depends_on', 'task'].edge_index = torch.empty((2, 0), dtype=torch.long)
            data['task', 'depends_on', 'task'].edge_attr = torch.empty((0, 1), dtype=torch.float)

        return data

    def get_action_mask(self) -> np.ndarray:
        """
        返回一个布尔掩码，指示哪些任务是合法的动作（即处于就绪状态）。
        """
        mask = np.zeros(self.num_tasks, dtype=bool)
        ready_tasks = self._get_ready_tasks()
        if ready_tasks:
            ready_indices = list(ready_tasks)
            mask[ready_indices] = True
        return mask

    def step(self, action: Tuple[int, int]):
        """
        在环境中执行一个动作（调度一个任务到处理器）。
        """
        task_id, proc_id = action

        if self.task_status[task_id] != 'ready':
            raise ValueError(f"任务 {task_id} 不处于就绪状态。当前状态: {self.task_status[task_id]}")

        data_arrival_time = 0
        for pred in self.dag.predecessors(task_id):
            pred_proc = self.task_assignments.get(pred)
            comm_cost = self.dag.edges[pred, task_id]['c'] if pred_proc != proc_id else 0
            dat = self.task_finish_time.get(pred, 0) + comm_cost
            data_arrival_time = max(data_arrival_time, dat)

        start_time = max(self.proc_available_time[proc_id], data_arrival_time)
        # 异构处理器：执行时间 = 工作量 / 处理器速度
        exec_time = self.dag.nodes[task_id]['w'] / self.proc_speeds[proc_id]
        finish_time = start_time + exec_time

        # 更新状态
        self.proc_available_time[proc_id] = finish_time
        self.task_finish_time[task_id] = finish_time
        self.task_assignments[task_id] = proc_id
        self.task_status[task_id] = 'completed'
        self.scheduled_tasks_count += 1

        self._update_ready_tasks()
        self._update_earliest_finish_times()

        done = self.scheduled_tasks_count == self.num_tasks

        reward = 0
        if done:
            reward = -self.get_makespan()

        return self._get_state(), reward, done

    def get_makespan(self):
        """返回当前部分或完整调度的makespan。"""
        return np.max(self.proc_available_time) if self.proc_available_time.size > 0 else 0