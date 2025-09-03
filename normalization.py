# utils/normalization.py

import numpy as np
import networkx as nx
import config


class Normalizer:
    """
    根据设计方案，实现基于问题实例的、分类型的动态归一化。
    """
    def __init__(self, dag: nx.DiGraph, heft_makespan: float, proc_speeds: np.ndarray):
        """
        在每个episode开始时，为该实例创建一个归一化器。

        Args:
            dag (nx.DiGraph): 当前问题的任务图。
            heft_makespan (float): HEFT算法计算出的makespan，用于归一化时间相关特征。
            proc_speeds (np.ndarray): 处理器速度数组。
        """
        self.heft_makespan = heft_makespan + config.EPSILON # 防止除以零

        # 1. 预计算原始和启发式特征的统计量
        workloads = np.array([data['w'] for _, data in dag.nodes(data=True)])
        comm_costs = np.array([data['c'] for _, _, data in dag.edges(data=True)])
        rank_u_vals = np.array([data['rank_u'] for _, data in dag.nodes(data=True)])

        self.stats = {
            'w': (np.mean(workloads), np.std(workloads) + config.EPSILON),
            'c': (np.mean(comm_costs), np.std(comm_costs) + config.EPSILON) if len(comm_costs) > 0 else (0, 1),
            'rank_u': (np.mean(rank_u_vals), np.std(rank_u_vals) + config.EPSILON),
            'speed': (np.mean(proc_speeds), np.std(proc_speeds) + config.EPSILON),
        }

    def z_score_normalize(self, value, feature_type: str):
        """
        对原始特征和启发式特征应用Z-score归一化。
        """
        mean, std = self.stats[feature_type]
        return (value - mean) / std

    def time_normalize(self, time_value):
        """
        对时间相关特征进行归一化。
        """
        return time_value / self.heft_makespan