# agent/replay_buffer.py

import random
import numpy as np


# 高效的SumTree数据结构
# SumTree 可以在 O(log N) 时间内完成采样和更新，显著提升性能
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        # 树的内部节点数量为 capacity - 1
        # 树的总大小为 2 * capacity - 1
        self.tree = np.zeros(2 * capacity - 1)
        # 叶子节点存储数据
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def add(self, priority, data):
        """向树中添加新的数据和其优先级。"""
        # 数据的索引，对应叶子节点的起始位置
        tree_idx = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        """更新节点的优先级，并向上传播变化。"""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # 向上回溯，更新父节点的值
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """根据给定的值v，在树中查找对应的叶子节点。"""
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            # 到达叶子节点
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        """返回树根节点的值，即总优先级。"""
        return self.tree[0]


class PrioritizedReplayBuffer:
    """
    基于SumTree的高效优先经验回放缓冲区。
    """

    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 1e-5  # 保证优先级不为0
        self.max_priority = 1.0

    def add(self, experience):
        """添加新的经验到缓冲区，并赋予最大优先级。"""
        # 新加入的经验赋予当前最大的优先级，以保证它们被采样的机会
        priority = self.max_priority
        self.tree.add(priority, experience)

    def sample(self, batch_size, beta=0.4):
        """根据优先级采样一批经验。"""
        batch_indices = np.empty(batch_size, dtype=np.int32)
        experiences = np.empty(batch_size, dtype=object)
        weights = np.empty(batch_size, dtype=np.float32)

        # 将总优先级分成 batch_size 个区间
        priority_segment = self.tree.total_priority / batch_size

        # 修正：仅在有效条目中计算 p_min，以获得正确的 max_weight
        if self.tree.n_entries > 0 and self.tree.total_priority > 0:
            # 树的叶子节点从索引 capacity - 1 开始
            start_idx = self.tree.capacity - 1
            end_idx = start_idx + self.tree.n_entries
            valid_priorities = self.tree.tree[start_idx:end_idx]
            p_min = np.min(valid_priorities) / self.tree.total_priority
            # 增加epsilon以防止 p_min * n_entries 为0导致的除零错误
            max_weight = ((p_min + self.epsilon) * self.tree.n_entries) ** (-beta)
        else:
            max_weight = 1.0

        for i in range(batch_size):
            # 在每个区间内均匀采样一个值
            a = priority_segment * i
            b = priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # 查找该值对应的叶子节点
            index, priority, data = self.tree.get_leaf(value)

            sampling_prob = priority / self.tree.total_priority
            # 增加epsilon以防止 sampling_prob 为0
            weights[i] = (self.tree.n_entries * (sampling_prob + self.epsilon)) ** (-beta)

            batch_indices[i] = index
            experiences[i] = data

        # 归一化重要性采样权重
        if max_weight > 0:
            weights /= max_weight

        return list(experiences), batch_indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        """更新采样过的经验的优先级。"""
        for idx, prio in zip(batch_indices, batch_priorities):
            # 保证优先级有一个小的正值
            priority = (prio + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

            # 更新全局最大优先级
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.tree.n_entries