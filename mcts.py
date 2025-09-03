# agent/mcts.py

import numpy as np
import torch
import torch.nn.functional as F
import math
from torch_geometric.data import Batch
import uuid
import time
from collections import namedtuple

from config import MCTS_C_PUCT, MCTS_DIRICHLET_ALPHA, MCTS_DIRICHLET_EPSILON

# 确保与main.py中的定义一致
InferenceRequest = namedtuple('InferenceRequest', ('request_id', 'state_data', 'action_mask'))


class Node:
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}  # action -> Node
        self.n_visits = 0
        self.q_value = 0
        self.u_value = 0
        self.p_value = prior_p

    def expand(self, action_priors):
        for action, p in action_priors.items():
            if action not in self.children:
                self.children[action] = Node(self, p)

    def select_child(self):
        best_score = -float('inf')
        best_action = None
        best_child = None
        for action, child in self.children.items():
            score = child.get_ucb_score()
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def get_ucb_score(self):
        self.u_value = (MCTS_C_PUCT * self.p_value *
                        math.sqrt(self.parent.n_visits) / (1 + self.n_visits))
        return self.q_value + self.u_value

    def update(self, value):
        self.n_visits += 1
        self.q_value += (value - self.q_value) / self.n_visits

    def is_leaf(self):
        return len(self.children) == 0


class MCTS:
    """
    蒙特卡洛树搜索主类。
    这个修正版本清晰地区分了本地模式和远程客户端模式。
    """

    def __init__(self, model=None, device=None, request_queue=None, result_dict=None):
        self.model = model
        self.device = device
        self.request_queue = request_queue
        self.result_dict = result_dict
        self.is_remote = model is None
        if not self.is_remote and device is None:
            raise ValueError("本地模式必须提供device。")
        if self.is_remote and (request_queue is None or result_dict is None):
            raise ValueError("远程模式必须提供推理队列和结果字典。")

    def _evaluate_state(self, state_data, action_mask):
        """
        根据模式评估状态，统一返回 (policy_dict, value)。
        """
        if not self.is_remote:
            # 本地模式 (用于Validator): 使用模型内置方法计算策略
            batch = Batch.from_data_list([state_data]).to(self.device)
            action_mask_tensor = torch.from_numpy(action_mask).to(self.device)

            with torch.no_grad():
                task_logits, value_pred, task_embeds, proc_embeds = self.model(batch)
                # 调用模型内置方法来计算策略，消除了此处的重复代码
                policy_dict = self.model.compute_policy_dict(
                    task_logits, task_embeds, proc_embeds, action_mask_tensor
                )

            return policy_dict, value_pred.item()
        else:
            # 远程模式（用于Actor）: 发送请求并等待结果
            req_id = uuid.uuid4()
            self.request_queue.put(InferenceRequest(req_id, state_data, action_mask))
            while req_id not in self.result_dict:
                time.sleep(0.001)
            return self.result_dict.pop(req_id)

    def search(self, env, num_simulations, heft_makespan,
               dirichlet_epsilon=MCTS_DIRICHLET_EPSILON,
               dirichlet_alpha=MCTS_DIRICHLET_ALPHA):

        root_state_data = env._get_state()
        action_mask = env.get_action_mask()
        action_priors, root_value = self._evaluate_state(root_state_data, action_mask)

        if not action_priors:
            return {}, root_value

        if dirichlet_epsilon > 0:
            actions = list(action_priors.keys())
            probs = np.array(list(action_priors.values()))
            noise = np.random.dirichlet([dirichlet_alpha] * len(probs))
            new_probs = (1 - dirichlet_epsilon) * probs + dirichlet_epsilon * noise
            action_priors = {action: prob for action, prob in zip(actions, new_probs)}

        root = Node(None, 1.0)
        root.expand(action_priors)

        for _ in range(num_simulations):
            node = root
            sim_env = env.clone()
            search_path = [node]
            while not node.is_leaf():
                action, node = node.select_child()
                _, _, done = sim_env.step(action)
                search_path.append(node)
                if done: break

            leaf_node = search_path[-1]
            if sim_env.scheduled_tasks_count == sim_env.num_tasks:
                final_makespan = sim_env.get_makespan()
                value = -final_makespan / heft_makespan if heft_makespan > 0 else -1.0
            else:
                leaf_state_data = sim_env._get_state()
                leaf_action_mask = sim_env.get_action_mask()
                leaf_priors, value = self._evaluate_state(leaf_state_data, leaf_action_mask)
                if leaf_priors:
                    leaf_node.expand(leaf_priors)

            for node_in_path in reversed(search_path):
                node_in_path.update(value)

        visit_counts = {action: child.n_visits for action, child in root.children.items()}
        total_visits = sum(visit_counts.values())
        pi = {action: count / total_visits for action, count in visit_counts.items()} if total_visits > 0 else {}

        return pi, root_value