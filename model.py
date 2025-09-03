# agent/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear, global_mean_pool


class GraphTransformerLayer(nn.Module):
    """
    一个完整的异构图Transformer层。

    此层对图中的'task'节点进行更新。它首先通过图注意力网络（GAT）
    聚合邻居信息（已更新为包含边特征），然后将多头注意力的输出投影回
    原始维度，最后应用残差连接和层归一化。
    """

    def __init__(self, embedding_dim, num_heads, edge_feature_dim):
        super().__init__()
        # 异构图卷积层，定义了 ('task', 'depends_on', 'task') 类型的消息传递
        # GATConv已更新，通过 edge_dim 参数来接收和使用边特征（如通信成本）
        self.conv = HeteroConv({
            ('task', 'depends_on', 'task'): GATConv(
                in_channels=-1,
                out_channels=embedding_dim,
                heads=num_heads,
                concat=True,
                add_self_loops=True,
                edge_dim=edge_feature_dim  # <-- 核心修改点
            ),
        }, aggr='sum')

        self.projection = Linear(num_heads * embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        """
        前向传播逻辑：GAT -> Projection -> ELU -> Residual -> LayerNorm
        签名已更新，以接收和传递 edge_attr_dict。
        """
        x_residual = x_dict['task']

        # 1. 通过GATConv进行消息传递，现在它会利用 edge_attr_dict
        x_dict_update = self.conv(x_dict, edge_index_dict, edge_attr_dict)
        task_update_concat = x_dict_update['task']

        task_update_projected = self.projection(task_update_concat)

        # 2. 应用激活函数、残差连接和层归一化
        task_output = self.norm(x_residual + F.elu(task_update_projected))

        x_dict_updated = x_dict.copy()
        x_dict_updated['task'] = task_output
        return x_dict_updated


class GraphTransformer(nn.Module):
    """
    基于异构图Transformer的策略与价值网络。
    该网络将调度问题的状态（一个异构图）编码为高维嵌入，
    并输出一个分解式的策略（任务选择 + 处理器选择）和一个状态价值。
    """

    def __init__(self, task_feature_dim, proc_feature_dim, edge_feature_dim,
                 embedding_dim, num_heads, num_layers):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.task_embed = Linear(task_feature_dim, embedding_dim)
        self.proc_embed = Linear(proc_feature_dim, embedding_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # 将 edge_feature_dim 传递给每个Transformer层
            self.layers.append(GraphTransformerLayer(embedding_dim, num_heads, edge_feature_dim))

        self.policy_head_task = nn.Sequential(
            Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            Linear(embedding_dim // 2, 1)
        )

        self.policy_head_proc = nn.Sequential(
            Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            Linear(embedding_dim, 1)
        )

        self.value_head = nn.Sequential(
            Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            Linear(embedding_dim, 1)
        )

    def forward(self, hetero_data):
        """
        模型的主前向传播。
        现在它会从hetero_data中提取并使用edge_attr_dict。
        """
        x_dict = hetero_data.x_dict
        edge_index_dict = hetero_data.edge_index_dict
        # 提取边特征字典
        edge_attr_dict = hetero_data.edge_attr_dict

        x_dict['task'] = self.task_embed(x_dict['task'])
        x_dict['proc'] = self.proc_embed(x_dict['proc'])

        for layer in self.layers:
            # 将边特征传递给每一层
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)

        task_embeds = x_dict['task']
        proc_embeds = x_dict['proc']

        task_logits = self.policy_head_task(task_embeds).squeeze(-1)

        task_pool = global_mean_pool(task_embeds, hetero_data['task'].batch)
        proc_pool = global_mean_pool(proc_embeds, hetero_data['proc'].batch)
        graph_embedding = torch.cat([task_pool, proc_pool], dim=1)
        value = self.value_head(graph_embedding)

        return task_logits, value, task_embeds, proc_embeds

    def compute_policy_dict(self, task_logits, task_embeds, proc_embeds, action_mask):
        """
        (新方法) 从模型的原始输出计算最终的策略字典。
        此方法封装了策略解码逻辑，以消除代码重复。
        """
        policy_dict = {}
        num_procs = proc_embeds.shape[0]

        # 只有当存在合法动作时才计算策略
        if torch.any(action_mask):
            # 1. 计算任务选择概率 P(t)
            masked_task_logits = task_logits[action_mask]
            task_probs = F.softmax(masked_task_logits, dim=0)

            # 2. 向量化计算处理器选择概率 P(p|t)
            valid_task_embeds = task_embeds[action_mask]
            num_valid_tasks = valid_task_embeds.shape[0]

            task_embeds_rep = valid_task_embeds.unsqueeze(1).expand(-1, num_procs, -1)
            proc_embeds_rep = proc_embeds.unsqueeze(0).expand(num_valid_tasks, -1, -1)

            combined_embeds = torch.cat([task_embeds_rep, proc_embeds_rep], dim=2).view(-1, self.embedding_dim * 2)
            proc_logits = self.policy_head_proc(combined_embeds).view(num_valid_tasks, num_procs)

            proc_probs_cond = F.softmax(proc_logits, dim=1)

            # 3. 计算联合概率 P(t, p) = P(t) * P(p|t)
            joint_probs = task_probs.unsqueeze(1) * proc_probs_cond

            # 4. 构建策略字典
            valid_task_indices = torch.where(action_mask)[0]
            for task_idx_in_batch, task_id in enumerate(valid_task_indices):
                for proc_id in range(num_procs):
                    action = (int(task_id), int(proc_id))
                    policy_dict[action] = joint_probs[task_idx_in_batch, proc_id].item()

        return policy_dict