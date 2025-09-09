# main.py

import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch
import torch.nn.functional as F
import time
import os
import json
import pickle
from collections import deque, namedtuple
import numpy as np
import queue
from tqdm import tqdm

import config
from model import GraphTransformer
from mcts import MCTS
from replay_buffer import PrioritizedReplayBuffer
from dag_generator import generate_dag
from scheduling_env import SchedulingEnv
from validator import Validator
from heuristics import heft_scheduler, calculate_rank_u
from normalization import Normalizer


# --- 专业日志记录器 ---
class Logger:
    """一个用于生成专业、美观、层次化日志的辅助类。"""

    def __init__(self):
        self._major_step = 0
        self._sub_step = 0

    def _reset_sub_step(self):
        self._sub_step = 0

    def log_major_step(self, message, icon="⚙️"):
        self._major_step += 1
        self._reset_sub_step()
        print(f"\n[{self._major_step}.] {icon}  {message}")
        print("-" * (len(message) + 6))

    def log_sub_step(self, message):
        self._sub_step += 1
        print(f"  {self._major_step}.{self._sub_step}  {message}")

    def log_info(self, message, indent=1, symbol="└─"):
        prefix = "   " * indent
        print(f"{prefix}{symbol} {message}")

    def log_event(self, message, icon="🔔", border_char="="):
        print(f"\n{border_char * 3} {icon}  {message}  {icon} {border_char * 3}")

    def print_header(self, title):
        print("\n" + "=" * 60)
        print(f"{' ' * ((60 - len(title)) // 2)}{title}")
        print("=" * 60)

    def print_footer(self):
        print("=" * 60)


# 定义经验元组
Experience = namedtuple('Experience', ('state', 'pi', 'value'))
# 定义推理请求元组，需包含action_mask
InferenceRequest = namedtuple('InferenceRequest', ('request_id', 'state_data', 'action_mask'))


def actor_process(actor_id, data_queue, curriculum_queue, stop_event,
                  inference_request_queue, inference_result_dict):
    """
    并行执行器 (Actor) 的工作流程。
    作为推理客户端，不持有模型，通过队列与推理工作器通信。
    """
    pid = os.getpid()
    print(f"[ACTOR {actor_id} | PID: {pid}] ▶️  Process started.")

    mcts = MCTS(request_queue=inference_request_queue,
                result_dict=inference_result_dict)
    current_curriculum = 0

    while not stop_event.is_set():
        # 非阻塞地检查课程更新
        while not curriculum_queue.empty():
            try:
                message = curriculum_queue.get_nowait()
                if 'curriculum' in message:
                    current_curriculum = message['curriculum']
                    print(f"[ACTOR {actor_id}] 🔄 Updated to Curriculum: {current_curriculum}")
            except queue.Empty:
                break

        # --- 生成一个调度问题实例 ---
        m_range, k_range, ccr_range = config.CURRICULUM_STAGES[current_curriculum]
        dag, k, proc_speeds = generate_dag(m_range, k_range, ccr_range)
        calculate_rank_u(dag, proc_speeds)
        heft_makespan = heft_scheduler(dag.copy(), k, proc_speeds)
        normalizer = Normalizer(dag, heft_makespan, proc_speeds)

        env = SchedulingEnv(dag, k, proc_speeds, normalizer)
        state = env.reset()
        game_history = []

        # --- 自我对弈 ---
        while True:
            action_mask = env.get_action_mask()
            if not np.any(action_mask):
                break  # 没有合法动作，游戏结束

            # 使用MCTS增强策略
            pi, _ = mcts.search(env, config.TRAINING_MCTS_SIMULATIONS, env.heft_makespan)

            if not pi:
                break  # MCTS未返回有效策略

            # 保存状态和MCTS输出的策略
            full_pi = np.zeros((env.num_tasks, env.k))
            for (task, proc), prob in pi.items():
                full_pi[task, proc] = prob
            game_history.append({'state': state.clone(), 'pi': full_pi})

            # 根据MCTS策略采样动作
            actions, probs = list(pi.keys()), list(pi.values())
            # 为np.random.choice确保概率和为1 (处理浮点数精度问题)
            probs = np.array(probs, dtype=np.float64)
            probs /= np.sum(probs)
            action_idx = np.random.choice(len(actions), p=probs)
            action = actions[action_idx]

            state, _, done = env.step(action)
            if done:
                break

        # --- 游戏结束后，整理并发送经验 ---
        if game_history:
            final_makespan = env.get_makespan()
            # 价值被归一化，作为所有步骤的共同目标
            normalized_value = -final_makespan / heft_makespan if heft_makespan > 0 else -1.0
            experiences_to_send = [Experience(transition['state'], transition['pi'], normalized_value)
                                   for transition in game_history]
            data_queue.put(experiences_to_send)


def inference_worker(model_dict, request_queue, result_dict, stop_event, model_params):
    """
    一个专门在GPU上运行模型推理的独立进程，作为推理服务器。
    此版本经过优化，使用模型内置方法来计算策略，以提高代码复用性。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pid = os.getpid()
    print(f"[INFERENCE WORKER | PID: {pid}] ▶️  Process started on {device}.")

    model = GraphTransformer(**model_params).to(device)
    model.eval()

    # 初始加载一次模型
    try:
        cpu_state_dict = model_dict.copy()
        if cpu_state_dict:
            model.load_state_dict(cpu_state_dict)
    except Exception as e:
        print(f"[INFERENCE WORKER] Initial model load failed: {e}")

    while not stop_event.is_set():
        # --- 安全地获取最新的模型状态 ---
        # model_dict.copy() 是原子操作, 保证了获取到的是一个完整的状态快照
        cpu_state_dict = model_dict.copy()
        if not cpu_state_dict:  # 如果learner还没来得及放入模型，则稍等
            time.sleep(0.1)
            continue
        model.load_state_dict(cpu_state_dict)

        # --- 批量收集推理请求 ---
        requests = []
        try:
            while len(requests) < config.BATCH_SIZE:
                req = request_queue.get_nowait()
                requests.append(req)
        except queue.Empty:
            if not requests:
                time.sleep(0.001)  # 队列为空时短暂休眠，避免CPU空转
                continue

        # --- 执行批量推理 ---
        if requests:
            batch = Batch.from_data_list([req.state_data for req in requests]).to(device)
            with torch.no_grad():
                task_logits_all, value_preds, task_embeds_all, proc_embeds_all = model(batch)

            task_ptr = batch['task'].ptr
            proc_ptr = batch['proc'].ptr

            # --- 解析批次结果并返回给各个Actor ---
            for i, req in enumerate(requests):
                start_task, end_task = task_ptr[i], task_ptr[i + 1]
                start_proc, end_proc = proc_ptr[i], proc_ptr[i + 1]

                single_task_logits = task_logits_all[start_task:end_task]
                single_value = value_preds[i].item()
                single_task_embeds = task_embeds_all[start_task:end_task]
                single_proc_embeds = proc_embeds_all[start_proc:end_proc]
                action_mask = torch.from_numpy(req.action_mask).to(device)

                # 调用模型内置方法计算策略字典，消除了此处的重复代码
                policy_dict = model.compute_policy_dict(
                    single_task_logits,
                    single_task_embeds,
                    single_proc_embeds,
                    action_mask
                )

                result_dict[req.request_id] = (policy_dict, single_value)


def save_checkpoint(state):
    """保存检查点，将replay buffer分开存储以避免内存问题。"""
    buffer_to_save = state.pop('replay_buffer', None)
    torch.save(state, config.CHECKPOINT_FILE)
    if buffer_to_save:
        with open(config.REPLAY_BUFFER_FILE, "wb") as f:
            pickle.dump(buffer_to_save, f)


def load_checkpoint(model, optimizer):
    """加载检查点，并分开加载replay buffer。"""
    if os.path.isfile(config.CHECKPOINT_FILE):
        try:
            # weights_only=False是默认值，但显式指出是为了清晰地表明我们要加载优化器状态等非权重信息
            checkpoint = torch.load(config.CHECKPOINT_FILE, map_location=config.DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if os.path.isfile(config.REPLAY_BUFFER_FILE):
                with open(config.REPLAY_BUFFER_FILE, "rb") as f:
                    replay_buffer = pickle.load(f)
                return checkpoint, replay_buffer, True
            else:
                print(f"  [WARNING] Checkpoint found but replay buffer file '{config.REPLAY_BUFFER_FILE}' is missing.")
                # 即使没有buffer，也返回checkpoint信息，以便恢复global_step等
                return checkpoint, None, True
        except (KeyError, pickle.UnpicklingError, RuntimeError) as e:
            print(
                f"  [WARNING] Failed to load checkpoint file. It might be corrupt or from an incompatible version: {e}")
            return None, None, False
        except Exception as e:
            print(f"  [WARNING] An unexpected error occurred while loading checkpoint: {e}")
            return None, None, False

    return None, None, False


def update_web_ui_data(data):
    """
    将训练状态写入JSON文件，供Web前端监控。
    此版本能正确处理嵌套的监控数据结构。
    """
    # 创建一个可序列化的副本
    serializable_data = {
        "global_step": data.get("global_step", 0),
        "current_curriculum": data.get("current_curriculum", 0),
        "training_metrics": {k: list(v) for k, v in data.get("training_metrics", {}).items()},
        "system_metrics": {k: list(v) for k, v in data.get("system_metrics", {}).items()}
    }

    # 单独处理嵌套的验证指标
    serializable_validation = {}
    validation_data = data.get("validation_metrics", {})
    for stage_key, stage_data in validation_data.items():
        serializable_validation[stage_key] = {k: list(v) for k, v in stage_data.items()}
    serializable_data["validation_metrics"] = serializable_validation

    try:
        with open(config.WEB_STATUS_FILE, 'w') as f:
            json.dump(serializable_data, f, indent=4)
    except IOError as e:
        print(f"  [WARNING] Web UI status file could not be written: {e}")


def learner_process():
    """学习器 (Learner) 的主工作流程。"""
    logger = Logger()
    logger.print_header("Reinforcement Learning Scheduler - Learner")

    # --- 1. 初始化 ---
    logger.log_major_step("System Initialization")
    logger.log_sub_step(f"Setting device to: {config.DEVICE}")
    logger.log_sub_step("Checking and creating result directories...")
    dirs_to_create = [config.RESULT_DIR, config.LOG_DIR, config.MODEL_SAVE_DIR, config.WEB_MONITOR_DIR]
    for d in dirs_to_create: os.makedirs(d, exist_ok=True)
    logger.log_info(f"All output will be saved in '{config.RESULT_DIR}'")

    # --- 2. 动态获取环境维度 ---
    logger.log_major_step("Getting Environment Dimensions", icon="📐")
    try:
        dummy_dag, dummy_k, dummy_speeds = generate_dag(m_range=(5, 5), k_range=(2, 2), ccr_range=(1, 1))
        calculate_rank_u(dummy_dag, dummy_speeds)
        dummy_env = SchedulingEnv(dummy_dag, dummy_k, dummy_speeds)
        dummy_state = dummy_env.reset()
        task_feature_dim = dummy_state['task'].x.shape[1]
        proc_feature_dim = dummy_state['proc'].x.shape[1]
        edge_feature_dim = dummy_state['task', 'depends_on', 'task'].edge_attr.shape[1]
        logger.log_sub_step(f"Task feature dimension: {task_feature_dim}")
        logger.log_sub_step(f"Processor feature dimension: {proc_feature_dim}")
        logger.log_sub_step(f"Edge feature dimension: {edge_feature_dim}")
    except Exception as e:
        logger.log_info(f"❌ CRITICAL: Failed to create dummy environment to get feature dims: {e}")
        return

    # --- 3. 模型与优化器设置 ---
    logger.log_major_step("Model & Optimizer Setup", icon="🧠")
    logger.log_sub_step("Creating GraphTransformer model...")
    model_params = {
        "task_feature_dim": task_feature_dim,
        "proc_feature_dim": proc_feature_dim,
        "edge_feature_dim": edge_feature_dim,
        "embedding_dim": config.EMBEDDING_DIM,
        "num_heads": config.NUM_ATTENTION_HEADS,
        "num_layers": config.NUM_ENCODER_LAYERS
    }
    model = GraphTransformer(**model_params).to(config.DEVICE)

    try:
        logger.log_sub_step("Initializing model with a dummy forward pass...")
        model.eval()
        with torch.no_grad():
            model(Batch.from_data_list([dummy_state]).to(config.DEVICE))
        model.train()
        logger.log_info("✅ Model layers initialized successfully.")
    except Exception as e:
        logger.log_info(f"❌ CRITICAL: Model initialization failed: {e}")
        return

    logger.log_sub_step(f"Creating Adam optimizer with learning rate {config.LEARNING_RATE}...")
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # --- 4. 恢复训练状态 ---
    logger.log_major_step("Restoring Training State", icon="💾")
    logger.log_sub_step(f"Attempting to load checkpoint from '{config.CHECKPOINT_FILE}'...")
    checkpoint, replay_buffer, success = load_checkpoint(model, optimizer)
    if success:
        global_step = checkpoint.get('global_step', 0)
        current_curriculum = checkpoint.get('current_curriculum', 0)
        slr_history = checkpoint.get('slr_history', deque(maxlen=config.PROMOTION_STABLE_EPOCHS))
        if replay_buffer is None:
            replay_buffer = PrioritizedReplayBuffer(config.REPLAY_BUFFER_SIZE, alpha=config.PER_ALPHA)
            logger.log_info("⚠️ Replay buffer file was missing. Initializing a new one.")
        logger.log_info(f"✅ Checkpoint loaded. Resuming from step {global_step}.")
    else:
        global_step, current_curriculum = 0, 0
        slr_history = deque(maxlen=config.PROMOTION_STABLE_EPOCHS)
        replay_buffer = PrioritizedReplayBuffer(config.REPLAY_BUFFER_SIZE, alpha=config.PER_ALPHA)
        logger.log_info(f"⚠️ No valid checkpoint found. Starting from scratch.")

    # --- 5. 初始化其他组件 ---
    logger.log_major_step("Initializing Components")
    writer = SummaryWriter(config.LOG_DIR)
    logger.log_sub_step("Setting up TensorBoard SummaryWriter...")
    validator = Validator(config.VALIDATION_SET_SIZE, config.CURRICULUM_STAGES)
    logger.log_sub_step("Setting up Validator with persistent dataset...")

    # 根据检查点是否存在来加载或创建 web_ui_data
    if success and 'web_ui_data' in checkpoint and checkpoint['web_ui_data']:
        logger.log_info("✅ Restoring historical monitoring data from checkpoint.")
        web_ui_data = checkpoint['web_ui_data']

        # 关键一步: torch.load 会将 deque 变回 list。
        # 我们需要递归地将所有数据列表转换回 deque，以便后续的 .append() 操作能正常工作。
        for metric_type in ['training_metrics', 'system_metrics']:
            if metric_type in web_ui_data:
                for key, value in web_ui_data[metric_type].items():
                    web_ui_data[metric_type][key] = deque(value)

        if 'validation_metrics' in web_ui_data:
            for stage_key, stage_data in web_ui_data['validation_metrics'].items():
                for key, value in stage_data.items():
                    web_ui_data['validation_metrics'][stage_key][key] = deque(value)

    else:
        if success:
            logger.log_info(
                "⚠️ Checkpoint found, but no historical monitoring data inside. Initializing a new monitoring session.")
        else:
            logger.log_info("⚠️ No checkpoint found. Initializing a new monitoring session.")

        # 如果没有检查点或检查点中没有监控数据，则使用原始的初始化代码
        web_ui_data = {
            "global_step": global_step,
            "current_curriculum": current_curriculum,
            "training_metrics": {
                "steps": deque(), "total_loss": deque(),
                "policy_loss": deque(), "value_loss": deque()
            },
            "system_metrics": {
                "steps": deque(),
                "buffer_size": deque()
            },
            "validation_metrics": {}
        }

    # --- 6. 启动并行进程 ---
    logger.log_major_step(f"Spawning {config.NUM_ACTORS} Actors & 1 Inference Worker", icon="🚀")
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        logger.log_info("Start method has already been set. Continuing...")
    manager = mp.Manager()

    cpu_model_state = {k: v.cpu().detach() for k, v in model.state_dict().items()}
    model_dict = manager.dict(cpu_model_state)
    data_queue = manager.Queue()
    curriculum_queues = [manager.Queue() for _ in range(config.NUM_ACTORS)]
    stop_event = manager.Event()
    inference_request_queue = manager.Queue()
    inference_result_dict = manager.dict()

    inference_proc = mp.Process(target=inference_worker,
                                args=(
                                    model_dict, inference_request_queue, inference_result_dict, stop_event,
                                    model_params))
    inference_proc.start()

    actors = []
    for i in range(config.NUM_ACTORS):
        curriculum_queues[i].put({'curriculum': current_curriculum})
        actor = mp.Process(target=actor_process,
                           args=(i, data_queue, curriculum_queues[i], stop_event,
                                 inference_request_queue, inference_result_dict))
        actor.start()
        actors.append(actor)
    logger.log_info("✅ All processes launched.")

    # --- 7. 主训练循环 ---
    logger.log_major_step("Starting Main Training Loop", icon="▶️")

    steps_until_val = config.VALIDATION_INTERVAL
    initial_step = global_step % steps_until_val
    next_val_step = global_step - initial_step + steps_until_val
    pbar = tqdm(total=steps_until_val, initial=initial_step,
                desc=f"Training (next val @ step {next_val_step})",
                bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")

    try:
        while True:
            # --- 数据收集 ---
            try:
                while not data_queue.empty():
                    experience_list = data_queue.get_nowait()
                    for exp in experience_list:
                        replay_buffer.add(exp)
            except queue.Empty:
                pass

            # --- 等待缓冲区填充 ---
            if len(replay_buffer) < config.MIN_BUFFER_SIZE_FOR_TRAINING:
                print(f"\r[INFO] Filling replay buffer... {len(replay_buffer)}/{config.MIN_BUFFER_SIZE_FOR_TRAINING}",
                      end="")
                time.sleep(0.1)
                continue

            if global_step == 0 and len(replay_buffer) >= config.MIN_BUFFER_SIZE_FOR_TRAINING:
                pbar.close()  # 关闭初始的bar
                print()  # 换行
                logger.log_info(f"✅ Replay Buffer filled. Starting training on {config.DEVICE}.")
                # 重新创建bar开始正式计数
                pbar = tqdm(total=steps_until_val, initial=0,
                            desc=f"Training (next val @ step {steps_until_val})",
                            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")

            # --- 采样与批次准备 ---
            model.train()
            beta = min(1.0,
                       config.PER_BETA_START + global_step * (1.0 - config.PER_BETA_START) / config.PER_BETA_FRAMES)
            experiences, indices, weights = replay_buffer.sample(config.BATCH_SIZE, beta)
            batch = Batch.from_data_list([exp.state for exp in experiences]).to(config.DEVICE)

            pis = [exp.pi for exp in experiences]
            max_m = max(pi.shape[0] for pi in pis)
            max_k = max(pi.shape[1] for pi in pis) if any(pi.ndim > 1 and pi.shape[1] > 0 for pi in pis) else 1
            padded_pis = np.zeros((len(pis), max_m, max_k), dtype=np.float32)
            for i, pi in enumerate(pis):
                if pi.ndim > 1 and pi.shape[1] > 0:
                    padded_pis[i, :pi.shape[0], :pi.shape[1]] = pi

            pi_targets_padded = torch.from_numpy(padded_pis).float().to(config.DEVICE)
            value_targets = torch.tensor([exp.value for exp in experiences], dtype=torch.float).view(-1, 1).to(
                config.DEVICE)
            weights_tensor = torch.tensor(weights, dtype=torch.float).view(-1, 1).to(config.DEVICE)

            # --- 前向传播与损失计算 ---
            task_logits_all, value_preds, task_embeds_all, proc_embeds_all = model(batch)

            value_loss = (weights_tensor * F.mse_loss(value_preds, value_targets, reduction='none')).mean()

            policy_losses = []
            task_ptr, proc_ptr = batch['task'].ptr, batch['proc'].ptr
            for i in range(batch.num_graphs):
                num_tasks, num_procs = task_ptr[i + 1] - task_ptr[i], proc_ptr[i + 1] - proc_ptr[i]
                task_logits = task_logits_all[task_ptr[i]:task_ptr[i + 1]]
                task_embeds = task_embeds_all[task_ptr[i]:task_ptr[i + 1]]
                proc_embeds = proc_embeds_all[proc_ptr[i]:proc_ptr[i + 1]]
                pi_target = pi_targets_padded[i, :num_tasks, :num_procs]
                pi_task_target = pi_target.sum(dim=1)

                loss_task_i = -torch.sum(pi_task_target * F.log_softmax(task_logits, dim=0))
                loss_proc_i = torch.tensor(0.0, device=config.DEVICE)

                valid_tasks_mask = pi_task_target > 1e-8
                if valid_tasks_mask.any():
                    valid_task_embeds = task_embeds[valid_tasks_mask]
                    num_valid_tasks = valid_task_embeds.shape[0]
                    task_embeds_rep = valid_task_embeds.unsqueeze(1).expand(-1, num_procs, -1)
                    proc_embeds_rep = proc_embeds.unsqueeze(0).expand(num_valid_tasks, -1, -1)

                    combined_embeds = torch.cat([task_embeds_rep, proc_embeds_rep], dim=2).view(-1,
                                                                                                config.EMBEDDING_DIM * 2)
                    proc_logits = model.policy_head_proc(combined_embeds).view(num_valid_tasks, num_procs)

                    log_pi_proc_pred = F.log_softmax(proc_logits, dim=1)
                    pi_proc_target_cond = pi_target[valid_tasks_mask] / (
                            pi_task_target[valid_tasks_mask].unsqueeze(1) + 1e-8)
                    ce_per_task = -torch.sum(pi_proc_target_cond * log_pi_proc_pred, dim=1)
                    loss_proc_i = torch.sum(pi_task_target[valid_tasks_mask] * ce_per_task)

                policy_losses.append(loss_task_i + loss_proc_i)

            policy_loss = (weights_tensor.squeeze() * torch.stack(policy_losses)).mean()
            total_loss = value_loss + policy_loss

            # --- 反向传播与优化 ---
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # --- 更新PER优先级 ---
            td_errors = torch.abs(value_preds - value_targets).detach().cpu().numpy().flatten()
            replay_buffer.update_priorities(indices, td_errors)

            # --- 日志记录 ---
            writer.add_scalar("Loss/Total", total_loss.item(), global_step)
            writer.add_scalar("Loss/Policy", policy_loss.item(), global_step)
            writer.add_scalar("Loss/Value", value_loss.item(), global_step)
            pbar.set_postfix_str(f"Loss={total_loss.item():.3f}")

            web_ui_data["global_step"] = global_step
            web_ui_data["training_metrics"]["steps"].append(global_step)
            web_ui_data["training_metrics"]["total_loss"].append(total_loss.item())
            web_ui_data["training_metrics"]["policy_loss"].append(policy_loss.item())
            web_ui_data["training_metrics"]["value_loss"].append(value_loss.item())
            web_ui_data["system_metrics"]["steps"].append(global_step)
            web_ui_data["system_metrics"]["buffer_size"].append(len(replay_buffer))

            # --- 定期验证 ---
            if global_step > 0 and global_step % config.VALIDATION_INTERVAL == 0:
                pbar.close()
                logger.log_event(f"Step {global_step}: Running Validation on All Learned Curricula", icon="📊")
                val_results_by_stage = validator.evaluate(model, config.DEVICE, current_curriculum, global_step)

                for stage, stage_results in val_results_by_stage.items():
                    # --- 更新TensorBoard日志 ---
                    writer.add_scalars(f"Validation_Makespan/C{stage}", {
                        'Agent': stage_results['Avg_Makespan_Agent'],
                        'HEFT': stage_results['Avg_Makespan_HEFT'],
                        'ILP': stage_results['Avg_Makespan_ILP'],
                    }, global_step)
                    writer.add_scalar(f"Validation_SLR/C{stage}", stage_results['Avg_SLR_vs_HEFT'], global_step)
                    writer.add_scalar(f"Validation_Gap/C{stage}", stage_results['Avg_Optimality_Gap'], global_step)
                    writer.add_scalar(f"Validation_WinRate/C{stage}", stage_results['Win_Rate_vs_HEFT'], global_step)

                    # --- 更新Web UI数据 ---
                    stage_key = f"C{stage}"
                    if stage_key not in web_ui_data["validation_metrics"]:
                        web_ui_data["validation_metrics"][stage_key] = {
                            "steps": deque(), "avg_makespan_agent": deque(),
                            "avg_makespan_heft": deque(), "avg_makespan_ilp": deque(),
                            "avg_slr": deque(), "avg_gap": deque(), "win_rate": deque()
                        }

                    ui_stage_data = web_ui_data["validation_metrics"][stage_key]
                    ui_stage_data["steps"].append(global_step)
                    ui_stage_data["avg_makespan_agent"].append(stage_results['Avg_Makespan_Agent'])
                    ui_stage_data["avg_makespan_heft"].append(stage_results['Avg_Makespan_HEFT'])
                    ui_stage_data["avg_makespan_ilp"].append(stage_results['Avg_Makespan_ILP'])
                    ui_stage_data["avg_slr"].append(stage_results['Avg_SLR_vs_HEFT'])
                    ui_stage_data["avg_gap"].append(stage_results['Avg_Optimality_Gap'])
                    ui_stage_data["win_rate"].append(stage_results['Win_Rate_vs_HEFT'])

                # 打印当前课程的摘要
                current_results = val_results_by_stage[current_curriculum]
                logger.log_info(f"Summary for Current Curriculum (C{current_curriculum}):", symbol="└─")
                agent_m = current_results['Avg_Makespan_Agent']
                heft_m = current_results['Avg_Makespan_HEFT']
                ilp_m = current_results['Avg_Makespan_ILP']
                makespan_summary_line = f"Avg. Makespans (Agent | HEFT   | ILP)   : {agent_m:>7.2f} | {heft_m:>7.2f} | {ilp_m:>7.2f}"
                logger.log_info(makespan_summary_line, indent=2)
                logger.log_info(f"Avg. SLR vs HEFT: {current_results['Avg_SLR_vs_HEFT']:.4f}", indent=2)
                logger.log_info(f"Avg. Optimality Gap: {current_results['Avg_Optimality_Gap']:.4f}", indent=2)

                # --- 课程学习晋级判断 (仅基于当前课程的表现) ---
                slr_history.append(current_results['Avg_SLR_vs_HEFT'])
                if len(slr_history) == config.PROMOTION_STABLE_EPOCHS and all(
                        s < config.PROMOTION_THRESHOLD_SLR for s in slr_history):
                    if current_curriculum < max(config.CURRICULUM_STAGES.keys()):
                        current_curriculum += 1
                        web_ui_data["current_curriculum"] = current_curriculum
                        logger.log_event(f"CURRICULUM PROMOTION! -> Stage {current_curriculum}", icon="⭐",
                                         border_char="⭐")
                        for q in curriculum_queues: q.put({'curriculum': current_curriculum})
                        slr_history.clear()

                pbar = tqdm(total=steps_until_val, initial=0,
                            desc=f"Training (next val @ step {global_step + steps_until_val})",
                            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")

            # --- 模型同步 ---
            if global_step > 0 and global_step % config.MODEL_SYNC_INTERVAL == 0:
                current_cpu_state = {k: v.cpu().detach() for k, v in model.state_dict().items()}
                model_dict.update(current_cpu_state)

            # --- 保存检查点 ---
            if global_step > 0 and global_step % config.CHECKPOINT_INTERVAL == 0:
                logger.log_event(f"Step {global_step}: Saving Checkpoint", icon="💾")
                save_checkpoint({
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'current_curriculum': current_curriculum,
                    'slr_history': slr_history,
                    'replay_buffer': replay_buffer,
                    'web_ui_data': web_ui_data
                })
                logger.log_info(f"✅ Checkpoint saved to '{config.CHECKPOINT_FILE}'.")

            update_web_ui_data(web_ui_data)
            global_step += 1
            if len(replay_buffer) >= config.MIN_BUFFER_SIZE_FOR_TRAINING:
                pbar.update(1)

    except KeyboardInterrupt:
        logger.log_event("KeyboardInterrupt caught. Initiating graceful shutdown...", icon="🛑")
    finally:
        if 'pbar' in locals() and pbar:
            pbar.close()
        logger.log_major_step("Shutdown Sequence", icon="🛑")
        logger.log_sub_step("Sending stop signal to all processes...")
        stop_event.set()

        logger.log_info("Waiting for Inference Worker to terminate...", indent=2)
        inference_proc.join(timeout=5)
        logger.log_info("✅ Inference Worker process has been terminated.")
        logger.log_info("Waiting for Actor processes to terminate...", indent=2)
        for actor in actors: actor.join(timeout=5)
        logger.log_info("✅ All Actor processes have been terminated.")

        logger.log_sub_step("Saving final model state and replay buffer...")
        save_checkpoint({
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'current_curriculum': current_curriculum,
            'slr_history': slr_history,
            'replay_buffer': replay_buffer,
            'web_ui_data': web_ui_data
        })
        logger.log_info(f"✅ Final state saved to '{config.CHECKPOINT_FILE}'.")
        logger.print_footer()


if __name__ == '__main__':
    learner_process()