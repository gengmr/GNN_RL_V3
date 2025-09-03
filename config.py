# config.py

import torch
import os
import numpy as np
import math

# ==============================================================================
# SECTION 1: PROBLEM DEFINITION PARAMETERS
# ------------------------------------------------------------------------------
# 此部分参数定义了待解决的DAG任务调度问题的核心属性和范围。
# 这些设置直接决定了在训练和评估过程中生成的随机问题实例的复杂度和多样性。
# ==============================================================================

# M_RANGE: 任务数量范围
# Description: 定义一个DAG图中任务（节点）数量的最小值和最大值。
# Role: 控制问题规模的“高度”或“长度”。
# Type: list[int]
M_RANGE = [5, 20]

# K_RANGE: 处理器数量范围
# Description: 定义异构处理器系统中的处理器数量的最小值和最大值。
# Role: 控制问题规模的“宽度”或可用资源量。
# Type: list[int]
K_RANGE = [2, 6]

# CCR_RANGE: 通信计算比 (Communication-to-Computation Ratio) 范围
# Description: CCR是衡量任务图通信密集程度的关键指标。它定义了平均通信成本与平均计算成本之间的比率。
# Role: 控制问题的类型。低CCR代表计算密集型问题，高CCR代表通信密集型问题。
# Type: list[float]
CCR_RANGE = [0.05, 0.8]

# WORKLOAD_RANGE: 单个任务的工作量范围
# Description: 定义每个任务节点的基础计算量（w_i）的随机范围。
# Role: 为问题实例增加多样性。实际执行时间还需除以处理器速度。
# Type: list[int]
WORKLOAD_RANGE = [1, 100]

# PROCESSOR_SPEED_RANGE: 异构处理器的速度范围
# Description: 定义每个处理器的相对计算速度。值为1.0代表标准速度。
# Role: 引入处理器异构性，这是问题的核心挑战之一。
# Type: list[float]
PROCESSOR_SPEED_RANGE = [0.5, 2.0]

# ==============================================================================
# SECTION 2: AUTOMATIC CURRICULUM LEARNING PARAMETERS
# ------------------------------------------------------------------------------
# 此部分参数用于控制自动课程学习机制。该机制通过从简单问题开始，
# 逐步过渡到更复杂的问题，来稳定和加速模型的训练过程。
# 新版支持根据问题规模跨度自适应地生成课程阶段。
# ==============================================================================

# --- 课程自适应生成控制参数 ---

# ADAPTIVE_CURRICULUM_ENABLED: 是否启用自适应课程数生成
# Description: 控制课程总阶段数是自动计算还是手动指定。True时，将根据问题规模的跨度和下方定义的难度步长自动计算课程数。False时，将使用 MANUAL_NUM_CURRICULUM_STAGES。
# Role: 提供灵活性，允许用户在固定课程数和动态课程数之间切换，以适应不同的研究或实验需求。
# Type: bool
ADAPTIVE_CURRICULUM_ENABLED = False

# M_DIFFICULTY_STEP: 任务数难度步长
# Description: 定义了模型在学习路径上，任务数量（M）每增加大约多少才构成一个显著的难度跃迁，从而需要一个新的课程阶段来专门学习和适应。
# Role: 作为自适应课程数计算的核心依据之一，它将课程的粒度与问题规模的“长度”或“深度”进行科学地关联。
# Type: int
M_DIFFICULTY_STEP = 8

# K_DIFFICULTY_STEP: 处理器数难度步长
# Description: 定义了模型在学习路径上，处理器数量（K）每增加大约多少才构成一个显著的难度跃迁。
# Role: 作为自适应课程数计算的核心依据之一，它将课程的粒度与问题规模的“宽度”或“资源复杂度”进行科学地关联。
# Type: int
K_DIFFICULTY_STEP = 2

# MIN_CURRICULUM_STAGES: 最少课程阶段数
# Description: 为自适应课程生成机制设定一个下限，确保即使对于难度跨度较小的问题，学习过程也被划分为足够多的阶段，以保证训练的稳定性和平滑性。
# Role: 防止因问题规模跨度过小而导致课程阶段过少，从而可能引发训练震荡或收敛困难。
# Type: int
MIN_CURRICULUM_STAGES = 3

# MANUAL_NUM_CURRICULUM_STAGES: 手动设定的课程总阶段数
# Description: 一个固定的整数，仅在 ADAPTIVE_CURRICULUM_ENABLED 设置为 False 时生效，用于指定课程学习的总阶段数。
# Role: 在需要进行精确控制或复现特定实验设置时，提供一种覆盖自适应机制的备用方案。
# Type: int
MANUAL_NUM_CURRICULUM_STAGES = 4


# --- 课程生成函数 (内部使用) ---

def _get_adaptive_num_stages():
    """根据问题规模跨度和难度步长，科学地计算所需的课程总数。"""
    if not ADAPTIVE_CURRICULUM_ENABLED:
        return MANUAL_NUM_CURRICULUM_STAGES

    m_span = M_RANGE[1] - M_RANGE[0]
    k_span = K_RANGE[1] - K_RANGE[0]

    m_stages_needed = math.ceil(m_span / M_DIFFICULTY_STEP) if M_DIFFICULTY_STEP > 0 else 1
    k_stages_needed = math.ceil(k_span / K_DIFFICULTY_STEP) if K_DIFFICULTY_STEP > 0 else 1

    num_stages = int(max(MIN_CURRICULUM_STAGES, m_stages_needed, k_stages_needed))
    return num_stages


def _generate_curriculum_stages(num_stages):
    """根据给定的阶段总数，通过线性插值法自动生成课程阶段。"""
    m_min, m_max = M_RANGE
    k_min, k_max = K_RANGE
    ccr_min, ccr_max = CCR_RANGE

    stages = {}

    for i in range(num_stages):
        progress = (i + 1) / num_stages

        current_m_max = int(np.round(m_min + (m_max - m_min) * progress))
        stage_m_range = [m_min, min(current_m_max, m_max)]

        current_k_max = int(np.round(k_min + (k_max - k_min) * progress))
        stage_k_range = [k_min, min(current_k_max, k_max)]

        current_ccr_max = round(ccr_min + (ccr_max - ccr_min) * progress, 2)
        stage_ccr_range = [ccr_min, current_ccr_max]

        stages[i] = (stage_m_range, stage_k_range, stage_ccr_range)

    stages[num_stages - 1] = (M_RANGE, K_RANGE, CCR_RANGE)
    return stages


# CURRICULUM_STAGES: 课程学习阶段定义
# Description: 一个字典，定义了从易到难的多个训练阶段。该字典由上方函数根据控制参数自动生成。
# Role: 结构化地引导智能体的学习路径，避免在训练初期被过难的问题压垮。
# Type: dict[int, tuple]
NUM_CURRICULUM_STAGES = _get_adaptive_num_stages()
CURRICULUM_STAGES = _generate_curriculum_stages(NUM_CURRICULUM_STAGES)

# PROMOTION_THRESHOLD_SLR: 课程晋级阈值 (基于SLR)
# Description: Schedule Length Ratio (SLR) 是智能体产生的makespan与HEFT算法产生的makespan之比。
#              当模型在验证集上的平均SLR低于此阈值时，才认为模型已掌握当前课程难度。
# Role: 作为课程晋级的核心性能指标。
# Type: float
PROMOTION_THRESHOLD_SLR = 1.0

# PROMOTION_STABLE_EPOCHS: 课程晋级所需稳定周期数
# Description: 模型需要在连续多少个验证周期内，其平均SLR都满足PROMOTION_THRESHOLD_SLR的要求，
#              才能触发课程晋级。
# Role: 防止因偶然的良好表现而过早晋级，确保模型性能的稳定性。
# Type: int
PROMOTION_STABLE_EPOCHS = 10

# ==============================================================================
# SECTION 3: TRAINING & REPLAY BUFFER PARAMETERS
# ------------------------------------------------------------------------------
# 此部分参数控制着学习器（Learner）的训练循环、优化器以及经验回放机制。
# ==============================================================================

# EPSILON: 用于数值稳定性的微小常数
# Description: 在归一化等计算中添加到分母，以防止因除以零而导致的程序崩溃。
# Type: float
EPSILON = 1e-8

# DEVICE: Pytorch计算设备
# Description: 自动选择可用的CUDA设备（GPU），否则回退到CPU。
# Role: 指定模型训练和推理的主要计算硬件。
# Type: torch.device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NUM_ACTORS: 并行执行器 (Actors) 的数量
# Description: 启动多少个并行的进程来生成自我对弈数据。
# Role: 加速训练数据的生成。数量应根据CPU核心数和内存大小来调整。
# Type: int
CPU_COUNT = os.cpu_count()
NUM_ACTORS = max(1, CPU_COUNT)

# LEARNING_RATE: 优化器的学习率
# Description: Adam优化器在更新模型权重时使用的步长。
# Role: 控制模型参数的更新速度，是训练中最关键的超参数之一。
# Type: float
LEARNING_RATE = 1e-4

# REPLAY_BUFFER_SIZE: 经验回放缓冲区的最大容量
# Description: 缓冲区能存储的最大经验（state, pi, value）数量。
# Role: 存储历史经验，通过随机采样打破数据的时间相关性，稳定训练过程。
# Type: int
REPLAY_BUFFER_SIZE = 100000

# BATCH_SIZE: 训练批次大小
# Description: 每次从经验回放缓冲区中采样多少条经验用于单次梯度更新。
# Role: 影响梯度估计的准确性和训练的稳定性。受限于GPU显存大小。
# Type: int
BATCH_SIZE = 128

# MIN_BUFFER_SIZE_FOR_TRAINING: 开始训练所需的最小经验数
# Description: 学习器在开始训练前，经验回放缓冲区必须达到的经验数量。
# Role: 确保在训练开始时，数据具有一定的多样性，避免早期过拟合。
# Type: int
MIN_BUFFER_SIZE_FOR_TRAINING = 5000

# GAMMA: 折扣因子 (Discount Factor)
# Description: 在计算累积回报时，未来奖励的权重。
# Role: 值为1.0表示不折扣未来奖励，适用于makespan最小化这种具有明确终止状态的片段式任务。
# Type: float
GAMMA = 1.0

# PER_ALPHA: 优先经验回放 (PER) 的优先级影响因子
# Description: 控制TD-error转换为优先级的程度。alpha=0表示均匀采样。
# Role: alpha越大，TD-error越大的样本被采样的概率就越高。
# Type: float
PER_ALPHA = 0.6

# PER_BETA_START: 优先经验回放 (PER) 的重要性采样权重起始值
# Description: 用于修正由非均匀采样带来的偏差。beta会从这个初始值随训练线性增长到1.0。
# Role: 在训练初期，对偏差的修正较小，随着训练进行，修正逐渐增强。
# Type: float
PER_BETA_START = 0.4

# PER_BETA_FRAMES: PER Beta从起始值增长到1.0所需的总训练步数
# Description: 定义了beta值的退火（annealing）周期。
# Role: 控制重要性采样修正强度的增长速度。
# Type: int
PER_BETA_FRAMES = 100000

# ==============================================================================
# SECTION 4: MCTS (Monte Carlo Tree Search) PARAMETERS
# ------------------------------------------------------------------------------
# 此部分参数控制MCTS算法的行为，该算法用于在自我对弈期间增强策略，
# 通过模拟未来的走法来寻找更好的动作。
# ==============================================================================

# MCTS_SIMULATIONS: 每次决策的MCTS模拟次数
# Description: 在选择一个动作之前，MCTS要进行的完整（选择-扩展-评估-反向传播）模拟次数。
# Role: 这是搜索深度和广度的直接体现。值越高，策略的质量越高，但耗时也越长。
#       由于处理器数量K增加到6，MCTS的搜索分支因子会变大。80作为一个初始值是合理的。
#       如果发现模型性能不佳，可以考虑适当增加此值（例如到100或120）。
# Type: int
MCTS_SIMULATIONS = 150

# MCTS_C_PUCT: UCB1公式中的探索常数
# Description: 在MCTS的节点选择阶段，用于平衡利用（exploitation）和探索（exploration）的常数。
# Role: 值越高，MCTS越倾向于探索访问次数较少的节点。
# Type: float
MCTS_C_PUCT = 1.5

# MCTS_DIRICHLET_ALPHA: 狄利克雷噪声的Alpha参数
# Description: 用于生成狄利克雷噪声的浓度参数。
# Role: 控制噪声的形状。
# Type: float
MCTS_DIRICHLET_ALPHA = 0.3

# MCTS_DIRICHLET_EPSILON: 狄利克雷噪声的权重
# Description: 在根节点的先验概率上混合多少比例的狄利克雷噪声。
# Role: 在自我对弈的起始阶段增加探索，防止MCTS过早地收敛到次优策略。
# Type: float
MCTS_DIRICHLET_EPSILON = 0.25

# ==============================================================================
# SECTION 5: MODEL (Graph Transformer) ARCHITECTURE PARAMETERS
# ------------------------------------------------------------------------------
# 此部分参数定义了神经网络（Graph Transformer）的结构和超参数。
# ==============================================================================

# EMBEDDING_DIM: 嵌入维度
# Description: 模型内部将所有节点和边的特征映射到的统一高维空间的维度。
# Role: 这是模型表示能力的核心。维度越高，模型能捕捉的特征越复杂，但计算量和参数量也越大。
# Type: int
EMBEDDING_DIM = 128

# NUM_ENCODER_LAYERS: Graph Transformer编码器层数
# Description: 模型中堆叠的GraphTransformerLayer的数量。
# Role: 增加层数可以使模型学习到图中节点之间更长距离的依赖关系。
# Type: int
NUM_ENCODER_LAYERS = 4

# NUM_ATTENTION_HEADS: 多头注意力机制中的头数
# Description: 在GATConv层中，并行计算的独立注意力机制的数量。
# Role: 允许模型在不同的表示子空间中同时关注不同的邻居信息，增强模型的表达能力。
# Type: int
NUM_ATTENTION_HEADS = 4

# FFN_HIDDEN_DIM: 前馈网络 (FFN) 的隐藏层维度
# Description: Transformer层中Point-wise前馈网络的隐藏层维度。
# Role: （注意：当前模型结构中未使用标准FFN，此参数为备用，保留以备未来模型迭代）。
# Type: int
FFN_HIDDEN_DIM = 512

# DROPOUT_RATE: Dropout比率
# Description: 在训练期间，随机将神经元输出置为零的概率。
# Role: 一种正则化技术，用于防止模型过拟合。（注意：当前模型结构中未使用，保留以备未来模型迭代）。
# Type: float
DROPOUT_RATE = 0.1

# ==============================================================================
# SECTION 6: LOGGING, VALIDATION & CHECKPOINTING
# ------------------------------------------------------------------------------
# 此部分参数定义了所有输出文件和目录的路径，以及各种事件的触发频率。
# 所有输出统一到 'result/' 文件夹下，便于管理。
# ==============================================================================

# --- 目录定义 ---
RESULT_DIR = "result/"
LOG_DIR = os.path.join(RESULT_DIR, "logs/")  # TensorBoard日志目录
MODEL_SAVE_DIR = os.path.join(RESULT_DIR, "models/")  # (可选) 存储单独的模型快照
WEB_MONITOR_DIR = os.path.join(RESULT_DIR, "web_monitor/")  # Web监控界面相关文件目录

# --- 文件路径定义 ---
CHECKPOINT_FILE = os.path.join(RESULT_DIR, "checkpoint.pth.tar")  # 主检查点文件，包含模型和优化器状态
REPLAY_BUFFER_FILE = os.path.join(RESULT_DIR, "replay_buffer.pkl")  # 经验回放缓冲区的持久化文件
VALIDATION_SET_FILE = os.path.join(RESULT_DIR, "validation_set.pkl")  # 固定的验证集，用于一致性评估
WEB_STATUS_FILE = os.path.join(WEB_MONITOR_DIR, "status.json")  # Web UI读取的实时状态文件
VALIDATION_LOG_FILE = os.path.join(RESULT_DIR, "validation_log.json")  # 存储每次验证的详细逐实例结果

# --- 频率控制 (以 global_step 为单位) ---
VALIDATION_INTERVAL = 500  # 每隔多少个训练步运行一次验证
CHECKPOINT_INTERVAL = 100  # 每隔多少个训练步保存一次检查点
MODEL_SYNC_INTERVAL = 100  # 每隔多少个训练步，学习器将最新的模型权重同步给执行器

# --- 验证与评估相关 ---
VALIDATION_SET_SIZE = 100  # 为每个课程阶段生成的固定验证问题实例的数量
ILP_TIME_LIMIT = 3000  # ILP求解器在生成验证集时，为每个实例分配的最大求解时间（秒）。
# 由于K增加到6，ILP问题会更难求解，保持合理的超时限制是必要的。

# ==============================================================================
# 脚本自测试区域
# ==============================================================================
if __name__ == '__main__':
    # 此部分代码仅在直接运行 `python config.py` 时执行，用于验证课程生成逻辑
    print("=" * 60)
    print("🔬 CONFIGURATION SCRIPT SELF-TEST 🔬")
    print("=" * 60)
    print(f"Problem Definition:")
    print(f"  - Task Range (M): {M_RANGE}")
    print(f"  - Processor Range (K): {K_RANGE}")
    print(f"  - CCR Range: {CCR_RANGE}")
    print("\nAdaptive Curriculum Settings:")
    print(f"  - Adaptive Enabled: {ADAPTIVE_CURRICULUM_ENABLED}")
    if ADAPTIVE_CURRICULUM_ENABLED:
        print(f"  - M Difficulty Step: {M_DIFFICULTY_STEP}")
        print(f"  - K Difficulty Step: {K_DIFFICULTY_STEP}")
        print(f"  - Minimum Stages: {MIN_CURRICULUM_STAGES}")

    print(f"\n✅ Calculated Total Curriculum Stages: {NUM_CURRICULUM_STAGES}")
    print("\nGenerated Curriculum Stages:")
    for stage_num, params in CURRICULUM_STAGES.items():
        print(f"  - Stage {stage_num}: M={params[0]}, K={params[1]}, CCR={params[2]}")
    print("=" * 60)