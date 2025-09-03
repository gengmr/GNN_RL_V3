# validation/validator.py

import numpy as np
from tqdm import tqdm
import os
import pickle
import json
import time

import config
from dag_generator import generate_dag
from scheduling_env import SchedulingEnv
from heuristics import heft_scheduler, calculate_rank_u
from normalization import Normalizer
from ilp_solver import solve_dag_scheduling_ilp
from mcts import MCTS
from config import MCTS_SIMULATIONS


class Validator:
    def __init__(self, val_set_size, curriculum_stages):
        """
        初始化验证器。
        此方法的核心逻辑是：
        1. 检查是否存在持久化的验证集文件。
        2. 如果存在，则快速加载，节约时间。
        3. 如果不存在，则执行一次性的、可能耗时较长的生成过程，并将其保存以备后用。
        """
        if os.path.exists(config.VALIDATION_SET_FILE):
            print(f"\n[VALIDATOR]  INFO: 发现已存在的验证集文件。")
            print(f"  └─ 正在从 '{config.VALIDATION_SET_FILE}' 加载...")
            start_time = time.time()
            with open(config.VALIDATION_SET_FILE, 'rb') as f:
                self.val_sets = pickle.load(f)
            duration = time.time() - start_time
            print(f"  └─ ✅ 验证集加载成功！(耗时: {duration:.2f} 秒)")
        else:
            print(f"\n[VALIDATOR] WARNING: 未找到已存在的验证集文件。")
            start_time = time.time()
            self.val_sets = {}

            for stage, params in curriculum_stages.items():
                m_r, k_r, ccr_r = params
                stage_set = []
                ilp_success_count = 0

                pbar = tqdm(range(val_set_size),
                            desc=f"  └─ 正在生成课程 {stage} 的验证实例",
                            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")

                for i in pbar:
                    dag, k, proc_speeds = generate_dag(m_r, k_r, ccr_r)
                    calculate_rank_u(dag, proc_speeds)
                    m_heft = heft_scheduler(dag.copy(), k, proc_speeds)

                    # --- 核心修改 ---
                    # 直接调用ILP求解器。根据新要求，如果求解失败，
                    # `solve_dag_scheduling_ilp`会抛出异常，这将中断验证集的生成过程，
                    # 从而使整个项目报错，符合预期。
                    pbar.set_postfix_str(f"正在求解 ILP (实例 c{stage}_i{i})...")
                    m_ilp = solve_dag_scheduling_ilp(dag.copy(), k, proc_speeds)
                    ilp_success_count += 1

                    stage_set.append({
                        'id': f'c{stage}_i{i}', 'dag': dag, 'k': k,
                        'proc_speeds': proc_speeds, 'm_heft': m_heft, 'm_ilp': m_ilp
                    })

                self.val_sets[stage] = stage_set
                print(f"  └─ 课程 {stage} 生成完毕。ILP求解成功: {ilp_success_count}/{val_set_size}")

            print(f"\n[VALIDATOR] INFO: 正在将新生成的验证集保存到 '{config.VALIDATION_SET_FILE}'...")
            with open(config.VALIDATION_SET_FILE, 'wb') as f:
                pickle.dump(self.val_sets, f)

            duration = time.time() - start_time
            print(f"  └─ ✅ 全新验证集生成并保存成功！(总耗时: {duration / 3600:.2f} 小时)")

    def _append_log(self, new_log_entry):
        """加载、追加并保存详细的验证日志。"""
        log_data = []
        if os.path.exists(config.VALIDATION_LOG_FILE):
            with open(config.VALIDATION_LOG_FILE, 'r') as f:
                try:
                    log_data = json.load(f)
                except json.JSONDecodeError:
                    pass

        log_data.append(new_log_entry)

        with open(config.VALIDATION_LOG_FILE, 'w') as f:
            json.dump(log_data, f, indent=4)

    def evaluate(self, model, device, curriculum_stage, global_step):
        """在当前课程对应的验证集上评估模型，并记录详细日志。"""
        model.eval()
        validation_set = self.val_sets[curriculum_stage]
        mcts_evaluator = MCTS(model, device)

        agent_makespans, heft_makespans, ilp_makespans = [], [], []
        win_count, optimal_count = 0, 0

        per_instance_results = []

        for instance in validation_set:
            dag, k, proc_speeds, m_heft, m_ilp = (
                instance['dag'], instance['k'], instance['proc_speeds'],
                instance['m_heft'], instance['m_ilp']
            )

            normalizer = Normalizer(dag, m_heft, proc_speeds)
            env = SchedulingEnv(dag.copy(), k, proc_speeds, normalizer=normalizer)
            _ = env.reset()
            done = False

            while not done:
                if not np.any(env.get_action_mask()): break
                pi, _ = mcts_evaluator.search(env, MCTS_SIMULATIONS, env.heft_makespan, dirichlet_epsilon=0.0)
                if not pi: break
                best_action = max(pi, key=pi.get)
                _, _, done = env.step(best_action)

            m_agent = env.get_makespan()
            agent_makespans.append(m_agent)
            heft_makespans.append(m_heft)

            if m_agent < m_heft: win_count += 1
            if m_ilp > 0:
                ilp_makespans.append(m_ilp)
                if np.isclose(m_agent, m_ilp): optimal_count += 1

            per_instance_results.append({
                'id': instance['id'],
                'agent_makespan': round(m_agent, 2),
                'heft_makespan': round(m_heft, 2),
                'ilp_makespan': round(m_ilp, 2) if m_ilp > 0 else -1,
                'slr': round(m_agent / m_heft, 4) if m_heft > 0 else 1.0,
                'gap': round(m_agent / m_ilp, 4) if m_ilp > 0 else -1.0
            })

        log_entry = {
            'global_step': global_step,
            'curriculum_stage': curriculum_stage,
            'results': per_instance_results
        }
        self._append_log(log_entry)

        avg_agent_makespan = np.mean(agent_makespans) if agent_makespans else 0.0
        avg_heft_makespan = np.mean(heft_makespans) if heft_makespans else 0.0
        avg_ilp_makespan = np.mean(ilp_makespans) if ilp_makespans else -1.0

        slr_list = [a / h for a, h in zip(agent_makespans, heft_makespans) if h > 0]
        avg_slr = np.mean(slr_list) if slr_list else 1.0

        gap_list = [a / i for a, i in zip(agent_makespans, ilp_makespans) if i > 0]
        avg_gap = np.mean(gap_list) if gap_list else -1.0

        win_rate = win_count / len(validation_set) if validation_set else 0.0
        optimality_rate = optimal_count / len(ilp_makespans) if ilp_makespans else -1.0

        return {
            "Avg_Makespan_Agent": avg_agent_makespan,
            "Avg_Makespan_HEFT": avg_heft_makespan,
            "Avg_Makespan_ILP": avg_ilp_makespan,
            "Avg_SLR_vs_HEFT": avg_slr,
            "Win_Rate_vs_HEFT": win_rate,
            "Avg_Optimality_Gap": avg_gap,
            "Optimality_Rate": optimality_rate
        }