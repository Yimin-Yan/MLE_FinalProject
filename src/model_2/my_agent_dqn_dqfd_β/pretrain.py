"""
pretrain.py —— DQfD 预训练

放进 agent_code/my_agent_dqn_dqfd/pretrain.py

用 rule_based_agent 的演示数据预训练 Q 网络，让它在开始自己探索之前
就已经会躲炸弹、收金币、炸箱子。动机来自 Hester et al. (AAAI 2018)：
纯 DQN 前几万局基本在瞎摸索，而专家数据能直接跳过这一段。

损失由两部分组成：

  TD 损失：保证 Q 函数满足 Bellman 方程，从而能在预训练后继续做
           标准的 RL 更新。

  大边际分类损失 J_E(Q) = max_a [Q(s,a) + l(a_E,a)] - Q(s,a_E)
           其中 a_E 是专家动作，l(a_E,a) 在 a=a_E 时为 0、否则为
           MARGIN。它强制非专家动作的 Q 值至少比专家动作低一个边际。

第二项是必需的：演示数据只覆盖状态空间的一小部分，很多状态-动作组合
从未出现过。若只用 TD 更新，网络会朝这些毫无依据的估计值更新，并把
错误传播到整个 Q 函数。
"""

import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .callbacks import ACTIONS, features_to_vector
from .symmetry import augment

DEMO_PATH = os.path.join(os.path.dirname(__file__), "demonstrations.pkl")

# --- 预训练超参数 ---
PRETRAIN_STEPS = 8000      # 梯度更新次数
PRETRAIN_BATCH = 64
MARGIN = 0.8               # 大边际损失里的 l(a_E, a)
LAMBDA_E = 0.1             # 监督损失权重
LEARNING_RATE = 5e-4
GAMMA = 0.95               # 与正式训练一致


def load_demonstrations(logger=None, use_symmetry=True):
    """读取演示数据并转成 (状态向量, 动作索引, 下一状态向量, 奖励)。

    对称增强在这里做：每条专家转移生成 8 个等价版本，
    88k 条原始数据因此变成约 70 万条。
    """
    if not os.path.isfile(DEMO_PATH):
        if logger:
            logger.warning(f"找不到演示数据 {DEMO_PATH}，跳过预训练")
        return []

    with open(DEMO_PATH, "rb") as f:
        demos = pickle.load(f)

    out = []
    for d in demos:
        a_idx = ACTIONS.index(d.action)
        if use_symmetry:
            variants = augment(d.state, a_idx, d.next_state)
        else:
            variants = [(d.state, a_idx, d.next_state)]

        for f_, a_, nf_ in variants:
            out.append((
                features_to_vector(f_),
                a_,
                None if nf_ is None else features_to_vector(nf_),
                d.reward,
            ))

    if logger:
        logger.info(f"演示数据：{len(demos)} 条原始 -> {len(out)} 条（含对称增强）")
    print(f"[DQfD] 演示数据 {len(demos)} -> {len(out)} 条")
    return out


def _make_batch(data, size):
    batch = random.sample(data, size)

    states = torch.from_numpy(np.stack([b[0] for b in batch]))
    actions = torch.tensor([b[1] for b in batch], dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor([b[3] for b in batch], dtype=torch.float32)

    non_final = torch.tensor([b[2] is not None for b in batch], dtype=torch.bool)
    next_states = [b[2] for b in batch if b[2] is not None]

    return states, actions, rewards, non_final, next_states


def dqfd_loss(model, target_model, states, actions, rewards,
              non_final, next_states, is_demo=True):
    """TD 损失 + （仅对演示数据的）大边际分类损失。

    返回 (总损失, 逐样本 TD 误差)。
    """
    batch_size = states.shape[0]
    q_all = model(states)
    q_pred = q_all.gather(1, actions).squeeze(1)

    q_next = torch.zeros(batch_size)
    if next_states:
        with torch.no_grad():
            nxt = torch.from_numpy(np.stack(next_states))
            q_next[non_final] = target_model(nxt).max(1).values
    target = rewards + GAMMA * q_next

    td_loss = nn.functional.smooth_l1_loss(q_pred, target)

    if not is_demo:
        return td_loss, (target - q_pred).detach()

    # 大边际分类损失：专家动作那一项的 margin 为 0，其余为 MARGIN
    margins = torch.full_like(q_all, MARGIN)
    margins.scatter_(1, actions, 0.0)
    j_e = (q_all + margins).max(1).values - q_pred

    return td_loss + LAMBDA_E * j_e.mean(), (target - q_pred).detach()


def pretrain(model, target_model, logger=None):
    """在演示数据上预训练。就地修改 model，返回是否成功执行。"""
    data = load_demonstrations(logger)
    if not data:
        return False

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()

    print(f"[DQfD] 开始预训练 {PRETRAIN_STEPS} 步...")
    for step in range(PRETRAIN_STEPS):
        states, actions, rewards, non_final, next_states = _make_batch(
            data, PRETRAIN_BATCH)

        loss, _ = dqfd_loss(model, target_model, states, actions,
                            rewards, non_final, next_states, is_demo=True)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        # 预训练期间也要定期同步目标网络，否则目标一直是随机初始值
        if (step + 1) % 500 == 0:
            target_model.load_state_dict(model.state_dict())
            msg = f"[DQfD] 预训练 {step + 1}/{PRETRAIN_STEPS}  loss={loss.item():.4f}"
            print(msg)
            if logger:
                logger.info(msg)

    model.eval()
    target_model.load_state_dict(model.state_dict())
    print("[DQfD] 预训练完成")
    return True
