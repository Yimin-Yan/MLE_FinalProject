"""
symmetry.py —— 对称性数据增强

放进 agent_code/my_agent_dqn_attack_sym/symmetry.py

Bomberman 的棋盘在四个旋转和镜像下等价：把整个局面旋转 90 度，
最优策略也只是跟着旋转。因此每条经验都可以生成 8 条等价样本，
相当于训练数据量翻八倍，而不需要多跑一局游戏。

这对 DQN 尤其有用 —— 网络要在十万局内学会 38 维输入的映射，
样本效率是主要瓶颈。

注意：变换必须同时作用于特征和动作，否则会产生自相矛盾的样本，
静默污染训练数据。下面的 self_test() 用来验证变换的正确性。
"""

import numpy as np

# DIR_ORDER = ['UP', 'RIGHT', 'DOWN', 'LEFT'] -> 0, 1, 2, 3
# 顺时针旋转 90 度：UP -> RIGHT，即 d -> (d + 1) % 4
# 水平镜像（左右翻转）：LEFT <-> RIGHT，UP/DOWN 不变
MIRROR = [0, 3, 2, 1]

# 特征元组中属于"方向"的下标（取值 0-3 为方向，4 表示无/不适用）
DIR_SLOTS = [5, 6, 10, 11]     # escape_dir, target_dir, last_dir, opp_dir

# walkable 占 [0:4]，按 DIR_ORDER 排列
WALK_SLICE = slice(0, 4)


def _map_dir(d, rot, mirror):
    """把一个方向索引按给定的旋转和镜像变换。

    值为 4（无目标 / 不适用）保持不变 —— 它不是方向。
    """
    if d == 4:
        return 4
    if mirror:
        d = MIRROR[d]
    return (d + rot) % 4


def transform_features(features, rot, mirror):
    """变换 14 维特征元组。

    方向类字段按 _map_dir 映射；walkable 四位要重排 ——
    变换后"向上是否可走"应当等于变换前"某个方向是否可走"。
    其余字段（危险等级、bomb_safe、crate_gain 等）与方向无关，原样保留。
    """
    f = list(features)

    # walkable 重排：新方向 d 的可通行性 = 旧方向 src 的可通行性
    old_walk = features[WALK_SLICE]
    new_walk = [0, 0, 0, 0]
    for d in range(4):
        src = _map_dir(d, rot, mirror)
        new_walk[src] = old_walk[d]
    f[0:4] = new_walk

    for slot in DIR_SLOTS:
        f[slot] = _map_dir(features[slot], rot, mirror)

    return tuple(f)


def transform_action(action_idx, rot, mirror):
    """变换动作索引。ACTIONS 前四项是方向，WAIT/BOMB 与方向无关。"""
    if action_idx >= 4:
        return action_idx
    return _map_dir(action_idx, rot, mirror)


def all_transforms():
    """返回全部 8 种 (rot, mirror) 组合。"""
    return [(r, m) for m in (False, True) for r in range(4)]


def augment(features, action_idx, next_features):
    """给一条转移生成 8 个等价版本。

    返回 [(feat, act, next_feat), ...]，其中 next_features 可以是 None
    （终止转移）。
    """
    out = []
    for rot, mirror in all_transforms():
        f = transform_features(features, rot, mirror)
        a = transform_action(action_idx, rot, mirror)
        nf = None if next_features is None else transform_features(
            next_features, rot, mirror)
        out.append((f, a, nf))
    return out


# ============================================================
# 自检：变换必须构成一个群，且恒等变换不改变任何东西
# ============================================================

def self_test():
    """跑一遍基本性质检查。改动这个文件后务必执行一次：
        python -c "from symmetry import self_test; self_test()"
    """
    feat = (1, 0, 1, 1,      # walkable: UP/DOWN/LEFT 可走，RIGHT 不可
            2,               # danger
            1,               # escape_dir = RIGHT
            3,               # target_dir = LEFT
            1, 1, 0,         # bomb_safe, bombs_available, crate_gain
            0,               # last_dir = UP
            4, 1, 0)         # opp_dir = 无, opp_near, opp_in_blast

    # 1. 恒等变换
    assert transform_features(feat, 0, False) == feat, "恒等变换改变了特征"
    assert transform_action(1, 0, False) == 1, "恒等变换改变了动作"

    # 2. 旋转四次回到原状
    f = feat
    for _ in range(4):
        f = transform_features(f, 1, False)
    assert f == feat, "旋转四次未回到原状"

    # 3. 镜像两次回到原状
    f = transform_features(transform_features(feat, 0, True), 0, True)
    assert f == feat, "镜像两次未回到原状"

    # 4. 方向值 4（无目标）在所有变换下保持 4
    for rot, mirror in all_transforms():
        assert transform_features(feat, rot, mirror)[11] == 4, \
            "opp_dir=4 在变换后被改变"

    # 5. walkable 的"可走方向数"是不变量
    n = sum(feat[0:4])
    for rot, mirror in all_transforms():
        assert sum(transform_features(feat, rot, mirror)[0:4]) == n, \
            "walkable 重排后可走方向数改变"

    # 6. 8 种变换互不相同（对一个非对称的特征而言）
    seen = {transform_features(feat, r, m) for r, m in all_transforms()}
    assert len(seen) == 8, f"8 种变换只产生了 {len(seen)} 种不同结果"

    print("symmetry self-test 全部通过")


if __name__ == "__main__":
    self_test()
