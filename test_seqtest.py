"""seqtest 的自测：unittest + 精确动态规划（枚举小规模 SPRT 的全部路径）。

运行：python3 -m unittest -v test_seqtest.py
"""

import math
import random
import unittest

from seqtest import (
    SPRTConfig,
    BernoulliPairSPRT,
    FamilySPRT,
    bonferroni_configs,
    bern_asn,
)


def exact_error_rates(cfg, horizon=None, prune=1e-14):
    """对截断/未截断 SPRT 做精确 DP，返回 (P0拒绝, P1接受)。

    状态 = 当前 log LLR（按 1e-9 网格取整合并，避免浮点键组合爆炸；
    权重 < prune 的状态被剪掉，总剪除质量可忽略）。
    horizon 给定时按 nmax=horizon 截断（撞顶记接受）；None 表示未截断。
    """
    a, b = cfg.log_upper, cfg.log_lower
    steps = (cfg.llr_pair(0, 0), cfg.llr_pair(0, 1),
             cfg.llr_pair(1, 0), cfg.llr_pair(1, 1))
    grid = 1e-9

    def probs(p_t, p_c):
        return ((1 - p_t) * (1 - p_c), (1 - p_t) * p_c,
                p_t * (1 - p_c), p_t * p_c)

    def run(p_t, p_c):
        ps = probs(p_t, p_c)
        states = {0: 1.0}
        reject = accept = 0.0
        cap = horizon if horizon is not None else 10_000
        for n in range(1, cap + 1):
            nxt = {}
            for k, w in states.items():
                z = k * grid
                for dz, pr in zip(steps, ps):
                    if pr == 0.0:
                        continue
                    nz = z + dz
                    if nz >= a:
                        reject += w * pr
                    elif nz <= b:
                        accept += w * pr
                    elif horizon is not None and n >= horizon:
                        accept += w * pr  # 撞顶记接受
                    else:
                        kk = round(nz / grid)
                        nxt[kk] = nxt.get(kk, 0.0) + w * pr
            states = nxt
            if not states:
                break
            if prune:
                states = {k: w for k, w in states.items() if w >= prune}
        return reject, accept

    p0_reject, _ = run(cfg.p0_t, cfg.p0_c)
    _, p1_accept = run(cfg.p1_t, cfg.p1_c)
    return p0_reject, p1_accept


class TestExactGuarantees(unittest.TestCase):
    def test_untruncated_alpha_beta_bounds(self):
        # 未截断 SPRT：精确错误率必须不超过名义 alpha、beta（Wald）
        cfg = SPRTConfig.for_effect(0.5, 0.2, alpha=0.1, beta=0.1, nmax=10_000)
        alpha_hat, beta_hat = exact_error_rates(cfg, horizon=None)
        self.assertLessEqual(alpha_hat, cfg.alpha + 1e-9)
        self.assertLessEqual(beta_hat, cfg.beta + 1e-9)

    def test_truncated_alpha_still_bounded(self):
        # 截断后 I 类错误仍严格 ≤ alpha（Ville），且截断越紧越保守
        cfg = SPRTConfig.for_effect(0.5, 0.2, alpha=0.1, beta=0.1)
        a_full, _ = exact_error_rates(cfg, horizon=60)
        a_tight, _ = exact_error_rates(cfg, horizon=10)
        self.assertLessEqual(a_full, cfg.alpha + 1e-12)
        self.assertLessEqual(a_tight, cfg.alpha + 1e-12)
        self.assertLess(a_tight, a_full)

    def test_nominal_005(self):
        cfg = SPRTConfig.for_effect(0.5, 0.2, alpha=0.05, beta=0.20, nmax=10_000)
        alpha_hat, beta_hat = exact_error_rates(cfg, horizon=None)
        self.assertLessEqual(alpha_hat, 0.05 + 1e-9)
        self.assertLessEqual(beta_hat, 0.20 + 1e-9)


class TestStateMachine(unittest.TestCase):
    def test_first_observation_boundary_hit(self):
        # 极大 alpha=0.55：上界 log A≈0.598 < llr(1,0)≈0.673，一对 (1,0) 即命中
        cfg = SPRTConfig(0.5, 0.5, 0.7, 0.3, alpha=0.55, beta=0.1, nmax=100)
        self.assertGreater(cfg.llr_pair(1, 0), cfg.log_upper)
        test = BernoulliPairSPRT(cfg)
        self.assertEqual(test.update(1, 0), "reject_h0")
        self.assertEqual(test.n, 1)
        with self.assertRaises(RuntimeError):
            test.update(0, 0)

    def test_lower_boundary_hit(self):
        cfg = SPRTConfig(0.5, 0.5, 0.7, 0.3, alpha=0.35, beta=0.6, nmax=100)
        test = BernoulliPairSPRT(cfg)
        self.assertEqual(test.update(0, 1), "accept_h0")

    def test_cap_forces_accept(self):
        cfg = SPRTConfig(0.5, 0.5, 0.7, 0.3, alpha=0.01, beta=0.01, nmax=3)
        test = BernoulliPairSPRT(cfg)
        for pair in ((0, 0), (1, 1), (0, 0)):  # 不越任何界
            d = test.update(*pair)
        self.assertEqual(test.n, 3)
        self.assertEqual(d, "accept_h0")  # 撞顶绝不记阳性

    def test_determinism_with_seeded_rng_but_lr_path_fixed(self):
        cfg = SPRTConfig.for_effect(0.5, 0.15, alpha=0.1, beta=0.1)
        stream = [(1, 0), (1, 1), (0, 0), (1, 0), (0, 1)]
        t1, t2 = BernoulliPairSPRT(cfg), BernoulliPairSPRT(cfg)
        for p in stream:
            t1.update(*p)
            t2.update(*p)
        self.assertEqual((t1.n, t1.log_lr, t1.decision),
                         (t2.n, t2.log_lr, t2.decision))

    def test_invalid_configs(self):
        with self.assertRaises(ValueError):
            SPRTConfig.for_effect(0.5, -0.6)  # p1_c 越界
        with self.assertRaises(ValueError):
            SPRTConfig(0.5, 0.5, 0.5, 0.5, alpha=0.1, beta=0.1)  # 无效应
        with self.assertRaises(ValueError):
            SPRTConfig(0.5, 0.5, 0.6, 0.4, alpha=0.9, beta=0.2)  # 和≥1

    def test_negative_effect(self):
        # 真实效应为负（处理组 0.3 < 对照组 0.7）：
        # 1) 原方向检验应快速判 accept_h0（不误报正向）；
        # 2) 交换双臂后同一数据应判 reject_h0（负效应被正确检出）。
        cfg = SPRTConfig(0.5, 0.5, 0.7, 0.3, alpha=0.1, beta=0.1)

        rng = random.Random(123)
        fwd = BernoulliPairSPRT(cfg)
        while not fwd.stopped:
            x_t = 1 if rng.random() < 0.3 else 0
            x_c = 1 if rng.random() < 0.7 else 0
            fwd.update(x_t, x_c)
        self.assertEqual(fwd.decision, "accept_h0")
        self.assertLess(fwd.n, cfg.nmax)

        rng = random.Random(123)
        rev = BernoulliPairSPRT(cfg)
        while not rev.stopped:
            x_t = 1 if rng.random() < 0.3 else 0
            x_c = 1 if rng.random() < 0.7 else 0
            rev.update(x_c, x_t)  # 交换双臂 => 对检验而言是强正向
        self.assertEqual(rev.decision, "reject_h0")
        self.assertLess(rev.n, cfg.nmax)


class TestBonferroni(unittest.TestCase):
    def test_adjusted_levels(self):
        cfgs = [SPRTConfig.for_effect(0.5, 0.1, nmax=100) for _ in range(4)]
        adj = bonferroni_configs(cfgs, alpha=0.05)
        self.assertTrue(all(abs(c.alpha - 0.0125) < 1e-12 for c in adj))
        self.assertTrue(all(c.nmax == 100 for c in adj))  # nmax 不变
        # 上界随之变高
        self.assertGreater(adj[0].log_upper, cfgs[0].log_upper)

    def test_family_all_null_exact_fwer(self):
        # K=2 个独立的小检验，全部为真 H0；精确 DP 求 FWER ≤ alpha
        sub = SPRTConfig.for_effect(0.5, 0.25, alpha=0.05 / 2, beta=0.2,
                                    nmax=40)
        fpr1, _ = exact_error_rates(sub, horizon=40)
        self.assertLessEqual(1 - (1 - fpr1) ** 2, 0.05 + 1e-12)

    def test_family_runs(self):
        cfgs = [SPRTConfig.for_effect(0.5, 0.1, nmax=50) for _ in range(2)]
        fam = FamilySPRT(cfgs, alpha=0.1, rng=random.Random(7))
        rng = random.Random(7)
        while not fam.stopped:
            for i in fam.active_indices():
                x_t = 1 if rng.random() < 0.5 else 0
                x_c = 1 if rng.random() < 0.5 else 0
                fam.update(i, x_t, x_c)
                if fam.stopped:
                    break
        self.assertIn(fam.decision, ("reject_h0", "accept_h0"))


class TestASN(unittest.TestCase):
    def test_asn_positive_finite(self):
        cfg = SPRTConfig.for_effect(0.5, 0.1, alpha=0.05, beta=0.2)
        self.assertGreater(cfg.asn_h1(), 0)
        self.assertGreater(cfg.asn_h0(), 0)
        # 默认 nmax = ceil(3 * ASN_H1)
        self.assertEqual(cfg.nmax, math.ceil(3 * bern_asn(0.6, 0.4, cfg)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
