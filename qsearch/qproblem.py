"""把量子线路搜索包装成 pymoo 的 Problem。

两个目标（均为最小化）：
    f1 = E - E0                 基态能量误差，全精度，不做舍入
    f2 = 线路代价                 默认门总数
"""
import logging

import numpy as np
from pymoo.core.problem import Problem

from qsearch import h2_hamiltonian
from qsearch.qencoding import bounds, decode
from qsearch.vqe_trainer import VQEConfig, evaluate

COST_METRICS = ("gates", "cx", "depth", "weighted")


def _cost_of(result, metric):
    if metric == "gates":
        return result.n_gates
    if metric == "cx":
        return result.n_cx
    if metric == "depth":
        return result.depth
    if metric == "weighted":
        # CX 在真实硬件上比单比特门贵得多，按 3 倍计
        return result.n_gates + 2 * result.n_cx
    raise ValueError("unknown cost metric: {}".format(metric))


class QAS(Problem):
    """量子线路架构搜索问题。"""

    def __init__(self, n_qubits=4, min_layers=1, max_layers=6,
                 cfg=None, cost_metric="gates", pop_hist=None, use_cache=True):
        if cost_metric not in COST_METRICS:
            raise ValueError(
                "cost_metric 必须是 {} 之一，收到 {}".format(COST_METRICS, cost_metric)
            )

        lb, ub = bounds(min_layers, max_layers, n_qubits)
        super().__init__(n_var=len(lb), n_obj=2, n_ieq_constr=0, vtype=int)

        # pymoo 0.6 会在 super().__init__ 里把 xl/xu 转成 float，
        # 后续所有解码都用 int(round()) 兜底，见 qencoding._as_int
        self.xl = lb
        self.xu = ub

        self._n_qubits = n_qubits
        self._min_layers = min_layers
        self._max_layers = max_layers
        self._cfg = cfg or VQEConfig()
        self._cost_metric = cost_metric
        self._pop_hist = pop_hist if pop_hist is not None else []
        self._use_cache = use_cache
        self._n_evaluated = 0
        self._ham = h2_hamiltonian.to_pennylane()

    def _evaluate(self, x, out, *args, **kwargs):
        objs = np.full((x.shape[0], self.n_obj), np.nan)

        for i in range(x.shape[0]):
            self._n_evaluated += 1
            arch_id = self._n_evaluated

            spec = decode(
                x[i, :],
                n_qubits=self._n_qubits,
                max_layers=self._max_layers,
                min_layers=self._min_layers,
            )

            logging.info("Architecture id = {}".format(arch_id))
            logging.info(spec.signature)

            try:
                result = evaluate(
                    spec, ham=self._ham, cfg=self._cfg, use_cache=self._use_cache
                )
                objs[i, 0] = result.error
                objs[i, 1] = _cost_of(result, self._cost_metric)

                logging.info(
                    "  energy = {:.10f} Ha, error = {:.3e} Ha".format(
                        result.energy, result.error
                    )
                )
                logging.info(
                    "  n_gates = {}, n_cx = {}, depth = {}, n_params = {}".format(
                        result.n_gates, result.n_cx, result.depth, result.n_params
                    )
                )

                self._pop_hist.append({
                    "id": arch_id,
                    "signature": spec.signature,
                    "n_layers": spec.n_layers,
                    "energy": result.energy,
                    "error": result.error,
                    "n_gates": result.n_gates,
                    "n_cx": result.n_cx,
                    "depth": result.depth,
                    "n_params": result.n_params,
                    "ok": result.ok,
                })
            except Exception as exc:  # 单个架构失败不应该中断整个搜索
                logging.warning("  evaluation failed: %s", exc)
                objs[i, 0] = np.nan
                objs[i, 1] = np.nan

        # 用最差值替换 NaN，保证进化能继续（NaN 会污染非支配排序）
        if np.isnan(objs).any():
            finite = np.nanmax(objs, axis=0) if np.isfinite(objs).any() else np.array([1.0, 1.0])
            fallback = np.where(np.isfinite(finite), finite, 1.0)
            objs = np.where(np.isnan(objs), fallback, objs)

        out["F"] = objs
