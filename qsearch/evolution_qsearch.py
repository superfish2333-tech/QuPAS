"""量子线路架构搜索（QAS）的进化搜索入口。

使用本项目内置的 NSGA-II 引擎（qsearch/nsga2.py），双目标为
f1 = E - E0（能量误差）与 f2 = 线路代价。

用法（先跑冒烟）：
    python -m qsearch.evolution_qsearch --pop_size 4 --n_gens 2 --n_offspring 4 \
        --vqe_steps 50 --max_layers 3
"""
import sys
import os

# 自动定位项目根目录（qsearch/ 上一级），保证 `python qsearch/evolution_qsearch.py` 可直接运行
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import logging
import argparse

import numpy as np

from pymoo.core.repair import Repair
from pymoo.optimize import minimize

from qsearch import nsga2 as engine
from qsearch import h2_hamiltonian
from qsearch.qencoding import canonicalize
from qsearch.qproblem import COST_METRICS, QAS
from qsearch.vqe_trainer import VQEConfig


class ArchRepair(Repair):
    """基因组规范化，让"尾部基因不同但线路相同"的个体在基因层面也相同。

    repair 在 pymoo 里作用于初始种群和每代子代，且在 eliminate_duplicates
    之前执行，所以归零死基因后去重才能真正生效。
    """

    def __init__(self, n_qubits, min_layers, max_layers):
        super().__init__(vtype=int)
        self._n_qubits = n_qubits
        self._min_layers = min_layers
        self._max_layers = max_layers

    def _do(self, problem, X, **kwargs):
        return canonicalize(
            X,
            n_qubits=self._n_qubits,
            max_layers=self._max_layers,
            min_layers=self._min_layers,
        )


def create_exp_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    print("Experiment dir : {}".format(path))


def build_parser():
    parser = argparse.ArgumentParser("NSGA-II for Quantum Architecture Search (VQE)")
    parser.add_argument("--save", type=str, default="QAS-VQE", help="experiment name")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    # 线路搜索空间
    parser.add_argument("--n_qubits", type=int, default=4, help="H2 固定 4 比特")
    parser.add_argument("--min_layers", type=int, default=1, help="最少层数")
    parser.add_argument("--max_layers", type=int, default=6, help="最多层数")

    # 进化参数
    parser.add_argument("--pop_size", type=int, default=20, help="population size")
    parser.add_argument("--n_gens", type=int, default=20, help="number of generations")
    parser.add_argument("--n_offspring", type=int, default=20,
                        help="每代子代数，建议与 pop_size 相同")

    # VQE 参数
    parser.add_argument("--vqe_steps", type=int, default=150, help="每个架构的参数优化步数")
    parser.add_argument("--vqe_lr", type=float, default=0.05, help="Adam 学习率")
    parser.add_argument("--vqe_restarts", type=int, default=2, help="参数随机重启次数")

    # 代价度量与模拟器
    parser.add_argument("--cost_metric", type=str, default="gates", choices=COST_METRICS,
                        help="线路代价的定义")
    parser.add_argument("--device", type=str, default="default.qubit", help="PennyLane 设备")
    parser.add_argument("--diff_method", type=str, default="backprop", help="微分方式")
    parser.add_argument("--no_cache", action="store_true", help="关闭架构级缓存")

    return parser


def do_every_generations(algorithm):
    gen = algorithm.n_gen
    pop_obj = algorithm.pop.get("F")

    logging.info("generation = {}".format(gen))
    logging.info(
        "population error: best = {:.3e}, mean = {:.3e}, median = {:.3e}, worst = {:.3e}".format(
            np.min(pop_obj[:, 0]), np.mean(pop_obj[:, 0]),
            np.median(pop_obj[:, 0]), np.max(pop_obj[:, 0]),
        )
    )
    logging.info(
        "population cost : best = {}, mean = {:.2f}, median = {:.2f}, worst = {}".format(
            np.min(pop_obj[:, 1]), np.mean(pop_obj[:, 1]),
            np.median(pop_obj[:, 1]), np.max(pop_obj[:, 1]),
        )
    )


def main():
    args = build_parser().parse_args()
    args.save = "qas-{}-{}".format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save)

    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m/%d %I:%M:%S %p")
    file_handler = logging.FileHandler(os.path.join(args.save, "log.txt"))
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)

    np.random.seed(args.seed)
    logging.info("args = %s", args)

    h2_hamiltonian.check()

    cfg = VQEConfig(
        steps=args.vqe_steps,
        lr=args.vqe_lr,
        restarts=args.vqe_restarts,
        device_name=args.device,
        diff_method=args.diff_method,
    )

    problem = QAS(
        n_qubits=args.n_qubits,
        min_layers=args.min_layers,
        max_layers=args.max_layers,
        cfg=cfg,
        cost_metric=args.cost_metric,
        use_cache=not args.no_cache,
    )

    method = engine.nsganet(
        pop_size=args.pop_size,
        n_offsprings=args.n_offspring,
        eliminate_duplicates=True,
        repair=ArchRepair(args.n_qubits, args.min_layers, args.max_layers),
    )

    res = minimize(
        problem,
        method,
        callback=do_every_generations,
        termination=("n_gen", args.n_gens),
        seed=args.seed,
        verbose=False,
    )

    logging.info("=" * 70)
    logging.info("Pareto front (error vs {})".format(args.cost_metric))
    fronts = res.opt.get("F")
    order = np.argsort(fronts[:, 0])
    for idx in order:
        logging.info(
            "  error = {:.3e} Ha, cost = {:.2f}".format(fronts[idx, 0], fronts[idx, 1])
        )
    logging.info("=" * 70)

    return res


if __name__ == "__main__":
    main()
