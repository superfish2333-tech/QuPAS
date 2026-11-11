"""VQE 参数优化与 fitness 评估。

对给定线路优化参数，返回基态能量、相对精确解的误差、以及线路代价。

为什么手写 Adam 而不用 PennyLane 自带优化器：
qml.optimize.AdamOptimizer / qml.GradientDescentOptimizer 在不同版本间反复
迁移和弃用，直接用 qml.grad 手写 20 行 Adam 可以跨版本稳定运行。

注意：f1 = E - E0 必须保留全精度。整个可搜索能量区间只有 0.02 Ha，
一旦舍入到 3 位小数就会大面积退化成 0，导致 Pareto 前沿塌缩。
"""
from dataclasses import dataclass, field

import numpy as np

from qsearch import h2_hamiltonian
from qsearch.qcircuit import build_qnode, count_resources, n_params


@dataclass
class VQEConfig:
    steps: int = 150
    lr: float = 0.05
    restarts: int = 2
    tol: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    device_name: str = "default.qubit"
    diff_method: str = "backprop"


@dataclass
class EvalResult:
    energy: float
    error: float
    n_gates: int
    n_cx: int
    depth: int
    n_params: int
    params: np.ndarray = field(repr=False)
    ok: bool


_CACHE = {}


def _adam(qnode, params, cfg):
    """手写 Adam，返回 (最优能量, 参数)。"""
    import pennylane as qml
    from pennylane import numpy as pnp

    grad_fn = qml.grad(qnode)

    m = pnp.zeros_like(params)
    v = pnp.zeros_like(params)

    energy = float(qnode(params))
    for step in range(1, cfg.steps + 1):
        grad = grad_fn(params)

        m = cfg.beta1 * m + (1 - cfg.beta1) * grad
        v = cfg.beta2 * v + (1 - cfg.beta2) * grad ** 2

        m_hat = m / (1 - cfg.beta1 ** step)
        v_hat = v / (1 - cfg.beta2 ** step)

        params = params - cfg.lr * m_hat / (pnp.sqrt(v_hat) + cfg.eps)

        new_energy = float(qnode(params))
        if abs(new_energy - energy) < cfg.tol:
            energy = new_energy
            break
        energy = new_energy

    return energy, params


def evaluate(spec, ham=None, cfg=None, seed=0, use_cache=True):
    """对线路做 VQE 优化，返回 EvalResult。"""
    from pennylane import numpy as pnp

    cfg = cfg or VQEConfig()
    key = (spec.signature, cfg.device_name)

    if use_cache and key in _CACHE:
        return _CACHE[key]

    if ham is None:
        ham = h2_hamiltonian.to_pennylane()

    qnode = build_qnode(
        spec,
        ham=ham,
        device_name=cfg.device_name,
        diff_method=cfg.diff_method,
    )
    n_param = n_params(spec)
    resources = count_resources(spec)

    if n_param == 0:
        energy = float(qnode([]))
        result = EvalResult(
            energy=energy,
            error=energy - h2_hamiltonian.E_EXACT,
            n_gates=resources.n_gates,
            n_cx=resources.n_cx,
            depth=resources.depth,
            n_params=0,
            params=np.zeros(0),
            ok=False,
        )
        if use_cache:
            _CACHE[key] = result
        return result

    best_energy = np.inf
    best_params = np.zeros(n_param)

    for restart in range(max(1, cfg.restarts)):
        rng = np.random.default_rng(seed * 1009 + restart)
        init = rng.uniform(-0.1, 0.1, size=n_param)
        params = pnp.array(init, requires_grad=True)

        energy, optimized = _adam(qnode, params, cfg)
        if energy < best_energy:
            best_energy = energy
            best_params = np.array(optimized, dtype=float)

    error = best_energy - h2_hamiltonian.E_EXACT

    result = EvalResult(
        energy=best_energy,
        error=error,
        n_gates=resources.n_gates,
        n_cx=resources.n_cx,
        depth=resources.depth,
        n_params=n_param,
        params=best_params,
        ok=bool(np.isfinite(error) and error < 1e-3),
    )

    if use_cache:
        _CACHE[key] = result
    return result


def clear_cache():
    _CACHE.clear()


if __name__ == "__main__":
    from qsearch.qcircuit import reference_spec

    cfg = VQEConfig()
    spec = reference_spec(n_layers=6, entangler="linear")

    print("reference ansatz: {} layers, {} qubits, {} params".format(
        spec.n_layers, spec.n_qubits, n_params(spec)
    ))
    print("VQE config:", cfg)

    result = evaluate(spec, cfg=cfg, seed=0)
    print("\n  energy    = {:.10f} Ha".format(result.energy))
    print("  exact     = {:.10f} Ha".format(h2_hamiltonian.E_EXACT))
    print("  HF        = {:.10f} Ha".format(h2_hamiltonian.E_HF))
    print("  error     = {:.3e} Ha".format(result.error))
    print("  n_gates   = {}".format(result.n_gates))
    print("  n_cx      = {}".format(result.n_cx))
    print("  depth     = {}".format(result.depth))

    assert result.error < 1e-3, (
        "参考 ansatz 的能量误差 {:.3e} 超过 1e-3，最小闭环未成立".format(
            result.error
        )
    )
    print("\nvqe_trainer 自检通过：|E - E0| < 1e-3")
