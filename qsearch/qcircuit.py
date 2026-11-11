"""把 CircuitSpec 组装成 PennyLane QNode，并统计线路代价。

线路结构（每层）：
    1. 按 gates 给每个比特施加单比特门（I 跳过，H 无参，RX/RY/RZ 带参）
    2. 按 entangler 施加 CX 链

初态固定为 Hartree-Fock 行列式 |1100>，与 h2_hamiltonian.HF_STATE 一致。
"""
from collections import namedtuple

import numpy as np

from qsearch.qencoding import (
    ENTANGLERS,
    ENTANGLER_PAIRS,
    GATE_SET,
    CircuitSpec,
    LayerSpec,
)

# 每种纠缠模式在 CX 链上的临界深度（4 比特下的最优调度层数）
ENTANGLER_DEPTH = {
    "none": 0,
    "linear": 3,
    "ring": 2,
    "star": 3,
    "full": 3,
}

PARAM_GATES = ("RX", "RY", "RZ")

ResourceCount = namedtuple(
    "ResourceCount", ["n_gates", "n_1q", "n_cx", "depth"]
)


def _hf_state(n_qubits):
    state = [1, 1] + [0] * max(0, n_qubits - 2)
    return np.array(state[:n_qubits], dtype=int)


def n_params(spec):
    """带参数的门数量，即 VQE 需要优化的参数个数。"""
    return sum(
        1
        for layer in spec.layers
        for g in layer.gates
        if GATE_SET[g] in PARAM_GATES
    )


def count_resources(spec):
    """解析式统计线路代价，零执行开销。"""
    n_1q = n_cx = depth = 0

    for layer in spec.layers:
        has_1q = any(GATE_SET[g] != "I" for g in layer.gates)
        if has_1q:
            n_1q += sum(1 for g in layer.gates if GATE_SET[g] != "I")

        entangler = ENTANGLERS[layer.entangler]
        pairs = [
            (a, b) for a, b in ENTANGLER_PAIRS[entangler]
            if a < spec.n_qubits and b < spec.n_qubits
        ]
        n_cx += len(pairs)

        if has_1q or pairs:
            depth += (1 if has_1q else 0) + ENTANGLER_DEPTH[entangler]

    return ResourceCount(
        n_gates=n_1q + n_cx,
        n_1q=n_1q,
        n_cx=n_cx,
        depth=depth,
    )


def build_qnode(spec, ham=None, device_name="default.qubit", diff_method="backprop"):
    """构造返回 <psi(theta)|H|psi(theta)> 的 QNode。"""
    import pennylane as qml

    if ham is None:
        from qsearch import h2_hamiltonian

        ham = h2_hamiltonian.to_pennylane()

    device = qml.device(device_name, wires=spec.n_qubits)
    basis = _hf_state(spec.n_qubits)

    def circuit(weights):
        qml.BasisState(basis, wires=list(range(spec.n_qubits)))

        index = 0
        for layer in spec.layers:
            for wire, gate in enumerate(layer.gates):
                name = GATE_SET[gate]
                if name == "I":
                    continue
                if name == "H":
                    qml.Hadamard(wires=wire)
                else:
                    getattr(qml, name)(weights[index], wires=wire)
                    index += 1

            entangler = ENTANGLERS[layer.entangler]
            for a, b in ENTANGLER_PAIRS[entangler]:
                if a < spec.n_qubits and b < spec.n_qubits:
                    qml.CNOT(wires=[a, b])

        return qml.expval(ham)

    try:
        return qml.QNode(circuit, device, interface="autograd", diff_method=diff_method)
    except (TypeError, ValueError):
        # 新版 PennyLane 可能不再接受显式 autograd 接口
        return qml.QNode(circuit, device, interface="auto", diff_method=diff_method)


def reference_spec(n_qubits=4, n_layers=6, gate="RY", entangler="linear"):
    """标准的 hardware-efficient ansatz，用于闭环自检。"""
    gate_idx = GATE_SET.index(gate)
    ent_idx = ENTANGLERS.index(entangler)

    layers = [
        LayerSpec(gates=[gate_idx] * n_qubits, entangler=ent_idx)
        for _ in range(n_layers)
    ]
    signature = "ref-L{}|".format(n_layers) + ";".join(
        "{},{}".format(",".join(str(g) for g in layer.gates), layer.entangler)
        for layer in layers
    )
    return CircuitSpec(
        n_qubits=n_qubits,
        n_layers=n_layers,
        layers=layers,
        signature=signature,
    )


if __name__ == "__main__":
    import pennylane as qml

    spec = reference_spec()
    counted = count_resources(spec)
    n_param = n_params(spec)

    print("reference ansatz: {} layers, {} qubits".format(spec.n_layers, spec.n_qubits))
    print("  params    =", n_param)
    print("  n_1q      =", counted.n_1q)
    print("  n_cx      =", counted.n_cx)
    print("  n_gates   =", counted.n_gates)
    print("  depth     =", counted.depth)

    qnode = build_qnode(spec)
    weights = np.zeros(n_param)
    energy = float(qnode(weights))
    resources = qml.specs(qnode)(weights)["resources"]

    print("\nPennyLane 实际统计:")
    print("  num_gates =", resources.num_gates)
    print("  depth     =", resources.depth)
    print("  energy    = {:.10f} Ha".format(energy))

    # BasisState 会被额外计为一个门，其余门数应完全一致
    assert resources.num_gates == counted.n_gates + 1, (
        "解析统计 {} 与实际 {} 不一致".format(counted.n_gates, resources.num_gates)
    )
    assert n_param == 6 * 4

    # 无纠缠 + 全零参数时，线路退化为恒等，能量应精确等于 HF 能量。
    # 注意：带 CX 的 ansatz 在零参数下也会改变态（CX 不是单位门），
    # 所以这里必须用 entangler="none" 才能做这个比对。
    from qsearch import h2_hamiltonian

    plain = reference_spec(n_layers=1, entangler="none")
    plain_energy = float(build_qnode(plain)(np.zeros(n_params(plain))))
    print("\n零参数无纠缠能量 = {:.10f} Ha".format(plain_energy))
    assert abs(plain_energy - h2_hamiltonian.E_HF) < 1e-9, (
        "全零参数能量 {:.10f} 应等于 HF 能量 {:.10f}".format(
            plain_energy, h2_hamiltonian.E_HF
        )
    )
    print("qcircuit 自检通过")
