"""分层门矩阵编码：定长 genome 表达变长量子线路。

pymoo 要求 n_var 固定，所以用定长数组 + 可变的层数基因来表达"线路可深可浅"：

    gene[0]                     层数 L，取值 [min_layers, max_layers]
    gene[1 + k*(Q+1) ... ]      第 k 层的 Q 个单比特门类型 + 1 个纠缠模式

超过 L 的尾部层在解码时被丢弃，不参与线路构建。

注意：pymoo 0.6 的 Problem 会把 xl/xu 转成 float，交叉变异后基因可能变成
浮点甚至越界。所有取值必须 `np.clip(int(round(...)))` —— 先 round 再 clip。
"""
from dataclasses import dataclass

import numpy as np

# 单比特门原语。I 让电路可以"跳过"某个比特，H 不带参数，RX/RY/RZ 带参数。
GATE_SET = ("I", "RX", "RY", "RZ", "H")

# 纠缠模式及其 CX 连接
ENTANGLERS = ("none", "linear", "ring", "star", "full")

ENTANGLER_PAIRS = {
    "none": [],
    "linear": [(0, 1), (1, 2), (2, 3)],
    "ring": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "star": [(0, 1), (0, 2), (0, 3)],
    "full": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
}


@dataclass
class LayerSpec:
    """单层：每个比特的门类型 + 该层末尾的纠缠模式。"""

    gates: list
    entangler: int


@dataclass
class CircuitSpec:
    """完整线路描述，signature 用于缓存和去重。"""

    n_qubits: int
    n_layers: int
    layers: list
    signature: str


def n_var(max_layers, n_qubits):
    """genome 长度：1 个层数基因 + 每层 (n_qubits + 1) 个基因。"""
    return 1 + max_layers * (n_qubits + 1)


def bounds(min_layers, max_layers, n_qubits):
    """返回 pymoo 需要的 (lb, ub)，均为一维 float 数组。"""
    size = n_var(max_layers, n_qubits)
    lb = np.zeros(size, dtype=float)
    ub = np.zeros(size, dtype=float)

    lb[0] = min_layers
    ub[0] = max_layers

    for k in range(max_layers):
        base = 1 + k * (n_qubits + 1)
        ub[base:base + n_qubits] = len(GATE_SET) - 1
        ub[base + n_qubits] = len(ENTANGLERS) - 1

    return lb, ub


def _as_int(value, low, high):
    """先 round 再 clip，避免浮点基因和越界索引。"""
    return int(np.clip(int(round(float(value))), low, high))


def canonicalize(x, n_qubits=4, max_layers=6, min_layers=1):
    """把基因组规范成唯一表示，支持一维或二维输入。

    decode 只读取前 L 层，第 L 层之后的基因是"死基因"：它们随意漂移不会改变
    线路，却会让同一个线路拥有多种基因表示，使 pymoo 的 eliminate_duplicates
    失效，种群里堆积大量等价架构。这里把死基因归零（I + none），并顺手做
    round + clip，保证整数化。
    """
    X = np.asarray(x, dtype=float)
    if X.size == 0:
        return X.astype(int)

    single = X.ndim == 1
    X = np.atleast_2d(X).copy()

    n_layers = np.clip(np.rint(X[:, 0]), min_layers, max_layers).astype(int)
    X[:, 0] = n_layers

    hi_gate = len(GATE_SET) - 1
    hi_ent = len(ENTANGLERS) - 1

    for k in range(max_layers):
        base = 1 + k * (n_qubits + 1)
        X[:, base:base + n_qubits] = np.clip(
            np.rint(X[:, base:base + n_qubits]), 0, hi_gate)
        X[:, base + n_qubits] = np.clip(
            np.rint(X[:, base + n_qubits]), 0, hi_ent)

        dead = n_layers <= k
        if dead.any():
            X[dead, base:base + n_qubits + 1] = 0

    X = X.astype(int)
    return X[0] if single else X


def decode(x, n_qubits=4, max_layers=6, min_layers=1):
    """把 genome 解码成 CircuitSpec。"""
    x = np.asarray(x, dtype=float).ravel()

    n_layers = _as_int(x[0], min_layers, max_layers)

    layers = []
    for k in range(n_layers):
        base = 1 + k * (n_qubits + 1)
        gates = [
            _as_int(x[base + w], 0, len(GATE_SET) - 1)
            for w in range(n_qubits)
        ]
        entangler = _as_int(x[base + n_qubits], 0, len(ENTANGLERS) - 1)
        layers.append(LayerSpec(gates=gates, entangler=entangler))

    signature = "L{}|".format(n_layers) + ";".join(
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
    n_qubits, max_layers = 4, 6
    lb, ub = bounds(1, max_layers, n_qubits)
    print("n_var =", n_var(max_layers, n_qubits))
    print("lb    =", lb.astype(int))
    print("ub    =", ub.astype(int))

    # 构造一个 3 层 genome：每层 RY(2)，纠缠模式 linear(1)
    ry, linear = GATE_SET.index("RY"), ENTANGLERS.index("linear")
    genome = np.zeros(n_var(max_layers, n_qubits))
    genome[0] = 3
    for k in range(max_layers):
        base = 1 + k * (n_qubits + 1)
        genome[base:base + n_qubits] = ry
        genome[base + n_qubits] = linear

    spec = decode(genome, n_qubits, max_layers)
    print("\nn_layers =", spec.n_layers)
    for i, layer in enumerate(spec.layers):
        print("  layer {}: gates={} entangler={}".format(
            i,
            [GATE_SET[g] for g in layer.gates],
            ENTANGLERS[layer.entangler],
        ))
    print("signature =", spec.signature)

    # 浮点 + 越界基因必须被正确吸收
    messy = genome + 0.4
    messy[0] = 2.6
    messy[1] = -3.7
    messy[2] = 99.0
    spec2 = decode(messy, n_qubits, max_layers)
    assert spec2.n_layers == 3
    assert GATE_SET[spec2.layers[0].gates[0]] == "I"
    assert GATE_SET[spec2.layers[0].gates[1]] == "H"
    print("\nmessy genome decoded safely:", spec2.signature[:40], "...")

    assert spec.signature == decode(genome, n_qubits, max_layers).signature
    print("decode 自检通过")
