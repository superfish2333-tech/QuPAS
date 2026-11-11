"""H2 分子哈密顿量（STO-3G 基组，键长 0.735 Angstrom，Jordan-Wigner 变换）。

硬编码 Pauli 项，避免引入 openfermion / pyscf 这类重依赖。
参考能量由 numpy 对角化得到，用作 VQE 的误差基准。

关键：四个四体项（XXYY / YYXX / XYYX / YXXY）的符号只要错一个，
基态能量就会偏离约 0.02 Ha —— 恰好是整个搜索信号的量级。
因此 build_hamiltonian() 之后必须用 exact_energy() 断言，见 __main__。
"""
import numpy as np

N_QUBITS = 4

# (系数, Pauli 串)；"I" 表示全局单位算符
H2_PAULI_TERMS = [
    (-0.09706627, "I"),
    (0.17141283, "Z0"),
    (0.17141283, "Z1"),
    (-0.22343154, "Z2"),
    (-0.22343154, "Z3"),
    (0.16868898, "Z0 Z1"),
    (0.17441288, "Z2 Z3"),
    (0.12062523, "Z0 Z2"),
    (0.12062523, "Z1 Z3"),
    (0.16592785, "Z0 Z3"),
    (0.16592785, "Z1 Z2"),
    (0.04530262, "X0 X1 Y2 Y3"),
    (0.04530262, "Y0 Y1 X2 X3"),
    (-0.04530262, "X0 Y1 Y2 X3"),
    (-0.04530262, "Y0 X1 X2 Y3"),
]

# 精确参考值（numpy eigvalsh 实算）
E_EXACT = -1.1372838344
E_HF = -1.11675931

# Hartree-Fock 参考态 |1100>（qubit0、qubit1 占据）
HF_STATE = [1, 1, 0, 0]

_PAULI_MATRICES = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _term_matrix(term):
    """把 "X0 X1 Y2 Y3" 这样的 Pauli 串转成 16x16 矩阵。

    kron 顺序为 qubit0 * qubit1 * qubit2 * qubit3，
    即 qubit0 是最高位，与 HF_STATE 的下标计算保持一致。
    """
    if term.strip() == "I":
        return np.eye(2 ** N_QUBITS, dtype=complex)

    ops = ["I"] * N_QUBITS
    for token in term.split():
        op, qubit = token[0], int(token[1:])
        ops[qubit] = op

    matrix = np.array([[1]], dtype=complex)
    for op in ops:
        matrix = np.kron(matrix, _PAULI_MATRICES[op])
    return matrix


def build_hamiltonian():
    """返回 16x16 的稠密哈密顿量矩阵。"""
    ham = np.zeros((2 ** N_QUBITS, 2 ** N_QUBITS), dtype=complex)
    for coeff, term in H2_PAULI_TERMS:
        ham += coeff * _term_matrix(term)
    return ham


def exact_energy():
    """精确基态能量（对角化的最小本征值）。"""
    return float(np.linalg.eigvalsh(build_hamiltonian())[0].real)


def hf_energy():
    """Hartree-Fock 参考态 |1100> 的能量期望值。"""
    ham = build_hamiltonian()
    index = 0
    for bit in HF_STATE:
        index = index * 2 + bit
    return float(ham[index, index].real)


def _pauli_operator(term):
    """把 "X0 X1 Y2 Y3" 这样的 Pauli 串转成 PennyLane 可观测量。"""
    import pennylane as qml

    matrices = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}

    if term.strip() == "I":
        return qml.Identity(0)

    factors = [matrices[token[0]](int(token[1:])) for token in term.split()]
    if not factors:
        return qml.Identity(0)
    if len(factors) == 1:
        return factors[0]
    return qml.prod(*factors)


def to_pennylane():
    """转成 PennyLane 可观测量，供 qml.expval 使用。"""
    import pennylane as qml

    coeffs, ops = [], []
    for coeff, term in H2_PAULI_TERMS:
        coeffs.append(coeff)
        ops.append(_pauli_operator(term))

    try:
        # 0.40+ 推荐 qml.dot，qml.Hamiltonian 已弃用
        return qml.dot(coeffs, ops)
    except Exception:
        return qml.Hamiltonian(coeffs, ops)


def check(verbose=True):
    """校验哈密顿量系数是否抄对，返回 (e_exact, e_hf)。"""
    e_exact = exact_energy()
    e_hf = hf_energy()

    if verbose:
        print("H2 (STO-3G, 0.735 A)")
        print("  exact ground state : {:.10f} Ha".format(e_exact))
        print("  Hartree-Fock       : {:.10f} Ha".format(e_hf))
        print("  correlation energy : {:.10f} Ha".format(e_hf - e_exact))

    assert abs(e_exact - E_EXACT) < 1e-6, (
        "基态能量偏离参考值 {:.10f}，实际 {:.10f}。"
        "四体项符号很可能抄错了。".format(E_EXACT, e_exact)
    )
    assert abs(e_hf - E_HF) < 1e-6, (
        "HF 能量偏离参考值 {:.10f}，实际 {:.10f}。"
        "对角项系数或 HF 态下标有问题。".format(E_HF, e_hf)
    )

    if verbose:
        print("  assertions passed")
    return e_exact, e_hf


if __name__ == "__main__":
    check()
