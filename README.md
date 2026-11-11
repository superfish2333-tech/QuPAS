# QAS：量子线路架构搜索（VQE on H2）

用 NSGA-II 搜索 VQE ansatz 的量子架构搜索（Quantum Architecture Search）项目。
搜索对象是分层门矩阵编码的硬件高效线路，两个最小化目标：

- `f1 = E - E0`：相对 H2 精确基态（`-1.1372838344 Ha`）的能量误差，保留全精度
- `f2`：线路代价（默认门总数，可选 CX 数 / 深度 / 加权代价）

项目完全自包含，不依赖任何 CNN/NAS 代码。

## 目录结构

```
qas/
├── requirements.txt
└── qsearch/
    ├── __init__.py
    ├── h2_hamiltonian.py     # H2 哈密顿量（STO-3G / JW）+ 精确能量与断言
    ├── qencoding.py          # 定长 genome <-> CircuitSpec 编解码
    ├── qcircuit.py           # CircuitSpec -> PennyLane QNode，线路代价统计
    ├── vqe_trainer.py        # 手写 Adam 的 VQE 参数优化与评估缓存
    ├── qproblem.py           # pymoo Problem：双目标评估
    ├── nsga2.py              # 内置 NSGA-II 引擎（锦标赛 + 非支配排序 + 拥挤距离）
    └── evolution_qsearch.py  # 进化搜索 CLI 入口
```

## 环境

Python 3.10+（推荐 3.11），已验证：`pennylane 0.41.1`、`numpy 1.26.4`、`pymoo 0.6.2`。

```bash
conda create -n qas python=3.11 -y
conda activate qas
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 分层自检

按依赖顺序逐个跑，任何一层失败都能直接定位问题：

```bash
python -m qsearch.h2_hamiltonian   # 哈密顿量系数断言（四体项符号）
python -m qsearch.qencoding        # genome 解码 / 死基因归零
python -m qsearch.qcircuit         # 门数解析统计 vs PennyLane 实际统计
python -m qsearch.vqe_trainer      # 参考 ansatz 收敛：|E - E0| < 1e-3 Ha
```

## 运行搜索

冒烟（约几十秒）：

```bash
python -m qsearch.evolution_qsearch \
    --pop_size 4 --n_gens 2 --n_offspring 4 \
    --vqe_steps 50 --max_layers 3 --save SmokeTest
```

常规搜索：

```bash
python -m qsearch.evolution_qsearch \
    --pop_size 20 --n_gens 20 --n_offspring 20 \
    --vqe_steps 150 --vqe_restarts 2 --cost_metric gates
```

输出目录为 `qas-<save>-<timestamp>/`，`log.txt` 中记录每个架构的基因型、
能量误差、门数/CX/深度/参数数，以及每代种群统计和最终 Pareto 前沿。

主要参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--n_qubits` | 4 | H2 固定 4 比特 |
| `--min_layers` / `--max_layers` | 1 / 6 | 线路层数范围（>6 易遇 barren plateau） |
| `--pop_size` / `--n_gens` / `--n_offspring` | 20 / 20 / 20 | 进化规模 |
| `--vqe_steps` / `--vqe_lr` / `--vqe_restarts` | 150 / 0.05 / 2 | VQE 参数优化 |
| `--cost_metric` | gates | `gates` / `cx` / `depth` / `weighted` |
| `--device` / `--diff_method` | default.qubit / backprop | PennyLane 执行后端 |
| `--no_cache` | 关 | 关闭按 signature 的评估缓存 |

## 搜索空间

| 基因段 | 含义 | 取值 |
|---|---|---|
| `gene[0]` | 层数 `L` | `[min_layers, max_layers]` |
| 每层 `n_qubits` 位 | 单比特门 | `I / RX / RY / RZ / H` |
| 每层 1 位 | 纠缠模式 | `none / linear / ring / star / full` |

`n_var = 1 + max_layers * (n_qubits + 1)`（4 比特 6 层为 31）。`L` 之后的
尾部基因在解码时丢弃，并通过 `canonicalize` 归零，保证同一线路只有唯一基因表示。

## 关键约定

- 基因取整必须 `np.clip(int(round(x)), lb, ub)`：先 round 后 clip，兼容 pymoo 的浮点基因。
- `f1` 全程不舍入：可搜索能量区间仅约 `0.02 Ha`，舍入会让 Pareto 前沿塌缩。
- 初态固定为 Hartree-Fock 行列式 `|1100>`。
- `h2_hamiltonian.check()` 会在搜索启动时执行，四体项符号抄错会立即断言失败。
