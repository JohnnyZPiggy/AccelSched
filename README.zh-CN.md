# AccelSched

[English](README.md) | [简体中文](README.zh-CN.md)

**异构 GPU/CPU 资源放置与算力容量优化引擎。**

AccelSched 是一个 Python 优化工具，面向异构计算节点的任务放置与容量决策。项目整合六种调度方法、事件驱动模拟、LP/MIP 求解、可认证求解器接口及资源规划报告。

![MIT](https://img.shields.io/badge/license-MIT-0f766e) ![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab) ![Version](https://img.shields.io/badge/version-1.0.0-0f766e) ![NumPy/SciPy](https://img.shields.io/badge/core-NumPy%20%2B%20SciPy-334155)

**快速导航：** [调度算法](#调度方法) · [快速开始](#安装与快速开始) · [容量规划](#容量规划) · [实验结果](#性能与验证) · [软件设计说明](docs/release/software-design.zh-CN.md)

## 核心能力

- **异构资源放置：** 统一建模整卡 GPU 数量、CPU 核数、主机内存与 GPU 家族兼容性。
- **六种调度方法：** First-Fit、Best-Fit、LP 引导的启发式、PGFP 与 HiGHS MIP。
- **动态调度模拟：** 任务到达、执行时长、完成释放与接纳率分析。
- **算力容量规划：** 比较现有集群和新增节点方案，输出静态及时间窗口报告。
- **优化求解器：** CPU HiGHS 与可选的 cuPDLPx、H-PDLP GPU LP 后端。
- **可验证优化：** 物理资源可行性检查、求解器状态、原模型数值认证与可复现实验。

## 优化架构

```text
节点资源 + 任务需求 + 硬件兼容性
                 │
                 ▼
         资源约束与输入校验
                 │
       ┌─────────┴──────────┐
       ▼                    ▼
    静态资源放置         动态事件模拟
       │                    │
       ├─ Greedy / PGFP      ├─ 任务到达
       ├─ LP 引导策略        ├─ 完成事件
       └─ HiGHS MIP         └─ 资源释放
       │                    │
       └─────────┬──────────┘
                 ▼
          独立物理可行性校验
                 ▼
         放置决策 + 资源利用率
                 ▼
             容量分析报告
```

可选 Solver Portfolio 支持 GPU LP 求解与 CPU 回退。核心安装只需要 NumPy 和 SciPy，GPU 求解器作为可选加速组件。

## 安装与快速开始

要求 Python 3.10+。

```bash
git clone https://github.com/JohnnyZPiggy/AccelSched.git
cd AccelSched
python -m pip install .
```

使用同一异构算力实例比较六种静态调度方法：

```python
from workload import heterogeneous_example, evaluate

results = evaluate(heterogeneous_example(seed=11), seed=11, time_limit=2.0)
for row in results["rows"]:
    print(row["method"], row["status"], row["admitted_count"])
```

底层 [`Instance` / `solve` 接口](accelsched.py) 可以输出任务放置、目标值、剩余资源、求解状态与运行耗时。

### 动态事件调度

```python
from rolling import Job, simulate

jobs = [
    Job("A", 0, 2, (1, 4, 16)),
    Job("B", 1, 3, (1, 4, 16)),
    Job("C", 2, 2, (1, 4, 16)),
]

report = simulate([[1, 8, 32]], jobs, [[True, True, True]], method="first_fit")
for decision in report["decisions"]:
    print(decision["job_id"], decision["status"], decision["node"])
```

同一时刻先释放已完成任务的资源，再处理新任务到达。

## 调度方法

| 方法 | 核心策略 |
| --- | --- |
| `first_fit` | 选择第一个满足资格与容量的节点 |
| `classic_best_fit` | 选择剩余资源较小的可行节点 |
| `lp_order_best_fit` | LP 决定任务优先级，结合全局 Best-Fit |
| `lp_best_fit` | LP 引导排序与 Shape-aware Best-Fit |
| `pgfp` | 使用影子价格与未来容量评分进行放置 |
| `highs_mip` | 物理节点混合整数优化 |

资源向量顺序为 `[GPU 数量, CPU 核数, 主机 RAM GiB]`。Shape 根据容量聚类，GPU 家族兼容性由 Eligibility 指定。最优性声明以经过验证的求解器证书和数值容差为依据。

## 容量规划

仓库附带合成示例，可直接运行静态与动态容量分析：

```bash
python -m examples.capacity_whatif --input examples/data/static.json
python -m examples.time_window_capacity --input examples/data/dynamic.json
python -m examples.capacity_decision_report --input examples/data/static.json --mode static --format json
python -m examples.capacity_decision_report --input examples/data/dynamic.json --mode dynamic --format json
```

- **P0 · 静态容量比较：** 比较基线与新增算力节点方案。
- **P1 · 时间窗口容量：** 分析接纳、完成与资源时间利用率。
- **P2 · 决策报告：** 生成可比较场景的文本或 JSON 结果。

更多说明：[输入格式](docs/examples/planner-input.md) · [静态规划](docs/examples/capacity-whatif.md) · [动态规划](docs/examples/time-window-capacity.md) · [容量报告](docs/examples/capacity-decision-report.md)。

## 性能与验证

项目保留了物理可行性、数值认证及调度性能实验。

| 研究场景 | 实测结果 | 证据 |
| --- | --- | --- |
| Alibaba PAI 派生的静态反事实实验 | 60/60 放置结果通过独立物理校验 | [实验记录](https://github.com/JohnnyZPiggy/AccelSched/issues/31) |
| 合成动态调度 | 54/54 模拟通过审计；313/313 MIP 到达批次在容差内达到最优 | [动态 Benchmark](docs/research/dynamic-scheduling-benchmark.md) |
| GPU LP + CPU 放置 | L4 上 1K 工作负载暖运行端到端约 0.738 秒，优于所测 GPU 常驻 Greedy 版本 | [P6-G0 研究](https://github.com/JohnnyZPiggy/AccelSched/issues/99) |
| 发行验收 | 记录环境中的 264 项 CPU 聚焦测试通过 | [验收清单](docs/release/validation.md) |

实验数据均对应明确的工作负载与测量范围；L4 数据来自特定同机实验，并不代表所有规模的性能表现。

## 应用场景

| 场景 | 可实现的功能 |
| --- | --- |
| AI 算力集群 | 比较异构节点上的资源放置策略 |
| 算力扩容 | 评估新增 GPU 节点带来的容量变化 |
| 批任务调度 | 研究任务到达与完成过程中的资源利用 |
| 优化算法研究 | 比较 LP 引导策略、MIP 与认证求解器 |
| 基础设施规划 | 输出静态和动态容量分析报告 |

## 项目文档

- [中文软件设计说明书](docs/release/software-design.zh-CN.md)
- [JSON 输入格式](docs/examples/planner-input.md)
- [发行说明](RELEASE_NOTES.md)
- [测试与验证记录](docs/release/validation.md)
- [软件登记准备清单](docs/release/registration-checklist.zh-CN.md)

运行核心 CPU 测试：

```bash
python -m pip install pytest
python -m pytest -q test_m0_oracle.py test_workload.py test_rolling.py test_capacity_whatif.py test_time_window_capacity.py test_capacity_decision_report.py
```

## 仓库结构与开发

测试实现统一在 	ests/，历史 V3/V3.1 脚本在 enchmarks/historical/；原根目录命令继续作为兼容入口。生产模块、示例、V4 子进程脚本及安装范围均不改变。规范命令、暂缓迁移的高风险入口、完整依赖清单和复现边界见[开发与目录指南](docs/development/repository-layout.md)。

## 开源许可

[MIT License](LICENSE) · Copyright © 2026 Ru Zhang。

项目使用 [SAE（Scientist–Architect–Engineer）协作框架](https://github.com/JohnnyZPiggy/SAE-Framework) 开发。
