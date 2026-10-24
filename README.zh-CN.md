# AccelSched

[English](README.md) · [静态放置](#静态放置) · [滚动批次](#滚动批次) · [基准研究](#基准研究)

在 CPU 上运行的 Python **异构 GPU/CPU/RAM 放置优化与事件调度模拟工具**。

在多资源容量约束下，将每个不可拆分作业放到一个符合资格的物理节点。可用六种方法比较静态批次，也可按作业持续时间和完成事件模拟立即启动的滚动批次。GPU 型号兼容性显式指定；本项目是研究工具，不是实时集群编排器。

## 安装

需要 Python 3.10+、NumPy 和 SciPy，无需 GPU 或外部调度器。

```bash
git clone https://github.com/JohnnyZPiggy/AccelSched.git
cd AccelSched
python -m pip install .
```

## 静态放置

资源向量依次为**整卡 GPU 数、CPU 核数、主机内存 GiB**。下面在同一个合成异构工作负载上比较六种方法：

```python
from workload import heterogeneous_example, evaluate

report = evaluate(heterogeneous_example(seed=11), seed=11, time_limit=2.0)
for row in report["rows"]:
    print(row["method"], row["status"], row["admitted_count"])
```

[`workload.py`](workload.py) 的 `build_workload` 接受自定义容量、需求、价值和固定的节点/作业 `eligible` 数组，`evaluate` 比较接纳数量与资源利用率。GPU 型号资格通过 `eligible` 表达；`shapes` 归并相同容量向量，不代表 GPU 型号。底层 [`Instance` / `solve` API](accelsched.py) 返回放置、接纳价值、剩余资源、耗时和状态。Trace 解析由外部处理。

## 滚动批次

[`rolling.Job` / `simulate`](rolling.py) 复用相同资源单位和六种方法。下面是单节点合成示例：

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

输出依次为 `A accepted 0`、`B not_scheduled None`、`C accepted 0`。到达时间为非负整数，持续时间为正整数，使用任意时间单位；接纳后立即启动。同一时间**先释放完成作业的资源**，再将同时到达的作业按输入顺序组成一个批次求解。资格固定；拒绝是永久决定，不排队、不重试、不迁移、不抢占。`solver_error` 与正常的 `not_scheduled` 拒绝分开记录。

## 六种方法

| 方法 | 策略 |
| --- | --- |
| `first_fit` | 按输入顺序选择首个可行节点 |
| `classic_best_fit` | 按输入顺序选择剩余资源总和最小的可行节点 |
| `lp_order_best_fit` | Shape-LP 作业优先级与全局 Best-Fit |
| `lp_best_fit` | Shape-LP 优先级与按容量形状选择的 Best-Fit |
| `pgfp` | Shape-LP 优先级与未来容量感知放置 |
| `highs_mip` | 物理节点级 HiGHS 混合整数优化 |

`optimal` 表示经过验证的 MIP 解**在求解器数值容差内认证最优**；`feasible_uncertified` 是有效可行解但无最优证明，`uncertified` 表示没有有效可行解。启发式的 `feasible` 不是最优证书。其他状态包括 `input_invalid`、`invalid` 和 `failed`；`time_limit` 参数设置 HiGHS 求解时间上限。

滚动模拟使用**中性的未来类型**，不从输入推断预测，因此不能据此宣称 PGFP 的预测收益。滚动 MIP 证书仅适用于当前到达批次，不证明整个时间范围的调度最优。

## 基准研究

### Alibaba Trace 来源静态反事实

[Issue #31 已验收研究](https://github.com/JohnnyZPiggy/AccelSched/issues/31#issuecomment-6377137390) 使用 Alibaba PAI GPU 2020 任务构造**带 ASSUMED 假设的空集群反事实**（`TRACE_DERIVED_ASSUMED`）。Engineer 报告 **60/60 放置结果独立验证可行**，**10/10 物理节点级 HiGHS MIP 返回 `optimal`**，受求解器容差约束。整卡 GPU、单任务单实例作业样本覆盖 T4/P100/V100/V100M32 显式资格、错误型号节点，以及单独标记为 `SYNTHETIC_CONTROL` 的 RAM 不足负对照；可行结果包括正确拒绝放置的对照。

RAM 对容量和需求一致采用两种 **ASSUMED** 解释：原始值已是 GiB，或原始值为十进制 GB、乘 `10**9/2**30` 转为 GiB。单位任务价值（`value=1`）衡量接纳数量，不是经济收益。样本按型号刻意分层并筛选为可被单节点容纳，存在选择偏差；结果不代表总体接纳率、阿里巴巴历史分配或生产性能。

### 有界 SYNTHETIC 动态基准

[Issue #36 研究报告](docs/research/dynamic-scheduling-benchmark.md) 记录九个固定工作负载：8 个名义容量异构节点，T=40，种子 11/29/47，GPU offered load 为 0.3/0.7/1.1。六种方法 × 九个工作负载得到 **54/54 次独立审计通过的模拟**，`solver_error` 为零，**313/313 个 `highs_mip` 到达批次在求解器容差内返回 `optimal`**。审计覆盖固定窗口 GPU/CPU/RAM 资源时间利用率、事件账本、资格和按时间加权的可放置性。

保留了低负载下接纳数量相同的结果；较高负载有时出现累计接纳差异。单批最优 MIP 不一定在整个时间范围内最好。此研究为 **SYNTHETIC**，不代表历史调度、加速比或生产性能。详细原始 JSON 仅保留在本地，报告提供路径和 SHA 索引，并未发布到 GitHub。

## 验证和范围

安装 pytest 后，可在检出目录运行 focused CPU 测试：

```bash
python -m pytest -q test_rolling.py test_dynamic_bench.py
```

参见[滚动模拟测试](test_rolling.py)和[动态指标独立 oracle](test_dynamic_bench.py)。模拟已支持作业持续时间及完成事件；不建模多节点/gang 放置、拓扑、设备显存、运行性能或实时 Kubernetes 集成。GPU 执行、排队、迁移和抢占不在范围内。

## 归属与许可证

[Alibaba PAI 2020 trace](https://github.com/alibaba/clusterdata/tree/3858992ca41bf92082e6d6522ddf0bebd14e838b/cluster-trace-gpu-v2020) 遵循 [CC BY 4.0](https://github.com/alibaba/clusterdata/blob/3858992ca41bf92082e6d6522ddf0bebd14e838b/cluster-trace-gpu-v2020/LICENSE)；论文归属：[Weng 等，NSDI 2022](https://www.usenix.org/conference/nsdi22/presentation/weng)。源数据许可证与 AccelSched 的 [MIT 代码许可证](LICENSE) 相互独立。
