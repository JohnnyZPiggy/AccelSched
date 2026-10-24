# AccelSched

[English](README.md)

轻量 Python GPU/CPU 资源放置优化工具：在节点容量和作业资格约束下，将不可拆分作业放到单一节点，并对比启发式方法与物理节点级 HiGHS 混合整数优化基线。

## 安装

需要 Python 3.10+、NumPy 和 SciPy，无需 GPU 或外部调度器。

```bash
git clone https://github.com/JohnnyZPiggy/AccelSched.git
cd AccelSched
python -m pip install .
```

## 快速开始

下面是三资源的**合成示例**：

```python
from workload import heterogeneous_example, evaluate

report = evaluate(heterogeneous_example(seed=11), seed=11, time_limit=2.0)
for row in report["rows"]:
    print(row["method"], row["status"], row["admitted_count"])
```

[`workload.py`](workload.py) 的 `build_workload` 接受按顺序排列的整卡 GPU 数、CPU 核数和主机内存 GiB 向量；`evaluate` 在同一实例上运行六种方法，重建接纳数量和资源利用率。GPU 型号资格通过 `eligible` 显式给定，`shapes` 只归并相同容量向量。Trace 解析由外部处理；底层 `Instance` / `solve` 用法见[英文示例](README.md#quick-start)。

## 六种方法

| 方法 | 策略 |
| --- | --- |
| `first_fit` | 按输入顺序选择首个可行节点 |
| `classic_best_fit` | 按输入顺序选择剩余资源总和最小的可行节点 |
| `lp_order_best_fit` | Shape-LP 作业优先级与全局 Best-Fit |
| `lp_best_fit` | Shape-LP 优先级与按容量形状选择的 Best-Fit |
| `pgfp` | Shape-LP 优先级与未来容量感知放置 |
| `highs_mip` | 物理节点级 HiGHS 混合整数优化 |

`optimal` 表示在求解器数值容差内认证最优；启发式的 `feasible` **不是最优证书**。`feasible_uncertified` 表示有效 MIP 可行解但无最优证明，`uncertified` 表示没有有效可行解；其他状态包括 `input_invalid`、`invalid` 和 `failed`。`solve(..., time_limit=10.0)` 设置 HiGHS 时间上限。

## 验证和范围

仓库根目录的 CPU 回归命令见[英文说明](README.md#validation-and-scope)。本工具对批次中每个作业进行单节点放置，提供三资源合成工作负载及下述 trace 来源反事实研究证据。它不是生产 Kubernetes 调度器；拓扑、作业生命周期、迁移、实时集群集成和 GPU 加速均不在范围内。

### Trace 来源反事实基准

[Issue #31 已验收研究](https://github.com/JohnnyZPiggy/AccelSched/issues/31#issuecomment-6377137390) 使用 Alibaba PAI GPU 2020 数据中的任务和名义容量，构建空集群静态反事实批次，标记为 `TRACE_DERIVED_ASSUMED`。整卡 GPU、单任务单实例作业样本覆盖 T4/P100/V100/V100M32 显式请求类型、错误型号节点，以及单独标记为 `SYNTHETIC_CONTROL` 的 RAM 不足负对照。Engineer 报告 **60/60 放置结果独立验证可行**，**10/10 物理节点级 HiGHS MIP 返回 `optimal`**，最优状态受求解器容差约束。可行结果包括正确拒绝放置的负对照。

RAM 采用两种明确的 **ASSUMED** 解释，并同时作用于容量和需求：原始值已是 GiB，或原始值为十进制 GB、乘 `10**9/2**30` 转成 GiB。`value=1` 是接纳数量代理，不是经济收益。样本按型号刻意分层，并筛选为可被单个节点容纳，存在选择偏差；这些静态结果不代表总体接纳率、阿里巴巴历史调度或生产性能。

源数据：[Alibaba PAI 2020 trace](https://github.com/alibaba/clusterdata/tree/3858992ca41bf92082e6d6522ddf0bebd14e838b/cluster-trace-gpu-v2020)，遵循 [CC BY 4.0](https://github.com/alibaba/clusterdata/blob/3858992ca41bf92082e6d6522ddf0bebd14e838b/cluster-trace-gpu-v2020/LICENSE)；论文归属：[Weng 等，NSDI 2022](https://www.usenix.org/conference/nsdi22/presentation/weng)。源数据许可证与 AccelSched 的 [MIT 代码许可证](LICENSE) 相互独立。
