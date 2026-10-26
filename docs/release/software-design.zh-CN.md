# AccelSched 异构算力资源优化调度软件 V1.0 — 软件设计说明书（登记准备稿）

包版本：1.0.0；展示版本：V1.0。准备日期：2026-10-26，不作为开发完成或首次发表日期。拟著作权人公开署名：ru zhang，据用户确认；身份及最终登记口径由申请人核验。本文件不是登记申请表、受理证明或法律权属结论。

## 1. 目的与范围

软件是可独立安装的离线 AIDC 资源优化引擎。给定名义节点库存、不可拆分任务、多资源需求、家族资格及可选到达/持续时间，计算任务到物理节点的放置，或比较新增库存方案。输出可用于研究、配置分析和可解释容量讨论，不执行真实集群任务，不接入生产控制器。

V1 的运行范围为 pyproject 列出的十个核心模块及 examples 包。实验性 GPU Probe、历史研究数据、FIFO 研究模块不属于此 wheel 的命令入口。可选认证 LP 后端属于显式 API，不替换默认调度行为；本次不开展 GPU 实验。

## 2. 运行环境与安装

声明 Python>=3.10、NumPy>=1.23、SciPy>=1.9。CPU 即可运行，不需要显卡、CUDA 或外部调度器。构建使用 setuptools>=77；该版本下界匹配 SPDX license-string 支持，不增加新的运行依赖。实际验收只对证据中列出的环境负责，不宣称覆盖全部 Python/操作系统版本。

在审阅后的源码目录执行 `python -m pip install .`；构建候选 wheel 使用 `python -m pip wheel --no-deps --wheel-dir dist .`，安装 `dist/accelsched-1.0.0-py3-none-any.whl`。从仓库外也可导入核心模块并运行 `python -m examples.capacity_whatif` 等既有入口。资源文件可通过 `importlib.resources.files('examples')` 读取。完整命令见双语 README。

## 3. 功能架构与模块

| 层次/模块 | 输入、责任与输出 |
|---|---|
| workload | 构造受验证 Instance，给出合成异构例和六方法对照 |
| m0_oracle | 物理验证、残余资源、收益、future_score、Shape-LP 与原 PGFP 参考 |
| accelsched | 六个静态方法的公开统一接口、输入检查和结果状态 |
| scipy_backend | 物理节点 MIP 与有效 incumbent/最优性状态处理 |
| rolling | 到达/完成事件与逐批立即启动，不排队、不迁移 |
| examples.planner_input | 严格 JSON v1 解析、限额、输入文件 SHA 与来源信息 |
| examples.capacity_whatif | P0 静态 Baseline/新增节点方案比较 |
| examples.time_window_capacity | P1 有限窗口模拟、资源时间与显式压力参考 |
| examples.capacity_decision_report | P2 报告；调用一次规划比较后汇总，不重复求解 |
| lp_contract / lp_certify | 不可变式原始 CSR、原模型数值/终止证书 |
| lp_backends / solver_portfolio | 显式 HiGHS/可选后端、顺序尝试、回退及墙钟记录 |
| certified_pgfp | 可选稀疏 LP 与有界 Shadow；严格顺序差异可拒绝 |

模块间通过 Python 数据结构交换；不增加后台服务、数据库或网络接口。仅用户主动安装依赖和读取指定文件。报告不包含真实设备管理或执行权限。

## 4. 数据结构与约束

资源列固定为整卡 GPU 数、CPU 核、主机 RAM GiB；不是 GPU 显存。任务不可拆分，只能放在一个 eligible 物理节点上。家族资格必须显式给出；Shape 按容量聚合，不等同于 GPU 型号、性能或互换性。资源/收益为有限非负量，按现有 API 验证。

JSON v1 为 UTF-8 对象，字段严格、无未知字段或重复键；拒绝 NaN/Infinity、把 bool 当数值及非法时刻。根字段包括 schema_version=1、mode、nodes、jobs、scenarios、可选 solver。静态输入不含到达/持续时间；动态输入增加 horizon、arrival、duration 和可选 pressure_references。恰有一个名为 Baseline 的零新增方案。

现有限额：文件<=1MiB，基线/方案总节点<=32，任务<=64，方案2–8；每向量三列，GPU<=1024、CPU/RAM<=1,000,000。ID/家族/方案名区分大小写、不规范化。JSON v1 solver 仅 highs_mip，seed 有界，每次 time_limit<=0.5 秒。Python API 的六方法不等于 JSON 入口支持任意方法。细则以 [JSON 指南](../examples/planner-input.md) 和现有验证器为准。

输出包括 placement、接纳数量/收益、残余资源、利用率、错误和证书状态。P0/P1/P2 输出还保留输入文件 SHA、输入顺序、求解参数和可取得的 Git 信息；仓库外 Git 信息可为空，不能伪称 wheel 对应一个未核对的源码提交。求解计时和等价最优解选择可以不同，不能要求 JSON 字节跨运行完全相同。

## 5. 静态算法流程

先验证容量、任务、资格和参数，再按所选既有方法处理。first_fit 按输入顺序找首个可行节点；classic_best_fit 选择余量较小节点；lp_order_best_fit、lp_best_fit 利用 Shape-LP 排序；pgfp 按价值与容量影子价格排序，并结合已有未来适配评分选择节点；highs_mip 直接使用物理节点整数模型。

每次提交必须满足原资格和容量约束。Shape 聚合 LP 忽略部分逐节点碎片与资格细节，不能替代物理认证。未来评分是给定参考类型的适配分数，不是观测收益或训练出的预测模型。可选认证 LP 的 OPTIMAL 必须经原模型复核；不同最优对偶价格可改变顺序，严格 Shadow 不得把同目标值误当作相同物理策略。

## 6. Rolling 与容量报告流程

Rolling 以时间排序处理完成和到达；同一时刻先释放已完成任务，再把同时到达任务按输入顺序作为一批求解。接纳立即启动，拒绝永久记录；没有自动重试、等待队列、迁移或抢占。solver_error 与正常 not_scheduled 区分。滚动未来为中性，不凭空推断未来负载。

P0 对同一静态任务集比较 Baseline 与各新增库存。P1 对同一任务流做各方案的有限窗口模拟，按明确窗口计算资源时间积分、完成数和压力。P2 将一次比较结果汇总成文本/JSON。新增节点是输入中的假设方案，不是历史实际空闲资源。报告 leader 仅指接纳数较大，不等于成本最优采购或 ROI。

## 7. 状态、验证与错误

静态 optimal 只在有效物理 MIP 解及容差内最优性证据支持时使用。feasible_uncertified 为可行但未证明最优的 incumbent，uncertified 无有效 incumbent；启发式 feasible 不代表最优。其他状态包括 input_invalid、invalid、failed。LP 认证 API 另有 NUMERIC_UNCERTIFIED、BACKEND_UNAVAILABLE 等明示状态，不递归回退。

JSON 输入错误在任何求解前报告 `INPUT_INVALID: 字段: 原因`，退出2；规划/报告失败与正常拒绝分开。Rolling MIP 证书仅涉及当次到达批，不证明全时间范围最优。外部验证同时检查资格、逐节点多资源容量、时间释放顺序和 finite JSON，不将原生求解器状态当作独立证明。

## 8. 操作示例与验收

随包 static.json 是 T4 名义单节点、两个同类请求和一个缺失 P100 家族请求；动态文件在窗口5内给出完成前释放的 A/B/C 例。Baseline/Add T4 的静态接纳数应1/2、动态应2/3；MIP 等价解的任务身份可变化，计数不变。所有数据为合成名义资源，非真实 trace 或厂商规格。

实际测试环境、命令、输出及哈希见 [验证清单](validation.md) 与 Issue #101 对应 Draft PR 的工程证据。干净 wheel 安装在仓库外运行，避免源码路径掩盖缺包；测试只覆盖既有 CPU 功能与发行缺口，不运行 GPU 或大型研究实验。不制作虚假界面截图：本软件是 API/CLI，真实终端输出可以作为操作证据。

## 9. 版权与材料冻结

MIT 授权正文保持不变；版权署名 ru zhang 按用户声明更新，EPFL 关联据本人声明为无。Git 作者昵称、账户控制、AI 协助或某次测试通过，均不能替代身份及完整权属证明。第三方依赖许可证与项目权属分别核查；不把 NumPy/SciPy、求解器源码、第三方 trace 或个人证件夹入程序鉴别材料。

正式提交前冻结经 Architect 审核的 commit/tree、版本、软件全称、文件范围及每文件 SHA，按[交存方案](source-deposit.zh-CN.md) 导出。生成工具仅生成可追溯草稿，不填登记号，不替申请人签署、提交或声称登记已获准。
