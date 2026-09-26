# 三个适应诊断实验：时机、范围、优化对象

状态：三个配置已实现，梯度/冻结范围检查及 4 步端到端短测通过。正式运行与结果位于 `outputs/adaptation_diagnostics_1024/run/`，具体进度以该目录 `status.json` 为准。当前优先诊断为什么 Frozen 的最终结果优于 Full Joint，不扩展网络模块。

## 共同起点

先使用已经完成首轮实验的同一个 airplane 和同一份 1024 点输入。完整先验仍指 Encoding + 中间表示与查询网络 + Decode。

复用 `outputs/evidence_1024/pilot/` 的 Frozen、Joint 两组作共享参照；只新增 DelayedJoint、DecoderOnly、ResidualSupport 三次运行。三组均继承相同预训练权重、初始支持、FPS 索引、场初始化与训练设置。新配置的初始支持和场状态必须与参照相符，不能重新采样输入。

| 项目 | 固定设置 |
|---|---|
| 输入 | 已有 airplane 的 1024 个观测 |
| 支持 | 2048 点，固定初始 FPS 索引，后续不重新选择 |
| 后端 | 原多尺度三平面 + 3D Hash Grid，无 GSHE |
| 初始化 | 支持使用同一完整官方预训练权重；实验种子 21、场初始化种子 22 |
| 优化 | Adam；field lr 1e-3，适应变量 lr 1e-5；分别裁剪两分支梯度 |
| 预算 | 每次 6000 步，不按 GT 选停止点，不开训练损失早停 |
| 场配置 | 512 个训练查询，六级编码，每 1000 步按原逻辑开启一级 |
| 导出 | 网格分辨率 128；初始状态及第 1500、3000、4500、6000 步 |
| 评价 | 复用原固定 GT 采样、归一化尺度和观测距离四分区 |

参照终点（均为归一化距离，越小越好）：

| 参照 | 支持 CD | 支持 GapCoverage | 网格 CD | 网格 GapCoverage |
|---|---:|---:|---:|---:|
| Frozen | 0.009543 | 0.017366 | 0.009760 | 0.008495 |
| Full Joint | 0.009570 | 0.020607 | 0.010954 | 0.011521 |

## A. Delayed Field-to-Support Feedback

问题：让场先适应高质量支持，再开放完整支持网络的更新，是否比从第一步联合更新更稳定？

- 第 1–3000 步：整个支持网络冻结，S=S0，只更新场。
- 第 3001–6000 步：解冻完整支持网络，采用原 Joint 目标与学习率，同时更新支持网络和场。
- 第 3001 步启用支持更新时，保留场权重、Adam 状态和全局编码步数；不重新初始化场，不重置渐进编码，不改变 S0 先验参照。
- warm-up 期间支持不受任何损失更新；不是先运行 One-way。

比较：Frozen / FullJoint / DelayedJoint。

重点：终点网格 CD 与 GapCoverage、支持 Coverage 与 GapCoverage、支持位移曲线。3000 步前支持位移必须为零；记录每步反馈开关和可训练参数数目，以确认实际启动边界。

解释：优于 FullJoint 表明 warm-up 策略减轻了适应损伤；进一步优于 Frozen 才表明该延迟适应带来额外重建收益。该配置也减少了支持更新次数，因此不能仅凭一次结果认定早期场不成熟是唯一原因。首轮不额外添加 warm-up 长度扫描。

## B. Decoder-only Adaptation

问题：保持预训练几何表示固定，只让几何解码适应当前形状，是否比修改整个先验更有效？

- 冻结 3.2：Encoder、REM，以及对应几何状态与归一化统计。
- 冻结 3.3：KGM、projector、attention。
- 只更新 3.4 的 `skip_mlp + disp_mlp`，即查询/锚点特征融合与偏移预测；保留原预训练权重，不新建随机头。
- 场正常更新，从第一步允许 pull 和零水平面损失反馈到上述解码参数。
- 其他目标、学习率、总步数与 Full Joint 相同。当前网络保持原来的 eval BN 统计策略。

注意：上游名为 `backbone.decoder` 的对象还包含 REM、KGM、attention，不能直接整体解冻；本实验的 Decoder 指 Method 3.4 的实际解码边界。

比较：Frozen / FullJoint / DecoderOnly。

重点：支持/网格质量、位移，以及冻结部分的参数和中间表示是否保持不变。记录锚点特征、核点表示、查询读取特征的漂移，预期为零；解码参数应有非零更新。

解释：优于 FullJoint、接近 Frozen 表明限制更新范围有帮助；超过 Frozen 才支持“固定表示 + 自适应解码”的最终设计。不另加一个 head-only 变体，以保持本轮只有三个新增配置。

## C. Explicit Support Residual Adaptation

问题：是否应该保持整个先验网络不变，只对它产生的显式几何做当前形状的修正？

先生成并固定 S0，为最终选定的 2048 个支持点分别建立三维残差：

`S = S0 + deltaS`，`deltaS_initial = 0`。

- 整个 Encoding / 表示 / 查询 / Decode 网络冻结。
- 只优化 2048×3 个残差变量与场参数，从第一步启用反馈。
- 残差定义在整形状归一化坐标中；不能混用 patch 坐标与输出世界坐标。
- 初始支持索引固定，不在迭代过程中重做 FPS。
- 不新增网络、confidence、attention 或其他模块。

目标沿用当前损失体系，保留原权重：pull、支持零面、观测贴面/覆盖、Eikonal、原有排斥。网络内部核点正则在本配置中为常量。

当前代码已经有 `0.2 * mean(||S-S0||^2)` 的 prior 项。此处它恰好等于 `0.2 * mean(||deltaS||^2)`，直接作为残差位移正则，不再叠加第二个 L_res。这是软位移惩罚，不是带固定半径的硬约束。

保留初始完整 patch 支持布局，将选中点的残差回填后计算原有排斥项；未选中位置固定。不能只修改 `points`，却让排斥项仍读取完全未更新的 `patch_points`。

首轮残差 lr 也固定为 1e-5，不做学习率扫描。但坐标变量与网络参数的数值尺度不同，相同学习率不代表相同位移；实际位移必须单独报告。

比较：Frozen / FullJoint / DecoderOnly / ResidualSupport。

重点：支持与网格质量、残差平均/P95位移、全部先验权重不变性、残差的非零梯度和更新。若几乎没有产生位移，只能解释为该配置适应很弱，不能据此判定残差适应的潜力。

## 结果输出与判断

每组输出：

1. 终点表：支持 CD、Accuracy、Coverage、GapCoverage；网格 CD、Coverage、GapCoverage、F-score。
2. 轨迹：同一组迭代节点的支持/网格误差、支持平均/P95位移；另在日志中标记 A 的开启时刻。
3. 相同视角下的翼尖、尾翼等局部对比，使用固定评价区域，不按方法分别挑区域。

按 A、B、C 分表，共享两个参照，不复制成更多训练变体。完整结果以 6000 步终点为主，中间节点用于诊断，不事后挑 GT 最佳步数充当最终结果。

先分别看网格收益和支持收益。如果网格改善而支持覆盖退化，可以说明该适应策略有利于当前后端重建，但不能写成“支持本身更准确且更完整”。

三个新增配置可在三张 GPU 上并行：逻辑按 A→B→C 解释，实验设置不依赖前一组胜负，不将 A+B 或 A+C 混成新的主配置。单个 airplane 完成后，再将有价值的候选与 Frozen / FullJoint 扩展到三个代表形状。

若三个新增配置均未超过 Frozen，当前版本优先采用稳定的冻结先验；这个判断限于已测目标、预算与形状，不推广为任何 test-time adaptation 都无效。

## 执行入口

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/verify_adaptation.py
python scripts/make_diagnostic_jobs.py \
  --baseline-root outputs/evidence_1024/pilot \
  --output outputs/adaptation_diagnostics_1024/jobs.json
python launch.py --jobs outputs/adaptation_diagnostics_1024/jobs.json \
  --gpus 1 2 3 --output outputs/adaptation_diagnostics_1024/run
```

Python 使用服务器 `lkx_papera` 环境；输出目录不得覆盖既有运行。作业生成器直接继承已完成 Frozen 的实际参数，并让每个新任务核对该参照的 step-zero checkpoint：观测与支持索引完全相同、支持坐标误差小于 1e-6、场参数逐项完全相同。

更新范围由 `ssr/adaptation.py` 管理；`reconstruct.py` 保存残差、S0、阶段标记和冻结状态审计。调度器完成后自动离线评价，并由 `scripts/summarize_diagnostics.py` 分 A/B/C 生成三组表格、PNG/PDF 曲线、CSV 和 JSON；共享参照直接读取已有结果，不重新训练。
