# 1024 点证据链实现验证

本轮新增 One-way、共同后端配置、分区评价、完整先验 checkpoint 和多 GPU 实验汇总。

## 已完成检查

- `scripts/verify_coupling.py`：One-way 中 pull / surface 对支持网络梯度均为 0，对场参数梯度非零；Joint 中两项均能反馈至支持网络。
- One-way 的总目标仍使 Encoding、REM、KGM、Decode 全部分支获得非零有限梯度。Frozen 支持不变。
- 相同网络、输入与查询下，One-way 和 Joint 的前向损失逐项相同，只改变梯度路径。
- `scripts/verify_complete_prior.py`：query / direct 分支经一次参数更新、保存完整网络、重新加载，输出误差均为 0；状态数分别为 481 / 485。错误 decoder/deformation 配置会被拒绝。
- 指标已用完全重合点集（距离 0、F-score 1）和沿垂直方向平移 0.1 的点集（Accuracy / Coverage / GapCoverage 均为 0.1）验证；四个分位区域样本数符合预期。
- 六组真实 1024 点输入的 4 步端到端短测全部通过：Frozen、One-way、Joint、Raw、NTPS 支持、BSDF 支持。每组支持、网格、checkpoint 与 11 项离线产物评价正常完成；汇总脚本已生成表格与图片。

机器记录保存在服务器项目根目录的 `verification_coupling.json`、`verification_complete_prior.json`，短测在 `outputs/evidence_1024/smoke/`。完整先验加载器在候选文件中隔离验证，避免改变正在运行的实验代码。

## 正式首轮

首轮 `outputs/evidence_1024/pilot/` 已按 `EXPERIMENTS.md` 启动，运行状态及结果以该目录的 JSON、receipt 和 REPORT 为准。每组 6000 步、2048 支持（Raw 为 1024）、512 查询、六级渐进编码、128 网格分辨率。GPU 1 / 2 / 3 并行。

历史 NTPS / BSDF 的同一 1024 点输入与 30 分钟支持已核对原始网格哈希和实际消费输入哈希，并保持相同坐标系；没有将 mesh 采样冒充网络中间支持。GT 只传入独立评价程序，重建任务参数不包含 GT。

这些实现检查不代表 Joint 必然优于 Frozen，也不表示完整方法已优于官方 NTPS / BSDF。效果结论来自正式结果，并分别看支持贴合、覆盖及最终网格。
