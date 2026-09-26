# 三个适应诊断的实现验证

CUDA 单元检查 `scripts/verify_adaptation.py` 通过：

- Delayed：短测 warm-up=2，第 1、2 步支持完全固定，第 3、4 步支持梯度非零；场 Adam 的步数连续，没有切换时重置。解冻时只恢复原本可训练的参数，固定核点和查询位置仍不可训练。
- Decoder-only：仅 `skip_mlp + disp_mlp` 可训练，共 49,667 个参数。编码、核点表示和查询特征漂移均为 0；冻结状态变化为 0，解码参数确有更新。
- Residual：完整先验状态变化为 0，残差有非零梯度和更新；支持坐标与回填后的 patch 对应位置一致，patch 排斥路径可回传到残差；原 prior 项与残差平方位移项数值一致。
- 所有测试反传有限；各自不该更新的状态均未改变。

三个真实 1024 点、2048 支持的 4 步端到端任务全部完成，初始支持与场通过旧 Frozen checkpoint 核对。支持、网格、checkpoint 和离线指标正常输出，自动报告成功读取旧 Frozen / Joint 参照，生成五个配置的三组比较。

机器验证记录：`verification_adaptation.json`。短测：`outputs/adaptation_diagnostics_1024/smoke/`。短测只验证执行语义，不代替 6000 步正式效果实验。
