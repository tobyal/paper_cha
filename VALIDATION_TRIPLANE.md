# 当前三平面多尺度后端验证

- 原 P3D SDF（只将 GSHE 换成直接求和）作为独立进程参照：在编码步数 0、2000、7000，场值与空间梯度最大误差均为 0。
- 新后端没有 GSHE 参数；XY、YZ、XZ、3D Grid 四个编码分支均获得非零有限梯度。
- 支持处的零面损失使 335 组支持网络参数张量获得非零梯度；联合目标可正常完成二阶反传，并更新支持网络与 SDF 参数。
- 300 点 airplane 的 joint / frozen / raw / external 四条路径在 GPU 2、3 上完成 4 步短测，第二步快照、最终支持、网格与 checkpoint 正常导出。
- frozen 模式的初始与最终支持完全一致。
- 本地和服务器当前 Python 源文件哈希一致。

机器记录：verification_triplane.json、VALIDATION_TRIPLANE.json。运行文件：outputs/triplane_modes/。

这些检查验证后端恢复与联合训练链路，不表示重建已收敛，也不是与 BSDF/NTPS 的效果比较。external 短测仅验证外部支持接口。
