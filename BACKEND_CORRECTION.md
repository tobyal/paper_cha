# 隐式后端修正：沿用 P3D-Mesh，只去掉 GSHE

原先新增的 SupportConditionedSDF/Fourier MLP 偏离了用户指定的后端，已从当前执行路径替换。

当前结构：

```text
空间查询 x ─┬─ XY/YZ/XZ 三平面多尺度 Hash 编码 ─┐
           └─ 三维多尺度 Hash Grid ────────────┤ 直接求和得到 32 维特征
                                              ↓
                       位置编码(x) + 空间特征 → 原 8 层 SDF 解码器 → fφ(x)
```

保留原配置：16 级、每级 2 通道、基础分辨率 16、目标分辨率 2048、最大哈希表 2^19；每 1000 步开放一级，上限随支持点数量自动确定。保留 8 频段坐标位置编码、256 隐藏维度、skip 第 4 层、几何初始化及 weight normalization。

去掉的仅是 GridGuidedHybridSpatialEncoder 的网格引导平面加权模块，融合改为 `xy + yz + xz + grid`。三维网格编码不是 GSHE 本身，仍然保留。

`ssr/p3d_sdf.py` 来自旧工程 gsdf_net.py，只调整导入路径并删除 GSHE 实例及其调用；`ssr/p3d_backend/multiscale.py` 保留原 Hash_triplane / Hash_grid 定义；hashencoder 的 CUDA 数值实现沿用原文件，构建目录改为新工程内部，避免相对工作目录依赖。

## 支持如何约束重建

支持生成器保持上一轮实现。生成的 Sθ 是动态监督，不再直接拼入 SDF 网络输入：

- Pull：场查询投影后靠近当前支持，最近邻目标坐标保留到支持网络的梯度。
- Zero surface：在生成的支持坐标上计算 |fφ(Sθ)|，梯度同时更新 θ 和 φ。
- 稀疏观测：零面及覆盖约束；Eikonal、几何先验位移和核点正则保持。

因此范式对应应写为“动态支持监督原 P3D 三平面多尺度隐式场”，不再写“支持作为额外条件特征输入场”。固定空间查询的 fφ(x) 不直接依赖 S；经过联合优化后，场参数 φ 受到 S 的约束。这不等于回到离线生成固定点云的两阶段流程。

## 版本与验证

旧 MLP 的源码快照位于工作区 audit/ssr_before_triplane_correction.tar.gz；旧 outputs/smoke_* 保留为历史。旧的 verification.json / VALIDATION.json / validation_artifacts.zip 不代表当前后端。

当前验证使用 verification_triplane.json、VALIDATION_TRIPLANE.json 和 outputs/triplane_*。独立进程运行原 P3D SDF，将 GSHE 替换为直接求和，比较相同权重在多个渐进编码阶段的场值及空间梯度；另外检查新的联合损失反传和实际网格导出。
