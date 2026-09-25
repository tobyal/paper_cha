# Method 3.2–3.5 与代码的对应

当前实现可以对应“建立局部表示 → 查询读取 → 三维支持解码 → 支持约束重建”的流程。本次整理只拆分计算职责，不修改网络、初始化、损失或优化策略。

## 3.1 Overview

`P → {R_i} → {z_ij} → S_θ → F_φ(x) → mesh`。

`ssr/support.py::SurfaceSupportLearner.forward` 串联 3.2–3.4；`reconstruct.py` 将支持网络与 3.5 的场网络接入联合优化。箭头 `S_θ → F_φ` 表示支持提供重建约束，不表示 SDF 接收支持特征作为输入。

| 章节 | 主要文件与入口 | 输入 → 输出 | 当前实现组件 |
|---|---|---|---|
| 3.2 Learning Local Geometric Representation Space | `ssr/representation.py::build_local_representation` | patch 内的锚点 → `LocalSurfaceRepresentation` | encoder + REM |
| 3.3 Query-based Surface Sampling in Representation Space | `ssr/query_sampling.py::sample_surface_queries` | 局部表示 → `SurfaceQueries` | KGM + projector + attention |
| 3.4 Feature-space Inverse Mapping to 3D Surface Support | `ssr/support_mapping.py::decode_surface_support` | 查询特征与锚点特征 → 显式支持 | skip fusion + displacement head + anchor residual |
| 3.5 Support-guided Implicit Surface Reconstruction | `ssr/implicit_reconstruction.py` | 支持和观测提供约束 → 优化后的隐式场 | P3D 多尺度三平面 / 3D Hash Grid + SDF + 联合目标 |

## 3.2 Learning Local Geometric Representation Space

### 3.2.1 Anchor-based Geometry Encoding

稀疏观测先经过整形状归一化，再由 `geometry.patch_layout` 构造重叠 patch，并进行 patch 内归一化。每个 patch 内的观测点是锚点 `a_i`，编码得到 `f_i = E(P_patch, a_i)`。同一个原始点可能出现在多个 patch。

### 3.2.2 Adaptive Local Surface Representation

REM 使用邻域几何和特征预测局部核点偏移，形成 `R_i = {(k̃_im, h_im)}`。`LocalSurfaceRepresentation` 保存：

- `anchors`：patch 坐标下的锚点 `[B,3,N]`。
- `anchor_features`：编码特征 `[B,64,N]`。
- `local_features`：REM 聚合后的局部特征 `[B,64,N]`。
- `kernel_offsets`：相对锚点的自适应核点坐标 `[B,3,N,M]`，默认 `M=15`。
- `kernel_features`：核点特征 `[B,128,N,M]`。
- `regularizer`：REM 原有的拟合与排斥正则。

这里的表示同时包含局部几何与学习特征。可称为 **spatially anchored learned geometric representation**，不能把现有代码表述为完全 coordinate-free 或已经建立具有连续曲面保证的纯 latent manifold。

## 3.3 Query-based Surface Sampling in Representation Space

KGM 在局部固定查询位置构造查询特征，随后共享 projector 和 attention 从自适应核点表示中读取表面信息，得到 `z_ij = A(q_j, R_i)`。

`SurfaceQueries.offsets` 是局部查询位置 `[3,r]`，`features` 是查询读取后的特征 `[B,128,N,r]`。默认每个锚点有 `r=4` 个有效查询，维持原权重的结构。

本节的 sampling 指有限查询槽位对表面信息的读取。查询位置仍参与特征构造；代码尚未实现任意密度的连续潜空间采样，也没有移除空间查询先验。论文公式中的 `q_j` 可概括查询槽位，但具体特征还依赖局部观测与 REM 输出。

## 3.4 Feature-space Inverse Mapping to 3D Surface Support

实际解码同时使用查询特征和锚点编码的跳接：

`Δs_ij = tanh(D([z_ij, f_i]))`，`s_ij = a_i + Δs_ij`。

因此，比只写 `D(z_ij)` 更贴近代码的公式是 `D(z_ij, f_i)`。随后撤销 patch 的平移与尺度归一化，得到整形状归一化坐标下的显式支持。世界坐标恢复由输出入口完成。

返回接口保持原样：`points`、`patch_points`、`regularizer`、`kernel_offsets`、`anchor_features`、`kernel_features`。所有中间张量保持梯度。

“Inverse mapping”在这里表示从表示域解码回三维几何的方向；没有一个被证明可逆的前向映射，也没有双射约束。若保留此节标题，正文应明确这一含义；更直接的标题是 **Decoding Local Representations into 3D Surface Support**。

## 3.5 Support-guided Implicit Surface Reconstruction

`MultiScaleTriPlaneSDF` 沿用 P3D 的多尺度 Hash 三平面、3D Hash Grid、位置编码与 SDF 解码器，保持 `xy + yz + xz + grid` 融合，无 GSHE。底层源文件仍在 `ssr/p3d_sdf.py` 和 `ssr/p3d_backend/`，便于追溯来源。

`reconstruction_loss` 与场定义放在同一章节文件，明确支持的两个直接作用：

1. **动态投影目标**：场查询按 `x' = x − F_φ(x) normalize(∇F_φ(x))` 投影，靠近当前支持中的最近邻目标。
2. **零水平面约束**：使 `F_φ(s)` 接近零，并通过支持坐标向生成网络回传梯度。

观测贴面 / 覆盖、Eikonal、初始支持先验、同锚点排斥、核点正则和可选法向约束维持原实现与权重。最近邻索引及训练查询的采样中心不求导，但被选中的目标支持坐标与零面查询保留梯度。

`reconstruct.py` 管理优化、支持预算和快照；`ssr/export.py` 通过 Marching Cubes 提取网格。这里得到的是受支持约束训练的 `F_φ(x)`，不是 `F_φ(x,S)` 的条件场。

## 文件职责与兼容性

```text
ssr/
  support.py                  # 3.1 编排、权重加载、已有消融开关
  representation.py           # 3.2 锚点编码与自适应局部表示
  query_sampling.py           # 3.3 表面查询读取
  support_mapping.py          # 3.4 三维支持解码
  implicit_reconstruction.py  # 3.5 场与支持引导的重建目标
  geometry.py                 # 坐标、patch、FPS、距离与输入输出
  export.py                   # 网格提取
  p3d_sdf.py / p3d_backend/    # 保留来源可核查的三平面后端
  field.py / objective.py     # 旧导入路径的兼容转发
```

RepKPU 源码仍完整保留在 `vendor/repkpu`，参数只注册在 `SurfaceSupportLearner.backbone` 下；拆分函数不重复注册模块。旧 checkpoint 的键、顺序及形状保持一致。

已有消融位置：`--no-deform` 对应 3.2.2；`--decoder direct` 绕过 3.3 并替换 3.4 解码头；`--mode frozen` 关闭 3.5 对支持网络的联合更新；`raw/external` 替换支持来源。没有新增消融机制或训练实验。

上述对应说明现有实现可以支撑这一方法叙述；组件重命名本身不构成新机制。支持质量和最终重建收益仍由实验验证。
