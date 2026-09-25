# paper_cha

Learning Surface Support Representations for Sparse Point Cloud Reconstruction.

代码仓库：https://github.com/tobyal/paper_cha

当前后端版本为 v2（三平面多尺度，无 GSHE）。之前 Fourier MLP 后端及其短测仅作为历史，不能用于验证当前后端。

本工程研究：从稀疏观测推断显式表面支持，并将支持生成与隐式场在当前形状上联合优化。RepKPU 提供真实的几何编码、可变形核点表示及查询解码组件；方法最终服务于中间表面支持引导的重建，而非独立上采样排名。

```text
稀疏观测 → RepKPU encoder → REM 局部核点表示 → KGM / attention → 动态支持 Sθ
                                                           ↕
                      观测约束 + 支持约束 + pull / Eikonal → Fφ(x) → 网格
```

## 实际实现与来源

- `vendor/repkpu` 保存官方默认版 RepKPU 源码及 MIT 许可，修订 `80e2f293410a69747c0c61afe6f1fa2f71655133`。不是论文原版 RepKPU_o。
- 默认加载官方 PU-GAN 预训练权重，481 个状态项严格匹配。来源与逐文件哈希见 `PROVENANCE.json`。
- `ssr/support.py` 直接复用 encoder、REM、KGM、attention 和偏移头；输出适应当前形状的支持坐标，同时可访问核点及特征。原模型与适配器前向等价性由 `verify.py` 检查。
- 目前每个锚点保留官方 4 个查询，以兼容已验证权重。KGM 仍有固定局部查询坐标，不声称已实现纯 latent-query 或完全无坐标先验。对外任务参数是支持预算，而非级联上采样倍率；内部 vendor 保留原命名以便核查。
- `ssr/field.py`、`ssr/p3d_sdf.py` 恢复原 P3D-Mesh 的多尺度 Hash 三平面、三维 Hash Grid、坐标位置编码及 8 层 SDF 解码器。只移除 GSHE，采用 xy + yz + xz + grid 直接求和。16 级、每级 2 通道、分辨率 16→2048、每 1000 步启用一级及按点数确定上限的逻辑均沿用原代码。
- 支持是动态 pulling 目标和零水平面监督，不作为额外条件特征输入 SDF。最近邻索引不求导，目标坐标与支持处的场查询保留梯度，重建损失同时更新支持网络与场网络。没有 PLY/NumPy 中转切断训练图。
- 单形状优化固定 BN 运行统计，仿射参数与其余网络权重仍可学习。
- 不加入 AFNet、原实验脚本、Cross 或 Agreement；先验证新框架的核心证据链。查询—核点 attention 属于原 RepKPU 组件，不称为新的 patch Cross 创新。

## 环境

服务器：`/mnt/sda/ubuntu/lkx/paper1/paper_cha`。

已验证环境：`/mnt/sda/ubuntu/anaconda3/envs/lkx_papera/bin/python`。使用已有 pointops / Chamfer CUDA 扩展。新环境还需编译：

```bash
python -m pip install -r requirements.txt
python -m pip install ./vendor/repkpu/models/pointops --no-build-isolation
python -m pip install ./vendor/repkpu/models/Chamfer3D --no-build-isolation
```

## 首次验证与单形状重建

以下命令在项目根目录执行；`python` 指上述环境。输出目录必须是新的，避免覆盖已有实验。

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/reference_p3d_field.py --p3d-root ../p3d-mesh --output /tmp/p3d_field_reference.pt
CUDA_VISIBLE_DEVICES=3 python verify.py --p3d-reference /tmp/p3d_field_reference.pt --output verification_triplane.json

CUDA_VISIBLE_DEVICES=3 python reconstruct.py \
  --input /mnt/sda/ubuntu/lkx/projects/Lkx_PaperA/data/den/raw_mesh/airplane/airplane_1.obj \
  --observations 300 --mode joint --output outputs/airplane300_joint \
  --steps 2000 --minutes 30 --snapshot-steps 100 500 1000 2000
```

传入 mesh 时只按面积采样指定数量的稀疏点，训练不再访问其面或完整表面。已有点云可传入 PLY/XYZ/NPY，或包含 `p[N,3]` 的 NPZ。所有输出恢复到输入世界坐标。大于 16 的稀疏输入均可分 patch，不要求输入恰好为 256 的倍数。

运行产物：稀疏输入、初始/迭代/最终支持、迭代/最终网格、带 field/support 权重和优化器状态的 checkpoint、逐步损失、配置与状态。预算中止不会伪称收敛；没有零水平面时会明确记录，绝不伪造网格。checkpoint 保存用于复核和后续恢复实现，当前 CLI 不提供 resume。

`--minutes` 限制优化循环，包含循环内快照时间；单次更新可能跨过边界，最终网格导出不计入该上限。实际时间、迭代次数单独记录。若需按损失平台停止，增加 `--early-stop`：至少 500 步，此后每 50 步检查 EMA，连续 10 次未改善 0.3% 停止。

## 默认重建目标

`L = Lpull + 0.3 Lsurface + 0.3 Lobs_sdf + Lobs_cover + 0.1 Leikonal + 0.2 Lprior + 0.05 Lrep + 0.001 Lkernel`。

- `pull`：将场查询沿 SDF 梯度投影到零面，并靠近动态支持。pulling 目标的支持坐标不 detach；生成支持处的零面损失也可以回传到支持网络。
- `surface`：当前支持处 SDF 接近零。
- `obs_sdf`、`obs_cover`：输入点贴近零面，并被支持覆盖。采用 P→S 单向距离，避免把缺失区域支持全部吸回稀疏输入。
- `prior`：预训练初始化的对应支持位移约束；随机初始化对照不使用此项。
- `rep`：同锚点的 4 个查询输出相互排斥，避免在同锚点塌缩；不声称单独保证全局覆盖。
- `kernel`：原 REM 的核点拟合与排斥正则。
- 提供对齐输入的观测法向 `.npy` 时可加 `--normals`，加入不依赖法向正负的梯度一致性损失；不从测试 GT 偷取法向。

这些权重是可运行的初始设计，不是已证明最优的论文结果。联合优化需要实验检验，不能因梯度连通就宣称效果超过 BSDF/NTPS。

## 最小消融与共同后端

相同 input、seed、field 和预算下分别运行：

| 对照 | CLI |
|---|---|
| 完整联合优化 | `--mode joint` |
| 固定预训练支持 | `--mode frozen` |
| 只有稀疏观测 | `--mode raw` |
| 外部 NTPS/BSDF 支持 | `--mode external --external-support /absolute/support.ply` |
| 无预训练先验 | `--mode joint --initialization random --support-lr 0.0001` |
| 无核点形变 | `--mode joint --no-deform` |
| 无查询解码 | `--mode joint --decoder direct --support-lr 0.0001` |

`direct` 是 local feature→偏移的新随机头，绕过 KGM/attention。它不能继承原查询解码权重，实验解释须包括这一训练起点差别。`no-deform` 将形变置零，保持原相对坐标和感受野；不直接切换上游会改变坐标处理的 rigid 分支。

共同后端比较使用 `frozen/raw/external`；不能把 jointly adapted 的支持与固定外部支持混作纯支持来源消融。外部支持需在同一个输入世界坐标系中；如果 baseline 保存的是归一化坐标，先用其变换恢复后再输入。支持超过预算时统一 FPS，低于预算时保持原数量，记录实际数量。

## 预训练与评估

本地和服务器运行目录已有官方预训练权重。Git 仓库不包含权重或数据；新克隆后按 `weights/README.md` 下载权重。若要重新学习几何先验：

```bash
CUDA_VISIBLE_DEVICES=3 python train_prior.py \
  --manifest /mnt/sda/ubuntu/lkx/paper1/paper_results/module_ladder_v2/split.json \
  --output outputs/prior_v1 --epochs 100
```

只读取 `train`；不使用验证或测试划分选权重。优先读取每项的 `samples` 路径，否则使用 `cache`；NPZ 应包含 `p` 和 `gt`。加载生成权重时使用 `--checkpoint outputs/prior_v1/last.pt`。完整表面 CD 仅用于这一先验训练入口。

```bash
python evaluate.py --prediction outputs/airplane300_joint/support_final.ply \
  --gt /mnt/sda/ubuntu/lkx/projects/Lkx_PaperA/data/den/raw_mesh/airplane/airplane_1.obj \
  --observations outputs/airplane300_joint/input.ply --output outputs/airplane300_joint/support_metrics.json
```

同样可用 `mesh_final.ply` 作 prediction。报告 Accuracy、Coverage、CD-L1、采样 Hausdorff、远离输入四分位区域的 GapCoverage，世界单位及归一化 CD 分开。当前不报告未经实现的 normal consistency，不把采样距离称为精确点到三角面距离。

## 多 GPU 与实验层次

`launch.py --jobs experiments/example_jobs.json --gpus 1 2 --output outputs/level0`，每个 GPU 同时一个单形状任务。启动前检查显存；不终止其他用户任务。日志、退出码与总状态写入输出目录。

先用一个 airplane、300 个点检查 joint / frozen / raw 与支持—场反馈；再扩展 300/1024、少量形状与三个结构消融；最后接入 BSDF/NTPS 外部支持和原生重建结果。详细计划见 `EXPERIMENTS.md`。工程创建期间的短测试只验证实现，不代替科学结论。
