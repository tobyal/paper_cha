# Method 3.2–3.5 代码整理验证

日期：2026-09-26。服务器 GPU 3，环境 `lkx_papera`。整理前参照为本仓库提交 `3c8c9f4bce4ff2e754cee3d8fa9c1f5a08479b87`。

## 与整理前的实现比较

`scripts/verify_method_refactor.py` 从 Git 历史独立加载旧实现，以相同权重和输入比较新实现：

| 配置 | 返回张量最大误差 | 参数梯度最大绝对差 | 比较的梯度张量数 |
|---|---:|---:|---:|
| 完整查询解码 | 0 | 2.13e-6 | 335 |
| 无形变 | 0 | 4.66e-6 | 326 |
| 直接解码 | 0 | 9.71e-6 | 290 |
| 随机初始化 | 0 | 1.17e-10 | 335 |

所有比较通过 `atol=1e-5, rtol=1e-4`；未将 CUDA 参数梯度比较表述为逐位一致。旧、新参数键与排列相同，支持权重严格加载成功；默认 481 项，带直接解码头 485 项。

九个损失分量（包含可选法向项）的数值误差均为 0，总损失对支持坐标的梯度误差为 0。旧 `ssr.field` / `ssr.objective` 导入指向新 3.5 的同一实现。静态检查确认两个损失函数的 AST 与旧实现一致。

## 联合重建链路

重新运行 `verify.py`，使用独立原 P3D 后端的参照：

- RepKPU 原生前向与支持生成器输出误差为 0。
- 编码步数 0、2000、7000 的 SDF 值与空间梯度误差均为 0。
- 无 GSHE；XY、YZ、XZ 与 3D Grid 四分支获得非零有限梯度。
- 支持处的零面损失回传到 335 个支持网络参数张量。
- 联合目标完成有限二阶反传，并更新支持网络与 SDF 参数。
- 无形变、直接解码、固定支持和随机初始化检查通过。

这些是结构整理的回归检查，不是新的效果实验或收敛结论。没有重新启动完整训练。

## 复现

在服务器项目根目录，用该环境的 Python 执行：

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/verify_method_refactor.py \
  --output verification_method_refactor.json

CUDA_VISIBLE_DEVICES=3 python verify.py \
  --p3d-reference outputs/p3d_field_reference.pt \
  --output verification_method_joint.json
```

新克隆仓库需要先按 README 安装环境并获取权重。参照文件可由 `scripts/reference_p3d_field.py` 生成。机器记录保存在本地和服务器项目根目录，JSON 不上传 Git；其中 `verification_method_refactor.json` 记录被测 `ssr/*.py` 的 SHA-256。
