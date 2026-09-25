# 1024 点实验入口

主要方案见项目根目录 `EXPERIMENTS.md`。所有生成的输入、作业 JSON 和结果存于 `outputs/evidence_1024/`，不上传数据与 checkpoint。

首轮复用已核对的 airplane / 1024 点 baseline 支持：

```bash
python scripts/prepare_evidence.py \
  --mesh /mnt/sda/ubuntu/lkx/projects/Lkx_PaperA/data/den/raw_mesh/airplane/airplane_1.obj \
  --reuse-time-support /mnt/sda/ubuntu/lkx/paper1/time_support/runs/airplane_1_minutes \
  --output outputs/evidence_1024/cases/airplane_1

python scripts/make_evidence_jobs.py \
  --case outputs/evidence_1024/cases/airplane_1/case.json \
  --output outputs/evidence_1024/pilot_jobs.json --steps 6000

python launch.py --jobs outputs/evidence_1024/pilot_jobs.json \
  --gpus 1 2 3 --output outputs/evidence_1024/pilot
```

现有目录不可覆盖；复跑应换一个输出目录。这里的 Python 指服务器 `lkx_papera` 环境。

调度器会逐任务完成重建、离线评价，最后生成 `REPORT.md`、`metrics.csv`、`summary.json` 和 PNG/PDF 曲线。训练代码归档为 `source_snapshot.zip`，配置记录其哈希。NTPS/BSDF 的输入、来源、计时和实现信息保存在 case 与 receipt 中。

不复用历史 baseline 时，省略 `--reuse-time-support`，从指定网格构造新的 1024 点输入与独立评价采样。此时生成 Frozen / One-way / Joint / Raw 四组任务；原生 baseline 需要先对这份固定输入生成支持，不能混入别的采样结果。
