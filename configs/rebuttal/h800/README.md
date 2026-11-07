# 新训练权重交付到双卡 H800

专用代码分支：`rebuttal/h800-trained-checkpoints`。
已有权重、target TP1/TP2、SSD 等正在进行的 H800 实验继续使用原来的输出目录。
`run_delivery.yaml` 只运行新训练的权重；匹配的既有 target baseline 可以复用。

## 训练机整理上传目录

训练守护进程会自动把已完成的权重整理到：

```text
outputs/rebuttal_h800_upload/
```

也可以查看或手动整理：

```bash
python scripts/rebuttal/export_h800.py
python scripts/rebuttal/export_h800.py --group llama --execute
```

默认对大权重建立硬链接，导出目录与训练输出应在同一文件系统上；跨文件系统用 `--copy`。
导出的目录发布后保持只读，上传客户端正常读取即可。

将已就绪的子目录逐批上传到同一个 ModelScope 模型仓库的根目录，保留下表中的目录名。
每个目录内包含模型权重（包括全部分片与索引）、config、generation config、tokenizer、chat template、必要模型代码及简短说明。

| 分组参数 | 上传目录 | 选取规则 |
|---|---|---|
| `llama` | `05_llama8_d3d1_ar1_s42`、`06_llama8_d3d1_ar0_s42` | 各自第三个 epoch 的最终权重 |
| `qwen17` | `01_q32_d3d17_ar1_s42` | 第三个 epoch 的最终权重 |
| `qwen4` | `07_q32_d3d4_ar1_s42` | 第三个 epoch 的最终权重 |
| `pard` | `03_q32_pard2_17_s42` | `epoch-3`，target-independent 推理 |
| `pard_ref` | `09_q32_pard2_17_ref_lr_s42` | `epoch-3`，target-independent 推理 |
| `pruning_acceptance` | `10_q32_8b_prune_acceptance_d34`、`10_q32_8b_prune_acceptance_d31` | 第 2／5 阶段，验证集选定的恢复 checkpoint |
| `pruning_uniform` | `11_q32_8b_prune_uniform_d34`、`11_q32_8b_prune_uniform_d31` | 第 2／5 阶段，验证集选定的恢复 checkpoint |

优化器、RNG、训练数据、中间周期 checkpoint、TensorBoard 日志留在训练机。PARD-2 的教师投影只用于训练，不属于 target-independent 推理包。
首次交付优先等待 Llama 两组都完成；也可用 `--case` 单独测已经就绪的一组。

## H800 机器拉取代码

在新工作目录建立分支，已有实验工作目录保持原样：

```bash
git fetch origin
git worktree add -b rebuttal/h800-trained-checkpoints ../D3D-h800-trained \
  origin/rebuttal/h800-trained-checkpoints
cd ../D3D-h800-trained
```

后续在这个工作目录更新：

```bash
git pull --ff-only origin rebuttal/h800-trained-checkpoints
```

复用当前已验证的推理环境。原生后端要求 torch 2.8.x、Transformers 4.57.3 与可用的 FlashAttention-2。
新分支包含 Llama 的 Q/K norm、mask/EOS 处理和 PARD-2 的 target-independent 适配。

新工作目录继续读取已有 H800 实验的评测数据缓存，避免换目录后重新下载出不同的 prompt。
如果旧目录使用 `data_cache/`、`answer_cache/`，将下面路径换成旧工作目录的绝对路径：

```bash
export D3D_DATA_CACHE=/path/to/existing-D3D/data_cache
export D3D_ANSWER_CACHE=/path/to/existing-D3D/answer_cache
```

## 下载权重并运行

把 ModelScope 仓库下载到一个独立目录，例如 `/models/d3d-rebuttal`，其直接子目录应为上述模型 ID。
target 模型放在另一个根目录，例如 `/models/targets`，包含：

```text
/models/targets/Qwen3-32B/
/models/targets/Llama-3.1-8B-Instruct/
```

Qwen3-32B 可复用现有文件。若 H800 尚无 Llama target，另行准备完整的 Llama-3.1-8B-Instruct；训练机对应路径为 `assets/rebuttal/models/Llama-3.1-8B-Instruct/`。

先打印命令检查路径，再增加 `--execute` 正式运行：

```bash
python scripts/rebuttal/h800.py \
  --config configs/rebuttal/h800/run_delivery.yaml \
  --group llama \
  --models-root /models/targets \
  --checkpoints-root /models/d3d-rebuttal \
  --output-root outputs/rebuttal_h800_trained \
  --gpus 0,1
```

把 `--group llama` 换成表中的其他分组即可。每组在 MATH500、HumanEval、MT-Bench 上运行同步和异步，完整任务集、三次计时重复；greedy、block size 16、max-new-tokens 2048。这里的输出长度与训练总长上限 8192 是不同参数。

同一对 H800 上顺序执行已有实验和新实验。固定 prompt、输出上限、EOS、target attention backend 和计时口径后复用已有 baseline。
若现有结果的这些条件不同，则为新权重补匹配 baseline；不将不同协议的计时直接相除。
仅在缺少同协议 target baseline 时，使用 `--config configs/rebuttal/h800/run_target_baselines.yaml`
和 `--group llama_baseline` 或 `--group qwen32_baseline`；这部分不会自动加入新权重的推理队列。

单个权重的命令示例：

```bash
python scripts/rebuttal/h800.py \
  --config configs/rebuttal/h800/run_delivery.yaml \
  --case 06_llama8_d3d1_ar0_s42_async \
  --models-root /models/targets --checkpoints-root /models/d3d-rebuttal \
  --output-root outputs/rebuttal_h800_trained --gpus 0,1 --execute
```

剪枝耗时分解另跑一次，在正式性能结果之外保存：

```bash
python scripts/rebuttal/h800.py \
  --config configs/rebuttal/h800/run_delivery.yaml \
  --group pruning_acceptance \
  --models-root /models/targets --checkpoints-root /models/d3d-rebuttal \
  --output-root outputs/rebuttal_h800_profile --gpus 0,1 \
  --profile --repeats 1 --execute
```

均匀剪枝改为 `--group pruning_uniform`。对照 36 层父模型使用已有、同协议的 `D3D-8B-32B` 结果。
`scripts/rebuttal/compare.py --baseline <target-result.json> --method <new-result.json>` 会先核对 prompt 和 greedy 输出 token，再报告 decode／端到端 speedup 及配对区间。
