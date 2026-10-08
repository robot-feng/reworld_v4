# LIBERO World

这是官方 `examples/LIBERO` 的残差世界模型增量层。常规数据准备和仿真评估入口保持同名，并直接委托给官方实现；自定义代码只负责动态未来帧、残差世界模型训练以及离线世界模型评估。

## 公平对比原则

训练启动时会按顺序合并：

1. `examples/LIBERO/train_files/starvla_cotrain_libero.yaml`（官方基线）；
2. `examples/LIBERO_World/train_files/starvla_qwen_residual_world.yaml`（仅自定义项）；
3. 与官方训练脚本一致的 CLI 运行参数。

因此数据集及权重、seed、batch、优化器、学习率调度、训练步数和普通 LIBERO 评估均继承官方设置。增量项只有 `QwenResidualWorld`、视觉编码器、残差世界模型、两段 self-forced 损失，以及 `[0,m,n]` 动态视频索引。

## 数据准备

```bash
DEST=/path/to/data bash examples/LIBERO_World/data_preparation.sh
```

该命令直接运行官方 `examples/LIBERO/data_preparation.sh`。

## 四卡训练

默认使用 TorchCodec 解码视频。Torch 2.7 对应 TorchCodec 0.3--0.5，并且需要带共享库的 FFmpeg：

```bash
conda install ffmpeg -c conda-forge
python -c "import torchcodec; print(torchcodec.__version__)"
```

如需临时回退到直接 PyAV 解码，可在启动时设置 `VIDEO_BACKEND=pyav`；不要再使用已被 torchvision 弃用的 `torchvision_av`。

```bash
NUM_PROCESSES=4 bash examples/LIBERO_World/train_files/run_libero_train.sh
```

Semantic Prefill 对比版本有一份可直接使用的完整配置，并保持相同数据、视觉编码器、动作头和训练超参数：

```bash
python starVLA/model/framework/VLM4A/QwenResidualWorldPrefill.py \
  --config_yaml examples/LIBERO_World/train_files/starvla_qwen_residual_world_prefill.yaml

NUM_PROCESSES=4 \
FRAMEWORK_NAME=QwenResidualWorldPrefill \
WORLD_OVERLAY=examples/LIBERO_World/train_files/starvla_qwen_residual_world_prefill.yaml \
RUN_ID=qwen_residual_world_prefill \
bash examples/LIBERO_World/train_files/run_libero_train.sh
```

该版本只替换 `SemanticPrefill -> ResidualBlock -> ActionResamplerBlock`。配置中没有
`action_context_dim` 和 `world_to_action` 学习率；动作头每层只读取 8 个紧凑 action queries。

常用覆盖仍采用环境变量，不需要改脚本：

```bash
NUM_PROCESSES=4 BATCH_SIZE=8 SAVE_INTERVAL=5000 \
RUN_ID=qwen_residual_world_v2 \
bash examples/LIBERO_World/train_files/run_libero_train.sh
```

每个样本由自定义数据集产生 `trajectory.images` 和
`trajectory.observation_indices=[0,m,n]`。设
`R=min(500, end-current)`，默认采样为：

```text
u₂, u₁ ~ Beta(2.5, 1.0)
n = 2 + floor(u₂ * (R - 1))
m = 1 + floor(u₁ * (n - 1))
```

两个位置使用相同参数的独立 Beta 采样，因此始终满足 `0 < m < n <= end-current`，且 `m` 能覆盖当前帧与目标帧之间的每个位置。若官方 mixture 抽到 episode 最后两个没有足够未来帧的位置，插件只在同一 episode 的合法 current 区间均匀重采样；官方索引缓存、Dataset、样本数和 mixture 代码均不改变。设置 `BETA_ALPHA=1 BETA_BETA=1` 后，`n` 在 `{2,...,R}` 上均匀采样，给定 `n` 时 `m` 在 `{1,...,n-1}` 上均匀采样；默认 horizon 为 500。

```bash
MAX_HORIZON=500 BETA_ALPHA=2.5 BETA_BETA=1.0 \
bash examples/LIBERO_World/train_files/run_libero_train.sh
```

## 官方方式仿真评估

与官方一样开两个终端，只是入口位于 `LIBERO_World`：

```bash
CKPT=/path/to/checkpoint GPU_ID=0 PORT=6694 \
bash examples/LIBERO_World/eval_files/run_policy_server.sh
```

```bash
LIBERO_HOME=/path/to/LIBERO CKPT=/path/to/checkpoint PORT=6694 \
TASK_SUITE_NAME=libero_10 NUM_TRIALS_PER_TASK=50 \
bash examples/LIBERO_World/eval_files/eval_libero.sh
```

两个脚本直接委托官方 `examples/LIBERO/eval_files`，所以仿真协议、动作后处理和成功率口径完全相同。

### 推理 horizon 对照

`QwenResidualWorldPrefill.predict_action()` 固定使用 `h=action_horizon=8`，与八步动作块训练和部署一致；独立调用 `predict_residual()` 时仍默认请求 `10000`，并吸收到 `500`。下面的请求层覆盖只用于旧 `QwenResidualWorld` 的 horizon 对照，不改变官方客户端：

```bash
WORLD_HORIZON=8 BASE_PORT=18600 \
bash examples/LIBERO_World/eval_files/launch_horizon_4way.sh /path/to/checkpoint
```

该命令在四张卡上分别运行 `libero_spatial/object/goal/10`，默认每任务 50 次、不录制视频。结果位于 checkpoint 运行目录下的 `evaluation/<step>/task_success_horizon_8_full_50_no_video/`。

## 世界模型离线评估（额外插件）

```bash
CKPT=/path/to/checkpoint MAX_BATCHES=100 \
bash examples/LIBERO_World/run_world_eval.sh
```

结果保存为 `world_metrics.json`、逐样本 `samples.jsonl` 和少量可视化。该入口只依赖框架的 `evaluate_world()`；设置 `RANDOM_CORE=1` 可用相同数据采样评估随机初始化的残差核心。

### Horizon 注意力扫描

```bash
CUDA_VISIBLE_DEVICES=0 python examples/LIBERO_World/eval_attention.py \
  --checkpoint /path/to/checkpoint --max-horizon 500
```

该只读插件复用第一帧的视觉编码和 semantic prefill，逐个扫描 `h=1...500`，记录每层 residual→current、residual→residual、action→residual 注意力及 task-change map。它输出 `attention_summary.png`、`attention_contact_sheet.png` 和包含完整数组的 `attention_sweep.npz`，不修改模型 forward。

## 模块边界

- `starVLA/model/framework/VLM4A/QwenResidualWorld.py`：唯一模型 API。
- `starVLA/model/framework/VLM4A/QwenResidualWorldPrefill.py`：Semantic Prefill 对比框架及独立冒烟入口。
- `starVLA/model/modules/latent_world_model/`：纯张量残差世界模型。
- `starVLA/model/modules/vision_encoder/`：可替换视觉编码器。
- `QwenResidualWorldPrefill` 直接复用官方 `LayerwiseFM_ActionHeader.py`；旧 `QwenResidualWorld` 的兼容适配器保持隔离。
- `starVLA/dataloader/gr00t_lerobot/trajectory_dataset.py`：按样本生成 `[0,m,n]`，官方数据集不变。

迁移到后续版本时只复制这些新增路径和 `examples/LIBERO_World`；如果 `git diff --name-status` 出现官方文件的 `M`，说明增量边界被破坏。
