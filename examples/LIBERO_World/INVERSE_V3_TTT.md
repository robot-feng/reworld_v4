# Inverse V3：LIBERO 延迟反馈记忆

V3 继承 V2 的骨干、参数名称和动作条件模式，新增 `motion_memory`。
这是一版线性 Fast Memory，借鉴 RoboTTT 的可微在线更新机制；没有直接
包装 `robo_ttt.TTTWrapper`，也没有修改其源码。

## 训练

从预训练编码器重新训练时，使用 [8/64 联合训练](V3_H8_H64_TRAINING.md)：
一次训练同时更新基础模型和记忆。以下是 `joint_training=false` 的旧记忆微调路径。

使用已经训练好的 V2；不要冻结随机初始化的世界模型来训练记忆。
入口仍在本目录，`train_inverse_v3_ttt.py` 复用原训练器与现有 LIBERO dataset，
只覆盖周期评估：以 `forward_ttt()` 计算训练批次反馈诊断，避免无帧号调用
在线 `predict_action()`。`train_feedback/*` 日志不是独立验证集结果。

```bash
V2_CKPT=/path/to/v2/checkpoints/steps_N_pytorch_model.pt \
V2_CONFIG=/path/to/v2/config.full.yaml \
STARVLA_PYTHON=/data/miniconda3/envs/ResWAM/bin/python \
NUM_PROCESSES=1 \
bash examples/LIBERO_World/train_files/run_inverse_v3_ttt.sh \
  --run_id libero_inverse_v3_ttt
```

配置生成器从该 V2 运行继承模型结构、视角、数据路径和动作设置，仅覆盖
V3 记忆、冻结项、学习率和 `[0,K,2K]` 采样，K 继承 V2 的动作 horizon（通常为8）。
`starvla_qwen_residual_world_inverse_v3.yaml`
是可编辑的完整模板；直接使用时必须填写 checkpoint 和本机路径。

V3 启用记忆时，`forward()` 只训练反馈分支：预测 0→8，到真实帧 8 写入历史
运动和基线误差，再监督 8→16。未来帧只用于最后的 loss。冻结的 V2 模块保持
`eval()`。Key/Value 和 Fast Weights 的梯度保留到后续预测损失。
短 episode 尾部仍遵循原采样器的缩短规则，例如 `[0,4,5]`；不是跨 episode 补帧。

`framework.ttt.enabled=false` 恢复 V2 的 Self-Forcing / action 训练路径。
这时若要重新训练骨干，须相应恢复配置中的 `trainer.freeze_modules` 与学习率。
V2 初始化时只加载五个原有模块；恢复 V3 完整权重时使用现有 resume 机制，
不要再次用 `reload_modules` 排除记忆。

## LIBERO 闭环评估

继续使用本目录的入口：

```bash
CKPT=/path/to/v3/checkpoints/steps_N_pytorch_model.pt \
STARVLA_PYTHON=/data/miniconda3/envs/ResWAM/bin/python \
GPU_ID=0 PORT=6694 \
bash examples/LIBERO_World/eval_files/run_policy_server.sh
```

另一终端使用已配置好 LIBERO / robosuite 的 Python 环境：

```bash
LIBERO_HOME=/path/to/LIBERO LIBERO_PYTHON=/path/to/libero/python \
CKPT=/path/to/v3/checkpoints/steps_N_pytorch_model.pt PORT=6694 \
TASK_SUITE_NAME=libero_goal NUM_TRIALS_PER_TASK=50 \
bash examples/LIBERO_World/eval_files/eval_libero.sh
```

已有 `run_eval_job.sh` 同样使用 V3 兼容服务端。官方仿真循环、动作分块、
反归一化、夹爪处理和成功率统计保持复用。World 客户端根据服务端 metadata
自动启用 TTT 协议：发送 session、episode 和环境帧号（0、8、16…）。
每次 `reset()` 更换 episode，服务端丢弃该 session 的旧状态。不同客户端
状态分离；Fast Weights 不通过 websocket 传输。默认最多保留 16 个 session，
长期批量评估超过此数量时重新启动服务端。

V3 须配套使用 World 客户端与服务端；直接使用官方无状态 server/client
不会建立反馈。原有非 TTT checkpoint 继续走无状态路径。

## 离线世界预测评估

```bash
CKPT=/path/to/v3/checkpoints/steps_N_pytorch_model.pt MAX_BATCHES=100 \
bash examples/LIBERO_World/run_world_eval.sh
```

原有 direct/self-forced 指标与可视化保持可用；额外报告：

- `ttt_base_future_mse`：真实中间帧出发的基线预测误差。
- `ttt_adapted_future_mse`：同一预测经历史反馈修正后的误差。
- `ttt_future_gain`：二者差值，正值表示改善。

这些是单次反馈指标，不能据此认定长期记忆或闭环成功率提高。

## 内部接口与边界

Python 直接调用时，`predict_action(..., step=frame_index, state=previous_state)`
返回 `ttt_state`；新 episode 传 `state=None`。多 horizon 同一时刻只结算一次。
反馈仅在目标帧准确到达时写入；错过的目标丢弃，重复/倒退帧号报错。
在线 horizon 必须不超过 `absorbing_horizon`，不把终态式预测当短期反馈。
LIBERO 客户端的显式 horizon 覆盖还须为动作 chunk 的整数倍，否则目标帧不会到达。
记忆修正、残差相加和监督差分保持 FP32，只在冻结动作条件入口转换回骨干精度。
小于该入口量化精度的修正仍可能不改变最终动作；FP32 loss 不等于控制收益。

`step_encoded()` 是训练和推理共用入口，也可供缓存特征的序列训练使用。
`EpisodeState.detach()` 同时截断 Fast Weights 和 pending keys，供 TBPTT 段边界使用。
首版只提供三帧训练；尚未增加长序列采样或完整 TBPTT 训练循环。
每个 state 的 batch 槽位必须始终对应同一 episode、相机顺序和特征布局。

## 验证

```bash
OMP_NUM_THREADS=1 /data/miniconda3/envs/ResWAM/bin/python -m pytest -q \
  starVLA/model/modules/memory/tests examples/LIBERO_World/tests
```

覆盖未来泄漏、写入梯度、训练/推理一致性、零门控回退、精确到期、
多记录聚合、episode 隔离、1000 次 FP32 更新、BF16 外层模型、
LIBERO 分块帧号/reset 协议、V2 注册与 checkpoint 兼容，以及离线指标接口。
这些测试使用小模型/替身，不代表实际 LIBERO rollout 成功率；后者需要训练后的 V3 checkpoint。

## 长任务适用范围

当前三帧版是一次反馈的原理验证。每次训练从空记忆开始，而在线记忆可连续
更新几十次；相同更新公式并不意味着训练覆盖了在线的记忆状态分布。
`short_long` 在线还会读写长 horizon，但三帧训练未覆盖相同过程。
在用于提升高分 V2 的 LIBERO long 成功率前，应完成连续锚点训练、跨阶段测试、
短/长 horizon 对齐，以及对冻结 Action Head 的动作误差/闭环成功率评估。
详细审查和已修复问题见 [INVERSE_V3_REVIEW.md](INVERSE_V3_REVIEW.md)。
