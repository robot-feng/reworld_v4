# V3 从预训练编码器开始：8 / 64

V2 文件不改动。没有可用的 96.7 策略权重，因此本次从本机预训练 Qwen3.5-0.8B、
C-RADIOv4-SO400M 开始，重新训练 V3 的世界模型和动作头，不是恢复 96.7 checkpoint。

启动入口：`train_files/run_inverse_v3_h8_h64.sh`，使用 GPU 0、1、2、3，先后运行两个阶段。
任何阶段失败都会停止脚本，不会用未完成的基础模型启动记忆训练。

| 阶段 | 配置 | 训练内容 | 默认规模 |
|---|---|---|---|
| 基础训练 | `inverse_v3_h8_h64_stage1.yaml` | Qwen、世界模型、动作头；冻结 RADIO 和未启用的记忆 | 40,000 steps，4×8=32 samples/step |
| 连续反馈训练 | `inverse_v3_h8_h64_stage2.yaml` | 冻结基础模型，只训练记忆；增加通过冻结动作头回传的动作损失 | 10,000 steps，4×2=8 sequences/step |

基础训练取 `[0,8,64]`，保留原 Self-Forcing 结构。尾部不足64帧时在同一 episode
重新选合法起点，不缩短动作监督对应的8帧目标。实际四套本机 LIBERO 数据没有少于65帧的 episode。

记忆训练查询的 horizon 是 **8 和64**，观测间隔是8帧。默认一段最多17个锚点
`[0,8,...,128]`，使64帧预测到期后仍有后续目标可监督；128是训练段跨度，不是第三个预测 horizon。
短 episode 使用较少的完整锚点，按长度分组，避免重复末帧和错误反馈。
同一 `step_encoded()` 执行训练和推理的到期结算；即使动作条件模式只消费短预测，
在线仍按配置收集64帧反馈，保持更新分布一致。

动作标签按每个真实观测锚点对齐。例如第24帧的条件监督第24–31帧动作，
不复用第0帧动作。动作输入沿用原 LIBERO 归一化；当前序列数据路径仅支持 `action_mode=abs`。
记忆 loss 为视觉残差 MSE + 0.1×冻结动作头的 flow-matching loss。
每4个锚点截断状态与 pending key 的梯度（保留数值），可调 `ttt.tbptt_steps`。

这种序列训练覆盖多次反馈和长短反馈混合，修复了此前三帧训练的关键缺口；
它仍不等于已经覆盖520帧整局、失败恢复或证明超过96.7。
成功率必须等训练后在 LIBERO 仿真测量。先前审查文档中的三帧局限适用于旧的 `sequence_training=false` 路径。

日志、权重目录：

- `playground/Checkpoints/inverse_v3_h8_h64_pipeline_20261009.log`
- `playground/Checkpoints/inverse_v3_h8_h64_stage1_20261009/`
- `playground/Checkpoints/inverse_v3_h8_h64_stage2_20261009/`

复用 `eval_files/run_policy_server.sh` 和 `eval_files/eval_libero.py` 评估训练后的模型。
基础阶段的策略不启用记忆；第二阶段启用 session/episode/帧号协议。
评估需要与训练一致地传 proprio；本次两阶段 `include_state=true`，使用 Python 评估入口时
加 `--args.include-state`。不要拿未传 state 的评估与传 state 的基线直接比较。

## 阶段性验证记录（2026-10-09）

- 单元及集成测试：38 项通过，覆盖因果性、长反馈、多次更新、动作对齐和训练/在线状态一致性。
- 真实数据基础训练：两个优化步通过，第二步 Qwen 梯度有限且非零。第一步世界模型残差输出层零初始化，使上游 Qwen 梯度为零，不能据此判断 VLM 冻结。
- 四卡基础预检：3 steps，含评估及保存通过；总参数 1,377.852M，可训练 945.417M。
- 四卡记忆预检：2 steps，含加载基础权重、序列训练、评估及保存通过。预检权重仅用于验证流程，不用于正式第二阶段初始化。
- 第一阶段 Qwen 学习率 1e-5（含 warmup），冻结列表仅为 `vision_encoder,motion_memory`。第二阶段才冻结 Qwen。
- 正式两阶段流水线已于本日启动；运行状态以日志为准。四卡采用 bf16 和 ZeRO-2，预检阶段每卡约30 GiB，显存未占满不能说明 VLM 未解冻。

当前离线 `world_eval` 图像可视化仍要求三帧输入，不能直接套用第二阶段的序列数据配置；
其单次反馈指标也不能代表多次在线更新。第二阶段训练内评估走完整序列，最终策略成功率走 LIBERO 在线评估。
