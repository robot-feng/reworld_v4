# V3：一次联合训练，horizon 8 / 64

本机没有96.7策略权重，从预训练 Qwen3.5-0.8B 和 C-RADIO 开始。
V2 源文件不修改。此前两阶段流水线已停止，当前推荐入口只有一个训练任务：

```bash
bash examples/LIBERO_World/train_files/run_inverse_v3_h8_h64.sh
```

等价入口是 `run_inverse_v3_joint.sh`，配置是 `inverse_v3_h8_h64_joint.yaml`。
旧 stage1/stage2 配置仅保留作实验参考，不再由推荐入口启动。

## 优化目标与梯度

- 每批序列最多9个锚点 `[0,8,...,64]`；TTT 记忆仅使用 short horizon 8。基础分支仍以 `[0,8,64]` 训练世界模型的短、长预测。
- 基础分支从同一序列选 `[0,8,64]`，保留原有世界模型 self-forcing 和动作损失。
- 连续反馈分支每隔8帧结算一次 short 预测误差，再用更新后的记忆预测下一块动作；首块动作由基础分支监督。
- 一次反向传播：`loss = base_loss + lambda(step) * (ttt_world_loss + ttt_action_loss)`。
- `lambda(step) = 0.25 * min(1, (completed_steps + 1) / 2000)`，TTT 从首步参与，辅助损失平滑增权；
  修正本身使用可学习的小门控（初值0.001），没有训练专用的额外推理倍率。
- Qwen、世界模型通过基础分支更新；TTT、动作条件器、动作头通过反馈分支更新，
  动作模块同时接收基础分支监督。RADIO 始终冻结。

反馈分支对 Qwen/世界模型使用当前权重、eval 模式和 stop-gradient，避免为每个锚点保留 VLM 反向图。
这是共享一个优化器的联合训练，但不是对全部历史 VLM 特征做完整时间反传。
基础分支仍有完整的 VLM 梯度，两路模型参数每一步一起更新，无需中间 checkpoint 或手工切换。
反馈状态每4个锚点截断梯度；数值保留，序列之间清空状态。

## 配置与观测

四卡，每卡有效32条序列，全局128条序列；共40,000 optimizer steps。显存 micro-batch 为8，梯度累积4步（每步处理8条/卡，四次累积后有效32条/卡）。
Qwen lr=1e-5，世界模型5e-5，动作模块与记忆1e-4，优化器另有2000步学习率 warmup。
总训练样本数/计算量不能与旧4×8的三帧配置直接等同；本次每条序列还监督多个动作锚点。

日志：`playground/Checkpoints/inverse_v3_h8_h64_short_ttt_b128_20261009.log`。
权重：`playground/Checkpoints/inverse_v3_h8_h64_short_ttt_b128_20261009/`。

梯度测试验证基础损失可更新 Qwen/世界模型，联合损失可更新全部目标模块，
且反馈动作损失单独可以回传到记忆。周期 `train_feedback/*` 是训练批次诊断，不是成功率。

最终使用 `eval_files/run_policy_server.sh` 和 `eval_files/eval_libero.py` 做 LIBERO 在线评估。
使用 Python 评估入口时加 `--args.include-state`，匹配训练的 proprio 输入。
当前离线 `world_eval` 图像可视化仍要求三帧输入，不能直接套用序列数据配置；
最终在线评估和训练内连续反馈评估不受此限制。

这项实验尚未证明超过 V2 的96.7，尤其需要单独报告 LIBERO long 的结果。

## 既有实现验证（2026-10-09）

- 40 项测试通过；辅助权重日志调整后，相关17项回归通过。
- 四卡真实 LIBERO 数据完成3步联合训练，周期反馈评估及 checkpoint 保存成功。
- 946.615M 可训练参数；Qwen 与记忆均在优化器参数组中。
- 预检末步 total_loss=1.294221，反馈修正初期极小，尚不能声称有收益。
- 预检日志：`playground/Checkpoints/inverse_v3_preflight/joint_four_gpu.log`。

## 序列缓存与显存

每条样本最多9个锚点，每隔8帧取一张观测，覆盖 `[0,64]`。每个时刻只编码当前帧并因果结算到期反馈；
`EpisodeState` 保存 `[B,dim,dim]` 的 fast weights 及尚未到期的少量 key/origin/base prediction，
不会预先把9帧的视觉特征堆到 GPU。数据集会在CPU侧加载整段图像。

训练显存主要来自需要反向传播的锚点损失：最多8个动作头调用和一串记忆查询/更新图。
short 反馈下一锚点就到期；TTT 不再为64帧预测保留待结算图。TBPTT每4个锚点截断 fast-weight/key 的梯度，
但各项 loss 的计算图要留到统一 backward，因此显存随 micro-batch 和有效锚点数增长。

四卡每卡32直接运行已实测 OOM；16 micro-batch 峰值约79 GiB，正式跑到第2步时因序列长度波动再次 OOM。
正式设置用8×4累积达到有效32/卡、全局128，给激活显存留出余量，且每次参数更新的全局样本量不变。
后续还可用 activation checkpoint 或分段 backward 降低峰值；分段 backward 需要仔细处理跨段64帧延迟监督的梯度，
不能简单切开计算，否则会改变训练信号。

四卡 8×4 累积预检已在原17锚点配置上完成一个有效优化步并保存 checkpoint，峰值显存约47 GiB/卡；
对应日志为 `playground/Checkpoints/inverse_v3_preflight/joint_micro8_accum4_four_gpu.log`。

short-only 9锚点正式配置已完成 step 0，日志中的 `world_long_direct_loss` 和 TTT 损失均正常；首步没有 OOM。
正式日志为 `playground/Checkpoints/inverse_v3_h8_h64_short_ttt_b128_20261009.log`。

TTT 记忆只用与 action chunk 等长的 short 8 帧预测误差写入。观测到下一锚点后，先结算刚完成的 chunk 反馈，再预测并条件化后续动作；long 64 帧继续由基础世界模型直接监督，不进入 TTT 反馈状态。
