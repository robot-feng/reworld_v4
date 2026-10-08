# Pred Residual 对照实验（①②③）

核心验证：**残差形式是否更有效**。

| ID | 文档写法 | `condition_mode` | Action 上下文 | run_id |
|----|----------|------------------|---------------|--------|
| ① | `Pred(O_{t+h}-O_t)+C-Radio(O_t)` | `residual_plus_current` | `(F_t, Δ_8)` | `compare_pred_residual_1` |
| ② | `Pred(O_{t+h})+C-Radio(O_t)` | `absolute_plus_current` | `(F_t, F_{t+8})` | `compare_pred_residual_2` |
| ③ | `Pred(O_{t+h})` | `absolute_only` | `(F_{t+8}, F_{t+8})` | `compare_pred_residual_3` |

锁死设定：`QwenResidualWorldInverseV2`，`action_horizon=8`，`trajectory_fixed_mid=8`，`action_mode=abs`，`seed=42`，`40k` steps，数据 `libero_residual_world_all`。

World 仍在特征空间预测残差 \(\Delta\)；差别只在 Action 条件通道语义（残差 vs 绝对、是否拼当前 \(F_t\)）。

### Freeze variants

| Variant | `freeze_modules` | run_id suffix |
|---------|------------------|---------------|
| Frozen VLM+RADIO（已完成） | `qwen_vl_interface,vision_encoder` | `compare_pred_residual_{1,2,3}` |
| Unfreeze VLM（RADIO 仍冻） | `vision_encoder` | `compare_pred_residual_{1,2,3}_unfreeze_vlm` |

## Train

```bash
# Frozen VLM+RADIO
MODE=1 GPUS=0,1 bash examples/LIBERO_World/train_files/run_pred_residual_compare.sh
bash examples/LIBERO_World/train_files/run_pred_residual_compare_all.sh

# Unfreeze VLM (RADIO frozen)
MODE=1 GPUS=0,1 bash examples/LIBERO_World/train_files/run_pred_residual_unfreeze_vlm.sh
bash examples/LIBERO_World/train_files/run_pred_residual_unfreeze_vlm_all.sh
```
