## Plan: Qwen MoE 滚动分阶段量化训练（补强版）

在现有分支上重构为仅面向 Qwen MoE 的阶段调度：A0(仅第0层attention) -> 对 i=0..L-2 执行 Ai(可选Ri路由校准 + Mi:第i层router+moe+第i+1层attn) -> Z(仅最后一层moe)。本版补强四个关键约束：跨层残差与RMSNorm路径、双block设备驻留策略、中间层teacher标签hook化、以及calib_router与expert shift均使用FP16 teacher。

**Steps**
1. Phase 1: 调度器与跨层前向图契约
1. 在 quantize/omniquant.py 中引入显式 stage dispatcher：A0 / Ri(i) / Mi(i) / Z，输出统一 stage 日志（stage_id、layer_id、device、param_count）。
1. 明确 Mi(i) 的计算图边界不是简单 layer(x)：必须走完整路径
1. x_i -> layer i attention/ffn(前半) -> moe_i 输出
1. residual add（与该层输入残差相加）
1. 进入 layer i+1 的输入前先经过 i+1 的 rms norm
1. 再进入 layer i+1 attention，loss 以 i+1 attention 输出为监督目标
1. A0 仅量化并训练 layer0 attention 对应路径，不触达 layer0 moe。
1. Z 仅处理最后一层 moe，显式跳过 next_attn 分支。

2. Phase 2: 参数冻结与优化器隔离
1. 新增 stage-aware 参数选择器（替代仅按名字包含 smooth/bound_factor 的全局选择），明确 Ri/Mi/Z 允许更新的参数集合。
1. Ri: 仅 router 主权重（TopK-MSE）可训练，禁止 LET/LWC、shared_gate、LoRA 更新。
1. Mi: 仅 LET/LWC + shared_gate + gate_lora 训练，router 主权重冻结。
1. Z: 全参数冻结，仅做最终 in-place quant。
1. 每个阶段切换都重建 optimizer 与 clip 参数集合，杜绝跨阶段参数泄漏。

3. Phase 3: 设备驻留与数据搬运策略（CPU主驻留 + 双block上卡）
1. 训练期间保持模型主体层默认驻留 CPU；当前参与训练的两个block（layer i 与 layer i+1）迁移到同一GPU。
1. Mi 前执行设备一致性断言：输入、残差、norm输入、attn输入与两个block参数必须在同一 device。
1. 阶段结束后将完成训练的block回迁 CPU，并释放临时缓存，避免显存碎片。
1. parallelize 开启时仍保持 no_split_module_classes 约束，防止一个 Mi 子图被拆到不兼容设备路径。

4. Phase 4: Teacher 标签重构（hook抓中间输出）
1. 重构 teacher label 采集：从“仅block输出标签”改为“支持中间节点标签”。
1. 对 Mi 训练目标，使用 hook 获取 layer i+1 attention 输出（以及需要时的pre-attn输入）作为 teacher。
1. hook 注册与移除遵循阶段生命周期，避免跨阶段残留与内存泄漏。
1. 新增标签schema，区分 router logits 标签、attn输出标签、以及expert shift标签。

5. Phase 5: FP16 teacher 一致性约束
1. calib_router 的 teacher label 仅由 FP16 teacher 模型生成，不允许使用量化分支标签。
1. expert shift 的 teacher label 同样仅由 FP16 teacher 模型生成。
1. 日志中显式打印 teacher 来源（fp16_teacher）与采样范围（full nsamples）。
1. 当 calibrate_router=True：记录 Pre-Ri / Post-Ri / Post-Mi；当 False：记录 Pre-Mi / Post-Mi。

6. Phase 6: LoRA/shared gate 生命周期
1. gate LoRA 仅在 Mi 训练；Ri 禁止训练 lora_A/lora_B。
1. LoRA merge 时机固定为“该层所有 Mi 完成后一次性 merge”，并校验 merge 前后模块引用一致。
1. shared_expert_gate 仅在 Mi 参数组中出现，沿用 shared_gate_lr 与 wd。

7. Phase 7: 仅 Qwen MoE 范围与入口约束
1. main.py 运行前断言：仅允许 Qwen MoE（A2.7B、30B-A3B 等），其他模型家族直接报错退出。
1. 保留现有命令行语义，不破坏你当前启动命令。

8. Phase 8: 验证与回归
1. 新增测试：
1. tests/test_stage_schedule_qwen_moe.py：验证 A0 -> (Ri+Mi)* -> Z 或 A0 -> Mi* -> Z。
1. tests/test_mi_crosslayer_path.py：验证 Mi 包含 residual add + next rms norm + next attn，且loss来自 next attn 输出。
1. tests/test_stage_device_consistency.py：验证双block同卡与 tensor 同设备断言。
1. tests/test_teacher_hook_labels.py：验证中间层 hook 标签抓取、shape 与生命周期。
1. tests/test_fp16_teacher_source.py：验证 calib_router 与 expert shift 均来自 FP16 teacher。
1. tests/test_stage_param_freeze.py：验证 Ri/Mi/Z requires_grad 白名单。
1. tests/test_lora_merge_consistency.py：验证 merge 后引用与前向一致。
1. 保留并扩展 test_compute_topk_mse_loss.py、test_expert_shift.py、test_attnbits.py。
1. 集成验证：先 Qwen1.5-MoE-A2.7B 冒烟，再 qwen30b-a3b 短周期全链路，最后长周期稳定性观察。

**Relevant files**
- d:/project/myomni/quantize/omniquant.py — 主调度、Mi跨层前向路径、设备迁移、阶段优化器切换。
- d:/project/myomni/quantize/utils.py — stage-aware 参数筛选、hook工具、teacher标签schema。
- d:/project/myomni/main.py — 仅Qwen MoE入口断言、训练日志与参数兼容。
- d:/project/myomni/tests/test_stage_schedule_qwen_moe.py — 阶段顺序。
- d:/project/myomni/tests/test_mi_crosslayer_path.py — 残差+rms norm+next attn路径与loss锚点。
- d:/project/myomni/tests/test_stage_device_consistency.py — 设备一致性。
- d:/project/myomni/tests/test_teacher_hook_labels.py — hook标签采集。
- d:/project/myomni/tests/test_fp16_teacher_source.py — FP16 teacher来源约束。
- d:/project/myomni/tests/test_stage_param_freeze.py — 冻结策略。
- d:/project/myomni/tests/test_lora_merge_consistency.py — merge一致性。

**Verification**
1. 日志显示阶段序列正确，且 Mi 日志包含 crosslayer_path=enabled、loss_anchor=next_attn_out。
1. 在 Mi 期间，layer i 与 i+1 均驻留同一GPU，输入与中间tensor设备一致，无 device mismatch。
1. calib_router 与 expert shift 日志均显示 teacher_source=fp16_teacher。
1. expert shift 字段与开关一致（三段/两段），默认 full nsamples。
1. A2.7B 与 30B-A3B 在 parallelize 条件下完整链路可跑。

**Decisions**
- 包含范围：仅 Qwen MoE（含 Qwen1.5-MoE-A2.7B 与 Qwen3-30B-A3B 类变体）。
- 排除范围：dense 模型与其他家族，不做兼容。
- calibrate_router 语义：仅决定 Ri 是否执行；Ri 独立于 Mi。
- Mi 监督锚点：固定为 layer i+1 attention 输出，不使用 block 末端统一输出替代。

**Further Considerations**
1. 是否增加可选开关控制 Mi teacher 标签粒度（仅attn输出 或 attn输出+pre-attn输入）。推荐默认仅attn输出，调试时再扩展。
2. qwen30b-a3b 验收是否要求包含 lm-eval 全任务。推荐先训练链路稳定，再执行任务评测。