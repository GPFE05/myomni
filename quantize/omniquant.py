import torch
import torch.nn as nn
import torch.nn.functional as F
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
from quantize.utils import (
    let_parameters, lwc_parameters, get_omni_parameters,
    omni_state_dict, register_scales_and_zeros, smooth_and_quant_temporary,
    smooth_and_quant_inplace, clear_temp_variable, set_quant_state,
    capture_router_labels_layerwise, compute_expert_shift_detailed,
    compute_topk_mse_loss, forward_with_router_logits, create_router_hook,
    call_layer_forward, extract_hidden_states, forward_with_module_input,
    set_stage_trainable_params, assert_tensors_on_same_device
)


class LoraLinear(nn.Module):
    """
    LoRA wrapper for nn.Linear layer.
    Freezes the original weight and trains low-rank adapters lora_A and lora_B.
    """
    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 16, seed: int = 42):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Copy original weight and bias, freeze them
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        
        # Initialize LoRA matrices with fixed seed for reproducibility
        # A with Gaussian (scaled), B with zeros
        generator = torch.Generator(device=linear.weight.device)
        generator.manual_seed(seed)
        self.lora_A = nn.Parameter(torch.randn(r, self.in_features, device=linear.weight.device, dtype=linear.weight.dtype, generator=generator) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r, device=linear.weight.device, dtype=linear.weight.dtype))
    
    def forward(self, x):
        # output = x @ (W + scaling * B @ A).T + bias
        # = x @ W.T + scaling * x @ A.T @ B.T + bias
        base_out = nn.functional.linear(x, self.weight, self.bias)
        lora_out = nn.functional.linear(nn.functional.linear(x, self.lora_A), self.lora_B)
        return base_out + self.scaling * lora_out
    
    def merge(self):
        """
        Merge LoRA weights into the original weight and return a standard nn.Linear.
        """
        # Merge: W_new = W + scaling * B @ A
        merged_weight = self.weight.data + self.scaling * (self.lora_B.data @ self.lora_A.data)
        
        linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None, 
                          device=merged_weight.device, dtype=merged_weight.dtype)
        linear.weight.data = merged_weight
        if self.bias is not None:
            linear.bias.data = self.bias.data.clone()
        
        return linear
try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)     


def stage_build_units(total_layers):
    """构建显式阶段序列：A0 -> Ai* -> Z。"""
    units = [("A0", 0)]
    for idx in range(total_layers - 1):
        units.append(("Ai", idx))
    units.append(("Z", total_layers - 1))
    return units


def stage_is_attn_linear(module_name):
    return module_name.startswith("self_attn.") and module_name.split(".")[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}


def stage_is_router_gate(module_name):
    return module_name.endswith(".gate") or module_name == "gate"


def stage_is_shared_expert_gate(module_name):
    return module_name.endswith("shared_expert_gate")


def stage_is_moe_linear(module_name):
    return module_name.startswith("mlp.")


def stage_should_quantize_linear(stage_name, module_name):
    """阶段化线性层量化规则。"""
    if stage_name == "A0":
        return stage_is_attn_linear(module_name)
    if stage_name == "Ai":
        return stage_is_moe_linear(module_name) and (not stage_is_attn_linear(module_name))
    if stage_name == "Z":
        return stage_is_moe_linear(module_name) and (not stage_is_attn_linear(module_name))
    return True

def omniquant(
    lm,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
    train_shared_gate=False,
    train_gate_lora=False,
    shared_gate_lr=1e-4,
    gate_lora_lr=1e-4,
    lora_r=8,
    lora_alpha=16,
    # Router Calibration parameters
    calibrate_router=False,
    router_lr=1e-3,
    router_epochs=5,
    k_loss=20,      # TopK for loss calculation (cached label size)
    k_routing=4,    # TopK for expert shift metric (actual routing k)
):
    logger.info("Starting ...")
    
    # WandB integration: import and initialize global step for continuous training curve
    wandb = None
    if getattr(args, 'enable_wandb', False):
        try:
            import wandb as _wandb
            wandb = _wandb
        except ImportError:
            logger.warning("WandB not installed but enable_wandb=True. Skipping WandB logging.")
    global_step = 0  # Global step counter for continuous WandB logging across layers
    final_loss = None  # Track the loss from the last epoch of the last layer
    expert_shift_data = []  # Collect expert shift data per layer for visualization
    
    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "o_proj":"out",
            "up_proj":"fc1"
        }
        layer_name_prefix = "model.layers"
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkv",
            "out_proj":"out",
            "fc1":"fc1"
        }
        layer_name_prefix = "model.decoder.layers"
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    elif 'mixtral' in args.net.lower():
        is_llama = True   # same to llama except ffn
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        layer_name_prefix = "model.layers"
    elif 'qwen' in args.net.lower():
        is_llama = True   # same to llama except ffn (MoE structure)
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        # Qwen MoE models only support the MoE/LWC path here, no DecoderLayer wrapper is needed.
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral/qwen now")
    
    
    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = lambda: torch.amp.autocast('cuda')
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["layer_kwargs"] = dict(kwargs)
            cache["attention_mask"] = kwargs.get("attention_mask")
            if self.is_llama:
                cache["position_ids"] = kwargs.get("position_ids")
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "llama" in args.net.lower() or "mixtral" in args.net.lower() or "qwen" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings =  model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral/qwen now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    fp_inps_2 = copy.deepcopy(inps) if args.aug_loss else None # take output of quantization model as input
    
    attention_mask = cache["attention_mask"]

    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1) if args.deactive_amp else attention_mask.repeat(args.batch_size,1,1,1).float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()
    layer_kwargs = dict(cache.get("layer_kwargs", {}))
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None



    if args.resume:
        omni_parameters = torch.load(args.resume, weights_only=False)
    else:
        omni_parameters = {}

    if "qwen" not in args.net.lower():
        raise ValueError("This staged MoE schedule currently supports Qwen MoE only.")

    logger.info("[Stage] Schedule initialized: A0 -> (Ri+Mi)* -> Z")

    # ======================== 阶段辅助函数（中文详细注释） ========================
    # 说明：当前代码仍基于“逐层主循环”演进，为了逐步收敛到 A0/(Ri+Mi)/Z，
    # 我们先把“阶段语义”显式化，后续再进一步抽离为独立 dispatcher。
    def _build_stage_units(total_layers):
        return stage_build_units(total_layers)

    def _is_attn_linear(module_name):
        return stage_is_attn_linear(module_name)

    def _is_router_gate(module_name):
        return stage_is_router_gate(module_name)

    def _is_shared_expert_gate(module_name):
        return stage_is_shared_expert_gate(module_name)

    def _is_moe_related_linear(module_name):
        return stage_is_moe_linear(module_name)

    def _should_quantize_linear_for_stage(stage_name, module_name):
        return stage_should_quantize_linear(stage_name, module_name)

    stage_units = _build_stage_units(len(layers))

    # 保存每层的 FP16 teacher 快照，保证 Ri 与 expert shift 标签不依赖已量化学生层。
    # 说明：
    # 1) 第 0 层在进入 A0 前就会缓存；
    # 2) 第 i+1 层会在 Ai(i) 时作为 next_layer_fp 首次出现并缓存；
    # 3) 后续任意阶段都优先使用该快照生成 teacher 标签。
    fp_teacher_layers = {}

    for stage_name, i in stage_units:
        layer_stage = stage_name
        stage_train_enabled = args.epochs > 0
        should_roll_buffers = stage_train_enabled and layer_stage == "Ai"
        stage_gate_train_enabled = layer_stage in {"Ai", "Z"}

        if layer_stage == "A0":
            logger.info(f"=== [Stage A0] Start stage on layer {i} (attention-only) ===")
        elif layer_stage == "Z":
            logger.info(f"=== [Stage Z] Start quantize layer {i} (last-moe-only) ===")
        else:
            logger.info(f"=== [Stage A{i}] Start quantize layer {i} (Ri+Mi unit) ===")

        layer = layers[i].to(dev)
        if i not in fp_teacher_layers:
            fp_teacher_layers[i] = copy.deepcopy(layer).cpu()

        teacher_layer = fp_teacher_layers[i].to(dev)
        teacher_layer.eval()
        for p in teacher_layer.parameters():
            p.requires_grad = False

        next_layer_fp = None
        next_layer_student = None
        if layer_stage == "Ai" and i < len(layers) - 1:
            next_layer_fp = layers[i + 1].to(dev)
            if (i + 1) not in fp_teacher_layers:
                fp_teacher_layers[i + 1] = copy.deepcopy(next_layer_fp).cpu()
            next_layer_fp.eval()
            for p in next_layer_fp.parameters():
                p.requires_grad = False
            logger.info(f"[Device] Layer {i} and Layer {i+1} moved to {dev} for Mi cross-layer path.")

            # ======================== 中文详细注释 ========================
            # Ai 阶段除了训练当前层 router+moe，还需要把“下一层 attention”纳入可训练分支。
            # 这里构造 next_layer_student：
            # 1) 以 next_layer_fp 为模板复制；
            # 2) 仅将 self_attn 的 q/k/v/o 线性替换为 QuantLinear；
            # 3) 训练时通过 next_layer_student 在 post_attention_layernorm 输入处与 FP16 teacher 对齐。
            next_layer_student = copy.deepcopy(next_layer_fp)
            for n_name, n_module in next_layer_student.named_modules():
                if isinstance(n_module, torch.nn.Linear) and _is_attn_linear(n_name):
                    n_weight_params = args.attn_weight_quant_params if args.attn_weight_quant_params is not None else args.weight_quant_params
                    n_quantlinear = QuantLinear(n_module, n_weight_params, args.act_quant_params)
                    add_new_module(n_name, next_layer_student, n_quantlinear)
            next_layer_student = next_layer_student.to(dev)
        if "mixtral" in args.net.lower() or "qwen" in args.net.lower():  
            # For MoE models (Mixtral, Qwen MoE), only the LWC-style path is supported.
            # Simply replace Linear with QuantLinear, do not quantize router (gate)
            qlayer = copy.deepcopy(layer)
            
            for name, module in qlayer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # Target 1: Shared Expert Gate - name ends with "shared_expert_gate"
                    is_shared_expert_gate = _is_shared_expert_gate(name)
                    
                    # Target 2: Router Gate - name ends with ".gate" (NOT "gate_proj")
                    # e.g., "mlp.gate" is router, but "mlp.experts.0.gate_proj" is NOT
                    is_router_gate = _is_router_gate(name)
                    
                    if is_shared_expert_gate:
                        if train_shared_gate and stage_gate_train_enabled:
                            # Keep as nn.Linear but make trainable
                            module.weight.requires_grad = True
                            if module.bias is not None:
                                module.bias.requires_grad = True
                        # else: skip, keep as frozen nn.Linear (default behavior)
                    elif is_router_gate:
                        if train_gate_lora and stage_gate_train_enabled:
                            # Replace with LoraLinear wrapper (use layer index as seed for reproducibility)
                            lora_linear = LoraLinear(module, r=lora_r, alpha=lora_alpha, seed=args.seed + i)
                            add_new_module(name, qlayer, lora_linear)
                        # else: skip, keep as frozen nn.Linear (default behavior)
                    else:
                        if not _should_quantize_linear_for_stage(layer_stage, name):
                            # 阶段过滤：例如 Z 阶段不应再碰最后层 attention 线性层。
                            continue

                        # Target 3: All other linear layers (including gate_proj)
                        # Replace with QuantLinear
                        is_attn_linear = _is_attn_linear(name)
                        weight_params = args.attn_weight_quant_params if (is_attn_linear and args.attn_weight_quant_params is not None) else args.weight_quant_params
                        quantlinear = QuantLinear(module, weight_params, args.act_quant_params)
                        add_new_module(name, qlayer, quantlinear)    
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)

        # =================================================================
        # Expert Shift Tracking for Qwen MoE
        # Flow: 
        #   1. Capture FP16 teacher labels from ORIGINAL layer
        #   2. Set quantization state on qlayer
        #   3. Compute Pre-LWC Expert Shift (quantized vs FP teacher)
        #   4. (Optional) Router calibration training if calibrate_router=True
        #   5. (Optional) Compute Post-Calibration Expert Shift
        #   6. LWC training
        #   7. Compute Post-LWC Expert Shift
        # =================================================================
        is_qwen_moe = "qwen" in args.net.lower()
        # Ri 只在 Ai 阶段可选执行；A0 和 Z 默认跳过。
        do_router_calibration = calibrate_router and layer_stage == "Ai"
        cached_router_labels = None
        pre_lwc_shift = None
        post_calib_shift = None
        
        run_router_related_stages = layer_stage in {"Ai", "Z"}

        if is_qwen_moe and run_router_related_stages:
            logger.info(f"[Expert Shift] Layer {i}: Starting expert shift tracking...")
            
            # Convert qlayer to FP32 for stable computation (same as LWC training)
            # This is required because:
            # 1. V100 doesn't support BF16, and autocast converts to FP16 which can cause precision issues
            # 2. GradScaler doesn't support FP16 gradients
            # 3. LWC training also uses qlayer.float() before training
            qlayer.float()
            
            # ============================================================
            # Phase A: Capture FP16 router labels from ORIGINAL layer
            # ============================================================
            logger.info(f"[Expert Shift] Layer {i}: Capturing FP16 router labels (topk={k_loss})...")
            logger.info(f"[Expert Shift] Layer {i}: teacher_source=fp16_teacher")
            cached_router_labels = capture_router_labels_layerwise(
                teacher_layer, fp_inps, dev, topk=k_loss, logger=logger, layer_kwargs=layer_kwargs
            )
            
            if cached_router_labels is not None:
                teacher_logits, teacher_indices = cached_router_labels
                logger.info(f"[Expert Shift] Layer {i}: teacher_logits shape={teacher_logits.shape}, teacher_indices shape={teacher_indices.shape}")
                seqlen = fp_inps.shape[1]
                
                # ============================================================
                # Phase B: Set quantization state and compute Pre-LWC Shift
                # ============================================================
                logger.info(f"[Expert Shift] Layer {i}: Setting quantization state...")
                
                # Set quantization state BEFORE computing expert shift
                # weight_quant=True: use quantized weights (temp_weight) to measure real quantization impact
                # act_quant=True: use quantized activations
                # This ensures we measure the true impact of quantization on router
                set_quant_state(qlayer, weight_quant=True, act_quant=True)
                
                # Create temp_weight ONCE for all samples (weights don't change during eval)
                smooth_and_quant_temporary(qlayer, args, is_llama)
                
                # Helper function to compute expert shift (temp_weight already set)
                def compute_shift_metrics(layer, inputs, teacher_idx, num_samples=None, desc=""):
                    """
                    Compute expert shift metrics using hook mechanism.
                    Assumes temp_weight is already set on the layer.
                    """
                    # 默认走全量样本统计，和你确认的“expert shift 默认全量”保持一致。
                    # 仅当调用方显式传入 num_samples 时，才会降采样。
                    if num_samples is None:
                        num_samples = inputs.shape[0]

                    shift_any_sum = 0.0
                    shift_half_sum = 0.0
                    shift_all_sum = 0.0
                    for j in range(min(num_samples, inputs.shape[0])):
                        # Use hook-based forward to get router logits
                        # Use autocast to handle dtype mismatch (input FP16, weights FP32)
                        with torch.amp.autocast('cuda'):
                            out, router_logits = forward_with_router_logits(
                                layer,
                                inputs[j].unsqueeze(0),
                                layer_kwargs=layer_kwargs,
                                attention_mask=attention_mask,
                                position_ids=position_ids
                            )
                        
                        if router_logits is not None:
                            if router_logits.dim() == 2:
                                router_logits = router_logits.unsqueeze(0)
                            
                            if teacher_idx.dim() == 2:
                                t_idx = teacher_idx[j*seqlen:(j+1)*seqlen].unsqueeze(0)
                            else:
                                t_idx = teacher_idx[j:j+1]
                            
                            metrics = compute_expert_shift_detailed(
                                router_logits, t_idx, k_routing, debug=(j == 0 and desc != "")
                            )
                            shift_any_sum += metrics["shift_any"]
                            shift_half_sum += metrics["shift_half"]
                            shift_all_sum += metrics["shift_all"]
                    
                    n = min(num_samples, inputs.shape[0])
                    return shift_any_sum / n, shift_half_sum / n, shift_all_sum / n
                
                # Compute Pre-LWC Expert Shift (temp_weight already set)
                with torch.no_grad():
                    pre_shift_any, pre_shift_half, pre_shift_all = compute_shift_metrics(
                        qlayer, quant_inps, teacher_indices, 
                        num_samples=args.nsamples, desc="Pre-LWC"
                    )
                    logger.info(f"[Expert Shift] Layer {i}: Pre-LWC Expert Shift (Quantized vs FP) - Any: {pre_shift_any:.4f}, Half: {pre_shift_half:.4f}, All: {pre_shift_all:.4f}")
                    pre_lwc_shift = (pre_shift_any, pre_shift_half, pre_shift_all)
                
                # ============================================================
                # Phase C: Router Calibration Training (Optional)
                # ============================================================
                # Only executed when calibrate_router=True
                # Goal: Train router to mimic FP16 expert selection UNDER QUANTIZED CONDITIONS
                # - Keep quantization enabled so router sees quantized hidden states
                # - temp_weight is already set from Phase B, no need to recreate
                if do_router_calibration:
                    logger.info(f"[Router Calibration] Layer {i}: Router calibration training (epochs={router_epochs}, lr={router_lr})...")
                    logger.info(f"[Router Calibration] Layer {i}: teacher_source=fp16_teacher")

                    # ======================== 关键修正（中文详细注释） ========================
                    # 这里先保存当前 requires_grad 状态，Ri 结束后需要完整恢复，
                    # 以保证后续 Mi 阶段（LWC/LoRA/shared_gate）不受污染。
                    saved_requires_grad = {
                        name: param.requires_grad for name, param in qlayer.named_parameters()
                    }

                    # 通过 stage-aware 白名单直接切换到 Ri：
                    # - 仅允许 router 主权重/偏置参与训练
                    # - 明确禁止 LoRA A/B、LET/LWC、shared gate 等参数更新
                    ri_named_params = set_stage_trainable_params(qlayer, stage="ri")

                    # 将白名单参数转为优化器参数列表，并做去重。
                    router_gate_params = []
                    seen_params = set()
                    for p_name, p in ri_named_params:
                        if id(p) in seen_params:
                            continue
                        seen_params.add(id(p))
                        router_gate_params.append(p)
                        logger.info(f"[Router Calibration] Layer {i}: Enabled gradient for {p_name}")
                    
                    if router_gate_params:
                        logger.info(f"[Router Calibration] Layer {i}: {len(router_gate_params)} unique parameters to optimize")
                        
                        # qlayer is already FP32 (converted at start of expert shift tracking)
                        router_optimizer = torch.optim.AdamW(router_gate_params, lr=router_lr, weight_decay=0)
                        
                        # Setup hook once for all training iterations using simple closure
                        hook_fn, get_logits, clear_logits = create_router_hook()
                        hook_handle = qlayer.mlp.gate.register_forward_hook(hook_fn)
                        
                        for epoch in range(router_epochs):
                            epoch_loss = 0.0
                            valid_samples = 0
                            for j in range(args.nsamples):
                                router_optimizer.zero_grad()
                                clear_logits()
                                
                                # Forward pass - qlayer is FP32, use autocast for efficiency
                                # Must call smooth_and_quant_temporary each iteration to recreate computation graph
                                with torch.amp.autocast('cuda'):
                                    smooth_and_quant_temporary(qlayer, args, is_llama)
                                    _ = call_layer_forward(
                                        qlayer,
                                        quant_inps[j].unsqueeze(0),
                                        layer_kwargs=layer_kwargs,
                                        attention_mask=attention_mask,
                                        position_ids=position_ids
                                    )
                                router_logits = get_logits()
                                
                                if router_logits is not None:
                                    if router_logits.dim() == 2:
                                        router_logits = router_logits.unsqueeze(0)
                                    
                                    if teacher_indices.dim() == 2:
                                        teacher_logit_sample = teacher_logits[j*seqlen:(j+1)*seqlen].unsqueeze(0)
                                        teacher_idx_sample = teacher_indices[j*seqlen:(j+1)*seqlen].unsqueeze(0)
                                    else:
                                        teacher_logit_sample = teacher_logits[j:j+1]
                                        teacher_idx_sample = teacher_indices[j:j+1]
                                    
                                    # Compute loss in FP32 for numerical stability
                                    loss = compute_topk_mse_loss(
                                        router_logits.float(),
                                        teacher_logit_sample,
                                        teacher_idx_sample,
                                        debug=(j == 0 and epoch == 0)
                                    )
                                    
                                    if not (torch.isnan(loss) or torch.isinf(loss)):
                                        loss.backward()
                                        
                                        # Clip gradients to prevent explosion
                                        torch.nn.utils.clip_grad_norm_(router_gate_params, max_norm=1.0)
                                        
                                        # Check for NaN gradients before stepping
                                        has_nan_grad = any(p.grad is not None and torch.isnan(p.grad).any() for p in router_gate_params)
                                        if not has_nan_grad:
                                            router_optimizer.step()
                                            epoch_loss += loss.item()
                                            valid_samples += 1
                            
                            avg_loss = epoch_loss / max(valid_samples, 1)
                            logger.info(f"[Router Calibration] Layer {i} Epoch {epoch}: TopK-MSE Loss = {avg_loss:.6f} ({valid_samples}/{args.nsamples} valid)")
                        
                        hook_handle.remove()
                        del router_optimizer
                    
                    # 恢复进入 Ri 之前的 requires_grad，确保阶段隔离。
                    for name, param in qlayer.named_parameters():
                        if name in saved_requires_grad:
                            param.requires_grad = saved_requires_grad[name]
                    
                    # ============================================================
                    # Phase D: Compute Post-Calibration Expert Shift
                    # temp_weight is still set from Phase B, reuse it
                    # ============================================================
                    with torch.no_grad():
                        post_shift_any, post_shift_half, post_shift_all = compute_shift_metrics(
                            qlayer, quant_inps, teacher_indices, 
                            num_samples=args.nsamples, desc=""
                        )
                        logger.info(f"[Router Calibration] Layer {i}: Post-Calib Expert Shift - Any: {post_shift_any:.4f}, Half: {post_shift_half:.4f}, All: {post_shift_all:.4f}")
                        logger.info(f"[Router Calibration] Layer {i}: Router Calib Improvement - Any: {pre_shift_any - post_shift_any:.4f}, Half: {pre_shift_half - post_shift_half:.4f}, All: {pre_shift_all - post_shift_all:.4f}")
                        post_calib_shift = (post_shift_any, post_shift_half, post_shift_all)
                    
                    logger.info(f"[Router Calibration] Layer {i}: Router calibration complete. Continuing to LWC training...")
                
                # Clean up temp_weight after expert shift tracking / router calibration
                clear_temp_variable(qlayer)
            else:
                logger.warning(f"[Expert Shift] Layer {i}: No router gate found, skipping expert shift tracking.")
        
        # ======================== Mi Teacher 构建（中文详细注释） ========================
        # 这里不再使用“整层最终输出”作为监督，而是使用 next block 中的中间锚点：
        #   layer i+1 的 post_attention_layernorm 输入。
        # 这个位置严格对应 Qwen3 forward 里的：
        #   hidden_states = residual + self_attn(...)
        #   residual = hidden_states
        #   hidden_states = post_attention_layernorm(hidden_states)
        # 即：我们监督的是 attention 残差相加后的流（进入 post_attention_layernorm 之前）。
        # 这样可以避免监督时机过早（比如仅抓 self_attn 原始输出）导致的语义偏差。
        use_next_attn_loss = is_qwen_moe and (next_layer_fp is not None) and (layer_stage == "Ai")
        teacher_next_attn_targets = None
        stage_fp_targets = fp_inps
        stage_fp_targets_2 = fp_inps_2
        if layer_stage == "A0" and stage_train_enabled:
            # A0 uses local targets only; global rolling buffers should not advance here.
            stage_fp_targets = torch.zeros_like(fp_inps)
            if args.aug_loss:
                stage_fp_targets_2 = torch.zeros_like(fp_inps_2)
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if stage_train_enabled:
            if use_next_attn_loss:
                teacher_next_attn_targets = torch.zeros_like(fp_inps)
                logger.info(f"[Mi] Layer {i}: teacher_source=fp16_teacher, loss_anchor=layer_{i+1}.post_attention_layernorm.input")

            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    for j in range(args.nsamples):
                        fp_layer_out = extract_hidden_states(call_layer_forward(
                            teacher_layer,
                            fp_inps[j].unsqueeze(0),
                            layer_kwargs=layer_kwargs,
                            attention_mask=attention_mask,
                            position_ids=position_ids
                        ))
                        if layer_stage == "A0":
                            stage_fp_targets[j] = fp_layer_out
                        else:
                            fp_inps[j] = fp_layer_out

                        if use_next_attn_loss:
                            # 设备一致性检查：teacher 目标张量和 next block 参数必须同设备。
                            # 否则会触发隐式拷贝或直接报错，训练过程不稳定。
                            assert_tensors_on_same_device(
                                {
                                    "fp_layer_out": fp_layer_out,
                                    "next_layer_weight": next(iter(next_layer_fp.parameters())).data,
                                },
                                logger=logger,
                                context=f"teacher-next-attn-layer-{i}",
                            )
                            # 通过 pre-hook 抓取 post_attention_layernorm 的输入，
                            # 保证 teacher 标签来自“attention残差相加之后”的精确时机。
                            _, teacher_attn_out = forward_with_module_input(
                                next_layer_fp,
                                fp_layer_out,
                                module_path="post_attention_layernorm",
                                layer_kwargs=layer_kwargs,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                detach_input=True,
                            )
                            teacher_next_attn_targets[j] = teacher_attn_out.squeeze(0).to(teacher_next_attn_targets.dtype)

                        if args.aug_loss:
                            fp_aug_out = extract_hidden_states(call_layer_forward(
                                teacher_layer,
                                quant_inps[j].unsqueeze(0),
                                layer_kwargs=layer_kwargs,
                                attention_mask=attention_mask,
                                position_ids=position_ids
                            ))
                            if layer_stage == "A0":
                                stage_fp_targets_2[j] = fp_aug_out
                            else:
                                fp_inps_2[j] = fp_aug_out
        # init smooth parameters
        set_quant_state(qlayer, weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        if is_llama or args.abits == 16:
            use_shift = False                   # deactivate channel-wise shifting for llama model and weight-only quantization
        if args.let:
            # init channel-wise scaling and shift
            qlayer.register_parameter("qkt_smooth_scale",torch.nn.Parameter(torch.ones(layer.self_attn.q_proj.out_features,device=dev, dtype=dtype)))
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.abs().max(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(args.alpha)/weight.pow(1-args.alpha)).clamp(min=1e-5)
                            if use_shift and not is_llama:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
                                
        if args.resume:
            qlayer.load_state_dict(omni_parameters[i], strict=False)
        

        if stage_train_enabled:
            with torch.no_grad():
                qlayer.float()      # required for AMP training
                if next_layer_student is not None:
                    next_layer_student.float()

            # Mi stage parameter whitelist: LET/LWC + optional shared gate + optional LoRA.
            set_stage_trainable_params(
                qlayer,
                stage="mi",
                use_shift=use_shift,
                train_shared_gate=(train_shared_gate and stage_gate_train_enabled),
                train_gate_lora=(train_gate_lora and stage_gate_train_enabled),
            )

            if next_layer_student is not None:
                # 下一层 attention 分支仅训练其量化参数（主要是 LWC 的 bound_factor）。
                # 这里不引入 shared_gate / LoRA，避免跨层参数语义污染。
                set_stage_trainable_params(
                    next_layer_student,
                    stage="mi",
                    use_shift=False,
                    train_shared_gate=False,
                    train_gate_lora=False,
                )

            # create optimizer with parameter groups
            # LET/LWC parameters use weight_decay=0 (fixed, not controlled by args.wd)
            param_groups = [
                {"params": let_parameters(qlayer, use_shift), "lr": args.let_lr, "weight_decay": 0},
                {"params": lwc_parameters(qlayer), "lr": args.lwc_lr, "weight_decay": 0}
            ]
            
            # Add shared_expert_gate parameters if training is enabled (uses args.wd)
            if train_shared_gate and stage_gate_train_enabled:
                shared_gate_params = []
                for name, module in qlayer.named_modules():
                    if name.endswith("shared_expert_gate") and isinstance(module, nn.Linear):
                        shared_gate_params.extend([p for p in module.parameters() if p.requires_grad])
                if shared_gate_params:
                    param_groups.append({"params": shared_gate_params, "lr": shared_gate_lr, "weight_decay": args.wd})
            
            # Add LoRA parameters if training is enabled (uses args.wd)
            if train_gate_lora and stage_gate_train_enabled:
                lora_params = []
                for name, module in qlayer.named_modules():
                    if isinstance(module, LoraLinear):
                        lora_params.extend([module.lora_A, module.lora_B])
                if lora_params:
                    param_groups.append({"params": lora_params, "lr": gate_lora_lr, "weight_decay": args.wd})

            # 将 next-attn 分支的量化参数加入优化器。
            if next_layer_student is not None:
                next_attn_params = list(get_omni_parameters(next_layer_student, use_shift=False))
                if next_attn_params:
                    param_groups.append({"params": next_attn_params, "lr": args.lwc_lr, "weight_decay": 0})
            
            # Default weight_decay=0 for optimizer (each group specifies its own)
            optimizer = torch.optim.AdamW(param_groups, weight_decay=0)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            # Collect all trainable parameters for gradient clipping
            # Start with quantization parameters (LET/LWC)
            clip_parameters = list(get_omni_parameters(qlayer, use_shift))
            
            # Add shared_expert_gate parameters if training is enabled
            if train_shared_gate and stage_gate_train_enabled:
                for name, module in qlayer.named_modules():
                    if name.endswith("shared_expert_gate") and isinstance(module, nn.Linear):
                        clip_parameters.extend([p for p in module.parameters() if p.requires_grad])
            
            # Add LoRA parameters if training is enabled
            if train_gate_lora and stage_gate_train_enabled:
                for name, module in qlayer.named_modules():
                    if isinstance(module, LoraLinear):
                        clip_parameters.extend([module.lora_A, module.lora_B])

            if next_layer_student is not None:
                clip_parameters.extend(list(get_omni_parameters(next_layer_student, use_shift=False)))
            
            # Log training configuration once per block (first layer only)
            if i == 0:
                if train_shared_gate and stage_gate_train_enabled:
                    logger.info(f"[Gate Training] shared_expert_gate training ENABLED with lr={shared_gate_lr}")
                if train_gate_lora and stage_gate_train_enabled:
                    logger.info(f"[Gate Training] router gate LoRA training ENABLED with r={lora_r}, alpha={lora_alpha}, lr={gate_lora_lr}")
                if not (train_shared_gate and stage_gate_train_enabled) and not (train_gate_lora and stage_gate_train_enabled):
                    logger.info("[Gate Training] All gate training DISABLED (default behavior)")
            
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):    
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with traincast():
                        smooth_and_quant_temporary(qlayer, args, is_llama)
                        quant_out = extract_hidden_states(call_layer_forward(
                            qlayer,
                            quant_inps[index:index+args.batch_size,],
                            layer_kwargs=layer_kwargs,
                            attention_mask=attention_mask_batch,
                            position_ids=position_ids
                        ))
                        if use_next_attn_loss:
                            # Student 分支同样在完全一致的锚点抓取，
                            # 保证 teacher/student 对齐到同一语义位置。
                            assert_tensors_on_same_device(
                                {
                                    "quant_out": quant_out,
                                    "teacher_next_attn_targets": teacher_next_attn_targets[index:index+args.batch_size,],
                                },
                                logger=logger,
                                context=f"mi-next-attn-loss-layer-{i}",
                            )
                            if next_layer_student is not None:
                                # 下一层 attention 学生分支同样需要临时量化权重。
                                smooth_and_quant_temporary(next_layer_student, args, is_llama)

                            _, student_next_attn_out = forward_with_module_input(
                                next_layer_student if next_layer_student is not None else next_layer_fp,
                                quant_out,
                                module_path="post_attention_layernorm",
                                layer_kwargs=layer_kwargs,
                                attention_mask=attention_mask_batch,
                                position_ids=position_ids,
                            )
                            loss = loss_func(
                                teacher_next_attn_targets[index:index+args.batch_size,],
                                student_next_attn_out,
                            )
                        else:
                            loss = loss_func(stage_fp_targets[index:index+args.batch_size,], quant_out)

                        if args.aug_loss and not use_next_attn_loss:
                            loss += loss_func(stage_fp_targets_2[index:index+args.batch_size,], quant_out)
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    # Use complete parameter list for gradient clipping
                    norm = loss_scaler(loss, optimizer, parameters=clip_parameters).cpu()
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                
                # Calculate and log gate training gradient norms
                gate_grad_info = ""
                shared_gate_grad_norm = 0.0
                lora_grad_norm = 0.0
                
                if train_shared_gate and stage_gate_train_enabled:
                    for name, module in qlayer.named_modules():
                        if name.endswith("shared_expert_gate") and isinstance(module, nn.Linear):
                            if module.weight.grad is not None:
                                shared_gate_grad_norm += module.weight.grad.norm().item() ** 2
                    shared_gate_grad_norm = shared_gate_grad_norm ** 0.5
                    gate_grad_info += f" shared_gate_grad:{shared_gate_grad_norm:.2e}"
                
                if train_gate_lora and stage_gate_train_enabled:
                    for name, module in qlayer.named_modules():
                        if isinstance(module, LoraLinear):
                            if module.lora_A.grad is not None:
                                lora_grad_norm += module.lora_A.grad.norm().item() ** 2
                            if module.lora_B.grad is not None:
                                lora_grad_norm += module.lora_B.grad.norm().item() ** 2
                    lora_grad_norm = lora_grad_norm ** 0.5
                    gate_grad_info += f" lora_grad:{lora_grad_norm:.2e}"
                
                logger.info(
                    f"stage {layer_stage} layer {i} iter {epochs} "
                    f"loss:{loss_mean} norm:{norm_mean}{gate_grad_info} "
                    f"max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} "
                )
                final_loss = loss_mean.item()  # always update; after loop ends this holds last layer's last epoch loss
                
                # WandB logging: Log metrics with strict naming schema for Router Collapse detection
                if wandb is not None:
                    # Extract learning rates from optimizer param_groups
                    lr_shared_gate = None
                    lr_router_lora = None
                    for pg in optimizer.param_groups:
                        # Identify groups by checking if they contain shared_gate or lora params
                        if len(pg['params']) > 0:
                            # Check if this is the shared gate group (3rd group, index 2)
                            if train_shared_gate and lr_shared_gate is None:
                                for name, module in qlayer.named_modules():
                                    if name.endswith("shared_expert_gate") and isinstance(module, nn.Linear):
                                        for p in module.parameters():
                                            if p.requires_grad and any(p is pp for pp in pg['params']):
                                                lr_shared_gate = pg['lr']
                                                break
                            # Check if this is the lora group (4th group, index 3)
                            if train_gate_lora and lr_router_lora is None:
                                for name, module in qlayer.named_modules():
                                    if isinstance(module, LoraLinear):
                                        if any(module.lora_A is pp or module.lora_B is pp for pp in pg['params']):
                                            lr_router_lora = pg['lr']
                                            break
                    
                    # Build metrics dict with strict naming schema
                    wandb_metrics = {
                        "train/loss": loss_mean.item(),
                        "train/layer_id": i,
                        "train/epoch": epochs,
                        "train/grad_norm_mean": norm_mean.item(),
                    }
                    
                    # Add gradient norms for Router Collapse detection
                    if train_shared_gate and stage_gate_train_enabled:
                        wandb_metrics["grad/shared_expert_norm"] = shared_gate_grad_norm
                    if train_gate_lora and stage_gate_train_enabled:
                        wandb_metrics["grad/router_lora_norm"] = lora_grad_norm
                    
                    # Add learning rates for hyperparameter tracking
                    if lr_shared_gate is not None:
                        wandb_metrics["lr/shared_gate"] = lr_shared_gate
                    if lr_router_lora is not None:
                        wandb_metrics["lr/router_lora"] = lr_router_lora
                    if calibrate_router:
                        wandb_metrics["lr/router_lr"] = router_lr
                    
                    wandb.log(wandb_metrics, step=global_step)
                    global_step += 1
            clear_temp_variable(qlayer)
            if next_layer_student is not None:
                clear_temp_variable(next_layer_student)
            del optimizer
            
            # Merge LoRA weights back into original Linear layers after training
            if train_gate_lora and stage_gate_train_enabled:
                for name, module in list(qlayer.named_modules()):
                    if isinstance(module, LoraLinear):
                        merged_linear = module.merge()
                        add_new_module(name, qlayer, merged_linear)
                        logger.info(f"Merged LoRA weights for {name}")
        
        # =================================================================
        # Post-LWC Expert Shift Check (always for Qwen MoE)
        # =================================================================
        if is_qwen_moe and cached_router_labels is not None:
            teacher_logits, teacher_indices = cached_router_labels
            seqlen = fp_inps.shape[1]
            
            # Create temp_weight ONCE for all samples
            smooth_and_quant_temporary(qlayer, args, is_llama)
            
            with torch.no_grad():
                post_lwc_shift_any_sum = 0.0
                post_lwc_shift_half_sum = 0.0
                post_lwc_shift_all_sum = 0.0
                num_samples = args.nsamples
                for j in range(num_samples):
                    # Use hook-based forward to get router logits
                    with torch.amp.autocast('cuda'):
                        out, router_logits = forward_with_router_logits(
                            qlayer,
                            quant_inps[j].unsqueeze(0),
                            layer_kwargs=layer_kwargs,
                            attention_mask=attention_mask,
                            position_ids=position_ids
                        )
                    
                    if router_logits is not None:
                        # Reshape router_logits if needed
                        if router_logits.dim() == 2:
                            router_logits = router_logits.unsqueeze(0)
                        
                        # Get correct teacher indices for this sample
                        if teacher_indices.dim() == 2:
                            teacher_idx_sample = teacher_indices[j*seqlen:(j+1)*seqlen].unsqueeze(0)
                        else:
                            teacher_idx_sample = teacher_indices[j:j+1]
                        
                        shift_metrics = compute_expert_shift_detailed(
                            router_logits,
                            teacher_idx_sample,
                            k_routing
                        )
                        post_lwc_shift_any_sum += shift_metrics["shift_any"]
                        post_lwc_shift_half_sum += shift_metrics["shift_half"]
                        post_lwc_shift_all_sum += shift_metrics["shift_all"]
                
                post_lwc_shift_any = post_lwc_shift_any_sum / num_samples
                post_lwc_shift_half = post_lwc_shift_half_sum / num_samples
                post_lwc_shift_all = post_lwc_shift_all_sum / num_samples
                logger.info(f"[Expert Shift] Layer {i}: Post-LWC Expert Shift - Any: {post_lwc_shift_any:.4f}, Half: {post_lwc_shift_half:.4f}, All: {post_lwc_shift_all:.4f}")
                
                # Log improvement from Pre-LWC to Post-LWC
                if pre_lwc_shift is not None:
                    lwc_improvement_any = pre_lwc_shift[0] - post_lwc_shift_any
                    lwc_improvement_half = pre_lwc_shift[1] - post_lwc_shift_half
                    lwc_improvement_all = pre_lwc_shift[2] - post_lwc_shift_all
                    logger.info(f"[Expert Shift] Layer {i}: LWC Improvement (Pre-LWC -> Post-LWC) - Any: {lwc_improvement_any:.4f}, Half: {lwc_improvement_half:.4f}, All: {lwc_improvement_all:.4f}")
                    
                    # Collect data for visualization (3 phases)
                    expert_shift_data.append({
                        "layer": i,
                        "initial_any": pre_lwc_shift[0],
                        "initial_half": pre_lwc_shift[1],
                        "initial_all": pre_lwc_shift[2],
                        "post_calib_any": post_calib_shift[0] if post_calib_shift else None,
                        "post_calib_half": post_calib_shift[1] if post_calib_shift else None,
                        "post_calib_all": post_calib_shift[2] if post_calib_shift else None,
                        "post_lwc_any": post_lwc_shift_any,
                        "post_lwc_half": post_lwc_shift_half,
                        "post_lwc_all": post_lwc_shift_all,
                    })
            
            # Clean up temp_weight
            clear_temp_variable(qlayer)
        
        qlayer.half() 
        # real smooth and quantization
        smooth_and_quant_inplace(qlayer, args, is_llama)
        if args.epochs>0:
            # update input of quantization model
            if should_roll_buffers:
                with torch.no_grad():
                    # with torch.cuda.amp.autocast():
                    with traincast():
                        for j in range(args.nsamples):
                            quant_inps[j] = extract_hidden_states(call_layer_forward(
                                qlayer,
                                quant_inps[j].unsqueeze(0),
                                layer_kwargs=layer_kwargs,
                                attention_mask=attention_mask,
                                position_ids=position_ids
                            ))
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")

        # Keep model body on CPU and only pin two active blocks on GPU.
        if next_layer_fp is not None:
            if next_layer_student is not None:
                # 将训练后的 next-attn 学生分支写回下一层，供后续阶段继续使用。
                register_scales_and_zeros(next_layer_student)
                layers[i + 1] = next_layer_student.to("cpu")
                del next_layer_student
            else:
                layers[i + 1] = next_layer_fp.to("cpu")
            del next_layer_fp

        if args.real_quant:
            assert args.wbits in [2,3,4] and args.abits >= 16   # only support weight-only quantization
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                if args.wbits == 3:
                    q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                else:
                    q_linear = qlinear_triton.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer, q_linear)       
                print(f"pack quantized {name} finished")
                del module        
        del layer

        # teacher 快照用完后立即回迁 CPU，避免多层 teacher 常驻 GPU 导致显存累计。
        fp_teacher_layers[i] = teacher_layer.to("cpu")
        del teacher_layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    
    # === WandB Expert Shift Visualization (Matplotlib, 3-Phase Line Charts) ===
    # 3 separate charts: shift_any, shift_half, shift_all
    # Each chart: X=layer_id, lines=Initial / Post-Calib(if exists) / Post-LWC
    if wandb is not None and len(expert_shift_data) > 0:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            has_calib = any(e.get("post_calib_any") is not None for e in expert_shift_data)
            layers = [e["layer"] for e in expert_shift_data]

            phase_defs = [
                ("initial", "Initial", {"color": "#1f77b4", "linestyle": "-", "marker": "o"}),
                ("post_calib", "Post-Calib", {"color": "#ff7f0e", "linestyle": "--", "marker": "s"}),
                ("post_lwc", "Post-LWC", {"color": "#2ca02c", "linestyle": "-.", "marker": "^"}),
            ]

            fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
            for metric, ax in zip(["any", "half", "all"], axes):
                for phase_key, phase_label, style in phase_defs:
                    if phase_key == "post_calib" and not has_calib:
                        continue
                    xs = []
                    ys = []
                    for entry in expert_shift_data:
                        value = entry.get(f"{phase_key}_{metric}")
                        if value is None:
                            continue
                        xs.append(entry["layer"])
                        ys.append(value)
                    if xs:
                        ax.plot(xs, ys, label=phase_label, **style)
                ax.set_title(f"shift_{metric}")
                ax.set_xlabel("Layer")
                ax.set_ylabel("Shift")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best")

            wandb.log({
                "expert_shift/three_phase": wandb.Image(
                    fig, caption="Expert Shift: Initial vs Post-Calib vs Post-LWC"
                )
            })
            plt.close(fig)
            logger.info(f"[Expert Shift] Uploaded Matplotlib visualization to WandB ({len(expert_shift_data)} layers)")
        except ImportError:
            logger.warning("[Expert Shift] Matplotlib not installed; skipping custom visualization.")
        except Exception as e:
            logger.warning(f"[Expert Shift] Failed to create Matplotlib visualization: {e}")
    
    return model, final_loss

