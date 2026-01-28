import torch
import torch.nn as nn
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
from quantize.utils import let_parameters, lwc_parameters, get_omni_parameters,\
                            omni_state_dict, register_scales_and_zeros,smooth_and_quant_temporary,\
                            smooth_and_quant_inplace,clear_temp_variable,set_quant_state


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
):
    logger.info("Starting ...")
    
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
        # Qwen2MoE only supports LWC quantization, no DecoderLayer needed
        layer_name_prefix = "model.layers"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral/qwen now")
    
    
    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
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
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
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
    if is_llama:
        position_ids = cache["position_ids"]
    else:
        position_ids = None



    if args.resume:
        omni_parameters = torch.load(args.resume, weights_only=False)
    else:
        omni_parameters = {}

    
    
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        if "mixtral" in args.net.lower() or "qwen" in args.net.lower():  
            # For MoE models (mixtral, qwen2moe), only LWC is supported
            # Simply replace Linear with QuantLinear, do not quantize router (gate)
            qlayer = copy.deepcopy(layer)
            
            for name, module in qlayer.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # Target 1: Shared Expert Gate - name ends with "shared_expert_gate"
                    is_shared_expert_gate = name.endswith("shared_expert_gate")
                    
                    # Target 2: Router Gate - name ends with ".gate" (NOT "gate_proj")
                    # e.g., "mlp.gate" is router, but "mlp.experts.0.gate_proj" is NOT
                    is_router_gate = name.endswith(".gate") or name == "gate"
                    
                    if is_shared_expert_gate:
                        if train_shared_gate:
                            # Keep as nn.Linear but make trainable
                            module.weight.requires_grad = True
                            if module.bias is not None:
                                module.bias.requires_grad = True
                        # else: skip, keep as frozen nn.Linear (default behavior)
                    elif is_router_gate:
                        if train_gate_lora:
                            # Replace with LoraLinear wrapper (use layer index as seed for reproducibility)
                            lora_linear = LoraLinear(module, r=lora_r, alpha=lora_alpha, seed=args.seed + i)
                            add_new_module(name, qlayer, lora_linear)
                        # else: skip, keep as frozen nn.Linear (default behavior)
                    else:
                        # Target 3: All other linear layers (including gate_proj)
                        # Replace with QuantLinear
                        quantlinear = QuantLinear(module, args.weight_quant_params, args.act_quant_params)
                        add_new_module(name, qlayer, quantlinear)    
        else:
            qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)

        
        # obtain output of full-precision model
        set_quant_state(qlayer, weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
                        if args.aug_loss:
                            fp_inps_2[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
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
        

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training
            # create optimizer with parameter groups
            param_groups = [
                {"params": let_parameters(qlayer, use_shift), "lr": args.let_lr},
                {"params": lwc_parameters(qlayer), "lr": args.lwc_lr}
            ]
            
            # Add shared_expert_gate parameters if training is enabled
            if train_shared_gate:
                shared_gate_params = []
                for name, module in qlayer.named_modules():
                    if name.endswith("shared_expert_gate") and isinstance(module, nn.Linear):
                        shared_gate_params.extend([p for p in module.parameters() if p.requires_grad])
                if shared_gate_params:
                    param_groups.append({"params": shared_gate_params, "lr": shared_gate_lr})
            
            # Add LoRA parameters if training is enabled
            if train_gate_lora:
                lora_params = []
                for name, module in qlayer.named_modules():
                    if isinstance(module, LoraLinear):
                        lora_params.extend([module.lora_A, module.lora_B])
                if lora_params:
                    param_groups.append({"params": lora_params, "lr": gate_lora_lr})
            
            optimizer = torch.optim.AdamW(param_groups, weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            # Log training configuration once per block (first layer only)
            if i == 0:
                if train_shared_gate:
                    logger.info(f"[Gate Training] shared_expert_gate training ENABLED with lr={shared_gate_lr}")
                if train_gate_lora:
                    logger.info(f"[Gate Training] router gate LoRA training ENABLED with r={lora_r}, alpha={lora_alpha}, lr={gate_lora_lr}")
                if not train_shared_gate and not train_gate_lora:
                    logger.info("[Gate Training] All gate training DISABLED (default behavior)")
            
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):    
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with traincast():
                        smooth_and_quant_temporary(qlayer, args, is_llama)
                        quant_out = qlayer(quant_inps[index:index+args.batch_size,], attention_mask=attention_mask_batch,position_ids=position_ids)[0]
                        loss = loss_func(fp_inps[index:index+args.batch_size,], quant_out)
                        if args.aug_loss:
                            loss += loss_func(fp_inps_2[index:index+args.batch_size,], quant_out)
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    loss_list.append(loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters= get_omni_parameters(qlayer, use_shift)).cpu()
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                
                # Calculate and log gate training gradient norms
                gate_grad_info = ""
                if train_shared_gate:
                    shared_gate_grad_norm = 0.0
                    for name, module in qlayer.named_modules():
                        if name.endswith("shared_expert_gate") and isinstance(module, nn.Linear):
                            if module.weight.grad is not None:
                                shared_gate_grad_norm += module.weight.grad.norm().item() ** 2
                    shared_gate_grad_norm = shared_gate_grad_norm ** 0.5
                    gate_grad_info += f" shared_gate_grad:{shared_gate_grad_norm:.2e}"
                
                if train_gate_lora:
                    lora_grad_norm = 0.0
                    for name, module in qlayer.named_modules():
                        if isinstance(module, LoraLinear):
                            if module.lora_A.grad is not None:
                                lora_grad_norm += module.lora_A.grad.norm().item() ** 2
                            if module.lora_B.grad is not None:
                                lora_grad_norm += module.lora_B.grad.norm().item() ** 2
                    lora_grad_norm = lora_grad_norm ** 0.5
                    gate_grad_info += f" lora_grad:{lora_grad_norm:.2e}"
                
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean}{gate_grad_info} max memory_allocated {torch.cuda.max_memory_allocated(lm._device) / 1024**2} ")
            clear_temp_variable(qlayer)
            del optimizer
            
            # Merge LoRA weights back into original Linear layers after training
            if train_gate_lora:
                for name, module in list(qlayer.named_modules()):
                    if isinstance(module, LoraLinear):
                        merged_linear = module.merge()
                        add_new_module(name, qlayer, merged_linear)
                        logger.info(f"Merged LoRA weights for {name}")
        
        qlayer.half() 
        # real smooth and quantization
        smooth_and_quant_inplace(qlayer, args, is_llama)
        if args.epochs>0:
            # update input of quantization model
            with torch.no_grad():
                # with torch.cuda.amp.autocast():
                with traincast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            omni_parameters[i] = omni_state_dict(qlayer)
            torch.save(omni_parameters, os.path.join(args.output_dir, f"omni_parameters.pth"))
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
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
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

