import torch
import torch.nn as nn
from types import SimpleNamespace

# 1. 模拟 QuantLinear 类
# 我们不需要真正的量化逻辑，只需要记录传入了什么参数
class MockQuantLinear:
    def __init__(self, module, weight_params, act_params):
        self.module = module
        self.weight_params = weight_params
        self.act_params = act_params
        # 方便后续打印验证
        self.n_bits = weight_params.get('n_bits', 'Unknown')

# 2. 测试主函数
def test_logic():
    print("=== 开始测试量化位宽分配逻辑 ===\n")

    # --- A. 模拟命令行参数 (args) ---
    args = SimpleNamespace()
    args.wbits = 4          # 默认位宽 (MLP等)
    args.attn_wbits = 8     # Attention 层特定位宽
    
    # 其他无关参数
    args.symmetric = False
    args.w_dynamic_method = "per_channel"
    args.group_size = 128
    args.lwc = True
    args.disable_zero_point = False
    args.act_quant_params = {} # 占位

    # --- B. 构造参数字典 (模拟 main.py 中的逻辑) ---
    print(f"设置: 默认 wbits={args.wbits}, Attention wbits={args.attn_wbits}")
    
    # 基础参数字典
    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc": args.lwc,
        "disable_zero_point": args.disable_zero_point
    }

    # Attention 参数字典 (验证字典解包覆盖逻辑)
    if args.attn_wbits is not None:
        args.attn_weight_quant_params = {
            **args.weight_quant_params, # 复制基础参数
            "n_bits": args.attn_wbits   # 覆盖 n_bits
        }
    else:
        args.attn_weight_quant_params = None
    
    print(f"-> 默认 params n_bits: {args.weight_quant_params['n_bits']}")
    print(f"-> Attn params n_bits: {args.attn_weight_quant_params['n_bits']}")
    print("-" * 60)

    # --- C. 定义测试用例 (层名称 vs 期望位宽) ---
    # 这些名称是相对于 DecoderLayer 的，正如 omniquant.py 中遍历的那样
    test_cases = [
        # 应该匹配 Attention 逻辑 (8bit)
        ("self_attn.q_proj", 8),
        ("self_attn.k_proj", 8),
        ("self_attn.v_proj", 8),
        ("self_attn.o_proj", 8),
        
        # 应该走默认逻辑 (4bit)
        ("mlp.gate_proj", 4),
        ("mlp.up_proj", 4),
        ("mlp.down_proj", 4),
        
        # MoE 结构测试
        ("mlp.experts.0.gate_proj", 4),
        
        # 边界情况：名字里有 self_attn 但不是投影层
        ("self_attn.rotary_emb", 4), 
        ("self_attn.dense", 4),      # 假设这是一个不在集合里的层
        
        # 其他层
        ("input_layernorm", 4),
    ]

    # --- D. 执行你的逻辑 ---
    for name, expected_bits in test_cases:
        # 模拟一个 module (只是为了传参，类型无所谓)
        module = nn.Linear(32, 32) 
        
        # =========== 你的核心逻辑 ===========
        is_attn_linear = name.startswith("self_attn.") and name.split(".")[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"}
        
        weight_params = args.attn_weight_quant_params if (is_attn_linear and args.attn_weight_quant_params is not None) else args.weight_quant_params
        
        quantlinear = MockQuantLinear(module, weight_params, args.act_quant_params)
        # ====================================

        # 验证结果
        actual_bits = quantlinear.n_bits
        result_str = "通过" if actual_bits == expected_bits else "失败"
        
        print(f"层名称: {name:<25} | 判定为Attn: {str(is_attn_linear):<5} | 分配位宽: {actual_bits} (期望: {expected_bits}) | {result_str}")

if __name__ == "__main__":
    test_logic()