"""Compact frozen expert weights for Gemma adapter tuning on a Colab L4.

Forward math is BF16. FP8 is a storage format for the frozen expert matrices.
The deployable checkpoint is exported from the original compact BF16 weights
plus the learned adapter.
"""

from __future__ import annotations


def pack_experts_for_l4(model) -> int:
    import torch
    from torch import nn
    from torch.nn import functional as F

    class FrozenExperts(nn.Module):
        def __init__(self, source) -> None:
            super().__init__()
            self.act_fn = source.act_fn
            self.num_experts = source.num_experts
            for name in ("gate_up_proj", "down_proj"):
                packed, scales = pack(getattr(source, name))
                self.register_buffer(name, packed, persistent=False)
                self.register_buffer(f"{name}_scale", scales, persistent=False)

        def forward(self, hidden_states, top_k_index, top_k_weights):
            result = torch.zeros_like(hidden_states)
            with torch.no_grad():
                assignments = torch.nn.functional.one_hot(top_k_index, self.num_experts).permute(2, 1, 0)
                active = assignments.any(dim=(-1, -2)).nonzero().flatten().tolist()
            dtype = hidden_states.dtype
            for expert in active:
                top_pos, token_pos = torch.where(assignments[expert])
                gate_up = self.gate_up_proj[expert].to(dtype) * self.gate_up_proj_scale[expert].to(dtype)
                down = self.down_proj[expert].to(dtype) * self.down_proj_scale[expert].to(dtype)
                gate, up = F.linear(hidden_states[token_pos], gate_up).chunk(2, dim=-1)
                values = F.linear(self.act_fn(gate) * up, down)
                values = (values * top_k_weights[token_pos, top_pos, None]).to(dtype)
                result.index_add_(0, token_pos, values)
            return result

    @torch.no_grad()
    def pack(weight):
        packed = torch.empty_like(weight, device="cuda", dtype=torch.float8_e4m3fn)
        scales = torch.empty((weight.shape[0], 1, 1), device="cuda", dtype=torch.float32)
        largest = float(torch.finfo(packed.dtype).max)
        for start in range(0, weight.shape[0], 4):
            block = weight[start:start + 4].float()
            scale = block.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6) / largest
            packed[start:start + 4].copy_((block / scale).to(dtype=packed.dtype))
            scales[start:start + 4].copy_(scale)
        return packed, scales

    layers = model.model.language_model.layers
    expert_weights = [p for layer in layers for p in layer.experts.parameters()]
    packed_bytes = sum(p.numel() for p in expert_weights)
    remaining = sum(p.numel() * p.element_size() for p in model.parameters()) - 2 * packed_bytes
    available, _total = torch.cuda.mem_get_info()
    reserve = 2 << 30
    if available < packed_bytes + remaining + reserve:
        raise RuntimeError(
            f"Packed expert fit needs {packed_bytes + remaining + reserve} bytes including activation reserve. "
            f"CUDA reports {available} bytes available."
        )
    del expert_weights
    for layer in layers:
        layer.experts = FrozenExperts(layer.experts)
    model.to("cuda")
    return packed_bytes
