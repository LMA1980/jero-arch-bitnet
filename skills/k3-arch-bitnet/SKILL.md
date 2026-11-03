---
name: k3-arch-bitnet
description: "Implement and validate BitNet (bitnet, bitnet-b1.58, and bitnet-25) ArchRecipe execution, SubLN int8 dynamic quantization, and on-the-fly TQ1_0/TQ2_0 BitLinear forward passes in k3-arch-bitnet."
---

# `k3-arch-bitnet` Architecture Recipe Invariants

- Implement `k3_core::ArchRecipe` for BitNet architectures (`bitnet`, `bitnet-b1.58`, and `bitnet-25`), enforcing mandatory SubLN (`subln_quantize_absmax_i8`) strictly prior to every `BitLinear` projection (`attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate`, `ffn_up`, `ffn_down`).
- Support both `ReLU^2` (`use_relu2: true` via `relu_squared_gate_f32`) and `SwiGLU` (`use_relu2: false` via `swiglu_f32`) gated FFN blocks.
- Support `QuantType::TQ1_0`, `QuantType::TQ2_0`, and `QuantType::Q8_0` weights, maintaining zero heap expansion and delegating to `kernels-rvv` (`bitlinear_i8_ime` with A100 `smt.vmadot` and `bitlinear_i8_rvv` with X100 RVV).
- Use `usize` for tensor slice indexing and memory offsets, and `u32` for architectural hyperparameters.
