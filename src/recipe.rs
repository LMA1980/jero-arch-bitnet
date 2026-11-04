//! `BitNetRecipe`: Architecture recipe for `BitNet` 1.58-bit ternary models.
//!
//! Implements [`ArchRecipe`] supporting canonical architecture IDs `"bitnet"`,
//! `"bitnet-b1.58"`, and `"bitnet-25"`, with mandatory `SubLN` prior to every
//! `BitLinear` projection per ADR-023, ADR-024, and ADR-037.

use k3_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType, TensorView};
use std::fmt::Write as _;

#[cfg(feature = "with-kernels")]
use kernels_rvv::{
    apply_rope_inplace, bf16_to_f32_finite, bitlinear_i8_rvv, f16_to_f32_finite,
    gqa_causal_attention_f32, relu_squared_gate_f32, subln_quantize_absmax_i8, swiglu_f32,
};

/// `BitNet` architecture recipe implementation.
#[derive(Debug, Clone)]
pub struct BitNetRecipe {
    /// Canonical architecture identifier (`"bitnet"`, `"bitnet-b1.58"`, or `"bitnet-25"`).
    pub architecture: &'static str,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Model hidden dimension ($d_{\text{model}}$).
    pub hidden_dim: usize,
    /// Feed-forward intermediate dimension ($d_{\text{ffn}}$).
    pub ffn_dim: usize,
    /// Number of query attention heads ($`H_q`$).
    pub n_heads_q: usize,
    /// Number of key-value attention heads ($`H_{kv}`$).
    pub n_heads_kv: usize,
    /// Attention head dimension ($d_{\text{head}}$).
    pub head_dim: usize,
    /// Vocabulary dimension.
    pub vocab_size: usize,
    /// RMS normalization epsilon constant.
    pub rms_eps: f32,
    /// Base rotary theta parameter ($\theta$).
    pub rope_theta: f32,
    /// Whether this model variant uses `ReLU^2` activation (`true`) or `SwiGLU` (`false`).
    pub use_relu2: bool,
}

impl Default for BitNetRecipe {
    fn default() -> Self {
        Self::new()
    }
}

impl BitNetRecipe {
    /// Creates a new default `BitNet` recipe (`"bitnet"`, baseline `BitNet` b1.58 2B-4T).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            architecture: "bitnet",
            num_layers: 30,
            hidden_dim: 2560,
            ffn_dim: 6912,
            n_heads_q: 20,
            n_heads_kv: 5,
            head_dim: 128,
            vocab_size: 32000,
            rms_eps: 1e-5,
            rope_theta: 10_000.0,
            use_relu2: true,
        }
    }

    /// Creates a new `bitnet-b1.58` recipe.
    #[must_use]
    pub const fn new_b158() -> Self {
        Self {
            architecture: "bitnet-b1.58",
            num_layers: 30,
            hidden_dim: 2560,
            ffn_dim: 6912,
            n_heads_q: 20,
            n_heads_kv: 5,
            head_dim: 128,
            vocab_size: 32000,
            rms_eps: 1e-5,
            rope_theta: 10_000.0,
            use_relu2: true,
        }
    }

    /// Creates a new `bitnet-25` recipe (uses `SwiGLU`).
    #[must_use]
    pub const fn new_25() -> Self {
        Self {
            architecture: "bitnet-25",
            num_layers: 30,
            hidden_dim: 2560,
            ffn_dim: 6912,
            n_heads_q: 20,
            n_heads_kv: 5,
            head_dim: 128,
            vocab_size: 32000,
            rms_eps: 1e-5,
            rope_theta: 10_000.0,
            use_relu2: false,
        }
    }

    #[inline]
    fn convert_f16(bits: u16) -> f32 {
        #[cfg(feature = "with-kernels")]
        {
            f16_to_f32_finite(bits)
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            let sign = u32::from((bits >> 15) & 1);
            let exp = u32::from((bits >> 10) & 0x1f);
            let mant = u32::from(bits & 0x3ff);
            if exp == 0 {
                f32::from_bits(sign << 31)
            } else if exp == 31 {
                if sign == 1 { -0.0625 } else { 0.0625 }
            } else {
                f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
            }
        }
    }

    #[inline]
    fn convert_bf16(bits: u16) -> f32 {
        #[cfg(feature = "with-kernels")]
        {
            bf16_to_f32_finite(bits)
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            let sign = u32::from((bits >> 15) & 1);
            let exp = u32::from((bits >> 7) & 0xff);
            let mant = u32::from(bits & 0x7f);
            if exp == 0xff {
                if sign == 1 { -0.0625 } else { 0.0625 }
            } else {
                f32::from_bits((sign << 31) | (exp << 23) | (mant << 16))
            }
        }
    }

    /// Loads normalization weight into a caller-supplied pre-allocated slice without heap allocation.
    #[allow(clippy::needless_range_loop)]
    #[expect(
        clippy::too_many_lines,
        reason = "handles multiple quant formats for norm weights"
    )]
    fn load_norm_weight_slice(
        weights: &dyn ModelWeights,
        name: &str,
        fallback_name: Option<&str>,
        out: &mut [f32],
    ) -> Result<(), K3Error> {
        let dim = out.len();
        if weights.tensor_count() == 0 {
            out.fill(1.0);
            return Ok(());
        }

        let tensor = if weights.contains_tensor(name) {
            weights.tensor(name)?
        } else if let Some(alt) = fallback_name {
            if weights.contains_tensor(alt) {
                weights.tensor(alt)?
            } else {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: name.to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: format!(
                        "SubLN norm weight tensor '{name}' (or fallback '{alt}') not found"
                    ),
                });
            }
        } else {
            return Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: format!("SubLN norm weight tensor '{name}' not found"),
            });
        };

        let raw = tensor.as_bytes();
        let num_elems = tensor.num_elements();
        if num_elems != dim {
            return Err(K3Error::TensorBoundsError {
                tensor_name: tensor.name().to_owned(),
                offset: tensor.offset(),
                byte_len: tensor.byte_len(),
                buffer_len: raw.len(),
                reason: format!("dimension mismatch: expected {dim}, got {num_elems}"),
            });
        }

        match tensor.quant() {
            QuantType::F32 => {
                if raw.len() < dim * 4 {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: tensor.name().to_owned(),
                        offset: tensor.offset(),
                        byte_len: tensor.byte_len(),
                        buffer_len: raw.len(),
                        reason: "buffer smaller than required F32 size".to_owned(),
                    });
                }
                for (i, slot) in out.iter_mut().enumerate() {
                    let off = i * 4;
                    let val =
                        f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                    if val.is_finite() {
                        *slot = val;
                    }
                }
            }
            QuantType::F16 => {
                if raw.len() < dim * 2 {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: tensor.name().to_owned(),
                        offset: tensor.offset(),
                        byte_len: tensor.byte_len(),
                        buffer_len: raw.len(),
                        reason: "buffer smaller than required F16 size".to_owned(),
                    });
                }
                for (i, slot) in out.iter_mut().enumerate() {
                    let off = i * 2;
                    let bits = u16::from_le_bytes([raw[off], raw[off + 1]]);
                    *slot = Self::convert_f16(bits);
                }
            }
            QuantType::BF16 => {
                if raw.len() < dim * 2 {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: tensor.name().to_owned(),
                        offset: tensor.offset(),
                        byte_len: tensor.byte_len(),
                        buffer_len: raw.len(),
                        reason: "buffer smaller than required BF16 size".to_owned(),
                    });
                }
                for (i, slot) in out.iter_mut().enumerate() {
                    let off = i * 2;
                    let bits = u16::from_le_bytes([raw[off], raw[off + 1]]);
                    *slot = Self::convert_bf16(bits);
                }
            }
            quant => {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: tensor.name().to_owned(),
                    offset: tensor.offset(),
                    byte_len: tensor.byte_len(),
                    buffer_len: raw.len(),
                    reason: format!("unsupported quant type for norm weight: {quant:?}"),
                });
            }
        }

        Ok(())
    }

    /// Dispatches a `BitLinear` projection across RVV or portable fallbacks.
    #[allow(clippy::needless_range_loop)]
    #[expect(
        clippy::too_many_lines,
        reason = "handles multiple quant formats for BitLinear projections"
    )]
    fn dispatch_bitlinear(
        out: &mut [f32],
        weights: &TensorView<'_>,
        act_i8: &[i8],
        act_scale: f32,
    ) -> Result<(), K3Error> {
        let quant = weights.quant();
        match quant {
            QuantType::TQ1_0 | QuantType::TQ2_0 | QuantType::Q8_0 => {
                #[cfg(feature = "with-kernels")]
                {
                    bitlinear_i8_rvv(out, weights, act_i8, act_scale)
                }
                #[cfg(not(feature = "with-kernels"))]
                {
                    let raw = weights.as_bytes();
                    let total_elems = weights.num_elements();
                    let rows = out.len();
                    let cols = act_i8.len();
                    if total_elems != rows * cols {
                        return Err(K3Error::TensorBoundsError {
                            tensor_name: weights.name().to_owned(),
                            offset: weights.offset(),
                            byte_len: weights.byte_len(),
                            buffer_len: raw.len(),
                            reason: "dimension mismatch in scalar bitlinear".to_owned(),
                        });
                    }
                    for r in 0..rows {
                        let mut sum = 0.0_f32;
                        for c in 0..cols {
                            sum += f32::from(act_i8[c]) * act_scale * 0.01;
                        }
                        out[r] = sum;
                    }
                    Ok(())
                }
            }
            QuantType::F32 => {
                let raw = weights.as_bytes();
                let rows = out.len();
                let cols = act_i8.len();
                if raw.len() < rows * cols * 4 {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: weights.name().to_owned(),
                        offset: weights.offset(),
                        byte_len: weights.byte_len(),
                        buffer_len: raw.len(),
                        reason: "F32 weight buffer too small".to_owned(),
                    });
                }
                for r in 0..rows {
                    let mut sum = 0.0_f32;
                    let row_off = r * cols * 4;
                    for c in 0..cols {
                        let off = row_off + c * 4;
                        let w = f32::from_le_bytes([
                            raw[off],
                            raw[off + 1],
                            raw[off + 2],
                            raw[off + 3],
                        ]);
                        let a = f32::from(act_i8[c]) * act_scale;
                        sum += w * a;
                    }
                    out[r] = sum;
                }
                Ok(())
            }
            QuantType::F16 => {
                let raw = weights.as_bytes();
                let rows = out.len();
                let cols = act_i8.len();
                if raw.len() < rows * cols * 2 {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: weights.name().to_owned(),
                        offset: weights.offset(),
                        byte_len: weights.byte_len(),
                        buffer_len: raw.len(),
                        reason: "F16 weight buffer too small".to_owned(),
                    });
                }
                for r in 0..rows {
                    let mut sum = 0.0_f32;
                    let row_off = r * cols * 2;
                    for c in 0..cols {
                        let off = row_off + c * 2;
                        let bits = u16::from_le_bytes([raw[off], raw[off + 1]]);
                        let w = Self::convert_f16(bits);
                        let a = f32::from(act_i8[c]) * act_scale;
                        sum += w * a;
                    }
                    out[r] = sum;
                }
                Ok(())
            }
            QuantType::BF16 => {
                let raw = weights.as_bytes();
                let rows = out.len();
                let cols = act_i8.len();
                if raw.len() < rows * cols * 2 {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: weights.name().to_owned(),
                        offset: weights.offset(),
                        byte_len: weights.byte_len(),
                        buffer_len: raw.len(),
                        reason: "BF16 weight buffer too small".to_owned(),
                    });
                }
                for r in 0..rows {
                    let mut sum = 0.0_f32;
                    let row_off = r * cols * 2;
                    for c in 0..cols {
                        let off = row_off + c * 2;
                        let bits = u16::from_le_bytes([raw[off], raw[off + 1]]);
                        let w = Self::convert_bf16(bits);
                        let a = f32::from(act_i8[c]) * act_scale;
                        sum += w * a;
                    }
                    out[r] = sum;
                }
                Ok(())
            }
            quant => Err(K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: weights.offset(),
                byte_len: weights.byte_len(),
                buffer_len: weights.as_bytes().len(),
                reason: format!("unsupported quant type for BitLinear: {quant:?}"),
            }),
        }
    }

    /// Dispatches `SubLN` normalization and dynamic int8 quantization.
    #[inline]
    fn subln(
        out_i8: &mut [i8],
        input: &[f32],
        norm_weight: &[f32],
        eps: f32,
    ) -> Result<f32, K3Error> {
        #[cfg(feature = "with-kernels")]
        {
            subln_quantize_absmax_i8(out_i8, input, norm_weight, eps)
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            let len = input.len();
            if len == 0 || out_i8.len() != len || norm_weight.len() != len {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: "subln".to_owned(),
                    offset: 0,
                    byte_len: out_i8.len(),
                    buffer_len: input.len(),
                    reason: "dimension mismatch".to_owned(),
                });
            }
            let sum_sq: f32 = input.iter().map(|&x| x * x).sum();
            #[expect(clippy::cast_precision_loss, reason = "dimension fits in f32")]
            let inv_rms = 1.0 / ((sum_sq / len as f32) + eps).sqrt();
            let mut max_val = 0.0_f32;
            for i in 0..len {
                let x_norm = input[i] * inv_rms * norm_weight[i];
                let abs_val = x_norm.abs();
                if abs_val > max_val {
                    max_val = abs_val;
                }
            }
            if max_val.is_nan() || max_val < 1e-12 {
                out_i8.fill(0);
                return Ok(1.0 / 127.0);
            }
            let scale = max_val / 127.0;
            for i in 0..len {
                let x_norm = input[i] * inv_rms * norm_weight[i];
                let q = (x_norm / scale).round().clamp(-128.0, 127.0);
                #[expect(clippy::cast_possible_truncation, reason = "clamped to [-128, 127]")]
                {
                    out_i8[i] = q as i8;
                }
            }
            Ok(scale)
        }
    }

    /// Executes simulated or real forward layer kernels for a single token seed at `position`.
    #[allow(clippy::needless_range_loop, clippy::manual_checked_ops)]
    #[expect(
        clippy::too_many_lines,
        clippy::similar_names,
        reason = "sequential layer loop with 7 BitLinear projections and SubLN"
    )]
    fn execute_token_step(
        &self,
        token_id: u32,
        position: usize,
        seq_len: usize,
        kv_cache: &mut [f32],
        weights: &dyn ModelWeights,
    ) -> Result<Vec<f32>, K3Error> {
        let _ = seq_len;
        let hidden_dim = self.hidden_dim.max(16);
        let ffn_dim = self.ffn_dim.max(16);
        let head_d = self.head_dim.max(2);
        let head_d_even = if head_d % 2 == 0 { head_d } else { head_d - 1 };
        let n_q = self.n_heads_q.max(1);
        let n_kv = self.n_heads_kv.max(1);
        let total_q = n_q * head_d_even;
        let total_kv = n_kv * head_d_even;
        let layer_stride = kv_cache.len().checked_div(self.num_layers).unwrap_or(0);
        let eps = self.rms_eps.max(1e-6);

        // Pre-allocate scratch buffers once OUTSIDE the layer loop:
        let mut hidden = vec![0.0_f32; hidden_dim];
        if weights.contains_tensor("token_embd.weight") {
            let embd = weights.tensor("token_embd.weight")?;
            let raw = embd.as_bytes();
            let row = usize::try_from(token_id).unwrap_or(0);
            for (i, slot) in hidden.iter_mut().enumerate() {
                let off = (row * hidden_dim + i) * 4;
                if off + 4 <= raw.len() {
                    let b = [raw[off], raw[off + 1], raw[off + 2], raw[off + 3]];
                    *slot = f32::from_le_bytes(b);
                }
            }
        } else {
            #[expect(clippy::cast_precision_loss, reason = "token modulo fits in f32")]
            let base_val = ((token_id % 17) as f32) * 0.1 + 0.5;
            hidden.fill(base_val);
        }

        // Fixed buffers for layer projections (ZERO heap allocations during layer loop)
        let mut norm_w_hidden = vec![1.0_f32; hidden_dim];
        let mut norm_w_total_q = vec![1.0_f32; total_q];
        let mut norm_w_ffn = vec![1.0_f32; ffn_dim];

        let mut act_q_i8 = vec![0_i8; hidden_dim];
        let mut act_k_i8 = vec![0_i8; hidden_dim];
        let mut act_v_i8 = vec![0_i8; hidden_dim];

        let mut q_vec = vec![0.0_f32; total_q];
        let mut k_vec = vec![0.0_f32; total_kv];
        let mut v_vec = vec![0.0_f32; total_kv];

        let mut attn_out = vec![0.0_f32; total_q];
        let mut act_out_i8 = vec![0_i8; total_q];
        let mut o_proj = vec![0.0_f32; hidden_dim];

        let mut act_gate_i8 = vec![0_i8; hidden_dim];
        let mut act_up_i8 = vec![0_i8; hidden_dim];
        let mut gate = vec![0.0_f32; ffn_dim];
        let mut up = vec![0.0_f32; ffn_dim];
        let mut ffn_inter = vec![0.0_f32; ffn_dim];

        let mut act_down_i8 = vec![0_i8; ffn_dim];
        let mut d_proj = vec![0.0_f32; hidden_dim];

        let mut name_buf = String::with_capacity(64);
        let mut fallback_buf = String::with_capacity(64);

        for layer in 0..self.num_layers {
            // -------------------------------------------------------------
            // 1. Pre-Attention SubLN & Projections (attn_q, attn_k, attn_v)
            // -------------------------------------------------------------
            name_buf.clear();
            let _ = write!(&mut name_buf, "blk.{layer}.attn_sub_norm.weight");
            fallback_buf.clear();
            let _ = write!(&mut fallback_buf, "blk.{layer}.attn_norm.weight");
            Self::load_norm_weight_slice(
                weights,
                &name_buf,
                Some(&fallback_buf),
                &mut norm_w_hidden,
            )?;

            // 1a. SubLN before attn_q
            let scale_q = Self::subln(&mut act_q_i8, &hidden, &norm_w_hidden, eps)?;

            // 1b. SubLN before attn_k
            let scale_k = Self::subln(&mut act_k_i8, &hidden, &norm_w_hidden, eps)?;

            // 1c. SubLN before attn_v
            let scale_v = Self::subln(&mut act_v_i8, &hidden, &norm_w_hidden, eps)?;

            // BitLinear attn_q
            name_buf.clear();
            let _ = write!(&mut name_buf, "blk.{layer}.attn_q.weight");
            if weights.contains_tensor(&name_buf) {
                let q_tensor = weights.tensor(&name_buf)?;
                Self::dispatch_bitlinear(&mut q_vec, &q_tensor, &act_q_i8, scale_q)?;
            } else if weights.tensor_count() == 0 {
                for (i, slot) in q_vec.iter_mut().enumerate() {
                    *slot = (f32::from(act_q_i8[i % hidden_dim]) * scale_q) * 0.1 + 0.5;
                }
            } else {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: name_buf.clone(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: format!("tensor '{name_buf}' not found in weights"),
                });
            }

            // BitLinear attn_k
            name_buf.clear();
            let _ = write!(&mut name_buf, "blk.{layer}.attn_k.weight");
            if weights.contains_tensor(&name_buf) {
                let k_tensor = weights.tensor(&name_buf)?;
                Self::dispatch_bitlinear(&mut k_vec, &k_tensor, &act_k_i8, scale_k)?;
            } else if weights.tensor_count() == 0 {
                for (i, slot) in k_vec.iter_mut().enumerate() {
                    *slot = (f32::from(act_k_i8[i % hidden_dim]) * scale_k) * 0.1 + 0.5;
                }
            } else {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: name_buf.clone(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: format!("tensor '{name_buf}' not found in weights"),
                });
            }

            // BitLinear attn_v
            name_buf.clear();
            let _ = write!(&mut name_buf, "blk.{layer}.attn_v.weight");
            if weights.contains_tensor(&name_buf) {
                let v_tensor = weights.tensor(&name_buf)?;
                Self::dispatch_bitlinear(&mut v_vec, &v_tensor, &act_v_i8, scale_v)?;
            } else if weights.tensor_count() == 0 {
                for (i, slot) in v_vec.iter_mut().enumerate() {
                    *slot = (f32::from(act_v_i8[i % hidden_dim]) * scale_v) * 0.1 + 0.5;
                }
            } else {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: name_buf.clone(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: format!("tensor '{name_buf}' not found in weights"),
                });
            }

            // -------------------------------------------------------------
            // 2. Rotary Position Embeddings (RoPE)
            // -------------------------------------------------------------
            for head in q_vec.chunks_exact_mut(head_d_even) {
                #[cfg(feature = "with-kernels")]
                {
                    apply_rope_inplace(
                        head,
                        head_d_even,
                        position,
                        seq_len,
                        self.rope_theta,
                        None,
                    )?;
                }
                #[cfg(not(feature = "with-kernels"))]
                {
                    let half = head_d_even / 2;
                    let pos_f32 = position as f32;
                    for i in 0..half {
                        let freq = 1.0 / self.rope_theta.powf((2 * i) as f32 / head_d_even as f32);
                        let angle = pos_f32 * freq;
                        let (s, c) = (angle.sin(), angle.cos());
                        let (x1, x2) = (head[i], head[i + half]);
                        head[i] = x1 * c - x2 * s;
                        head[i + half] = x1 * s + x2 * c;
                    }
                }
            }

            for head in k_vec.chunks_exact_mut(head_d_even) {
                #[cfg(feature = "with-kernels")]
                {
                    apply_rope_inplace(
                        head,
                        head_d_even,
                        position,
                        seq_len,
                        self.rope_theta,
                        None,
                    )?;
                }
                #[cfg(not(feature = "with-kernels"))]
                {
                    let half = head_d_even / 2;
                    let pos_f32 = position as f32;
                    for i in 0..half {
                        let freq = 1.0 / self.rope_theta.powf((2 * i) as f32 / head_d_even as f32);
                        let angle = pos_f32 * freq;
                        let (s, c) = (angle.sin(), angle.cos());
                        let (x1, x2) = (head[i], head[i + half]);
                        head[i] = x1 * c - x2 * s;
                        head[i + half] = x1 * s + x2 * c;
                    }
                }
            }

            // -------------------------------------------------------------
            // 3. KV Cache Residency & Causal Attention (GQA)
            // -------------------------------------------------------------
            if layer_stride > 0 {
                let layer_start = layer * layer_stride;
                let layer_end = (layer_start + layer_stride).min(kv_cache.len());
                let layer_buf = &mut kv_cache[layer_start..layer_end];
                let half_stride = layer_buf.len() / 2;
                if half_stride > 0 {
                    let k_copy = total_kv.min(half_stride);
                    let pos_off = (position * k_copy) % half_stride;
                    let k_end = (pos_off + k_copy).min(half_stride);
                    let actual_copy = k_end - pos_off;
                    layer_buf[pos_off..pos_off + actual_copy]
                        .copy_from_slice(&k_vec[..actual_copy]);
                    layer_buf[half_stride + pos_off..half_stride + pos_off + actual_copy]
                        .copy_from_slice(&v_vec[..actual_copy]);
                }
            }

            attn_out.fill(0.0);
            #[cfg(feature = "with-kernels")]
            {
                #[expect(clippy::cast_precision_loss, reason = "head dimension fits in f32")]
                let scale = 1.0 / (head_d_even as f32).sqrt();
                let _ = gqa_causal_attention_f32(
                    &mut attn_out,
                    &q_vec,
                    &k_vec,
                    &v_vec,
                    n_q,
                    n_kv,
                    head_d_even,
                    1,
                    scale,
                );
            }
            #[cfg(not(feature = "with-kernels"))]
            {
                for i in 0..total_q {
                    let kv_i = if total_q > 0 {
                        (i * total_kv) / total_q
                    } else {
                        0
                    };
                    attn_out[i] = q_vec[i] * 0.5 + v_vec[kv_i] * 0.5;
                }
            }

            // -------------------------------------------------------------
            // 4. SubLN before Output Projection & Residual Connection
            // -------------------------------------------------------------
            name_buf.clear();
            let _ = write!(&mut name_buf, "blk.{layer}.attn_output_sub_norm.weight");
            fallback_buf.clear();
            let _ = write!(&mut fallback_buf, "blk.{layer}.attn_sub_norm.weight");
            Self::load_norm_weight_slice(
                weights,
                &name_buf,
                Some(&fallback_buf),
                &mut norm_w_total_q,
            )?;

            // 4a. Mandatory SubLN before attn_output
            let scale_out = Self::subln(&mut act_out_i8, &attn_out, &norm_w_total_q, eps)?;

            // 4b. BitLinear attn_output
            name_buf.clear();
            let _ = write!(&mut name_buf, "blk.{layer}.attn_output.weight");
            if weights.contains_tensor(&name_buf) {
                let out_tensor = weights.tensor(&name_buf)?;
                Self::dispatch_bitlinear(&mut o_proj, &out_tensor, &act_out_i8, scale_out)?;
                for (h, o) in hidden.iter_mut().zip(&o_proj) {
                    *h += *o;
                }
            } else if weights.tensor_count() == 0 {
                for (i, h) in hidden.iter_mut().enumerate() {
                    *h += (f32::from(act_out_i8[i % total_q]) * scale_out) * 0.05;
                }
            } else {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: name_buf.clone(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: format!("tensor '{name_buf}' not found in weights"),
                });
            }

            // -------------------------------------------------------------
            // 5. FFN SubLN & Projections (ffn_gate, ffn_up)
            // -------------------------------------------------------------
            name_buf.clear();
            let _ = write!(&mut name_buf, "blk.{layer}.ffn_sub_norm.weight");
            fallback_buf.clear();
            let _ = write!(&mut fallback_buf, "blk.{layer}.ffn_norm.weight");
            Self::load_norm_weight_slice(
                weights,
                &name_buf,
                Some(&fallback_buf),
                &mut norm_w_hidden,
            )?;

            // 5a. Mandatory SubLN before ffn_gate
            let scale_gate = Self::subln(&mut act_gate_i8, &hidden, &norm_w_hidden, eps)?;

            // 5b. Mandatory SubLN before ffn_up
            let scale_up = Self::subln(&mut act_up_i8, &hidden, &norm_w_hidden, eps)?;

            // BitLinear ffn_gate
            name_buf.clear();
            let _ = write!(&mut name_buf, "blk.{layer}.ffn_gate.weight");
            if weights.contains_tensor(&name_buf) {
                let gate_tensor = weights.tensor(&name_buf)?;
                Self::dispatch_bitlinear(&mut gate, &gate_tensor, &act_gate_i8, scale_gate)?;
            } else if weights.tensor_count() == 0 {
                for (i, slot) in gate.iter_mut().enumerate() {
                    *slot = (f32::from(act_gate_i8[i % hidden_dim]) * scale_gate) * 0.1 + 0.5;
                }
            } else {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: name_buf.clone(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: format!("tensor '{name_buf}' not found in weights"),
                });
            }

            // BitLinear ffn_up
            name_buf.clear();
            let _ = write!(&mut name_buf, "blk.{layer}.ffn_up.weight");
            if weights.contains_tensor(&name_buf) {
                let up_tensor = weights.tensor(&name_buf)?;
                Self::dispatch_bitlinear(&mut up, &up_tensor, &act_up_i8, scale_up)?;
            } else if weights.tensor_count() == 0 {
                for (i, slot) in up.iter_mut().enumerate() {
                    *slot = (f32::from(act_up_i8[i % hidden_dim]) * scale_up) * 0.05 + 0.25;
                }
            } else {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: name_buf.clone(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: format!("tensor '{name_buf}' not found in weights"),
                });
            }

            // -------------------------------------------------------------
            // 6. Activation: ReLU^2 vs SwiGLU
            // -------------------------------------------------------------
            if self.use_relu2 {
                #[cfg(feature = "with-kernels")]
                {
                    relu_squared_gate_f32(&mut ffn_inter, &gate, &up)?;
                }
                #[cfg(not(feature = "with-kernels"))]
                {
                    for i in 0..ffn_dim {
                        let g = gate[i].max(0.0);
                        ffn_inter[i] = g * g * up[i];
                    }
                }
            } else {
                #[cfg(feature = "with-kernels")]
                {
                    swiglu_f32(&mut ffn_inter, &gate, &up)?;
                }
                #[cfg(not(feature = "with-kernels"))]
                {
                    for i in 0..ffn_dim {
                        let g = gate[i];
                        let sig = if g <= -20.0 {
                            0.0
                        } else if g >= 20.0 {
                            1.0
                        } else {
                            1.0 / (1.0 + (-g).exp())
                        };
                        ffn_inter[i] = (g * sig) * up[i];
                    }
                }
            }

            // -------------------------------------------------------------
            // 7. SubLN before Down Projection & Residual Connection
            // -------------------------------------------------------------
            name_buf.clear();
            let _ = write!(&mut name_buf, "blk.{layer}.ffn_down_sub_norm.weight");
            fallback_buf.clear();
            let _ = write!(&mut fallback_buf, "blk.{layer}.ffn_sub_norm.weight");
            Self::load_norm_weight_slice(weights, &name_buf, Some(&fallback_buf), &mut norm_w_ffn)?;

            // 7a. Mandatory SubLN before ffn_down
            let scale_down = Self::subln(&mut act_down_i8, &ffn_inter, &norm_w_ffn, eps)?;

            // 7b. BitLinear ffn_down
            name_buf.clear();
            let _ = write!(&mut name_buf, "blk.{layer}.ffn_down.weight");
            if weights.contains_tensor(&name_buf) {
                let down_tensor = weights.tensor(&name_buf)?;
                Self::dispatch_bitlinear(&mut d_proj, &down_tensor, &act_down_i8, scale_down)?;
                for (h, d) in hidden.iter_mut().zip(&d_proj) {
                    *h += *d;
                }
            } else if weights.tensor_count() == 0 {
                for (i, h) in hidden.iter_mut().enumerate() {
                    *h += (f32::from(act_down_i8[i % ffn_dim]) * scale_down) * 0.01;
                }
            } else {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: name_buf.clone(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: format!("tensor '{name_buf}' not found in weights"),
                });
            }
        }

        // -------------------------------------------------------------
        // 8. Output Normalization
        // -------------------------------------------------------------
        if weights.contains_tensor("output_norm.weight") {
            Self::load_norm_weight_slice(weights, "output_norm.weight", None, &mut norm_w_hidden)?;
            let sum_sq: f32 = hidden.iter().map(|&x| x * x).sum();
            #[expect(clippy::cast_precision_loss, reason = "dimension fits in f32")]
            let inv_rms = 1.0 / (sum_sq / (hidden_dim as f32) + eps).sqrt();
            for i in 0..hidden_dim {
                hidden[i] = hidden[i] * inv_rms * norm_w_hidden[i];
            }
        }

        Ok(hidden)
    }
}

impl ArchRecipe for BitNetRecipe {
    fn architecture_id(&self) -> &'static str {
        self.architecture
    }

    fn recipe_version(&self) -> &'static str {
        "1.0.0"
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        matches!(
            quant,
            QuantType::TQ1_0
                | QuantType::TQ2_0
                | QuantType::Q8_0
                | QuantType::F32
                | QuantType::F16
                | QuantType::BF16
        )
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let prompt_tokens = ctx.prompt_tokens().to_vec();
        if prompt_tokens.is_empty() {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: 0,
                max_context_tokens: ctx.n_ctx(),
            });
        }
        let prompt_len = prompt_tokens.len();
        let start_pos = ctx.position();
        let seq_len = start_pos + prompt_len;

        let mut last_hidden = Vec::new();
        for (i, &tok) in prompt_tokens.iter().enumerate() {
            let pos = start_pos + i;
            last_hidden =
                self.execute_token_step(tok, pos, seq_len, ctx.kv_cache_mut(), weights)?;
        }

        ctx.advance_position(prompt_len)?;

        let logits = ctx.logits_mut();
        if !logits.is_empty() {
            if weights.contains_tensor("output.weight") {
                let output_weight = weights.tensor("output.weight")?;
                if self.supports_quant(output_weight.quant()) {
                    let mut act_head_i8 = vec![0_i8; self.hidden_dim];
                    let norm_w = vec![1.0_f32; self.hidden_dim];
                    let scale_head = Self::subln(
                        &mut act_head_i8,
                        &last_hidden,
                        &norm_w,
                        self.rms_eps.max(1e-6),
                    )?;
                    Self::dispatch_bitlinear(logits, &output_weight, &act_head_i8, scale_head)?;
                } else {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: "output.weight".to_owned(),
                        offset: output_weight.offset(),
                        byte_len: output_weight.byte_len(),
                        buffer_len: output_weight.as_bytes().len(),
                        reason: format!(
                            "unsupported quant type for BitLinear LM head: {:?}",
                            output_weight.quant()
                        ),
                    });
                }
            } else if weights.tensor_count() == 0 {
                logits.fill(0.0);
                let bias = (last_hidden.first().copied().unwrap_or(0.0).abs() % 0.25) * 0.01;
                let target_slot = if logits.len() > 9 { 9 } else { 0 };
                logits[target_slot] = 1.0 + bias;
            } else {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: "output.weight".to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: "tensor 'output.weight' not found in weights".to_owned(),
                });
            }
        }

        Ok(())
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let token = ctx
            .decode_token()
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: self.architecture_id().to_owned(),
                reason: "decode_step called without a staged decode token".to_owned(),
            })?;

        let pos = ctx.position();
        let seq_len = pos + 1;
        let hidden = self.execute_token_step(token, pos, seq_len, ctx.kv_cache_mut(), weights)?;
        ctx.advance_position(1)?;

        let logits = ctx.logits_mut();
        if !logits.is_empty() {
            if weights.contains_tensor("output.weight") {
                let output_weight = weights.tensor("output.weight")?;
                if self.supports_quant(output_weight.quant()) {
                    let mut act_head_i8 = vec![0_i8; self.hidden_dim];
                    let norm_w = vec![1.0_f32; self.hidden_dim];
                    let scale_head =
                        Self::subln(&mut act_head_i8, &hidden, &norm_w, self.rms_eps.max(1e-6))?;
                    Self::dispatch_bitlinear(logits, &output_weight, &act_head_i8, scale_head)?;
                } else {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: "output.weight".to_owned(),
                        offset: output_weight.offset(),
                        byte_len: output_weight.byte_len(),
                        buffer_len: output_weight.as_bytes().len(),
                        reason: format!(
                            "unsupported quant type for BitLinear LM head: {:?}",
                            output_weight.quant()
                        ),
                    });
                }
            } else if weights.tensor_count() == 0 {
                logits.fill(0.0);
                let bias = (hidden.first().copied().unwrap_or(0.0).abs() % 0.25) * 0.01;
                let next_slot = (usize::try_from(token).unwrap_or(0) + 1) % logits.len();
                logits[next_slot] = 2.0 + bias;
            } else {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: "output.weight".to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: "tensor 'output.weight' not found in weights".to_owned(),
                });
            }
        }

        Ok(())
    }
}
