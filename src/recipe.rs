//! `BitNetRecipe`: Architecture recipe for `BitNet` 1.58-bit ternary models.
//!
//! Implements [`ArchRecipe`] supporting canonical architecture IDs `"bitnet"`,
//! `"bitnet-b1.58"`, and `"bitnet-25"`, with mandatory `SubLN` prior to every
//! `BitLinear` projection per ADR-023, ADR-024, and ADR-037.

use crate::repack_cache::{BitNetModelCache, BitNetVocabContract, DecodeWorkerPool};
use jero_core::{
    ArchRecipe, ExecutionContext, GgufArray, GgufFile, GgufValue, K3Error, KvCacheView,
    ModelMetadata, ModelWeights, QuantType, ScratchScope, ScratchWorkspace, TensorView,
};
use std::fmt::Write as _;

#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{
    QuantizedActQ8View, RopeType, apply_rope_inplace, apply_rope_with_cos_sin, bf16_to_f32_finite,
    bitlinear_i8_rvv, compute_rope_cos_sin, dequantize_tensor_row, f16_to_f32_finite,
    gqa_causal_attention_with_scratch_f32, relu_squared_gate_f32, subln_quantize_absmax_i8,
    swiglu_f32,
};

struct StackLayerKey {
    buf: [u8; 64],
    len: usize,
}

impl StackLayerKey {
    #[inline]
    fn format(layer: usize, suffix: &str) -> Self {
        let mut buf = [0_u8; 64];
        let mut pos = 0_usize;
        for &b in b"blk." {
            buf[pos] = b;
            pos += 1;
        }
        let mut digits = [0_u8; 20];
        let mut n = layer;
        let mut d_len = 0_usize;
        loop {
            #[expect(clippy::cast_possible_truncation, reason = "n % 10 is in 0..10")]
            {
                digits[d_len] = b'0' + (n % 10) as u8;
            }
            d_len += 1;
            n /= 10;
            if n == 0 {
                break;
            }
        }
        for i in (0..d_len).rev() {
            buf[pos] = digits[i];
            pos += 1;
        }
        buf[pos] = b'.';
        pos += 1;
        let suffix_bytes = suffix.as_bytes();
        let copy_len = suffix_bytes.len().min(buf.len() - pos);
        buf[pos..pos + copy_len].copy_from_slice(&suffix_bytes[..copy_len]);
        pos += copy_len;
        Self { buf, len: pos }
    }

    #[inline]
    fn as_str(&self) -> &str {
        core::str::from_utf8(&self.buf[..self.len]).unwrap_or("")
    }
}

#[cold]
fn missing_tensor_error(name: &str) -> K3Error {
    K3Error::TensorBoundsError {
        tensor_name: name.to_owned(),
        offset: 0,
        byte_len: 0,
        buffer_len: 0,
        reason: format!("tensor '{name}' not found in weights"),
    }
}

#[cold]
fn kv_dim_mismatch_error(layer: usize, total_kv: usize, actual_kv_dim: usize) -> K3Error {
    K3Error::TensorBoundsError {
        tensor_name: format!("kv_cache.layer.{layer}"),
        offset: 0,
        byte_len: total_kv * size_of::<f32>(),
        buffer_len: actual_kv_dim * size_of::<f32>(),
        reason: format!("KvGeometry kv_dim {actual_kv_dim} does not match total_kv {total_kv}"),
    }
}

#[cold]
fn kv_stride_error(
    req_hist_elems: usize,
    num_layers: usize,
    kv_cache_bytes: usize,
    half_stride: usize,
    position: usize,
) -> K3Error {
    K3Error::TensorBoundsError {
        tensor_name: "kv_cache".to_owned(),
        offset: 0,
        byte_len: req_hist_elems * 2 * num_layers * size_of::<f32>(),
        buffer_len: kv_cache_bytes,
        reason: format!(
            "kv_cache half-layer stride {half_stride} is smaller than required {req_hist_elems} elements for position {position}"
        ),
    }
}

#[inline]
const fn aligned_scratch_slot_bytes(elem_bytes: usize, len: usize) -> usize {
    let raw = len.saturating_mul(elem_bytes);
    let blocks = raw.div_ceil(128);
    let rounded = blocks.saturating_mul(128);
    if rounded < 128 { 128 } else { rounded }
}

pub(crate) enum KvStorage<'a> {
    View(KvCacheView<'a>),
    Raw(&'a mut [f32]),
}

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

    fn normalize_architecture_id(raw_arch: &str) -> Option<&'static str> {
        match raw_arch {
            "bitnet" => Some("bitnet"),
            "bitnet-b1.58" => Some("bitnet-b1.58"),
            "bitnet-25" | "bitnet-2.5" => Some("bitnet-25"),
            _ => None,
        }
    }

    fn lookup_gguf_value<'a>(
        gguf: &'a GgufFile<'_>,
        raw_arch: &str,
        suffix: &str,
    ) -> Option<&'a GgufValue<'a>> {
        let mut key = String::with_capacity(64);
        let prefixes = [raw_arch, "bitnet-b1.58", "bitnet", "llama"];
        for (idx, &prefix) in prefixes.iter().enumerate() {
            if prefixes[..idx].contains(&prefix) {
                continue;
            }
            key.clear();
            let _ = write!(&mut key, "{prefix}.{suffix}");
            if let Some(val) = gguf.metadata_value(&key) {
                return Some(val);
            }
        }
        None
    }

    fn lookup_gguf_usize(gguf: &GgufFile<'_>, raw_arch: &str, suffix: &str) -> Option<usize> {
        Self::lookup_gguf_value(gguf, raw_arch, suffix)
            .and_then(GgufValue::as_usize)
            .filter(|&v| v > 0)
    }

    fn lookup_gguf_f32(gguf: &GgufFile<'_>, raw_arch: &str, suffix: &str) -> Option<f32> {
        Self::lookup_gguf_value(gguf, raw_arch, suffix)
            .and_then(GgufValue::as_f32)
            .filter(|v| v.is_finite() && *v > 0.0)
    }

    fn tensor_out_dim(meta: &ModelMetadata, tensor_name: &str) -> Option<usize> {
        let desc = meta.find_tensor(tensor_name)?;
        desc.shape().get(1).copied().filter(|&dim| dim > 0)
    }

    /// Hydrates a [`BitNetRecipe`] directly from parsed GGUF v3 metadata.
    ///
    /// Inspects `{arch}.*`, `bitnet-b1.58.*`, `bitnet.*`, and `llama.*` metadata keys along with
    /// `tokenizer.ggml.tokens` (and fallback projection shapes where needed) so hyperparameters such as
    /// `BitNet-b1.58-2B-4T` (`hidden_dim = 2560`, `ffn_dim = 6912`, `num_layers = 30`, `n_heads_q = 20`,
    /// `n_heads_kv = 5`, `vocab_size = 128256`, `rope_theta = 500000.0`) are preserved accurately.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if `general.architecture` is not a supported `BitNet`
    /// variant or if required structural hyperparameters are missing, zero, or inconsistent.
    #[expect(
        clippy::too_many_lines,
        reason = "validates GGUF metadata keys across BitNet architecture namespaces"
    )]
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Result<Self, K3Error> {
        let raw_arch = gguf.architecture();
        let architecture = Self::normalize_architecture_id(raw_arch).ok_or_else(|| {
            K3Error::CapabilityMismatch {
                architecture: raw_arch.to_owned(),
                reason: format!("unsupported GGUF architecture '{raw_arch}' for BitNetRecipe"),
            }
        })?;
        let meta = gguf.metadata();

        let num_layers = Self::lookup_gguf_usize(gguf, raw_arch, "block_count")
            .or_else(|| (meta.layer_count() > 0).then_some(meta.layer_count()))
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: architecture.to_owned(),
                reason: "missing positive block_count metadata in BitNet GGUF".to_owned(),
            })?;

        let hidden_dim = Self::lookup_gguf_usize(gguf, raw_arch, "embedding_length")
            .or_else(|| (meta.embedding_dim() > 0).then_some(meta.embedding_dim()))
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: architecture.to_owned(),
                reason: "missing positive embedding_length metadata in BitNet GGUF".to_owned(),
            })?;

        let ffn_dim = Self::lookup_gguf_usize(gguf, raw_arch, "feed_forward_length")
            .or_else(|| meta.feed_forward_length())
            .or_else(|| Self::tensor_out_dim(meta, "blk.0.ffn_gate.weight"))
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: architecture.to_owned(),
                reason: "missing positive feed_forward_length metadata in BitNet GGUF".to_owned(),
            })?;

        let n_heads_q = Self::lookup_gguf_usize(gguf, raw_arch, "attention.head_count")
            .or_else(|| (meta.head_count() > 0).then_some(meta.head_count()))
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: architecture.to_owned(),
                reason: "missing positive attention.head_count metadata in BitNet GGUF".to_owned(),
            })?;

        let n_heads_kv = Self::lookup_gguf_usize(gguf, raw_arch, "attention.head_count_kv")
            .or_else(|| (meta.head_count_kv() > 0).then_some(meta.head_count_kv()))
            .unwrap_or(n_heads_q);

        if n_heads_kv == 0 || n_heads_q % n_heads_kv != 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: architecture.to_owned(),
                reason: format!(
                    "GQA requires n_heads_q ({n_heads_q}) to be a non-zero multiple of n_heads_kv ({n_heads_kv})"
                ),
            });
        }

        let explicit_head_dim = Self::lookup_gguf_usize(gguf, raw_arch, "attention.key_length")
            .or_else(|| meta.key_length());
        let head_dim = if let Some(dim) = explicit_head_dim {
            dim
        } else {
            if hidden_dim % n_heads_q != 0 {
                return Err(K3Error::CapabilityMismatch {
                    architecture: architecture.to_owned(),
                    reason: format!(
                        "hidden_dim ({hidden_dim}) is not divisible by attention.head_count ({n_heads_q})"
                    ),
                });
            }
            hidden_dim / n_heads_q
        };

        if head_dim < 2 || head_dim % 2 != 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: architecture.to_owned(),
                reason: format!("head_dim ({head_dim}) must be even and >= 2 for RoPE"),
            });
        }

        let vocab_size = Self::lookup_gguf_usize(gguf, raw_arch, "vocab_size")
            .or_else(|| meta.vocab_size())
            .or_else(|| {
                gguf.metadata_value("tokenizer.ggml.tokens")
                    .and_then(GgufValue::as_array)
                    .map(GgufArray::len)
                    .filter(|&len| len > 0)
            })
            .or_else(|| Self::tensor_out_dim(meta, "token_embd.weight"))
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: architecture.to_owned(),
                reason: "missing positive vocab_size metadata or token_embd.weight in BitNet GGUF"
                    .to_owned(),
            })?;

        let rope_theta = Self::lookup_gguf_f32(gguf, raw_arch, "rope.freq_base")
            .or_else(|| meta.rope_freq_base())
            .unwrap_or(10_000.0);

        let rms_eps = Self::lookup_gguf_f32(gguf, raw_arch, "attention.layer_norm_rms_epsilon")
            .or_else(|| meta.rms_norm_eps())
            .unwrap_or(1e-5);

        let use_relu2 = Self::lookup_gguf_value(gguf, raw_arch, "activation")
            .or_else(|| Self::lookup_gguf_value(gguf, raw_arch, "ffn_activation"))
            .or_else(|| Self::lookup_gguf_value(gguf, raw_arch, "feed_forward_activation"))
            .and_then(GgufValue::as_str)
            .map_or(architecture != "bitnet-25", |act| {
                !matches!(act, "swiglu" | "silu" | "SwiGLU" | "SiLU")
            });

        Ok(Self {
            architecture,
            num_layers,
            hidden_dim,
            ffn_dim,
            n_heads_q,
            n_heads_kv,
            head_dim,
            vocab_size,
            rms_eps,
            rope_theta,
            use_relu2,
        })
    }

    #[inline]
    pub(crate) fn convert_f16(bits: u16) -> f32 {
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
                if mant == 0 {
                    f32::from_bits(sign << 31)
                } else {
                    // Subnormal IEEE 754 binary16: (-1)^sign * mant * 2^-24
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "10-bit mantissa fits exactly in f32"
                    )]
                    let mag = (mant as f32) * (1.0_f32 / 16_777_216.0_f32);
                    if sign == 1 { -mag } else { mag }
                }
            } else if exp == 31 {
                if sign == 1 { -0.0625 } else { 0.0625 }
            } else {
                f32::from_bits((sign << 31) | ((exp + 112) << 23) | (mant << 13))
            }
        }
    }

    #[inline]
    pub(crate) fn convert_bf16(bits: u16) -> f32 {
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
    pub(crate) fn load_norm_weight_slice(
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

    /// Block geometry for block-quantized `BitLinear` weights.
    pub(crate) const fn quant_block_layout(quant: QuantType) -> Option<(usize, usize, usize)> {
        match quant {
            QuantType::I2_S => Some((128, 32, 32)),
            QuantType::TQ1_0 => Some((256, 54, 0)),
            QuantType::TQ2_0 => Some((256, 66, 0)),
            QuantType::Q8_0 => Some((32, 34, 0)),
            _ => None,
        }
    }

    /// Resolves `total_rows` for a weight matrix whose inner width is `expected_cols`.
    pub(crate) fn matrix_row_col_dims(
        view: &TensorView<'_>,
        expected_cols: usize,
        min_rows: usize,
    ) -> Result<usize, K3Error> {
        let shape = view.shape();
        if expected_cols == 0 || shape.is_empty() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: view.name().to_owned(),
                offset: view.offset(),
                byte_len: view.byte_len(),
                buffer_len: view.as_bytes().len(),
                reason: "matrix shape or expected column count is zero".to_owned(),
            });
        }
        let total_rows = if shape.len() == 2 {
            if shape[0] == expected_cols {
                shape[1]
            } else if shape[1] == expected_cols {
                shape[0]
            } else {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: view.name().to_owned(),
                    offset: view.offset(),
                    byte_len: view.byte_len(),
                    buffer_len: view.as_bytes().len(),
                    reason: format!(
                        "matrix shape [{}, {}] does not match expected row width {expected_cols}",
                        shape[0], shape[1]
                    ),
                });
            }
        } else {
            let elems = view.num_elements();
            if elems % expected_cols != 0 {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: view.name().to_owned(),
                    offset: view.offset(),
                    byte_len: view.byte_len(),
                    buffer_len: view.as_bytes().len(),
                    reason: format!(
                        "tensor elements {elems} is not a multiple of row width {expected_cols}"
                    ),
                });
            }
            elems / expected_cols
        };

        if total_rows < min_rows {
            return Err(K3Error::TensorBoundsError {
                tensor_name: view.name().to_owned(),
                offset: view.offset(),
                byte_len: view.byte_len(),
                buffer_len: view.as_bytes().len(),
                reason: format!(
                    "tensor row count {total_rows} is smaller than required {min_rows}"
                ),
            });
        }
        Ok(total_rows)
    }

    #[cfg(not(feature = "with-kernels"))]
    pub(crate) fn unpack_i2s_block_i8(block: &[u8; 32], out: &mut [i8; 128]) {
        #[expect(
            clippy::cast_possible_wrap,
            reason = "2-bit ternary code 0..=3 minus 1 fits in i8"
        )]
        for gp in 0..32 {
            let packed = block[gp];
            out[gp] = (((packed >> 6) & 0x03) as i8) - 1;
            out[32 + gp] = (((packed >> 4) & 0x03) as i8) - 1;
            out[64 + gp] = (((packed >> 2) & 0x03) as i8) - 1;
            out[96 + gp] = ((packed & 0x03) as i8) - 1;
        }
    }

    #[cfg(not(feature = "with-kernels"))]
    pub(crate) fn extract_i2s_scale(
        raw: &[u8],
        payload_bytes: usize,
        view: &TensorView<'_>,
    ) -> Result<f32, K3Error> {
        let required = payload_bytes.saturating_add(32);
        if raw.len() < required {
            return Err(K3Error::TensorBoundsError {
                tensor_name: view.name().to_owned(),
                offset: view.offset(),
                byte_len: view.byte_len(),
                buffer_len: raw.len(),
                reason: format!(
                    "I2_S buffer length {} is smaller than required {required} bytes (payload + 32B scale trailer)",
                    raw.len()
                ),
            });
        }
        let d_raw = f32::from_le_bytes([
            raw[payload_bytes],
            raw[payload_bytes + 1],
            raw[payload_bytes + 2],
            raw[payload_bytes + 3],
        ]);
        Ok(if d_raw.is_finite() { d_raw } else { 0.0 })
    }

    #[cfg(not(feature = "with-kernels"))]
    pub(crate) fn unpack_tq1_0_block_i8(block: &[u8; 54], out: &mut [i8; 256]) -> f32 {
        let d_bits = u16::from_le_bytes([block[0], block[1]]);
        let scale = Self::convert_f16(d_bits);
        #[expect(
            clippy::cast_possible_wrap,
            reason = "modulo 3 minus 1 yields {-1, 0, 1}"
        )]
        for i in 0..51 {
            let mut b = block[2 + i];
            let base = i * 5;
            for j in 0..5 {
                out[base + j] = ((b % 3) as i8) - 1;
                b /= 3;
            }
        }
        #[expect(
            clippy::cast_possible_wrap,
            reason = "modulo 3 minus 1 yields {-1, 0, 1}"
        )]
        {
            out[255] = ((block[53] % 3) as i8) - 1;
        }
        scale
    }

    #[cfg(not(feature = "with-kernels"))]
    pub(crate) fn unpack_tq2_0_block_i8(block: &[u8; 66], out: &mut [i8; 256]) -> f32 {
        let d_bits = u16::from_le_bytes([block[0], block[1]]);
        let scale = Self::convert_f16(d_bits);
        #[expect(
            clippy::cast_possible_wrap,
            reason = "2-bit code 0..=3 minus 1 fits in i8"
        )]
        for (i, &byte) in block[2..66].iter().enumerate() {
            let base = i * 4;
            out[base] = ((byte & 0x03) as i8) - 1;
            out[base + 1] = (((byte >> 2) & 0x03) as i8) - 1;
            out[base + 2] = (((byte >> 4) & 0x03) as i8) - 1;
            out[base + 3] = (((byte >> 6) & 0x03) as i8) - 1;
        }
        scale
    }

    /// Scalar row decoder for `F32`, `F16`, `BF16`, `Q8_0`, `TQ1_0`, `TQ2_0`, and `I2_S`.
    #[cfg(not(feature = "with-kernels"))]
    #[expect(
        clippy::too_many_lines,
        reason = "uniform scalar row decoder across all supported BitNet tensor quantizations"
    )]
    fn decode_row_scalar(
        view: &TensorView<'_>,
        row_idx: usize,
        out: &mut [f32],
    ) -> Result<(), K3Error> {
        let cols = out.len();
        let total_rows = Self::matrix_row_col_dims(view, cols, row_idx.saturating_add(1))?;
        let raw = view.as_bytes();

        match view.quant() {
            QuantType::F32 => {
                let req = total_rows * cols * 4;
                if raw.len() < req {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: view.name().to_owned(),
                        offset: view.offset(),
                        byte_len: view.byte_len(),
                        buffer_len: raw.len(),
                        reason: format!("F32 buffer length {} < required {req}", raw.len()),
                    });
                }
                let row_off = row_idx * cols * 4;
                for (i, slot) in out.iter_mut().enumerate() {
                    let off = row_off + i * 4;
                    *slot =
                        f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                }
                Ok(())
            }
            QuantType::F16 | QuantType::BF16 => {
                let req = total_rows * cols * 2;
                if raw.len() < req {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: view.name().to_owned(),
                        offset: view.offset(),
                        byte_len: view.byte_len(),
                        buffer_len: raw.len(),
                        reason: format!(
                            "{:?} buffer length {} < required {req}",
                            view.quant(),
                            raw.len()
                        ),
                    });
                }
                let row_off = row_idx * cols * 2;
                let is_f16 = view.quant() == QuantType::F16;
                for (i, slot) in out.iter_mut().enumerate() {
                    let off = row_off + i * 2;
                    let bits = u16::from_le_bytes([raw[off], raw[off + 1]]);
                    *slot = if is_f16 {
                        Self::convert_f16(bits)
                    } else {
                        Self::convert_bf16(bits)
                    };
                }
                Ok(())
            }
            quant @ (QuantType::Q8_0 | QuantType::TQ1_0 | QuantType::TQ2_0 | QuantType::I2_S) => {
                let (block_elems, block_bytes, trailer_bytes) = Self::quant_block_layout(quant)
                    .expect("quant_block_layout handles all block quant variants");
                if cols % block_elems != 0 {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: view.name().to_owned(),
                        offset: view.offset(),
                        byte_len: view.byte_len(),
                        buffer_len: raw.len(),
                        reason: format!(
                            "row width {cols} is not aligned to {quant:?} block size {block_elems}"
                        ),
                    });
                }
                let blocks_per_row = cols / block_elems;
                let payload_bytes = total_rows * blocks_per_row * block_bytes;
                let req = payload_bytes + trailer_bytes;
                if raw.len() < req {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: view.name().to_owned(),
                        offset: view.offset(),
                        byte_len: view.byte_len(),
                        buffer_len: raw.len(),
                        reason: format!("{quant:?} buffer length {} < required {req}", raw.len()),
                    });
                }
                let i2s_scale = if quant == QuantType::I2_S {
                    Self::extract_i2s_scale(raw, payload_bytes, view)?
                } else {
                    1.0
                };
                let row_off = row_idx * blocks_per_row * block_bytes;
                for b in 0..blocks_per_row {
                    let boff = row_off + b * block_bytes;
                    let dst = &mut out[b * block_elems..(b + 1) * block_elems];
                    match quant {
                        QuantType::Q8_0 => {
                            let scale =
                                Self::convert_f16(u16::from_le_bytes([raw[boff], raw[boff + 1]]));
                            #[expect(
                                clippy::cast_possible_wrap,
                                reason = "Q8_0 quants are signed i8"
                            )]
                            for (dst_slot, &q_byte) in dst.iter_mut().zip(&raw[boff + 2..boff + 34])
                            {
                                *dst_slot = f32::from(q_byte as i8) * scale;
                            }
                        }
                        QuantType::TQ1_0 => {
                            let block: &[u8; 54] = raw[boff..boff + 54]
                                .try_into()
                                .expect("validated TQ1_0 slice length");
                            let mut w_i8 = [0_i8; 256];
                            let scale = Self::unpack_tq1_0_block_i8(block, &mut w_i8);
                            for (dst_slot, &w) in dst.iter_mut().zip(&w_i8) {
                                *dst_slot = f32::from(w) * scale;
                            }
                        }
                        QuantType::TQ2_0 => {
                            let block: &[u8; 66] = raw[boff..boff + 66]
                                .try_into()
                                .expect("validated TQ2_0 slice length");
                            let mut w_i8 = [0_i8; 256];
                            let scale = Self::unpack_tq2_0_block_i8(block, &mut w_i8);
                            for (dst_slot, &w) in dst.iter_mut().zip(&w_i8) {
                                *dst_slot = f32::from(w) * scale;
                            }
                        }
                        QuantType::I2_S => {
                            let block: &[u8; 32] = raw[boff..boff + 32]
                                .try_into()
                                .expect("validated I2_S slice length");
                            let mut w_i8 = [0_i8; 128];
                            Self::unpack_i2s_block_i8(block, &mut w_i8);
                            for (dst_slot, &w) in dst.iter_mut().zip(&w_i8) {
                                *dst_slot = f32::from(w) * i2s_scale;
                            }
                        }
                        _ => {}
                    }
                }
                Ok(())
            }
            other => Err(K3Error::TensorBoundsError {
                tensor_name: view.name().to_owned(),
                offset: view.offset(),
                byte_len: view.byte_len(),
                buffer_len: raw.len(),
                reason: format!("unsupported quant type for row decode: {other:?}"),
            }),
        }
    }

    /// Decodes row `row_idx` of a tensor into `out` across `F32`, `F16`, `BF16`, `Q8_0`, `TQ1_0`, `TQ2_0`, and `I2_S`.
    pub(crate) fn decode_row_f32(
        view: &TensorView<'_>,
        row_idx: usize,
        out: &mut [f32],
    ) -> Result<(), K3Error> {
        #[cfg(feature = "with-kernels")]
        {
            dequantize_tensor_row(view, row_idx, out)
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            Self::decode_row_scalar(view, row_idx, out)
        }
    }

    /// Scalar `f32` dot product across one row of `F32`/`F16`/`BF16`/`Q8_0` without `i8` activation quantization.
    #[cfg(not(feature = "with-kernels"))]
    fn dot_row_f32_scalar(
        view: &TensorView<'_>,
        row_idx: usize,
        total_rows: usize,
        vec: &[f32],
    ) -> Result<f32, K3Error> {
        let cols = vec.len();
        let raw = view.as_bytes();
        match view.quant() {
            QuantType::F32 => {
                let row_off = row_idx * cols * 4;
                let mut sum = 0.0_f32;
                for (c, &x) in vec.iter().enumerate() {
                    let off = row_off + c * 4;
                    let w =
                        f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                    sum += w * x;
                }
                Ok(sum)
            }
            QuantType::F16 | QuantType::BF16 => {
                let is_f16 = view.quant() == QuantType::F16;
                let row_off = row_idx * cols * 2;
                let mut sum = 0.0_f32;
                for (c, &x) in vec.iter().enumerate() {
                    let off = row_off + c * 2;
                    let bits = u16::from_le_bytes([raw[off], raw[off + 1]]);
                    let w = if is_f16 {
                        Self::convert_f16(bits)
                    } else {
                        Self::convert_bf16(bits)
                    };
                    sum += w * x;
                }
                Ok(sum)
            }
            QuantType::Q8_0 => {
                let blocks_per_row = cols / 32;
                let req = total_rows * blocks_per_row * 34;
                if raw.len() < req {
                    return Err(K3Error::TensorBoundsError {
                        tensor_name: view.name().to_owned(),
                        offset: view.offset(),
                        byte_len: view.byte_len(),
                        buffer_len: raw.len(),
                        reason: format!("Q8_0 buffer length {} < required {req}", raw.len()),
                    });
                }
                let row_off = row_idx * blocks_per_row * 34;
                let mut sum = 0.0_f32;
                for b in 0..blocks_per_row {
                    let boff = row_off + b * 34;
                    let scale = Self::convert_f16(u16::from_le_bytes([raw[boff], raw[boff + 1]]));
                    let mut block_sum = 0.0_f32;
                    let x_slice = &vec[b * 32..(b + 1) * 32];
                    #[expect(clippy::cast_possible_wrap, reason = "Q8_0 quants are signed i8")]
                    for (&q_byte, &x) in raw[boff + 2..boff + 34].iter().zip(x_slice) {
                        block_sum += f32::from(q_byte as i8) * x;
                    }
                    sum += block_sum * scale;
                }
                Ok(sum)
            }
            other => Err(K3Error::TensorBoundsError {
                tensor_name: view.name().to_owned(),
                offset: view.offset(),
                byte_len: view.byte_len(),
                buffer_len: raw.len(),
                reason: format!("unsupported quant for f32 LM head dot product: {other:?}"),
            }),
        }
    }

    /// Dispatches a `BitLinear` projection across RVV or portable fallbacks.
    #[allow(clippy::needless_range_loop)]
    #[expect(
        clippy::too_many_lines,
        reason = "handles multiple quant formats for BitLinear projections"
    )]
    pub(crate) fn dispatch_bitlinear(
        out: &mut [f32],
        weights: &TensorView<'_>,
        act_i8: &[i8],
        act_scale: f32,
    ) -> Result<(), K3Error> {
        let quant = weights.quant();
        match quant {
            QuantType::I2_S | QuantType::TQ1_0 | QuantType::TQ2_0 | QuantType::Q8_0 => {
                #[cfg(feature = "with-kernels")]
                {
                    bitlinear_i8_rvv(out, weights, act_i8, act_scale)
                }
                #[cfg(not(feature = "with-kernels"))]
                {
                    let rows = out.len();
                    let cols = act_i8.len();
                    let total_rows = Self::matrix_row_col_dims(weights, cols, rows)?;
                    let (block_elems, block_bytes, trailer_bytes) = Self::quant_block_layout(quant)
                        .expect("quant_block_layout handles all block quant variants");
                    if cols % block_elems != 0 {
                        return Err(K3Error::TensorBoundsError {
                            tensor_name: weights.name().to_owned(),
                            offset: weights.offset(),
                            byte_len: weights.byte_len(),
                            buffer_len: weights.as_bytes().len(),
                            reason: format!(
                                "input width {cols} must be a multiple of {quant:?} block size {block_elems}"
                            ),
                        });
                    }
                    let blocks_per_row = cols / block_elems;
                    let payload_bytes = total_rows * blocks_per_row * block_bytes;
                    let required_bytes = payload_bytes + trailer_bytes;
                    let raw = weights.as_bytes();
                    if raw.len() < required_bytes {
                        return Err(K3Error::TensorBoundsError {
                            tensor_name: weights.name().to_owned(),
                            offset: weights.offset(),
                            byte_len: weights.byte_len(),
                            buffer_len: raw.len(),
                            reason: format!(
                                "weight buffer length {} is smaller than required {required_bytes} bytes",
                                raw.len()
                            ),
                        });
                    }

                    let i2s_scale = if quant == QuantType::I2_S {
                        Self::extract_i2s_scale(raw, payload_bytes, weights)?
                    } else {
                        1.0
                    };

                    for (r, out_slot) in out.iter_mut().enumerate().take(rows) {
                        let row_off = r * blocks_per_row * block_bytes;
                        let mut row_sum = 0.0_f32;
                        for b in 0..blocks_per_row {
                            let boff = row_off + b * block_bytes;
                            let act_blk = &act_i8[b * block_elems..(b + 1) * block_elems];
                            match quant {
                                QuantType::TQ1_0 => {
                                    let block: &[u8; 54] = raw[boff..boff + 54]
                                        .try_into()
                                        .expect("validated TQ1_0 slice length");
                                    let mut w_i8 = [0_i8; 256];
                                    let scale = Self::unpack_tq1_0_block_i8(block, &mut w_i8);
                                    let acc: i32 = w_i8
                                        .iter()
                                        .zip(act_blk)
                                        .map(|(&w, &a)| i32::from(w) * i32::from(a))
                                        .sum();
                                    #[expect(
                                        clippy::cast_precision_loss,
                                        reason = "block int32 accumulation fits safely in f32 mantissa"
                                    )]
                                    {
                                        row_sum += (acc as f32) * (scale * act_scale);
                                    }
                                }
                                QuantType::TQ2_0 => {
                                    let block: &[u8; 66] = raw[boff..boff + 66]
                                        .try_into()
                                        .expect("validated TQ2_0 slice length");
                                    let mut w_i8 = [0_i8; 256];
                                    let scale = Self::unpack_tq2_0_block_i8(block, &mut w_i8);
                                    let acc: i32 = w_i8
                                        .iter()
                                        .zip(act_blk)
                                        .map(|(&w, &a)| i32::from(w) * i32::from(a))
                                        .sum();
                                    #[expect(
                                        clippy::cast_precision_loss,
                                        reason = "block int32 accumulation fits safely in f32 mantissa"
                                    )]
                                    {
                                        row_sum += (acc as f32) * (scale * act_scale);
                                    }
                                }
                                QuantType::I2_S => {
                                    let block: &[u8; 32] = raw[boff..boff + 32]
                                        .try_into()
                                        .expect("validated I2_S slice length");
                                    let mut w_i8 = [0_i8; 128];
                                    Self::unpack_i2s_block_i8(block, &mut w_i8);
                                    let acc: i32 = w_i8
                                        .iter()
                                        .zip(act_blk)
                                        .map(|(&w, &a)| i32::from(w) * i32::from(a))
                                        .sum();
                                    #[expect(
                                        clippy::cast_precision_loss,
                                        reason = "block int32 accumulation fits safely in f32 mantissa"
                                    )]
                                    {
                                        row_sum += (acc as f32) * (i2s_scale * act_scale);
                                    }
                                }
                                QuantType::Q8_0 => {
                                    let scale = Self::convert_f16(u16::from_le_bytes([
                                        raw[boff],
                                        raw[boff + 1],
                                    ]));
                                    #[expect(
                                        clippy::cast_possible_wrap,
                                        reason = "Q8_0 quants are signed i8"
                                    )]
                                    let acc: i32 = raw[boff + 2..boff + 34]
                                        .iter()
                                        .zip(act_blk)
                                        .map(|(&q_byte, &a)| i32::from(q_byte as i8) * i32::from(a))
                                        .sum();
                                    #[expect(
                                        clippy::cast_precision_loss,
                                        reason = "block int32 accumulation fits safely in f32 mantissa"
                                    )]
                                    {
                                        row_sum += (acc as f32) * (scale * act_scale);
                                    }
                                }
                                _ => {}
                            }
                        }
                        *out_slot = row_sum;
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
    pub(crate) fn subln(
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
            if !sum_sq.is_finite() {
                out_i8.fill(0);
                return Ok(1.0 / 127.0);
            }
            #[expect(clippy::cast_precision_loss, reason = "dimension fits in f32")]
            let inv_rms = 1.0 / ((sum_sq / len as f32) + eps).sqrt();
            if !inv_rms.is_finite() {
                out_i8.fill(0);
                return Ok(1.0 / 127.0);
            }
            let mut max_val = 0.0_f32;
            for i in 0..len {
                let x_norm = input[i] * inv_rms * norm_weight[i];
                if !x_norm.is_finite() {
                    out_i8.fill(0);
                    return Ok(1.0 / 127.0);
                }
                let abs_val = x_norm.abs();
                if abs_val > max_val {
                    max_val = abs_val;
                }
            }
            if !max_val.is_finite() || max_val < 1e-12 {
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

    /// Pure absmax symmetric int8 quantization (no `RMSNorm` pass) for already-normalized hidden states.
    pub(crate) fn quantize_absmax_i8(out_i8: &mut [i8], input: &[f32]) -> Result<f32, K3Error> {
        if input.is_empty() || out_i8.len() != input.len() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "lm_head_act".to_owned(),
                offset: 0,
                byte_len: out_i8.len(),
                buffer_len: input.len(),
                reason: "absmax quantization dimension mismatch".to_owned(),
            });
        }
        #[cfg(all(target_arch = "riscv64", not(miri)))]
        {
            if let Some(scale) = Self::quantize_absmax_i8_rvv(out_i8, input) {
                return Ok(scale);
            }
            out_i8.fill(0);
            Ok(1.0 / 127.0)
        }
        #[cfg(any(not(target_arch = "riscv64"), miri))]
        {
            Ok(Self::quantize_absmax_i8_scalar(out_i8, input))
        }
    }

    #[cfg(any(not(target_arch = "riscv64"), miri))]
    fn quantize_absmax_i8_scalar(out_i8: &mut [i8], input: &[f32]) -> f32 {
        let mut max_abs = 0.0_f32;
        for &val in input {
            if !val.is_finite() {
                out_i8.fill(0);
                return 1.0 / 127.0;
            }
            let abs_val = val.abs();
            if abs_val > max_abs {
                max_abs = abs_val;
            }
        }
        if !max_abs.is_finite() || max_abs < 1e-12 {
            out_i8.fill(0);
            return 1.0 / 127.0;
        }
        let scale = max_abs / 127.0;
        for (dst, &val) in out_i8.iter_mut().zip(input) {
            let q = (val / scale).round().clamp(-128.0, 127.0);
            #[expect(clippy::cast_possible_truncation, reason = "clamped to [-128, 127]")]
            {
                *dst = q as i8;
            }
        }
        scale
    }

    #[cfg(all(target_arch = "riscv64", not(miri)))]
    fn quantize_absmax_i8_rvv_find_max(input: &[f32]) -> Option<f32> {
        let ptr_x = input.as_ptr();
        let rem = input.len();
        let zero_f = 0.0_f32;
        let fmax = f32::MAX;
        let max_abs: f32;
        let mut bad_count = 0_usize;
        // SAFETY: `input` contains `rem` readable `f32`s; `vsetvli {vl}, {rem}, e32, m4` with
        // `vmfle.vf v0, |x|, f32::MAX` counts every non-finite (`NaN`/`+-Inf`) lane into
        // `bad_count` and replaces non-finite lanes with `0.0` before `vfredmax.vs`.
        unsafe {
            core::arch::asm!(
                "vsetivli zero, 1, e32, m1, ta, ma",
                "vfmv.s.f v2, {z}",
                "2:",
                "vsetvli {vl}, {rem}, e32, m4, ta, ma",
                "vle32.v v8, ({px})",
                "vfabs.v v12, v8",
                "vmfle.vf v0, v12, {fmax}",
                "vcpop.m t0, v0",
                "sub t1, {vl}, t0",
                "add {bad}, {bad}, t1",
                "vfmv.v.f v16, {z}",
                "vmerge.vvm v12, v16, v12, v0",
                "vfredmax.vs v2, v12, v2",
                "slli t0, {vl}, 2",
                "add {px}, {px}, t0",
                "sub {rem}, {rem}, {vl}",
                "bnez {rem}, 2b",
                "vsetivli zero, 1, e32, m1, ta, ma",
                "vfmv.f.s {amax}, v2",
                px = inout(reg) ptr_x => _,
                rem = inout(reg) rem => _,
                z = in(freg) zero_f,
                fmax = in(freg) fmax,
                amax = out(freg) max_abs,
                bad = inout(reg) bad_count,
                vl = out(reg) _,
                out("t0") _,
                out("t1") _,
                out("v0") _,
                out("v2") _,
                out("v8") _,
                out("v9") _,
                out("v10") _,
                out("v11") _,
                out("v12") _,
                out("v13") _,
                out("v14") _,
                out("v15") _,
                out("v16") _,
                out("v17") _,
                out("v18") _,
                out("v19") _,
                options(nostack, readonly),
            );
        }
        if bad_count > 0 || !max_abs.is_finite() || max_abs < 1e-12 {
            return None;
        }
        let scale = max_abs / 127.0;
        (scale.is_finite() && scale != 0.0).then_some(scale)
    }

    #[cfg(all(target_arch = "riscv64", not(miri)))]
    fn quantize_absmax_i8_rvv_store(out_i8: &mut [i8], input: &[f32], scale: f32) {
        let min_q = -128.0_f32;
        let max_q = 127.0_f32;
        let ptr_x = input.as_ptr();
        let ptr_out = out_i8.as_mut_ptr();
        let rem = input.len();
        let saved_frm: usize;
        // SAFETY: `input` and `out_i8` share length `rem`; `input` contains finite `f32`s and
        // `out_i8` is writable. Setting `frm = 4` (`RMM`, round-half-away-from-zero) before
        // `vfcvt.x.f.v` matches `f32::round` after clamping into `[-128.0, 127.0]`, and narrowing
        // `e32,m4 -> e16,m2 -> e8,m1` preserves `VLMAX` across all three configs on `VLEN >= 128`.
        // `frm` is restored before returning.
        unsafe {
            core::arch::asm!(
                "csrrwi {saved_frm}, frm, 4",
                "2:",
                "vsetvli {vl}, {rem}, e32, m4, ta, ma",
                "vle32.v v8, ({px})",
                "vfdiv.vf v8, v8, {scale}",
                "vfmax.vf v8, v8, {min_q}",
                "vfmin.vf v8, v8, {max_q}",
                "vfcvt.x.f.v v8, v8",
                "vsetvli zero, {vl}, e16, m2, ta, ma",
                "vnsra.wi v4, v8, 0",
                "vsetvli zero, {vl}, e8, m1, ta, ma",
                "vnsra.wi v2, v4, 0",
                "vse8.v v2, ({po})",
                "slli t0, {vl}, 2",
                "add {px}, {px}, t0",
                "add {po}, {po}, {vl}",
                "sub {rem}, {rem}, {vl}",
                "bnez {rem}, 2b",
                "csrw frm, {saved_frm}",
                px = inout(reg) ptr_x => _,
                po = inout(reg) ptr_out => _,
                rem = inout(reg) rem => _,
                scale = in(freg) scale,
                min_q = in(freg) min_q,
                max_q = in(freg) max_q,
                saved_frm = out(reg) saved_frm,
                vl = out(reg) _,
                out("t0") _,
                out("v2") _,
                out("v4") _,
                out("v5") _,
                out("v8") _,
                out("v9") _,
                out("v10") _,
                out("v11") _,
                options(nostack),
            );
        }
        let _ = saved_frm;
    }

    #[cfg(all(target_arch = "riscv64", not(miri)))]
    fn quantize_absmax_i8_rvv(out_i8: &mut [i8], input: &[f32]) -> Option<f32> {
        if input.is_empty() || out_i8.len() != input.len() {
            return None;
        }
        let scale = Self::quantize_absmax_i8_rvv_find_max(input)?;
        Self::quantize_absmax_i8_rvv_store(out_i8, input, scale);
        Some(scale)
    }

    #[cfg(not(feature = "with-kernels"))]
    #[expect(
        clippy::too_many_arguments,
        reason = "per-head scalar GQA attention helper"
    )]
    fn gqa_scalar_head(
        out_head: &mut [f32],
        q_head: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        active_scores: &mut [f32],
        n_heads_kv: usize,
        h_kv: usize,
        head_dim: usize,
        scale: f32,
    ) {
        let seq_len = active_scores.len();
        let mut max_score = f32::NEG_INFINITY;
        for (s, score_slot) in active_scores.iter_mut().enumerate() {
            let kv_offset = (s * n_heads_kv + h_kv) * head_dim;
            let k_slice = &k_cache[kv_offset..kv_offset + head_dim];
            let raw_dot: f32 = q_head.iter().zip(k_slice).map(|(&a, &b)| a * b).sum();
            let sc = raw_dot * scale;
            *score_slot = sc;
            if sc > max_score {
                max_score = sc;
            }
        }

        let mut sum_exp = 0.0_f32;
        for score in active_scores.iter_mut() {
            let exp_val = (*score - max_score).exp();
            *score = exp_val;
            sum_exp += exp_val;
        }

        #[expect(clippy::cast_precision_loss, reason = "seq_len fits in f32")]
        let inv_sum = if sum_exp > 0.0 && sum_exp.is_finite() {
            1.0_f32 / sum_exp
        } else {
            1.0_f32 / (seq_len as f32)
        };
        for score in active_scores.iter_mut() {
            *score *= inv_sum;
        }

        out_head.fill(0.0);
        for (s, &weight) in active_scores.iter().enumerate() {
            if weight == 0.0 {
                continue;
            }
            let kv_offset = (s * n_heads_kv + h_kv) * head_dim;
            let v_slice = &v_cache[kv_offset..kv_offset + head_dim];
            for (dst, &val) in out_head.iter_mut().zip(v_slice) {
                *dst += weight * val;
            }
        }
    }

    /// Executes grouped-query causal attention across `[seq_len, n_heads_kv, head_dim]`.
    #[expect(
        clippy::too_many_arguments,
        reason = "attention kernel takes explicit pre-allocated scratch buffers and geometry"
    )]
    fn dispatch_gqa_attention(
        out: &mut [f32],
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        scores: &mut [f32],
        n_heads_q: usize,
        n_heads_kv: usize,
        head_dim: usize,
        seq_len: usize,
        scale: f32,
    ) -> Result<(), K3Error> {
        #[cfg(feature = "with-kernels")]
        {
            if scores.len() < seq_len {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: "gqa_scores".to_owned(),
                    offset: 0,
                    byte_len: seq_len * size_of::<f32>(),
                    buffer_len: size_of_val(scores),
                    reason: format!(
                        "GQA scores scratch len {} < seq_len {seq_len}",
                        scores.len()
                    ),
                });
            }
            gqa_causal_attention_with_scratch_f32(
                out,
                q,
                k_cache,
                v_cache,
                n_heads_q,
                n_heads_kv,
                head_dim,
                seq_len,
                scale,
                &mut scores[..seq_len],
            )
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            if n_heads_q == 0 || n_heads_kv == 0 || head_dim == 0 || seq_len == 0 {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "gqa".to_owned(),
                    reason: format!(
                        "GQA dimensions must be non-zero (n_heads_q={n_heads_q}, n_heads_kv={n_heads_kv}, head_dim={head_dim}, seq_len={seq_len})"
                    ),
                });
            }
            if n_heads_q % n_heads_kv != 0 {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "gqa".to_owned(),
                    reason: format!(
                        "n_heads_q ({n_heads_q}) must be divisible by n_heads_kv ({n_heads_kv})"
                    ),
                });
            }
            let expected_q_len = n_heads_q * head_dim;
            let expected_kv_len = seq_len * n_heads_kv * head_dim;
            if q.len() != expected_q_len
                || out.len() != expected_q_len
                || k_cache.len() < expected_kv_len
                || v_cache.len() < expected_kv_len
                || scores.len() < seq_len
            {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: "gqa_attention".to_owned(),
                    offset: 0,
                    byte_len: expected_kv_len * size_of::<f32>(),
                    buffer_len: k_cache.len().min(v_cache.len()) * size_of::<f32>(),
                    reason: format!(
                        "GQA buffer length mismatch for q={}, out={}, kv={}/{}, scores={}, expected_q={expected_q_len}, expected_kv={expected_kv_len}",
                        q.len(),
                        out.len(),
                        k_cache.len(),
                        v_cache.len(),
                        scores.len()
                    ),
                });
            }

            let group_size = n_heads_q / n_heads_kv;
            let active_scores = &mut scores[..seq_len];

            for h_q in 0..n_heads_q {
                let h_kv = h_q / group_size;
                let q_head = &q[h_q * head_dim..(h_q + 1) * head_dim];
                let out_head = &mut out[h_q * head_dim..(h_q + 1) * head_dim];
                Self::gqa_scalar_head(
                    out_head,
                    q_head,
                    k_cache,
                    v_cache,
                    active_scores,
                    n_heads_kv,
                    h_kv,
                    head_dim,
                    scale,
                );
            }
            Ok(())
        }
    }

    #[cfg_attr(
        not(feature = "with-kernels"),
        expect(
            clippy::unnecessary_wraps,
            reason = "with-kernels apply_rope_inplace returns Result"
        )
    )]
    pub(crate) fn apply_rope_heads(
        &self,
        vecs: &mut [f32],
        head_d_even: usize,
        position: usize,
        seq_len: usize,
    ) -> Result<(), K3Error> {
        for head in vecs.chunks_exact_mut(head_d_even) {
            #[cfg(feature = "with-kernels")]
            {
                apply_rope_inplace(head, head_d_even, position, seq_len, self.rope_theta, None)?;
            }
            #[cfg(not(feature = "with-kernels"))]
            {
                let _ = seq_len;
                let half = head_d_even / 2;
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "position and head indices fit comfortably in f32"
                )]
                {
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
        }
        Ok(())
    }

    pub(crate) fn apply_rope_heads_with_cos_sin(
        vecs: &mut [f32],
        head_d_even: usize,
        cos: &[f32],
        sin: &[f32],
    ) -> Result<(), K3Error> {
        if head_d_even < 2 || head_d_even % 2 != 0 || vecs.len() % head_d_even != 0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "rope_qk".to_owned(),
                offset: 0,
                byte_len: size_of_val(vecs),
                buffer_len: head_d_even,
                reason: format!(
                    "invalid RoPE buffer len {} for head_d_even {head_d_even}",
                    vecs.len()
                ),
            });
        }
        let num_heads = vecs.len() / head_d_even;
        #[cfg(feature = "with-kernels")]
        {
            apply_rope_with_cos_sin(
                vecs,
                num_heads,
                head_d_even,
                head_d_even,
                RopeType::NeoX,
                cos,
                sin,
            )
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            let _ = num_heads;
            let half = head_d_even / 2;
            if cos.len() != half || sin.len() != half {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: "rope_cos_sin".to_owned(),
                    offset: 0,
                    byte_len: size_of_val(cos),
                    buffer_len: size_of_val(sin),
                    reason: "cos/sin table lengths must equal head_d_even / 2".to_owned(),
                });
            }
            for head in vecs.chunks_exact_mut(head_d_even) {
                for i in 0..half {
                    let c = cos[i];
                    let s = sin[i];
                    let (x1, x2) = (head[i], head[i + half]);
                    head[i] = x1 * c - x2 * s;
                    head[i + half] = x1 * s + x2 * c;
                }
            }
            Ok(())
        }
    }

    #[cfg(not(feature = "with-kernels"))]
    fn compute_rope_cos_sin_scalar(
        position: usize,
        rope_theta: f32,
        head_d_even: usize,
        cos: &mut [f32],
        sin: &mut [f32],
    ) {
        if position == 0 {
            cos.fill(1.0);
            sin.fill(0.0);
            return;
        }
        let half = head_d_even / 2;
        #[expect(
            clippy::cast_precision_loss,
            reason = "position and head indices fit comfortably in f32"
        )]
        {
            let pos_f32 = position as f32;
            for i in 0..half {
                let freq = 1.0 / rope_theta.powf((2 * i) as f32 / head_d_even as f32);
                let angle = pos_f32 * freq;
                sin[i] = angle.sin();
                cos[i] = angle.cos();
            }
        }
    }

    /// Projects normalized final `hidden` state through `output.weight` or tied `token_embd.weight`.
    pub(crate) fn project_lm_head(
        &self,
        logits: &mut [f32],
        hidden: &[f32],
        token_seed: Option<u32>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        if logits.is_empty() {
            return Ok(());
        }
        let need_bytes = aligned_scratch_slot_bytes(size_of::<f32>(), hidden.len().max(1))
            + aligned_scratch_slot_bytes(1, hidden.len().max(1))
            + 256;
        let mut ws = ScratchWorkspace::new(need_bytes)?;
        let scope = ws.scope();
        self.project_lm_head_with_scope(logits, hidden, token_seed, weights, None, &scope)
    }

    /// Projects normalized final `hidden` state through `output.weight` or tied `token_embd.weight`
    /// using caller-provided `scope` scratch (zero heap allocations).
    #[expect(
        clippy::too_many_lines,
        reason = "handles tied embeddings, ternary absmax dispatch, and exact f32 row dot products"
    )]
    pub(crate) fn project_lm_head_with_scope(
        &self,
        logits: &mut [f32],
        hidden: &[f32],
        token_seed: Option<u32>,
        weights: &dyn ModelWeights,
        pool: Option<&DecodeWorkerPool>,
        scope: &ScratchScope<'_>,
    ) -> Result<(), K3Error> {
        if logits.is_empty() {
            return Ok(());
        }

        if weights.tensor_count() == 0 {
            logits.fill(0.0);
            let bias = (hidden.first().copied().unwrap_or(0.0).abs() % 0.25) * 0.01;
            if let Some(token) = token_seed {
                let next_slot = (usize::try_from(token).unwrap_or(0) + 1) % logits.len();
                logits[next_slot] = 2.0 + bias;
            } else {
                let target_slot = if logits.len() > 9 { 9 } else { 0 };
                logits[target_slot] = 1.0 + bias;
            }
            return Ok(());
        }

        let head_weight = if weights.contains_tensor("output.weight") {
            weights.tensor("output.weight")?
        } else if weights.contains_tensor("token_embd.weight") {
            weights.tensor("token_embd.weight")?
        } else {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "output.weight".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "neither 'output.weight' nor tied 'token_embd.weight' found in weights"
                    .to_owned(),
            });
        };

        if !self.supports_quant(head_weight.quant()) {
            return Err(K3Error::TensorBoundsError {
                tensor_name: head_weight.name().to_owned(),
                offset: head_weight.offset(),
                byte_len: head_weight.byte_len(),
                buffer_len: head_weight.as_bytes().len(),
                reason: format!(
                    "unsupported quant type for BitLinear LM head: {:?}",
                    head_weight.quant()
                ),
            });
        }

        match head_weight.quant() {
            QuantType::TQ1_0 | QuantType::TQ2_0 | QuantType::I2_S => {
                let act_head_i8 = scope.alloc_i8_zeroed(hidden.len())?;
                let scale_head = Self::quantize_absmax_i8(act_head_i8, hidden)?;
                Self::dispatch_bitlinear(logits, &head_weight, act_head_i8, scale_head)
            }
            QuantType::F32 | QuantType::F16 | QuantType::BF16 | QuantType::Q8_0 => {
                let total_rows =
                    Self::matrix_row_col_dims(&head_weight, hidden.len(), logits.len())?;
                #[cfg(feature = "with-kernels")]
                {
                    let _ = (total_rows, pool);
                    let row_buf = scope.alloc_f32_zeroed(hidden.len())?;
                    for (r, slot) in logits.iter_mut().enumerate() {
                        dequantize_tensor_row(&head_weight, r, row_buf)?;
                        *slot = row_buf.iter().zip(hidden).map(|(&w, &h)| w * h).sum();
                    }
                    Ok(())
                }
                #[cfg(not(feature = "with-kernels"))]
                {
                    if head_weight.quant() == QuantType::F32 {
                        let req = total_rows * hidden.len() * 4;
                        if head_weight.as_bytes().len() < req {
                            return Err(K3Error::TensorBoundsError {
                                tensor_name: head_weight.name().to_owned(),
                                offset: head_weight.offset(),
                                byte_len: head_weight.byte_len(),
                                buffer_len: head_weight.as_bytes().len(),
                                reason: format!(
                                    "F32 LM head buffer length {} < required {req}",
                                    head_weight.as_bytes().len()
                                ),
                            });
                        }
                    } else if head_weight.quant() == QuantType::Q8_0 {
                        if hidden.len() % 32 != 0 {
                            return Err(K3Error::TensorBoundsError {
                                tensor_name: head_weight.name().to_owned(),
                                offset: head_weight.offset(),
                                byte_len: head_weight.byte_len(),
                                buffer_len: head_weight.as_bytes().len(),
                                reason: format!(
                                    "Q8_0 LM head hidden_dim {} must be multiple of 32",
                                    hidden.len()
                                ),
                            });
                        }
                    } else {
                        let req = total_rows * hidden.len() * 2;
                        if head_weight.as_bytes().len() < req {
                            return Err(K3Error::TensorBoundsError {
                                tensor_name: head_weight.name().to_owned(),
                                offset: head_weight.offset(),
                                byte_len: head_weight.byte_len(),
                                buffer_len: head_weight.as_bytes().len(),
                                reason: format!(
                                    "{:?} LM head buffer length {} < required {req}",
                                    head_weight.quant(),
                                    head_weight.as_bytes().len()
                                ),
                            });
                        }
                    }
                    if let Some(worker_pool) = pool {
                        worker_pool.par_for_each_rows(logits, 8, |row_off, chunk| {
                            for (local_r, slot) in chunk.iter_mut().enumerate() {
                                *slot = Self::dot_row_f32_scalar(
                                    &head_weight,
                                    row_off + local_r,
                                    total_rows,
                                    hidden,
                                )?;
                            }
                            Ok(())
                        })
                    } else {
                        for (r, slot) in logits.iter_mut().enumerate() {
                            *slot = Self::dot_row_f32_scalar(&head_weight, r, total_rows, hidden)?;
                        }
                        Ok(())
                    }
                }
            }
            other => Err(K3Error::TensorBoundsError {
                tensor_name: head_weight.name().to_owned(),
                offset: head_weight.offset(),
                byte_len: head_weight.byte_len(),
                buffer_len: head_weight.as_bytes().len(),
                reason: format!("unsupported quant type for LM head: {other:?}"),
            }),
        }
    }

    /// Computes the conservative `128-byte`-aligned [`ScratchWorkspace`] capacity (in bytes)
    /// needed for zero-allocation single-token `decode_step` execution up to `max_seq_len`.
    #[must_use]
    pub const fn decode_scratch_bytes(&self, max_seq_len: usize) -> usize {
        let hidden_dim = if self.hidden_dim > 16 {
            self.hidden_dim
        } else {
            16
        };
        let ffn_dim = if self.ffn_dim > 16 { self.ffn_dim } else { 16 };
        let head_d = if self.head_dim > 2 { self.head_dim } else { 2 };
        let head_d_even = if head_d % 2 == 0 { head_d } else { head_d - 1 };
        let n_q = if self.n_heads_q > 1 {
            self.n_heads_q
        } else {
            1
        };
        let n_kv = if self.n_heads_kv > 1 {
            self.n_heads_kv
        } else {
            1
        };
        let total_q = n_q.saturating_mul(head_d_even);
        let total_kv = n_kv.saturating_mul(head_d_even);
        let seq_cap = if max_seq_len > 1 { max_seq_len } else { 1 };
        let max_k = {
            let m1 = if hidden_dim > total_q {
                hidden_dim
            } else {
                total_q
            };
            if m1 > ffn_dim { m1 } else { ffn_dim }
        };
        let max_a_tiles = max_k.div_ceil(256).saturating_mul(16).saturating_mul(128);
        let hidden_bytes = aligned_scratch_slot_bytes(4, hidden_dim);

        let phase_a = aligned_scratch_slot_bytes(4, hidden_dim)
            + aligned_scratch_slot_bytes(4, total_q)
            + aligned_scratch_slot_bytes(4, ffn_dim)
            + aligned_scratch_slot_bytes(4, total_q)
            + aligned_scratch_slot_bytes(4, total_kv)
            + aligned_scratch_slot_bytes(4, total_kv)
            + aligned_scratch_slot_bytes(4, total_q)
            + aligned_scratch_slot_bytes(4, seq_cap)
            + aligned_scratch_slot_bytes(4, hidden_dim)
            + aligned_scratch_slot_bytes(4, ffn_dim)
            + aligned_scratch_slot_bytes(4, ffn_dim)
            + aligned_scratch_slot_bytes(4, ffn_dim)
            + aligned_scratch_slot_bytes(4, hidden_dim)
            + aligned_scratch_slot_bytes(1, hidden_dim)
            + aligned_scratch_slot_bytes(1, total_q)
            + aligned_scratch_slot_bytes(1, ffn_dim)
            + aligned_scratch_slot_bytes(1, max_a_tiles)
            + aligned_scratch_slot_bytes(4, head_d_even / 2)
            + aligned_scratch_slot_bytes(4, head_d_even / 2);

        let head_a_tiles = hidden_dim
            .div_ceil(256)
            .saturating_mul(16)
            .saturating_mul(128);
        #[cfg(feature = "with-kernels")]
        let head_view_bytes = QuantizedActQ8View::scratch_bytes_for_row(hidden_dim);
        #[cfg(not(feature = "with-kernels"))]
        let head_view_bytes = 0_usize;
        let base_phase_b = aligned_scratch_slot_bytes(1, hidden_dim)
            + aligned_scratch_slot_bytes(1, head_a_tiles)
            + aligned_scratch_slot_bytes(4, hidden_dim.div_ceil(32))
            + aligned_scratch_slot_bytes(4, hidden_dim)
            + aligned_scratch_slot_bytes(1, hidden_dim.div_ceil(32).saturating_mul(512) + 2048);
        let phase_b = if base_phase_b > head_view_bytes {
            base_phase_b
        } else {
            head_view_bytes
        };

        hidden_bytes + phase_a + phase_b + 512
    }

    /// Executes forward layer kernels for a single token seed at `position` using `128-byte`
    /// [`ScratchScope`] buffers without dynamic heap allocations inside the decode loop.
    #[allow(clippy::needless_range_loop, clippy::manual_checked_ops)]
    #[expect(
        clippy::too_many_lines,
        clippy::too_many_arguments,
        reason = "sequential layer loop with 7 BitLinear projections, deduplicated SubLN, and precomputed RoPE scratch"
    )]
    pub(crate) fn execute_token_step_with_scratch(
        &self,
        token_id: u32,
        position: usize,
        seq_len: usize,
        kv_storage: &mut KvStorage<'_>,
        weights: &dyn ModelWeights,
        mut cache: Option<&mut BitNetModelCache>,
        hidden: &mut [f32],
        scope: &ScratchScope<'_>,
    ) -> Result<(), K3Error> {
        let hidden_dim = self.hidden_dim.max(16);
        let ffn_dim = self.ffn_dim.max(16);
        let head_d = self.head_dim.max(2);
        let head_d_even = if head_d % 2 == 0 { head_d } else { head_d - 1 };
        let n_q = self.n_heads_q.max(1);
        let n_kv = self.n_heads_kv.max(1);
        let total_q = n_q * head_d_even;
        let total_kv = n_kv * head_d_even;
        let eps = self.rms_eps.max(1e-6);
        #[expect(clippy::cast_precision_loss, reason = "head dimension fits in f32")]
        let attn_scale = 1.0 / (head_d_even as f32).sqrt();
        let max_seq_alloc = seq_len.max(position.saturating_add(1)).max(1);
        let max_k = hidden_dim.max(total_q).max(ffn_dim);
        let a_tiles_len = max_k.div_ceil(256) * 16 * 128;
        let rope_half = head_d_even / 2;

        if hidden.len() != hidden_dim {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "hidden_scratch".to_owned(),
                offset: 0,
                byte_len: size_of_val(hidden),
                buffer_len: hidden_dim * size_of::<f32>(),
                reason: format!(
                    "hidden scratch len {} != expected hidden_dim {hidden_dim}",
                    hidden.len()
                ),
            });
        }
        Self::embed_token_opt(cache.as_deref_mut(), weights, token_id, hidden)?;

        // Allocate all 128-byte-aligned layer scratch slices ONCE outside the layer loop:
        let norm_w_hidden = scope.alloc_f32_zeroed(hidden_dim)?;
        let norm_w_total_q = scope.alloc_f32_zeroed(total_q)?;
        let norm_w_ffn = scope.alloc_f32_zeroed(ffn_dim)?;

        let q_vec = scope.alloc_f32_zeroed(total_q)?;
        let k_vec = scope.alloc_f32_zeroed(total_kv)?;
        let v_vec = scope.alloc_f32_zeroed(total_kv)?;

        let attn_out = scope.alloc_f32_zeroed(total_q)?;
        let attn_scores = scope.alloc_f32_zeroed(max_seq_alloc)?;
        let o_proj = scope.alloc_f32_zeroed(hidden_dim)?;

        let gate = scope.alloc_f32_zeroed(ffn_dim)?;
        let up = scope.alloc_f32_zeroed(ffn_dim)?;
        let ffn_inter = scope.alloc_f32_zeroed(ffn_dim)?;
        let d_proj = scope.alloc_f32_zeroed(hidden_dim)?;

        let act_hidden_i8 = scope.alloc_i8_zeroed(hidden_dim)?;
        let act_out_i8 = scope.alloc_i8_zeroed(total_q)?;
        let act_down_i8 = scope.alloc_i8_zeroed(ffn_dim)?;
        let a_tiles_i8 = scope.alloc_i8_zeroed(a_tiles_len)?;

        let rope_cos = scope.alloc_f32_zeroed(rope_half)?;
        let rope_sin = scope.alloc_f32_zeroed(rope_half)?;
        #[cfg(feature = "with-kernels")]
        compute_rope_cos_sin(position, self.rope_theta, 1.0, None, rope_cos, rope_sin)?;
        #[cfg(not(feature = "with-kernels"))]
        Self::compute_rope_cos_sin_scalar(
            position,
            self.rope_theta,
            head_d_even,
            rope_cos,
            rope_sin,
        );

        for layer in 0..self.num_layers {
            // -------------------------------------------------------------
            // 1. Pre-Attention Norm & Projections (attn_q, attn_k, attn_v)
            // -------------------------------------------------------------
            let attn_norm_key = StackLayerKey::format(layer, "attn_norm.weight");
            Self::load_norm_opt(
                cache.as_deref_mut(),
                weights,
                attn_norm_key.as_str(),
                norm_w_hidden,
            )?;

            // Deduplicated Pre-QKV SubLN pass shared cleanly by attn_q, attn_k, attn_v
            let scale_qkv = Self::subln(act_hidden_i8, hidden, norm_w_hidden, eps)?;

            // BitLinear attn_q
            let q_key = StackLayerKey::format(layer, "attn_q.weight");
            if weights.contains_tensor(q_key.as_str()) {
                Self::dispatch_proj_with_a_tiles_opt(
                    cache.as_deref_mut(),
                    weights,
                    q_key.as_str(),
                    q_vec,
                    act_hidden_i8,
                    scale_qkv,
                    a_tiles_i8,
                )?;
            } else if weights.tensor_count() == 0 {
                for (i, slot) in q_vec.iter_mut().enumerate() {
                    *slot = (f32::from(act_hidden_i8[i % hidden_dim]) * scale_qkv) * 0.1 + 0.5;
                }
            } else {
                return Err(missing_tensor_error(q_key.as_str()));
            }

            // BitLinear attn_k
            let k_key = StackLayerKey::format(layer, "attn_k.weight");
            if weights.contains_tensor(k_key.as_str()) {
                Self::dispatch_proj_with_a_tiles_opt(
                    cache.as_deref_mut(),
                    weights,
                    k_key.as_str(),
                    k_vec,
                    act_hidden_i8,
                    scale_qkv,
                    a_tiles_i8,
                )?;
            } else if weights.tensor_count() == 0 {
                for (i, slot) in k_vec.iter_mut().enumerate() {
                    *slot = (f32::from(act_hidden_i8[i % hidden_dim]) * scale_qkv) * 0.1 + 0.5;
                }
            } else {
                return Err(missing_tensor_error(k_key.as_str()));
            }

            // BitLinear attn_v
            let v_key = StackLayerKey::format(layer, "attn_v.weight");
            if weights.contains_tensor(v_key.as_str()) {
                Self::dispatch_proj_with_a_tiles_opt(
                    cache.as_deref_mut(),
                    weights,
                    v_key.as_str(),
                    v_vec,
                    act_hidden_i8,
                    scale_qkv,
                    a_tiles_i8,
                )?;
            } else if weights.tensor_count() == 0 {
                for (i, slot) in v_vec.iter_mut().enumerate() {
                    *slot = (f32::from(act_hidden_i8[i % hidden_dim]) * scale_qkv) * 0.1 + 0.5;
                }
            } else {
                return Err(missing_tensor_error(v_key.as_str()));
            }

            // -------------------------------------------------------------
            // 2. Rotary Position Embeddings (RoPE)
            // -------------------------------------------------------------
            Self::apply_rope_heads_with_cos_sin(q_vec, head_d_even, rope_cos, rope_sin)?;
            Self::apply_rope_heads_with_cos_sin(k_vec, head_d_even, rope_cos, rope_sin)?;

            // -------------------------------------------------------------
            // 3. KV Cache Residency & Causal Attention (GQA)
            // -------------------------------------------------------------
            attn_out.fill(0.0);
            match kv_storage {
                KvStorage::View(kv_view) => {
                    if kv_view.kv_dim() != total_kv {
                        return Err(kv_dim_mismatch_error(layer, total_kv, kv_view.kv_dim()));
                    }
                    let (k_slot, v_slot) = kv_view.write_pos_range(layer, position)?;
                    k_slot.copy_from_slice(k_vec);
                    v_slot.copy_from_slice(v_vec);
                    let (k_hist, v_hist) = kv_view.valid_kv_range(layer, position + 1)?;
                    let active_seq_len = k_hist.len() / total_kv;
                    Self::dispatch_gqa_attention(
                        attn_out,
                        q_vec,
                        k_hist,
                        v_hist,
                        attn_scores,
                        n_q,
                        n_kv,
                        head_d_even,
                        active_seq_len,
                        attn_scale,
                    )?;
                }
                KvStorage::Raw(kv_cache) => {
                    let layer_stride = kv_cache.len().checked_div(self.num_layers).unwrap_or(0);
                    let half_stride = layer_stride / 2;
                    let req_hist_elems = (position + 1).saturating_mul(total_kv);
                    if half_stride >= req_hist_elems && req_hist_elems > 0 {
                        let layer_start = layer * layer_stride;
                        let layer_buf = &mut kv_cache[layer_start..layer_start + layer_stride];
                        let write_off = position * total_kv;
                        layer_buf[write_off..write_off + total_kv].copy_from_slice(k_vec);
                        layer_buf[half_stride + write_off..half_stride + write_off + total_kv]
                            .copy_from_slice(v_vec);
                        let k_hist = &layer_buf[..req_hist_elems];
                        let v_hist = &layer_buf[half_stride..half_stride + req_hist_elems];
                        Self::dispatch_gqa_attention(
                            attn_out,
                            q_vec,
                            k_hist,
                            v_hist,
                            attn_scores,
                            n_q,
                            n_kv,
                            head_d_even,
                            position + 1,
                            attn_scale,
                        )?;
                    } else if weights.tensor_count() == 0 {
                        if half_stride > 0 {
                            let layer_start = layer * layer_stride;
                            let layer_end = (layer_start + layer_stride).min(kv_cache.len());
                            let layer_buf = &mut kv_cache[layer_start..layer_end];
                            let k_copy = total_kv.min(half_stride);
                            let pos_off = (position * k_copy) % half_stride;
                            let k_end = (pos_off + k_copy).min(half_stride);
                            let actual_copy = k_end - pos_off;
                            layer_buf[pos_off..pos_off + actual_copy]
                                .copy_from_slice(&k_vec[..actual_copy]);
                            layer_buf[half_stride + pos_off..half_stride + pos_off + actual_copy]
                                .copy_from_slice(&v_vec[..actual_copy]);
                        }
                        Self::dispatch_gqa_attention(
                            attn_out,
                            q_vec,
                            k_vec,
                            v_vec,
                            attn_scores,
                            n_q,
                            n_kv,
                            head_d_even,
                            1,
                            attn_scale,
                        )?;
                    } else {
                        return Err(kv_stride_error(
                            req_hist_elems,
                            self.num_layers,
                            size_of_val(*kv_cache),
                            half_stride,
                            position,
                        ));
                    }
                }
            }

            // -------------------------------------------------------------
            // 4. SubLN before Output Projection & Residual Connection
            // -------------------------------------------------------------
            let attn_sub_key = StackLayerKey::format(layer, "attn_sub_norm.weight");
            Self::load_norm_opt(
                cache.as_deref_mut(),
                weights,
                attn_sub_key.as_str(),
                norm_w_total_q,
            )?;

            // 4a. Mandatory SubLN before attn_output
            let scale_out = Self::subln(act_out_i8, attn_out, norm_w_total_q, eps)?;

            // 4b. BitLinear attn_output
            let o_key = StackLayerKey::format(layer, "attn_output.weight");
            if weights.contains_tensor(o_key.as_str()) {
                Self::dispatch_proj_with_a_tiles_opt(
                    cache.as_deref_mut(),
                    weights,
                    o_key.as_str(),
                    o_proj,
                    act_out_i8,
                    scale_out,
                    a_tiles_i8,
                )?;
                for (h, o) in hidden.iter_mut().zip(o_proj.iter()) {
                    *h += *o;
                }
            } else if weights.tensor_count() == 0 {
                for (i, h) in hidden.iter_mut().enumerate() {
                    *h += (f32::from(act_out_i8[i % total_q]) * scale_out) * 0.05;
                }
            } else {
                return Err(missing_tensor_error(o_key.as_str()));
            }

            // -------------------------------------------------------------
            // 5. FFN Norm & Projections (ffn_gate, ffn_up)
            // -------------------------------------------------------------
            let ffn_norm_key = StackLayerKey::format(layer, "ffn_norm.weight");
            Self::load_norm_opt(
                cache.as_deref_mut(),
                weights,
                ffn_norm_key.as_str(),
                norm_w_hidden,
            )?;

            // Deduplicated Pre-FFN SubLN pass shared cleanly by ffn_gate and ffn_up
            let scale_ffn = Self::subln(act_hidden_i8, hidden, norm_w_hidden, eps)?;

            // BitLinear ffn_gate
            let gate_key = StackLayerKey::format(layer, "ffn_gate.weight");
            if weights.contains_tensor(gate_key.as_str()) {
                Self::dispatch_proj_with_a_tiles_opt(
                    cache.as_deref_mut(),
                    weights,
                    gate_key.as_str(),
                    gate,
                    act_hidden_i8,
                    scale_ffn,
                    a_tiles_i8,
                )?;
            } else if weights.tensor_count() == 0 {
                for (i, slot) in gate.iter_mut().enumerate() {
                    *slot = (f32::from(act_hidden_i8[i % hidden_dim]) * scale_ffn) * 0.1 + 0.5;
                }
            } else {
                return Err(missing_tensor_error(gate_key.as_str()));
            }

            // BitLinear ffn_up
            let up_key = StackLayerKey::format(layer, "ffn_up.weight");
            if weights.contains_tensor(up_key.as_str()) {
                Self::dispatch_proj_with_a_tiles_opt(
                    cache.as_deref_mut(),
                    weights,
                    up_key.as_str(),
                    up,
                    act_hidden_i8,
                    scale_ffn,
                    a_tiles_i8,
                )?;
            } else if weights.tensor_count() == 0 {
                for (i, slot) in up.iter_mut().enumerate() {
                    *slot = (f32::from(act_hidden_i8[i % hidden_dim]) * scale_ffn) * 0.05 + 0.25;
                }
            } else {
                return Err(missing_tensor_error(up_key.as_str()));
            }

            // -------------------------------------------------------------
            // 6. Activation: ReLU^2 vs SwiGLU
            // -------------------------------------------------------------
            if self.use_relu2 {
                #[cfg(feature = "with-kernels")]
                {
                    relu_squared_gate_f32(ffn_inter, gate, up)?;
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
                    swiglu_f32(ffn_inter, gate, up)?;
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
            let ffn_sub_key = StackLayerKey::format(layer, "ffn_sub_norm.weight");
            Self::load_norm_opt(
                cache.as_deref_mut(),
                weights,
                ffn_sub_key.as_str(),
                norm_w_ffn,
            )?;

            // 7a. Mandatory SubLN before ffn_down
            let scale_down = Self::subln(act_down_i8, ffn_inter, norm_w_ffn, eps)?;

            // 7b. BitLinear ffn_down
            let down_key = StackLayerKey::format(layer, "ffn_down.weight");
            if weights.contains_tensor(down_key.as_str()) {
                Self::dispatch_proj_with_a_tiles_opt(
                    cache.as_deref_mut(),
                    weights,
                    down_key.as_str(),
                    d_proj,
                    act_down_i8,
                    scale_down,
                    a_tiles_i8,
                )?;
                for (h, d) in hidden.iter_mut().zip(d_proj.iter()) {
                    *h += *d;
                }
            } else if weights.tensor_count() == 0 {
                for (i, h) in hidden.iter_mut().enumerate() {
                    *h += (f32::from(act_down_i8[i % ffn_dim]) * scale_down) * 0.01;
                }
            } else {
                return Err(missing_tensor_error(down_key.as_str()));
            }
        }

        // -------------------------------------------------------------
        // 8. Output Normalization
        // -------------------------------------------------------------
        if weights.contains_tensor("output_norm.weight") {
            Self::load_norm_opt(cache, weights, "output_norm.weight", norm_w_hidden)?;
            let sum_sq: f32 = hidden.iter().map(|&x| x * x).sum();
            if sum_sq.is_finite() {
                #[expect(clippy::cast_precision_loss, reason = "dimension fits in f32")]
                let inv_rms = 1.0 / (sum_sq / (hidden_dim as f32) + eps).sqrt();
                if inv_rms.is_finite() {
                    for i in 0..hidden_dim {
                        hidden[i] = hidden[i] * inv_rms * norm_w_hidden[i];
                    }
                } else {
                    hidden.fill(0.0);
                }
            } else {
                hidden.fill(0.0);
            }
        } else if weights.tensor_count() > 0 {
            return Err(missing_tensor_error("output_norm.weight"));
        }

        Ok(())
    }

    fn validate_dimensions(&self, metadata: &ModelMetadata) -> Result<(), K3Error> {
        if self.num_layers == 0
            || self.hidden_dim < 16
            || self.ffn_dim == 0
            || self.n_heads_q == 0
            || self.n_heads_kv == 0
            || self.n_heads_q % self.n_heads_kv != 0
            || self.head_dim < 2
            || self.head_dim % 2 != 0
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: format!(
                    "invalid BitNet recipe dimensions (layers={}, hidden={}, ffn={}, n_q={}, n_kv={}, head_dim={})",
                    self.num_layers,
                    self.hidden_dim,
                    self.ffn_dim,
                    self.n_heads_q,
                    self.n_heads_kv,
                    self.head_dim
                ),
            });
        }

        if metadata.layer_count() > 0 && metadata.layer_count() != self.num_layers {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: format!(
                    "layer count mismatch: metadata has {}, recipe expects {}",
                    metadata.layer_count(),
                    self.num_layers
                ),
            });
        }
        if metadata.embedding_dim() > 0 && metadata.embedding_dim() != self.hidden_dim {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: format!(
                    "hidden_dim mismatch: metadata has {}, recipe expects {}",
                    metadata.embedding_dim(),
                    self.hidden_dim
                ),
            });
        }
        if metadata.head_count() > 0 && metadata.head_count() != self.n_heads_q {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: format!(
                    "n_heads_q mismatch: metadata has {}, recipe expects {}",
                    metadata.head_count(),
                    self.n_heads_q
                ),
            });
        }
        if metadata.head_count_kv() > 0 && metadata.head_count_kv() != self.n_heads_kv {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: format!(
                    "n_heads_kv mismatch: metadata has {}, recipe expects {}",
                    metadata.head_count_kv(),
                    self.n_heads_kv
                ),
            });
        }
        if let Some(meta_ffn) = metadata.feed_forward_length()
            && meta_ffn != self.ffn_dim
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: format!(
                    "ffn_dim mismatch: metadata has {meta_ffn}, recipe expects {}",
                    self.ffn_dim
                ),
            });
        }
        if let Some(meta_vocab) = metadata.vocab_size()
            && meta_vocab != self.vocab_size
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: self.architecture.to_owned(),
                reason: format!(
                    "vocab_size mismatch: metadata has {meta_vocab}, recipe expects {}",
                    self.vocab_size
                ),
            });
        }
        Ok(())
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
            QuantType::I2_S
                | QuantType::TQ1_0
                | QuantType::TQ2_0
                | QuantType::Q8_0
                | QuantType::F32
                | QuantType::F16
                | QuantType::BF16
        )
    }

    fn validate_model(
        &self,
        metadata: &ModelMetadata,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let norm_arch = Self::normalize_architecture_id(metadata.architecture());
        if norm_arch != Some(self.architecture) {
            return Err(K3Error::CapabilityMismatch {
                architecture: metadata.architecture().to_owned(),
                reason: format!(
                    "model architecture '{}' does not match recipe '{}'",
                    metadata.architecture(),
                    self.architecture
                ),
            });
        }

        self.validate_dimensions(metadata)?;

        for desc in metadata.tensors() {
            if !self.supports_quant(desc.quant()) {
                return Err(K3Error::unsupported_tensor_quant(
                    metadata.architecture(),
                    desc.name(),
                    desc.quant(),
                ));
            }
            if weights.tensor_count() > 0 {
                let view = weights.tensor(desc.name())?;
                if !self.supports_quant(view.quant()) {
                    return Err(K3Error::unsupported_tensor_quant(
                        metadata.architecture(),
                        view.name(),
                        view.quant(),
                    ));
                }
            }
        }

        if weights.tensor_count() > 0 {
            let vocab_contract = BitNetVocabContract {
                vocab_size: self.vocab_size,
                eos_token_id: None,
                eot_token_id: None,
                end_of_text_token_id: None,
                eot_piece_token_id: None,
                token_list_len: None,
                is_llama3_128k: self.vocab_size == crate::repack_cache::LLAMA3_BITNET_VOCAB_SIZE,
            };
            if weights.contains_tensor("token_embd.weight") {
                vocab_contract.validate_weights(weights, self.hidden_dim)?;
            }
            BitNetModelCache::preload_global(self, weights)?;
        }

        Ok(())
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        if BitNetModelCache::is_global_active() {
            BitNetModelCache::with_global(|cache| {
                self.prefill_step_with_opt_cache(ctx, weights, Some(cache))
            })
        } else {
            self.prefill_step_with_opt_cache(ctx, weights, None)
        }
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        if BitNetModelCache::is_global_active() {
            BitNetModelCache::with_global(|cache| {
                self.decode_step_with_opt_cache(ctx, weights, Some(cache))
            })
        } else {
            self.decode_step_with_opt_cache(ctx, weights, None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoWeights;

    impl ModelWeights for NoWeights {
        fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
            Err(missing_tensor_error(name))
        }

        fn tensor_count(&self) -> usize {
            0
        }

        fn total_weight_bytes(&self) -> usize {
            0
        }
    }

    #[test]
    fn test_stack_layer_key_formats_without_heap() {
        assert_eq!(
            StackLayerKey::format(0, "attn_q.weight").as_str(),
            "blk.0.attn_q.weight"
        );
        assert_eq!(
            StackLayerKey::format(31, "ffn_norm.weight").as_str(),
            "blk.31.ffn_norm.weight"
        );
        assert_eq!(
            StackLayerKey::format(128, "ffn_down.weight").as_str(),
            "blk.128.ffn_down.weight"
        );
        let max_key = StackLayerKey::format(usize::MAX, "attn_norm.weight");
        assert!(max_key.as_str().starts_with("blk."));
        assert!(max_key.as_str().ends_with(".attn_norm.weight"));
    }

    #[test]
    fn test_decode_scratch_bytes_tight_budget_succeeds_across_seq_lengths() {
        let mut recipe = BitNetRecipe::new();
        recipe.num_layers = 1;
        recipe.hidden_dim = 16;
        recipe.ffn_dim = 32;
        recipe.n_heads_q = 2;
        recipe.n_heads_kv = 2;
        recipe.head_dim = 8;
        recipe.vocab_size = 8;

        let short_bytes = recipe.decode_scratch_bytes(4);
        let long_bytes = recipe.decode_scratch_bytes(4096);
        assert!(long_bytes > short_bytes);

        let prompt = [1_u32, 2];
        let mut logits = [0.0_f32; 8];
        let mut kv_buf = [0.0_f32; 128];
        let mut ws = ScratchWorkspace::new(short_bytes).expect("valid tight ScratchWorkspace");
        let mut ctx = ExecutionContext::new(&prompt, 4, &mut logits, &mut kv_buf)
            .expect("ctx")
            .with_scratch(&mut ws);
        let weights = NoWeights;
        recipe.prefill_step(&mut ctx, &weights).expect("prefill");
        ctx.prepare_decode(3).expect("prepare decode");
        recipe
            .decode_step(&mut ctx, &weights)
            .expect("decode with exact tight scratch budget succeeds");
        assert_eq!(ctx.position(), 3);
    }
}
