//! Empirical Adversarial Challenger Test Suite for BitNetRecipe (M4 Epic 2).
//!
//! Validates SubLN normalization invariants, multi-layer sequential forward execution,
//! numerical stability across extreme float regimes, quantization format compatibility,
//! and activation function differentiation (ReLU^2 vs SwiGLU).

#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::undocumented_unsafe_blocks
)]

use k3_arch_bitnet::BitNetRecipe;
use k3_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType, TensorView};
use kernels_rvv::{bitlinear_i8_rvv, subln_quantize_absmax_i8};
use std::collections::HashMap;

/// In-memory synthetic weights container for BitNet test scenarios.
#[derive(Debug, Default)]
pub struct SyntheticBitNetWeights {
    pub tensor_data: HashMap<String, (QuantType, Vec<usize>, Vec<u8>)>,
}

impl SyntheticBitNetWeights {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert_tensor(
        &mut self,
        name: impl Into<String>,
        quant: QuantType,
        shape: Vec<usize>,
        data: Vec<u8>,
    ) {
        self.tensor_data.insert(name.into(), (quant, shape, data));
    }
}

impl ModelWeights for SyntheticBitNetWeights {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        let (key, (quant, shape, data)) =
            self.tensor_data
                .get_key_value(name)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: name.to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: format!("tensor '{name}' not found in synthetic weights"),
                })?;
        TensorView::from_bytes(key.as_str(), *quant, shape.as_slice(), 0, data.as_slice())
    }

    fn tensor_count(&self) -> usize {
        self.tensor_data.len()
    }

    fn total_weight_bytes(&self) -> usize {
        self.tensor_data.values().map(|(_, _, d)| d.len()).sum()
    }
}

/// Creates a synthetic TQ1_0 tensor buffer with a custom row/column pattern.
pub fn create_synthetic_tq1_0_pattern(
    rows: usize,
    cols: usize,
    pattern: impl Fn(usize, usize) -> i8,
) -> Vec<u8> {
    assert_eq!(cols % 256, 0);
    let blocks_per_row = cols / 256;
    let mut raw = vec![0u8; rows * blocks_per_row * 54];

    for r in 0..rows {
        for b in 0..blocks_per_row {
            let offset = (r * blocks_per_row + b) * 54;
            // FP16 scale = 1.0 (0x3C00)
            raw[offset..offset + 2].copy_from_slice(&[0x00, 0x3C]);
            let col_base = b * 256;
            for byte_idx in 0..51 {
                let mut byte_val = 0u8;
                let mut mul = 1u8;
                for trit_idx in 0..5 {
                    let c = col_base + byte_idx * 5 + trit_idx;
                    let val = pattern(r, c);
                    let t = match val {
                        -1 => 0u8,
                        0 => 1u8,
                        1 => 2u8,
                        _ => 1u8,
                    };
                    byte_val += t * mul;
                    mul = mul.saturating_mul(3);
                }
                raw[offset + 2 + byte_idx] = byte_val;
            }
            let last_val = pattern(r, col_base + 255);
            raw[offset + 53] = match last_val {
                -1 => 0u8,
                0 => 1u8,
                1 => 2u8,
                _ => 1u8,
            };
        }
    }
    raw
}

/// Creates a synthetic TQ1_0 tensor buffer with constant element value.
pub fn create_synthetic_tq1_0_raw(rows: usize, cols: usize, val: i8) -> Vec<u8> {
    create_synthetic_tq1_0_pattern(rows, cols, |_r, _c| val)
}

/// Creates a synthetic TQ2_0 tensor buffer.
/// `cols` must be a multiple of 256. For each row, 66 bytes are allocated per 256 elements.
pub fn create_synthetic_tq2_0_raw(rows: usize, cols: usize, val: i8) -> Vec<u8> {
    assert_eq!(cols % 256, 0);
    let blocks_per_row = cols / 256;
    let mut raw = vec![0u8; rows * blocks_per_row * 66];
    let q = match val {
        -1 => 0u8,
        0 => 1u8,
        1 => 2u8,
        _ => 1u8,
    };
    let packed_byte = q | (q << 2) | (q << 4) | (q << 6);

    for r in 0..rows {
        for b in 0..blocks_per_row {
            let offset = (r * blocks_per_row + b) * 66;
            // FP16 scale = 1.0 (0x3C00)
            raw[offset..offset + 2].copy_from_slice(&[0x00, 0x3C]);
            // 64 bytes of packed 2-bit quants
            raw[offset + 2..offset + 66].fill(packed_byte);
        }
    }
    raw
}

/// Creates a synthetic F32 vector buffer (for RMSNorm and SubLN weights).
pub fn create_synthetic_f32_raw(dim: usize, val: f32) -> Vec<u8> {
    let mut raw = vec![0u8; dim * 4];
    for i in 0..dim {
        raw[i * 4..(i + 1) * 4].copy_from_slice(&val.to_le_bytes());
    }
    raw
}

/// Builds a complete single-layer synthetic BitNet model weights collection.
pub fn build_synthetic_single_layer_weights(
    hidden_dim: usize,
    ffn_dim: usize,
    total_q: usize,
    total_kv: usize,
    attn_sub_norm_val: f32,
    ffn_sub_norm_val: f32,
) -> SyntheticBitNetWeights {
    let mut weights = SyntheticBitNetWeights::new();

    // Layer 0 SubLN and normalization weights
    weights.insert_tensor(
        "blk.0.attn_sub_norm.weight",
        QuantType::F32,
        vec![hidden_dim],
        create_synthetic_f32_raw(hidden_dim, attn_sub_norm_val),
    );
    weights.insert_tensor(
        "blk.0.attn_output_sub_norm.weight",
        QuantType::F32,
        vec![total_q],
        create_synthetic_f32_raw(total_q, 1.0),
    );
    weights.insert_tensor(
        "blk.0.ffn_sub_norm.weight",
        QuantType::F32,
        vec![hidden_dim],
        create_synthetic_f32_raw(hidden_dim, ffn_sub_norm_val),
    );
    weights.insert_tensor(
        "blk.0.ffn_down_sub_norm.weight",
        QuantType::F32,
        vec![ffn_dim],
        create_synthetic_f32_raw(ffn_dim, 1.0),
    );
    weights.insert_tensor(
        "output_norm.weight",
        QuantType::F32,
        vec![hidden_dim],
        create_synthetic_f32_raw(hidden_dim, 1.0),
    );

    // Layer 0 BitLinear projection matrices (TQ1_0)
    // Q: [cols=256, rows=256]
    weights.insert_tensor(
        "blk.0.attn_q.weight",
        QuantType::TQ1_0,
        vec![hidden_dim, total_q],
        create_synthetic_tq1_0_raw(total_q, hidden_dim, 1),
    );
    // K: [cols=256, rows=128]
    weights.insert_tensor(
        "blk.0.attn_k.weight",
        QuantType::TQ1_0,
        vec![hidden_dim, total_kv],
        create_synthetic_tq1_0_raw(total_kv, hidden_dim, 1),
    );
    // V: [cols=256, rows=128]
    weights.insert_tensor(
        "blk.0.attn_v.weight",
        QuantType::TQ1_0,
        vec![hidden_dim, total_kv],
        create_synthetic_tq1_0_raw(total_kv, hidden_dim, 1),
    );
    // Output: [cols=256, rows=256]
    weights.insert_tensor(
        "blk.0.attn_output.weight",
        QuantType::TQ1_0,
        vec![total_q, hidden_dim],
        create_synthetic_tq1_0_raw(hidden_dim, total_q, 1),
    );
    // Gate: [cols=256, rows=512] with alternating positive/negative rows
    weights.insert_tensor(
        "blk.0.ffn_gate.weight",
        QuantType::TQ1_0,
        vec![hidden_dim, ffn_dim],
        create_synthetic_tq1_0_pattern(
            ffn_dim,
            hidden_dim,
            |r, _c| {
                if r % 2 == 0 { 1 } else { -1 }
            },
        ),
    );
    // Up: [cols=256, rows=512]
    weights.insert_tensor(
        "blk.0.ffn_up.weight",
        QuantType::TQ1_0,
        vec![hidden_dim, ffn_dim],
        create_synthetic_tq1_0_raw(ffn_dim, hidden_dim, 1),
    );
    // Down: [cols=512, rows=256]
    weights.insert_tensor(
        "blk.0.ffn_down.weight",
        QuantType::TQ1_0,
        vec![ffn_dim, hidden_dim],
        create_synthetic_tq1_0_raw(hidden_dim, ffn_dim, 1),
    );

    // LM head output projection matrix
    weights.insert_tensor(
        "output.weight",
        QuantType::TQ1_0,
        vec![hidden_dim, 16],
        create_synthetic_tq1_0_raw(16, hidden_dim, 1),
    );

    weights
}

#[test]
fn test_empirical_bitnet_recipe_supports_quant() {
    let recipes = [
        BitNetRecipe::new(),
        BitNetRecipe::new_b158(),
        BitNetRecipe::new_25(),
    ];

    for recipe in &recipes {
        assert!(
            recipe.supports_quant(QuantType::TQ1_0),
            "{} must support TQ1_0",
            recipe.architecture_id()
        );
        assert!(
            recipe.supports_quant(QuantType::TQ2_0),
            "{} must support TQ2_0",
            recipe.architecture_id()
        );

        assert!(recipe.supports_quant(QuantType::Q8_0));
        assert!(recipe.supports_quant(QuantType::F32));
        assert!(recipe.supports_quant(QuantType::F16));
        assert!(recipe.supports_quant(QuantType::BF16));

        assert!(!recipe.supports_quant(QuantType::Q4_0));
        assert!(!recipe.supports_quant(QuantType::Q4_K_M));
        assert!(!recipe.supports_quant(QuantType::Q6_K));
    }
}

/// Suite 4: Recipe descriptors verification.
#[test]
fn test_bitnet_recipe_descriptors() {
    let b = BitNetRecipe::new();
    assert_eq!(b.architecture_id(), "bitnet");
    assert_eq!(b.recipe_version(), "1.0.0");
    assert_eq!(b.target_layouts(), &["native", "ime_tile_8x16"]);
    assert!(b.supports_quant(QuantType::TQ1_0));
    assert!(b.supports_quant(QuantType::TQ2_0));
    assert!(b.supports_quant(QuantType::Q8_0));
    assert!(b.supports_quant(QuantType::F32));
    assert!(b.supports_quant(QuantType::F16));
    assert!(b.supports_quant(QuantType::BF16));
    assert!(!b.supports_quant(QuantType::Q4_0));
    assert!(!b.supports_quant(QuantType::Q4_K_M));
    assert!(!b.supports_quant(QuantType::Q6_K));
    assert!(b.use_relu2);

    let b158 = BitNetRecipe::new_b158();
    assert_eq!(b158.architecture_id(), "bitnet-b1.58");
    assert_eq!(b158.recipe_version(), "1.0.0");
    assert!(b158.supports_quant(QuantType::TQ1_0));
    assert!(b158.supports_quant(QuantType::TQ2_0));
    assert!(b158.use_relu2);

    let b25 = BitNetRecipe::new_25();
    assert_eq!(b25.architecture_id(), "bitnet-25");
    assert_eq!(b25.recipe_version(), "1.0.0");
    assert!(b25.supports_quant(QuantType::TQ1_0));
    assert!(b25.supports_quant(QuantType::TQ2_0));
    assert!(!b25.use_relu2);
}

/// Suite 1: SubLN normalization invariants and projection sensitivity.
#[test]
fn test_bitnet_recipe_subln_invariants() {
    let hidden_dim = 256;
    let ffn_dim = 512;
    let n_heads_q = 4;
    let n_heads_kv = 2;
    let head_dim = 64;
    let total_q = n_heads_q * head_dim; // 256
    let total_kv = n_heads_kv * head_dim; // 128

    // 1. Direct SubLN + BitLinear projection scale verification:
    // When norm_weight scales by factor 2.0 or 0.5, the projection output scales proportionally
    let input = vec![1.5f32; 256];
    let w_norm_1 = vec![1.0f32; 256];
    let w_norm_2 = vec![2.0f32; 256];
    let w_norm_half = vec![0.5f32; 256];
    let eps = 1e-5;

    let mut act_1 = vec![0i8; 256];
    let mut act_2 = vec![0i8; 256];
    let mut act_half = vec![0i8; 256];

    let s_1 = subln_quantize_absmax_i8(&mut act_1, &input, &w_norm_1, eps).unwrap();
    let s_2 = subln_quantize_absmax_i8(&mut act_2, &input, &w_norm_2, eps).unwrap();
    let s_half = subln_quantize_absmax_i8(&mut act_half, &input, &w_norm_half, eps).unwrap();

    assert!(
        (s_2 / s_1 - 2.0).abs() < 1e-4,
        "scale must double when norm weight doubles"
    );
    assert!(
        (s_half / s_1 - 0.5).abs() < 1e-4,
        "scale must halve when norm weight halves"
    );

    let raw_tq1 = create_synthetic_tq1_0_raw(1, 256, 1);
    let shape = [256, 1];
    let view = TensorView::from_bytes("test_proj", QuantType::TQ1_0, &shape, 0, &raw_tq1).unwrap();

    let mut proj_1 = [0.0f32; 1];
    let mut proj_2 = [0.0f32; 1];
    let mut proj_half = [0.0f32; 1];

    bitlinear_i8_rvv(&mut proj_1, &view, &act_1, s_1).unwrap();
    bitlinear_i8_rvv(&mut proj_2, &view, &act_2, s_2).unwrap();
    bitlinear_i8_rvv(&mut proj_half, &view, &act_half, s_half).unwrap();

    assert!(
        (proj_2[0] / proj_1[0] - 2.0).abs() < 1e-3,
        "projection output must double"
    );
    assert!(
        (proj_half[0] / proj_1[0] - 0.5).abs() < 1e-3,
        "projection output must halve"
    );

    // 2. Full recipe execution with baseline weights
    let recipe = BitNetRecipe {
        architecture: "bitnet",
        num_layers: 1,
        hidden_dim,
        ffn_dim,
        n_heads_q,
        n_heads_kv,
        head_dim,
        vocab_size: 1024,
        rms_eps: 1e-5,
        rope_theta: 10_000.0,
        use_relu2: true,
    };

    let weights_base =
        build_synthetic_single_layer_weights(hidden_dim, ffn_dim, total_q, total_kv, 1.0, 1.0);

    let prompt = [5u32, 12u32];
    let mut logits_base = vec![0.0f32; 16];
    let mut kv_cache = vec![0.0f32; 1024];
    let mut ctx_base = ExecutionContext::new(&prompt, 32, &mut logits_base, &mut kv_cache)
        .expect("context creation");

    recipe
        .prefill_step(&mut ctx_base, &weights_base)
        .expect("prefill succeeds");
    assert_eq!(ctx_base.position(), 2);
    assert!(logits_base.iter().all(|x| x.is_finite()));
    assert!(logits_base.iter().any(|&x| x != 0.0));

    // 3. Omitting blk.0.attn_sub_norm.weight fails closed
    let mut bad_weights =
        build_synthetic_single_layer_weights(hidden_dim, ffn_dim, total_q, total_kv, 1.0, 1.0);
    bad_weights.tensor_data.remove("blk.0.attn_sub_norm.weight");
    let mut logits_err = vec![0.0f32; 16];
    let mut kv_err = vec![0.0f32; 1024];
    let mut ctx_err =
        ExecutionContext::new(&prompt, 32, &mut logits_err, &mut kv_err).expect("context creation");
    let err = recipe
        .prefill_step(&mut ctx_err, &bad_weights)
        .expect_err("must fail closed on missing attn_sub_norm");
    assert!(matches!(err, K3Error::TensorBoundsError { .. }));

    // 4. Omitting blk.0.ffn_sub_norm.weight fails closed
    let mut bad_weights_ffn =
        build_synthetic_single_layer_weights(hidden_dim, ffn_dim, total_q, total_kv, 1.0, 1.0);
    bad_weights_ffn
        .tensor_data
        .remove("blk.0.ffn_sub_norm.weight");
    let mut logits_err_ffn = vec![0.0f32; 16];
    let mut kv_err_ffn = vec![0.0f32; 1024];
    let mut ctx_err_ffn = ExecutionContext::new(&prompt, 32, &mut logits_err_ffn, &mut kv_err_ffn)
        .expect("context creation");
    let err = recipe
        .prefill_step(&mut ctx_err_ffn, &bad_weights_ffn)
        .expect_err("must fail closed on missing ffn_sub_norm");
    assert!(matches!(err, K3Error::TensorBoundsError { .. }));

    // 5. Corrupted SubLN weight dimension (128 != 256) fails closed
    let mut corrupted_weights =
        build_synthetic_single_layer_weights(hidden_dim, ffn_dim, total_q, total_kv, 1.0, 1.0);
    corrupted_weights.insert_tensor(
        "blk.0.attn_sub_norm.weight",
        QuantType::F32,
        vec![128],
        create_synthetic_f32_raw(128, 1.0),
    );
    let mut logits_corrupt = vec![0.0f32; 16];
    let mut kv_corrupt = vec![0.0f32; 1024];
    let mut ctx_corrupt = ExecutionContext::new(&prompt, 32, &mut logits_corrupt, &mut kv_corrupt)
        .expect("context creation");
    let err = recipe
        .prefill_step(&mut ctx_corrupt, &corrupted_weights)
        .expect_err("must fail closed on corrupted SubLN dimension");
    assert!(matches!(err, K3Error::TensorBoundsError { .. }));
}

/// Suite 2: Multi-layer sequential forward execution (prefill_step and decode_step).
#[test]
fn test_bitnet_recipe_forward() {
    let hidden_dim = 256;
    let ffn_dim = 512;
    let n_heads_q = 4;
    let n_heads_kv = 2;
    let head_dim = 64;
    let total_q = n_heads_q * head_dim; // 256
    let total_kv = n_heads_kv * head_dim; // 128
    let vocab_size = 1024;

    let recipes = [
        BitNetRecipe {
            architecture: "bitnet",
            num_layers: 2,
            hidden_dim,
            ffn_dim,
            n_heads_q,
            n_heads_kv,
            head_dim,
            vocab_size,
            rms_eps: 1e-5,
            rope_theta: 10_000.0,
            use_relu2: true,
        },
        BitNetRecipe {
            architecture: "bitnet-b1.58",
            num_layers: 2,
            hidden_dim,
            ffn_dim,
            n_heads_q,
            n_heads_kv,
            head_dim,
            vocab_size,
            rms_eps: 1e-5,
            rope_theta: 10_000.0,
            use_relu2: true,
        },
        BitNetRecipe {
            architecture: "bitnet-25",
            num_layers: 2,
            hidden_dim,
            ffn_dim,
            n_heads_q,
            n_heads_kv,
            head_dim,
            vocab_size,
            rms_eps: 1e-5,
            rope_theta: 10_000.0,
            use_relu2: false,
        },
    ];

    // Build 2-layer weights (Layer 0: TQ1_0, Layer 1: TQ2_0)
    let mut weights = SyntheticBitNetWeights::new();
    weights.insert_tensor(
        "output_norm.weight",
        QuantType::F32,
        vec![hidden_dim],
        create_synthetic_f32_raw(hidden_dim, 1.0),
    );
    weights.insert_tensor(
        "output.weight",
        QuantType::TQ1_0,
        vec![hidden_dim, 16],
        create_synthetic_tq1_0_raw(16, hidden_dim, 1),
    );

    for l in 0..2 {
        weights.insert_tensor(
            format!("blk.{l}.attn_sub_norm.weight"),
            QuantType::F32,
            vec![hidden_dim],
            create_synthetic_f32_raw(hidden_dim, 1.0),
        );
        weights.insert_tensor(
            format!("blk.{l}.attn_output_sub_norm.weight"),
            QuantType::F32,
            vec![total_q],
            create_synthetic_f32_raw(total_q, 1.0),
        );
        weights.insert_tensor(
            format!("blk.{l}.ffn_sub_norm.weight"),
            QuantType::F32,
            vec![hidden_dim],
            create_synthetic_f32_raw(hidden_dim, 1.0),
        );
        weights.insert_tensor(
            format!("blk.{l}.ffn_down_sub_norm.weight"),
            QuantType::F32,
            vec![ffn_dim],
            create_synthetic_f32_raw(ffn_dim, 1.0),
        );

        if l == 0 {
            weights.insert_tensor(
                format!("blk.{l}.attn_q.weight"),
                QuantType::TQ1_0,
                vec![hidden_dim, total_q],
                create_synthetic_tq1_0_raw(total_q, hidden_dim, 1),
            );
            weights.insert_tensor(
                format!("blk.{l}.attn_k.weight"),
                QuantType::TQ1_0,
                vec![hidden_dim, total_kv],
                create_synthetic_tq1_0_raw(total_kv, hidden_dim, 1),
            );
            weights.insert_tensor(
                format!("blk.{l}.attn_v.weight"),
                QuantType::TQ1_0,
                vec![hidden_dim, total_kv],
                create_synthetic_tq1_0_raw(total_kv, hidden_dim, 1),
            );
            weights.insert_tensor(
                format!("blk.{l}.attn_output.weight"),
                QuantType::TQ1_0,
                vec![total_q, hidden_dim],
                create_synthetic_tq1_0_raw(hidden_dim, total_q, 1),
            );
            weights.insert_tensor(
                format!("blk.{l}.ffn_gate.weight"),
                QuantType::TQ1_0,
                vec![hidden_dim, ffn_dim],
                create_synthetic_tq1_0_pattern(ffn_dim, hidden_dim, |r, _c| {
                    if r % 2 == 0 { 1 } else { -1 }
                }),
            );
            weights.insert_tensor(
                format!("blk.{l}.ffn_up.weight"),
                QuantType::TQ1_0,
                vec![hidden_dim, ffn_dim],
                create_synthetic_tq1_0_raw(ffn_dim, hidden_dim, 1),
            );
            weights.insert_tensor(
                format!("blk.{l}.ffn_down.weight"),
                QuantType::TQ1_0,
                vec![ffn_dim, hidden_dim],
                create_synthetic_tq1_0_raw(hidden_dim, ffn_dim, 1),
            );
        } else {
            weights.insert_tensor(
                format!("blk.{l}.attn_q.weight"),
                QuantType::TQ2_0,
                vec![hidden_dim, total_q],
                create_synthetic_tq2_0_raw(total_q, hidden_dim, 1),
            );
            weights.insert_tensor(
                format!("blk.{l}.attn_k.weight"),
                QuantType::TQ2_0,
                vec![hidden_dim, total_kv],
                create_synthetic_tq2_0_raw(total_kv, hidden_dim, 1),
            );
            weights.insert_tensor(
                format!("blk.{l}.attn_v.weight"),
                QuantType::TQ2_0,
                vec![hidden_dim, total_kv],
                create_synthetic_tq2_0_raw(total_kv, hidden_dim, 1),
            );
            weights.insert_tensor(
                format!("blk.{l}.attn_output.weight"),
                QuantType::TQ2_0,
                vec![total_q, hidden_dim],
                create_synthetic_tq2_0_raw(hidden_dim, total_q, 1),
            );
            weights.insert_tensor(
                format!("blk.{l}.ffn_gate.weight"),
                QuantType::TQ2_0,
                vec![hidden_dim, ffn_dim],
                create_synthetic_tq2_0_raw(ffn_dim, hidden_dim, 1),
            );
            weights.insert_tensor(
                format!("blk.{l}.ffn_up.weight"),
                QuantType::TQ2_0,
                vec![hidden_dim, ffn_dim],
                create_synthetic_tq2_0_raw(ffn_dim, hidden_dim, 1),
            );
            weights.insert_tensor(
                format!("blk.{l}.ffn_down.weight"),
                QuantType::TQ2_0,
                vec![ffn_dim, hidden_dim],
                create_synthetic_tq2_0_raw(hidden_dim, ffn_dim, 1),
            );
        }
    }

    for recipe in &recipes {
        let prompt = [10u32, 20u32, 30u32];
        let mut logits = vec![0.0f32; 16];
        let mut kv_cache = vec![0.0f32; 2048];
        let mut ctx = ExecutionContext::new(&prompt, 32, &mut logits, &mut kv_cache)
            .expect("context creation");

        // 1. Prefill step
        recipe
            .prefill_step(&mut ctx, &weights)
            .unwrap_or_else(|e| panic!("prefill must succeed for {}: {e:?}", recipe.architecture));
        assert_eq!(ctx.position(), 3);
        assert!(ctx.logits().iter().all(|x| x.is_finite()));
        assert!(ctx.logits().iter().any(|&x| x != 0.0));

        // 2. Decode step
        ctx.prepare_decode(42).expect("prepare decode");
        recipe
            .decode_step(&mut ctx, &weights)
            .unwrap_or_else(|e| panic!("decode must succeed for {}: {e:?}", recipe.architecture));
        assert_eq!(ctx.position(), 4);
        assert!(ctx.logits().iter().all(|x| x.is_finite()));

        // 3. Edge cases
        let empty_prompt: [u32; 0] = [];
        let mut logits_empty = vec![0.0f32; 16];
        let mut kv_empty = vec![0.0f32; 2048];
        let mut ctx_empty =
            ExecutionContext::new(&empty_prompt, 32, &mut logits_empty, &mut kv_empty)
                .expect("context creation");
        let err = recipe
            .prefill_step(&mut ctx_empty, &weights)
            .expect_err("must reject empty prompt");
        assert!(matches!(err, K3Error::ContextLengthExceeded { .. }));

        let mut ctx_no_decode =
            ExecutionContext::new(&prompt, 32, &mut logits, &mut kv_cache).expect("ctx");
        let err = recipe
            .decode_step(&mut ctx_no_decode, &weights)
            .expect_err("must reject decode without staged token");
        assert!(matches!(err, K3Error::CapabilityMismatch { .. }));
    }
}

/// Suite 3: Numerical stability across float extremes and activation differentiation.
#[test]
fn test_bitnet_subln_and_recipe() {
    let dim = 256;
    let norm_weight = vec![1.0f32; dim];
    let eps = 1e-5;

    // 1. Extreme high activation (1e4)
    let input_high = vec![1e4f32; dim];
    let mut out_high = vec![0i8; dim];
    let scale_high = subln_quantize_absmax_i8(&mut out_high, &input_high, &norm_weight, eps)
        .expect("high activation subln succeeds");
    assert!(scale_high.is_finite());
    assert!(scale_high > 0.0);
    assert!(out_high.iter().any(|&x| x != 0));

    // 2. Extreme small activation (1e-15)
    let input_small = vec![1e-15f32; dim];
    let mut out_small = vec![0i8; dim];
    let scale_small = subln_quantize_absmax_i8(&mut out_small, &input_small, &norm_weight, eps)
        .expect("small activation subln succeeds");
    assert!(scale_small.is_finite());
    assert!(scale_small > 0.0);

    // 3. All-zero activation
    let input_zero = vec![0.0f32; dim];
    let mut out_zero = vec![0i8; dim];
    let scale_zero = subln_quantize_absmax_i8(&mut out_zero, &input_zero, &norm_weight, eps)
        .expect("zero activation subln succeeds");
    assert_eq!(scale_zero, 1.0 / 127.0);
    assert!(out_zero.iter().all(|&x| x == 0));

    // 4. Exact scale propagation accuracy
    let input_2 = vec![2.0f32; dim];
    let mut act_i8 = vec![0i8; dim];
    let scale =
        subln_quantize_absmax_i8(&mut act_i8, &input_2, &norm_weight, eps).expect("subln succeeds");
    assert!((scale - (1.0 / 127.0)).abs() < 1e-5);
    assert_eq!(act_i8[0], 127);

    let raw_tq1 = create_synthetic_tq1_0_raw(1, dim, 1);
    let shape = [dim, 1];
    let view = TensorView::from_bytes("test_w", QuantType::TQ1_0, &shape, 0, &raw_tq1)
        .expect("tensor view");
    let mut out = [0.0f32; 1];
    bitlinear_i8_rvv(&mut out, &view, &act_i8, scale).expect("bitlinear succeeds");
    assert!(
        (out[0] - 256.0).abs() < 1e-3,
        "expected output 256.0, got {}",
        out[0]
    );

    // 5. Activation function differentiation (ReLU^2 vs SwiGLU)
    // Direct kernel activation differentiation on mixed positive and negative values:
    let g = [-2.0f32, -1.0, 0.5, 2.0];
    let u = [1.0f32; 4];
    let mut out_r = [0.0f32; 4];
    let mut out_s = [0.0f32; 4];
    kernels_rvv::relu_squared_gate_f32(&mut out_r, &g, &u).expect("relu2");
    kernels_rvv::swiglu_f32(&mut out_s, &g, &u).expect("swiglu");
    assert_eq!(out_r[0], 0.0);
    assert_eq!(out_r[1], 0.0);
    assert!((out_s[1] - (-0.26894)).abs() < 1e-3);
    assert_ne!(out_r, out_s);

    let recipe_relu = BitNetRecipe {
        architecture: "bitnet",
        num_layers: 1,
        hidden_dim: 256,
        ffn_dim: 512,
        n_heads_q: 4,
        n_heads_kv: 2,
        head_dim: 64,
        vocab_size: 1024,
        rms_eps: 1e-5,
        rope_theta: 10_000.0,
        use_relu2: true,
    };

    let recipe_swiglu = BitNetRecipe {
        architecture: "bitnet-25",
        num_layers: 1,
        hidden_dim: 256,
        ffn_dim: 512,
        n_heads_q: 4,
        n_heads_kv: 2,
        head_dim: 64,
        vocab_size: 1024,
        rms_eps: 1e-5,
        rope_theta: 10_000.0,
        use_relu2: false,
    };

    let mut weights = build_synthetic_single_layer_weights(256, 512, 256, 128, 1.0, 1.0);
    let mut embd_raw = vec![0u8; 1024 * 256 * 4];
    for t in 0..1024 {
        for i in 0..256 {
            let off = (t * 256 + i) * 4;
            let val = ((t * 7 + i * 13) % 100) as f32 * 0.02 - 1.0;
            embd_raw[off..off + 4].copy_from_slice(&val.to_le_bytes());
        }
    }
    weights.insert_tensor(
        "token_embd.weight",
        QuantType::F32,
        vec![256, 1024],
        embd_raw,
    );

    let mut out_norm_raw = vec![0u8; 256 * 4];
    for i in 0..256 {
        let val = 1.0 + (i as f32) * 0.005;
        out_norm_raw[i * 4..(i + 1) * 4].copy_from_slice(&val.to_le_bytes());
    }
    weights.insert_tensor(
        "output_norm.weight",
        QuantType::F32,
        vec![256],
        out_norm_raw,
    );

    let prompt = [3u32, 7u32];

    let mut logits_relu = vec![0.0f32; 16];
    let mut kv_relu = vec![0.0f32; 1024];
    let mut ctx_relu =
        ExecutionContext::new(&prompt, 32, &mut logits_relu, &mut kv_relu).expect("ctx");
    recipe_relu
        .prefill_step(&mut ctx_relu, &weights)
        .expect("prefill");

    let mut logits_swiglu = vec![0.0f32; 16];
    let mut kv_swiglu = vec![0.0f32; 1024];
    let mut ctx_swiglu =
        ExecutionContext::new(&prompt, 32, &mut logits_swiglu, &mut kv_swiglu).expect("ctx");
    recipe_swiglu
        .prefill_step(&mut ctx_swiglu, &weights)
        .expect("prefill");

    assert!(logits_relu.iter().all(|x| x.is_finite()));
    assert!(logits_swiglu.iter().all(|x| x.is_finite()));
    assert_ne!(logits_relu, logits_swiglu);
}
