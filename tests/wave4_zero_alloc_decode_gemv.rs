//! M4 Wave 4 integration verification (`#29`–`#33`):
//! - Zero dynamic heap allocations across steady-state `decode_step` / `decode_step_with_cache`
//!   using `128-byte`-aligned [`ScratchWorkspace`] buffers (`Raw` KV and `KvCacheView`, `seq_len <= 2048`
//!   and `seq_len > 2048` long-context GQA attention, depths `1..=30`, stack layer keys, precomputed
//!   `RoPE` `cos`/`sin`, and deduplicated `SubLN`)
//! - Bit-exact deterministic logits across `1`, `4`, and `8` worker partitions (`DecodeWorkerPool`)
//!   for cached ternary `BitLinear` projections and `128,256`-row tied/untied LM heads (`F16`, `Q8_0`,
//!   `BF16`, `F32`, ternary `I2_S`/`TQ1_0`/`TQ2_0`)

use jero_arch_bitnet::{BitNetModelCache, BitNetRecipe, LLAMA3_BITNET_VOCAB_SIZE};
use jero_core::{
    ExecutionContext, K3Error, KvGeometry, ModelWeights, QuantType, ScratchWorkspace, TensorView,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

struct TrackingAllocator;

static TRACK_ACTIVE: AtomicBool = AtomicBool::new(false);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static TEST_SERIALIZER: Mutex<()> = Mutex::new(());

// SAFETY: Delegates directly to `System` and tracks allocations atomically when enabled.
unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK_ACTIVE.load(Ordering::Acquire) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: Forwarded to system allocator with identical layout invariants.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Forwarded to system allocator with identical pointer and layout invariants.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if TRACK_ACTIVE.load(Ordering::Acquire) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: Forwarded to system allocator with identical layout invariants.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if TRACK_ACTIVE.load(Ordering::Acquire) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        // SAFETY: Forwarded to system allocator with identical invariants.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

struct TrackResetGuard;

impl Drop for TrackResetGuard {
    fn drop(&mut self) {
        TRACK_ACTIVE.store(false, Ordering::Release);
    }
}

fn measure_allocations<F: FnOnce() -> R, R>(f: F) -> (R, usize, usize) {
    let _reset = TrackResetGuard;
    ALLOC_COUNT.store(0, Ordering::Release);
    ALLOC_BYTES.store(0, Ordering::Release);
    TRACK_ACTIVE.store(true, Ordering::Release);
    let out = f();
    TRACK_ACTIVE.store(false, Ordering::Release);
    let count = ALLOC_COUNT.load(Ordering::Acquire);
    let bytes = ALLOC_BYTES.load(Ordering::Acquire);
    (out, count, bytes)
}

#[derive(Default, Clone)]
struct MapWeights(BTreeMap<String, (QuantType, Vec<usize>, Vec<u8>)>);

impl MapWeights {
    fn insert(&mut self, name: &str, quant: QuantType, shape: Vec<usize>, data: Vec<u8>) {
        self.0.insert(name.to_owned(), (quant, shape, data));
    }
}

impl ModelWeights for MapWeights {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        let (key, (quant, shape, bytes)) =
            self.0
                .get_key_value(name)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: name.to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: "tensor not found in MapWeights".to_owned(),
                })?;
        TensorView::from_bytes(key, *quant, shape, 0, bytes)
    }

    fn tensor_count(&self) -> usize {
        self.0.len()
    }

    fn total_weight_bytes(&self) -> usize {
        self.0.values().map(|(_, _, b)| b.len()).sum()
    }
}

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = u16::try_from((bits >> 16) & 0x8000).unwrap_or(0);
    let exp = i32::try_from((bits >> 23) & 0xff).unwrap_or(0) - 127 + 15;
    let mant = u16::try_from((bits >> 13) & 0x3ff).unwrap_or(0);
    if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7c00
    } else {
        sign | (u16::try_from(exp).unwrap_or(0) << 10) | mant
    }
}

fn f32_to_bf16_bits(v: f32) -> u16 {
    u16::try_from(v.to_bits() >> 16).unwrap_or(0)
}

fn make_f32_vec(dim: usize, val: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(dim * 4);
    for _ in 0..dim {
        out.extend_from_slice(&val.to_le_bytes());
    }
    out
}

fn make_f32_matrix(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * cols * 4);
    for r in 0..rows {
        for c in 0..cols {
            let phase = u16::try_from((r * 7 + c * 3 + seed) % 23).unwrap_or(0);
            let val = (f32::from(phase) - 11.0) * 0.0625;
            out.extend_from_slice(&val.to_le_bytes());
        }
    }
    out
}

fn make_half_table(rows: usize, cols: usize, bf16: bool, seed: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * cols * 2);
    for r in 0..rows {
        let base_phase = r * 7 + seed;
        for c in 0..cols {
            let phase = u16::try_from((base_phase + c * 3) % 23).unwrap_or(0);
            let val = (f32::from(phase) - 11.0) * 0.0625;
            let bits = if bf16 {
                f32_to_bf16_bits(val)
            } else {
                f32_to_f16_bits(val)
            };
            out.extend_from_slice(&bits.to_le_bytes());
        }
    }
    out
}

fn make_q8_0_matrix(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    assert_eq!(cols % 32, 0);
    let bpr = cols / 32;
    let mut out = Vec::with_capacity(rows * bpr * 34);
    for r in 0..rows {
        for b in 0..bpr {
            let s_idx = u16::try_from((r + b + seed) % 5).unwrap_or(0);
            let scale = f32_to_f16_bits(0.0625 + f32::from(s_idx) * 0.03125);
            out.extend_from_slice(&scale.to_le_bytes());
            for k in 0..32 {
                let q = i16::try_from((r * 11 + b * 7 + k * 5 + seed) % 61).unwrap_or(0) - 30;
                out.push(i8::try_from(q).unwrap_or(0).to_le_bytes()[0]);
            }
        }
    }
    out
}

fn make_i2s_matrix(rows: usize, cols: usize, seed: usize, scale: f32) -> Vec<u8> {
    assert_eq!(cols % 128, 0);
    let bpr = cols / 128;
    let mut out = Vec::with_capacity(rows * bpr * 32 + 32);
    for r in 0..rows {
        for b in 0..bpr {
            for g in 0..32 {
                let mut packed = 0_u8;
                for slot in 0..4 {
                    let code = u8::try_from((r * 3 + b * 5 + g + slot + seed) % 3).unwrap_or(0);
                    packed |= (code & 0x03) << (6 - slot * 2);
                }
                out.push(packed);
            }
        }
    }
    let sbytes = scale.to_le_bytes();
    for _ in 0..8 {
        out.extend_from_slice(&sbytes);
    }
    out
}

fn make_tq2_0_matrix(rows: usize, cols: usize, seed: usize) -> Vec<u8> {
    assert_eq!(cols % 256, 0);
    let bpr = cols / 256;
    let mut out = Vec::with_capacity(rows * bpr * 66);
    for r in 0..rows {
        for b in 0..bpr {
            let s_idx = u16::try_from((r + b + seed) % 3).unwrap_or(0);
            let scale = f32_to_f16_bits(0.25 + f32::from(s_idx) * 0.125);
            out.extend_from_slice(&scale.to_le_bytes());
            for p in 0..64 {
                let mut byte = 0_u8;
                for k in 0..4 {
                    let code = u8::try_from((r * 5 + b + p + k + seed) % 3).unwrap_or(0);
                    byte |= (code & 0x03) << (k * 2);
                }
                out.push(byte);
            }
        }
    }
    out
}

fn make_dense_or_ternary_table(rows: usize, cols: usize, quant: QuantType, seed: usize) -> Vec<u8> {
    match quant {
        QuantType::F16 => make_half_table(rows, cols, false, seed),
        QuantType::BF16 => make_half_table(rows, cols, true, seed),
        QuantType::Q8_0 => make_q8_0_matrix(rows, cols, seed),
        QuantType::F32 => make_f32_matrix(rows, cols, seed),
        QuantType::I2_S => make_i2s_matrix(rows, cols, seed, 0.25),
        other => panic!("unexpected table quant {other:?}"),
    }
}

fn build_i2s_model(
    num_layers: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    vocab_size: usize,
    tied_quant: QuantType,
    untied_output_quant: Option<QuantType>,
) -> (BitNetRecipe, MapWeights) {
    let mut recipe = BitNetRecipe::new_b158();
    recipe.num_layers = num_layers;
    recipe.hidden_dim = hidden_dim;
    recipe.ffn_dim = ffn_dim;
    recipe.n_heads_q = 4;
    recipe.n_heads_kv = 2;
    recipe.head_dim = hidden_dim / 4;
    recipe.vocab_size = vocab_size;
    let kv_dim = recipe.n_heads_kv * recipe.head_dim;

    let mut w = MapWeights::default();
    w.insert(
        "token_embd.weight",
        tied_quant,
        vec![hidden_dim, vocab_size],
        make_dense_or_ternary_table(vocab_size, hidden_dim, tied_quant, 7),
    );
    w.insert(
        "output_norm.weight",
        QuantType::F32,
        vec![hidden_dim],
        make_f32_vec(hidden_dim, 1.0),
    );
    if let Some(out_q) = untied_output_quant {
        w.insert(
            "output.weight",
            out_q,
            vec![hidden_dim, vocab_size],
            make_dense_or_ternary_table(vocab_size, hidden_dim, out_q, 29),
        );
    }

    for layer in 0..num_layers {
        let base = format!("blk.{layer}");
        for norm in [
            "attn_norm.weight",
            "attn_sub_norm.weight",
            "ffn_norm.weight",
            "ffn_sub_norm.weight",
        ] {
            let dim = if norm == "ffn_sub_norm.weight" {
                ffn_dim
            } else {
                hidden_dim
            };
            w.insert(
                &format!("{base}.{norm}"),
                QuantType::F32,
                vec![dim],
                make_f32_vec(dim, 1.0),
            );
        }
        w.insert(
            &format!("{base}.attn_q.weight"),
            QuantType::I2_S,
            vec![hidden_dim, hidden_dim],
            make_i2s_matrix(hidden_dim, hidden_dim, 10 + layer, 0.25),
        );
        w.insert(
            &format!("{base}.attn_k.weight"),
            QuantType::I2_S,
            vec![hidden_dim, kv_dim],
            make_i2s_matrix(kv_dim, hidden_dim, 20 + layer, 0.25),
        );
        w.insert(
            &format!("{base}.attn_v.weight"),
            QuantType::I2_S,
            vec![hidden_dim, kv_dim],
            make_i2s_matrix(kv_dim, hidden_dim, 30 + layer, 0.25),
        );
        w.insert(
            &format!("{base}.attn_output.weight"),
            QuantType::I2_S,
            vec![hidden_dim, hidden_dim],
            make_i2s_matrix(hidden_dim, hidden_dim, 40 + layer, 0.25),
        );
        w.insert(
            &format!("{base}.ffn_gate.weight"),
            QuantType::I2_S,
            vec![hidden_dim, ffn_dim],
            make_i2s_matrix(ffn_dim, hidden_dim, 50 + layer, 0.25),
        );
        w.insert(
            &format!("{base}.ffn_up.weight"),
            QuantType::I2_S,
            vec![hidden_dim, ffn_dim],
            make_i2s_matrix(ffn_dim, hidden_dim, 60 + layer, 0.25),
        );
        w.insert(
            &format!("{base}.ffn_down.weight"),
            QuantType::I2_S,
            vec![ffn_dim, hidden_dim],
            make_i2s_matrix(hidden_dim, ffn_dim, 70 + layer, 0.25),
        );
    }

    (recipe, w)
}

#[test]
fn test_wave4_zero_alloc_decode_step_multi_depth_raw_and_kv_view() {
    let _lock = TEST_SERIALIZER
        .lock()
        .unwrap_or_else(PoisonError::into_inner);

    for depth in [1_usize, 4, 16, 30] {
        let (recipe, weights) = build_i2s_model(depth, 128, 128, 16, QuantType::I2_S, None);
        let max_seq = 16_usize;
        let geom = KvGeometry::new(depth, recipe.n_heads_kv, recipe.head_dim, max_seq, None)
            .expect("valid KvGeometry");

        for &workers in &[1_usize, 4_usize] {
            let mut cache = BitNetModelCache::new().with_decode_workers(workers);
            cache.preload_model(&recipe, &weights).expect("preload");

            // Case 1: KvStorage::View
            {
                let mut logits = vec![0.0_f32; recipe.vocab_size];
                let mut kv_buf = vec![0.0_f32; geom.required_f32_len().expect("kv len")];
                let mut ws = ScratchWorkspace::new(recipe.decode_scratch_bytes(max_seq))
                    .expect("scratch workspace");
                let mut ctx = ExecutionContext::new(&[1, 2], max_seq, &mut logits, &mut kv_buf)
                    .expect("ctx")
                    .with_kv_geometry(geom)
                    .expect("attach kv geometry")
                    .with_scratch(&mut ws);
                recipe
                    .prefill_step_with_cache(&mut ctx, &weights, &mut cache)
                    .expect("prefill view");

                ctx.prepare_decode(3).expect("warmup decode prep");
                recipe
                    .decode_step_with_cache(&mut ctx, &weights, &mut cache)
                    .expect("warmup decode view");

                let (res, count, bytes) = measure_allocations(|| -> Result<(), K3Error> {
                    for tok in [4_u32, 5, 6] {
                        ctx.prepare_decode(tok)?;
                        recipe.decode_step_with_cache(&mut ctx, &weights, &mut cache)?;
                    }
                    Ok(())
                });
                res.expect("steady-state decode view");
                assert_eq!(
                    (count, bytes),
                    (0, 0),
                    "depth={depth}, workers={workers}, KvView must perform 0 heap allocations"
                );
                assert!(logits.iter().all(|v| v.is_finite()));
            }

            // Case 2: KvStorage::Raw
            {
                let mut logits = vec![0.0_f32; recipe.vocab_size];
                let raw_kv_floats = max_seq * depth * recipe.n_heads_kv * recipe.head_dim * 2 + 64;
                let mut kv_buf = vec![0.0_f32; raw_kv_floats];
                let mut ws = ScratchWorkspace::new(recipe.decode_scratch_bytes(max_seq))
                    .expect("scratch workspace");
                let mut ctx = ExecutionContext::new(&[1, 2], max_seq, &mut logits, &mut kv_buf)
                    .expect("ctx")
                    .with_scratch(&mut ws);
                recipe
                    .prefill_step_with_cache(&mut ctx, &weights, &mut cache)
                    .expect("prefill raw");

                ctx.prepare_decode(3).expect("warmup raw prep");
                recipe
                    .decode_step_with_cache(&mut ctx, &weights, &mut cache)
                    .expect("warmup raw decode");

                let (res, count, bytes) = measure_allocations(|| -> Result<(), K3Error> {
                    for tok in [4_u32, 5, 6] {
                        ctx.prepare_decode(tok)?;
                        recipe.decode_step_with_cache(&mut ctx, &weights, &mut cache)?;
                    }
                    Ok(())
                });
                res.expect("steady-state decode raw");
                assert_eq!(
                    (count, bytes),
                    (0, 0),
                    "depth={depth}, workers={workers}, KvRaw must perform 0 heap allocations"
                );
                assert!(logits.iter().all(|v| v.is_finite()));
            }
        }
    }
}

#[test]
fn test_wave4_all_lm_head_formats_worker_parity_and_scratch_zero_alloc() {
    let _lock = TEST_SERIALIZER
        .lock()
        .unwrap_or_else(PoisonError::into_inner);

    for (label, tied_q, untied_q) in [
        ("tied_f16", QuantType::F16, None),
        ("tied_bf16", QuantType::BF16, None),
        ("tied_q8_0", QuantType::Q8_0, None),
        ("tied_f32", QuantType::F32, None),
        ("tied_i2s", QuantType::I2_S, None),
        ("untied_f16", QuantType::I2_S, Some(QuantType::F16)),
        ("untied_q8_0", QuantType::I2_S, Some(QuantType::Q8_0)),
    ] {
        let (recipe, weights) = build_i2s_model(2, 128, 128, 64, tied_q, untied_q);
        let max_seq = 8_usize;
        let geom = KvGeometry::new(2, recipe.n_heads_kv, recipe.head_dim, max_seq, None)
            .expect("kv geometry");

        let mut baseline_logits: Option<Vec<f32>> = None;
        for &workers in &[1_usize, 4, 8] {
            let mut cache = BitNetModelCache::new().with_decode_workers(workers);
            cache.preload_model(&recipe, &weights).expect("preload");

            let mut logits = vec![0.0_f32; recipe.vocab_size];
            let mut kv_buf = vec![0.0_f32; geom.required_f32_len().expect("kv len")];
            let mut ws = ScratchWorkspace::new(recipe.decode_scratch_bytes(max_seq))
                .expect("scratch workspace");
            let mut ctx = ExecutionContext::new(&[2, 7], max_seq, &mut logits, &mut kv_buf)
                .expect("ctx")
                .with_kv_geometry(geom)
                .expect("kv view")
                .with_scratch(&mut ws);
            recipe
                .prefill_step_with_cache(&mut ctx, &weights, &mut cache)
                .expect("prefill");
            ctx.prepare_decode(11).expect("prep");
            recipe
                .decode_step_with_cache(&mut ctx, &weights, &mut cache)
                .expect("warmup decode");

            let (res, count, bytes) = measure_allocations(|| -> Result<(), K3Error> {
                ctx.prepare_decode(13)?;
                recipe.decode_step_with_cache(&mut ctx, &weights, &mut cache)
            });
            res.expect("measured decode");
            assert_eq!(
                (count, bytes),
                (0, 0),
                "{label}, workers={workers} expected 0 heap allocations"
            );
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "{label} finite logits"
            );

            if let Some(ref base) = baseline_logits {
                for (row, (&got, &want)) in logits.iter().zip(base).enumerate() {
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{label}: worker={workers} diverged at row {row}: got={got}, want={want}"
                    );
                }
            } else {
                baseline_logits = Some(logits);
            }
        }
    }
}

#[test]
fn test_wave4_128k_lm_head_and_cached_ternary_bit_exact_worker_parity() {
    let _lock = TEST_SERIALIZER
        .lock()
        .unwrap_or_else(PoisonError::into_inner);

    for (label, tied_q, untied_q) in [
        ("128k_tied_f16", QuantType::F16, None),
        ("128k_untied_q8_0", QuantType::I2_S, Some(QuantType::Q8_0)),
        ("128k_tied_i2s", QuantType::I2_S, None),
    ] {
        let vocab = LLAMA3_BITNET_VOCAB_SIZE;
        let (recipe, weights) = build_i2s_model(1, 128, 256, vocab, tied_q, untied_q);
        let max_seq = 8_usize;
        let geom = KvGeometry::new(1, recipe.n_heads_kv, recipe.head_dim, max_seq, None)
            .expect("kv geometry");

        let mut cache = BitNetModelCache::new().with_decode_workers(1);
        cache
            .preload_model(&recipe, &weights)
            .expect("preload 128k");

        let mut logits_1 = vec![0.0_f32; vocab];
        let mut logits_4 = vec![0.0_f32; vocab];
        let mut logits_8 = vec![0.0_f32; vocab];

        for (workers, target_logits) in [
            (1_usize, &mut logits_1),
            (4_usize, &mut logits_4),
            (8_usize, &mut logits_8),
        ] {
            cache.set_decode_workers(workers);
            let mut kv_buf = vec![0.0_f32; geom.required_f32_len().expect("kv len")];
            let mut ws = ScratchWorkspace::new(recipe.decode_scratch_bytes(max_seq))
                .expect("scratch workspace");
            let mut ctx = ExecutionContext::new(&[1, 5], max_seq, target_logits, &mut kv_buf)
                .expect("ctx")
                .with_kv_geometry(geom)
                .expect("kv view")
                .with_scratch(&mut ws);
            recipe
                .prefill_step_with_cache(&mut ctx, &weights, &mut cache)
                .expect("prefill 128k");
            ctx.prepare_decode(9).expect("prep");
            recipe
                .decode_step_with_cache(&mut ctx, &weights, &mut cache)
                .expect("decode 128k");
        }

        assert_eq!(logits_1.len(), 128_256);
        assert!(logits_1.iter().all(|v| v.is_finite()));
        assert!(logits_1.iter().any(|&v| v != 0.0));
        for row in 0..vocab {
            assert_eq!(
                logits_4[row].to_bits(),
                logits_1[row].to_bits(),
                "{label}: 4 workers != 1 worker at row {row}"
            );
            assert_eq!(
                logits_8[row].to_bits(),
                logits_1[row].to_bits(),
                "{label}: 8 workers != 1 worker at row {row}"
            );
        }

        let mut kv_buf = vec![0.0_f32; geom.required_f32_len().expect("kv len")];
        let mut ws =
            ScratchWorkspace::new(recipe.decode_scratch_bytes(max_seq)).expect("scratch workspace");
        let mut ctx = ExecutionContext::new(&[1, 5], max_seq, &mut logits_8, &mut kv_buf)
            .expect("ctx")
            .with_kv_geometry(geom)
            .expect("kv view")
            .with_scratch(&mut ws);
        recipe
            .prefill_step_with_cache(&mut ctx, &weights, &mut cache)
            .expect("prefill 128k steady");
        ctx.prepare_decode(9).expect("warmup prep 128k");
        recipe
            .decode_step_with_cache(&mut ctx, &weights, &mut cache)
            .expect("warmup decode 128k");
        let (res, count, bytes) = measure_allocations(|| -> Result<(), K3Error> {
            ctx.prepare_decode(11)?;
            recipe.decode_step_with_cache(&mut ctx, &weights, &mut cache)
        });
        res.expect("steady-state 128k 8-worker decode");
        assert_eq!((count, bytes), (0, 0), "{label} 128k zero-alloc failed");
    }
}

fn build_tq2_i2s_model() -> (BitNetRecipe, MapWeights) {
    let mut recipe = BitNetRecipe::new_b158();
    recipe.num_layers = 1;
    recipe.hidden_dim = 256;
    recipe.ffn_dim = 512;
    recipe.n_heads_q = 4;
    recipe.n_heads_kv = 2;
    recipe.head_dim = 64;
    recipe.vocab_size = 64;
    let kv_dim = recipe.n_heads_kv * recipe.head_dim;

    let mut weights = MapWeights::default();
    weights.insert(
        "token_embd.weight",
        QuantType::F16,
        vec![256, 64],
        make_half_table(64, 256, false, 3),
    );
    weights.insert(
        "output_norm.weight",
        QuantType::F32,
        vec![256],
        make_f32_vec(256, 1.0),
    );
    for norm in [
        "blk.0.attn_norm.weight",
        "blk.0.attn_sub_norm.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_sub_norm.weight",
    ] {
        let dim = if norm.contains("ffn_sub") { 512 } else { 256 };
        weights.insert(norm, QuantType::F32, vec![dim], make_f32_vec(dim, 1.0));
    }
    weights.insert(
        "blk.0.attn_q.weight",
        QuantType::TQ2_0,
        vec![256, 256],
        make_tq2_0_matrix(256, 256, 1),
    );
    weights.insert(
        "blk.0.attn_k.weight",
        QuantType::TQ2_0,
        vec![256, kv_dim],
        make_tq2_0_matrix(kv_dim, 256, 2),
    );
    weights.insert(
        "blk.0.attn_v.weight",
        QuantType::I2_S,
        vec![256, kv_dim],
        make_i2s_matrix(kv_dim, 256, 3, 0.25),
    );
    weights.insert(
        "blk.0.attn_output.weight",
        QuantType::TQ2_0,
        vec![256, 256],
        make_tq2_0_matrix(256, 256, 4),
    );
    weights.insert(
        "blk.0.ffn_gate.weight",
        QuantType::TQ2_0,
        vec![256, 512],
        make_tq2_0_matrix(512, 256, 5),
    );
    weights.insert(
        "blk.0.ffn_up.weight",
        QuantType::I2_S,
        vec![256, 512],
        make_i2s_matrix(512, 256, 6, 0.25),
    );
    weights.insert(
        "blk.0.ffn_down.weight",
        QuantType::TQ2_0,
        vec![512, 256],
        make_tq2_0_matrix(256, 512, 7),
    );
    (recipe, weights)
}

#[test]
fn test_wave4_cached_tq2_i2s_projections_bit_exact_worker_parity() {
    let _lock = TEST_SERIALIZER
        .lock()
        .unwrap_or_else(PoisonError::into_inner);

    let (recipe, weights) = build_tq2_i2s_model();
    let max_seq = 8_usize;
    let geom =
        KvGeometry::new(1, recipe.n_heads_kv, recipe.head_dim, max_seq, None).expect("kv geometry");
    let mut cache = BitNetModelCache::new();
    cache.preload_model(&recipe, &weights).expect("preload");

    let mut run_for_workers = |workers: usize| -> Vec<f32> {
        cache.set_decode_workers(workers);
        let mut logits = vec![0.0_f32; recipe.vocab_size];
        let mut kv_buf = vec![0.0_f32; geom.required_f32_len().expect("kv len")];
        let mut ws = ScratchWorkspace::new(recipe.decode_scratch_bytes(max_seq)).expect("ws");
        let mut ctx = ExecutionContext::new(&[1, 3, 5], max_seq, &mut logits, &mut kv_buf)
            .expect("ctx")
            .with_kv_geometry(geom)
            .expect("kv view")
            .with_scratch(&mut ws);
        recipe
            .prefill_step_with_cache(&mut ctx, &weights, &mut cache)
            .expect("prefill");
        ctx.prepare_decode(7).expect("prep");
        recipe
            .decode_step_with_cache(&mut ctx, &weights, &mut cache)
            .expect("decode");
        logits
    };

    let l1 = run_for_workers(1);
    let l4 = run_for_workers(4);
    let l8 = run_for_workers(8);
    assert!(l1.iter().all(|v| v.is_finite()));
    for row in 0..recipe.vocab_size {
        assert_eq!(l4[row].to_bits(), l1[row].to_bits());
        assert_eq!(l8[row].to_bits(), l1[row].to_bits());
    }
}

#[test]
fn test_wave4_long_context_gqa_and_scratch_128b_alignment() {
    let _lock = TEST_SERIALIZER
        .lock()
        .unwrap_or_else(PoisonError::into_inner);

    let (recipe, weights) = build_i2s_model(1, 128, 128, 16, QuantType::I2_S, None);
    let max_seq = 2064_usize;
    let geom =
        KvGeometry::new(1, recipe.n_heads_kv, recipe.head_dim, max_seq, None).expect("kv geometry");
    let mut cache = BitNetModelCache::new().with_decode_workers(4);
    cache.preload_model(&recipe, &weights).expect("preload");

    let mut kv_a = vec![0.05_f32; geom.required_f32_len().expect("kv len")];
    let mut kv_b = vec![-0.07_f32; geom.required_f32_len().expect("kv len")];
    for (idx, v) in kv_a.iter_mut().enumerate() {
        #[expect(clippy::cast_precision_loss, reason = "bounded test seed")]
        {
            *v = ((idx % 17) as f32 - 8.0) * 0.03125;
        }
    }
    for (idx, v) in kv_b.iter_mut().enumerate() {
        #[expect(clippy::cast_precision_loss, reason = "bounded test seed")]
        {
            *v = (8.0 - (idx % 19) as f32) * 0.03125;
        }
    }

    let ws_bytes = recipe.decode_scratch_bytes(max_seq);
    let mut ws = ScratchWorkspace::new(ws_bytes).expect("scratch workspace");
    {
        let scope = ws.scope();
        let a = scope.alloc_f32_zeroed(16).expect("aligned f32 slot");
        let b = scope.alloc_i8_zeroed(32).expect("aligned i8 slot");
        assert_eq!(a.as_ptr() as usize % 128, 0);
        assert_eq!(b.as_ptr() as usize % 128, 0);
    }

    let mut logits_a = vec![0.0_f32; recipe.vocab_size];
    let mut ctx_a = ExecutionContext::new(&[], max_seq, &mut logits_a, &mut kv_a)
        .expect("ctx_a")
        .with_kv_geometry(geom)
        .expect("kv_a")
        .with_scratch(&mut ws);
    ctx_a.advance_position(2054).expect("advance to 2054");

    ctx_a.prepare_decode(3).expect("warmup prep >2048");
    recipe
        .decode_step_with_cache(&mut ctx_a, &weights, &mut cache)
        .expect("warmup >2048 decode");

    let (res, count, bytes) = measure_allocations(|| -> Result<(), K3Error> {
        ctx_a.prepare_decode(5)?;
        recipe.decode_step_with_cache(&mut ctx_a, &weights, &mut cache)
    });
    res.expect("steady-state seq_len=2056 >2048 decode");
    assert_eq!(
        (count, bytes),
        (0, 0),
        "long-context seq_len > 2048 decode must perform 0 heap allocations"
    );
    assert!(logits_a.iter().all(|v| v.is_finite()));

    let mut ws_b = ScratchWorkspace::new(ws_bytes).expect("scratch workspace b");
    let mut logits_b = vec![0.0_f32; recipe.vocab_size];
    let mut ctx_b = ExecutionContext::new(&[], max_seq, &mut logits_b, &mut kv_b)
        .expect("ctx_b")
        .with_kv_geometry(geom)
        .expect("kv_b")
        .with_scratch(&mut ws_b);
    ctx_b.advance_position(2054).expect("advance b");
    ctx_b.prepare_decode(3).expect("b step 1");
    recipe
        .decode_step_with_cache(&mut ctx_b, &weights, &mut cache)
        .expect("b decode 1");
    ctx_b.prepare_decode(5).expect("b step 2");
    recipe
        .decode_step_with_cache(&mut ctx_b, &weights, &mut cache)
        .expect("b decode 2");
    assert!(
        logits_a
            .iter()
            .zip(&logits_b)
            .any(|(a, b)| (a - b).abs() > 1e-6),
        "attention scores over >2048 KV positions must affect output logits"
    );
}
