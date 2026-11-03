//! `k3-arch-bitnet`: `BitNet` architecture recipe for the `SpacemiT` K3 platform.
//!
//! Implements [`ArchRecipe`] for Microsoft `BitNet` architectures (`bitnet`, `bitnet-b1.58`,
//! and `bitnet-25`), supporting `SubLN` (`RMSNorm` followed by dynamic per-token int8 `absmax`
//! quantization), `ReLU^2` / `SwiGLU` gated FFNs, and on-the-fly ternary unpacking
//! (`TQ1_0`, `TQ2_0`) for zero-expansion `BitLinear` execution on `A100` (`smt.vmadot`)
//! and `X100` (RVV) compute clusters per ADR-023 and ADR-024.

pub mod recipe;

pub use recipe::BitNetRecipe;

#[cfg(test)]
mod tests {
    use super::*;
    use k3_core::{ArchRecipe, ExecutionContext, QuantType};

    #[derive(Debug)]
    struct EmptyWeights;

    impl k3_core::ModelWeights for EmptyWeights {
        fn tensor(&self, name: &str) -> Result<k3_core::TensorView<'_>, k3_core::K3Error> {
            Err(k3_core::K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "not found in dummy weights".to_owned(),
            })
        }
        fn tensor_count(&self) -> usize {
            0
        }
        fn total_weight_bytes(&self) -> usize {
            0
        }
    }

    #[test]
    fn test_bitnet_recipe_architecture_ids() {
        let bitnet = BitNetRecipe::new();
        assert_eq!(bitnet.architecture_id(), "bitnet");
        assert_eq!(bitnet.recipe_version(), "1.0.0");
        assert_eq!(bitnet.target_layouts(), &["native", "ime_tile_8x16"]);
        assert!(bitnet.supports_quant(QuantType::Q8_0));
        assert!(bitnet.supports_quant(QuantType::F32));
        assert!(bitnet.supports_quant(QuantType::F16));
        assert!(bitnet.supports_quant(QuantType::BF16));
        assert!(bitnet.use_relu2);

        let b158 = BitNetRecipe::new_b158();
        assert_eq!(b158.architecture_id(), "bitnet-b1.58");
        assert_eq!(b158.recipe_version(), "1.0.0");
        assert!(b158.use_relu2);

        let b25 = BitNetRecipe::new_25();
        assert_eq!(b25.architecture_id(), "bitnet-25");
        assert_eq!(b25.recipe_version(), "1.0.0");
        assert!(!b25.use_relu2);
    }

    #[test]
    fn test_bitnet_recipe_execution_flow() {
        let recipe = BitNetRecipe::new();
        let prompt_tokens = [1u32, 2, 3];
        let mut logits = [0.0f32; 16];
        let mut kv_cache = [0.0f32; 64];
        let mut ctx = ExecutionContext::new(&prompt_tokens, 32, &mut logits, &mut kv_cache)
            .expect("valid context");

        let weights = EmptyWeights;
        recipe
            .prefill_step(&mut ctx, &weights)
            .expect("prefill succeeds");
        assert_eq!(ctx.position(), 3);

        ctx.prepare_decode(42).expect("prepare decode succeeds");
        recipe
            .decode_step(&mut ctx, &weights)
            .expect("decode succeeds");
        assert_eq!(ctx.position(), 4);
    }

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
}
