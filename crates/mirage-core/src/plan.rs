// SPDX-License-Identifier: Apache-2.0
use crate::control::{Device, HeadMask, PrecisionTier};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpertBinding {
    Dense,
    Resident(Vec<u16>),
    Stream { handle: u32, experts: Vec<u16> },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerOp {
    pub layer_idx: u16,
    pub precision: PrecisionTier,
    pub head_mask: HeadMask,
    pub experts: ExpertBinding,
    pub device: Device,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecPlan {
    pub token_id: u32,
    pub position: u32,
    pub ops: Vec<LayerOp>,
    pub exit_after: u16,
    pub fallback_level: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelShape {
    pub n_layers: u16,
    pub n_heads: u16,
    pub n_experts: u16, // 0 => dense
    pub top_k: u16,
    pub hidden: u32,
    pub vocab: u32,
}

impl ModelShape {
    /// Synthetic shape approximating Qwen3-30B-A3B (MoE).
    pub fn qwen3_30b_a3b() -> Self {
        ModelShape {
            n_layers: 48,
            n_heads: 32,
            n_experts: 128,
            top_k: 8,
            hidden: 2048,
            vocab: 151_936,
        }
    }
    pub fn is_moe(&self) -> bool {
        self.n_experts > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qwen_shape_is_moe_with_128_experts() {
        let s = ModelShape::qwen3_30b_a3b();
        assert!(s.is_moe());
        assert_eq!(s.n_experts, 128);
        assert_eq!(s.top_k, 8);
        assert_eq!(s.n_layers, 48);
    }
}
