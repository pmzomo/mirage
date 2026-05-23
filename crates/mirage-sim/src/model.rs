// SPDX-License-Identifier: Apache-2.0
use mirage_core::plan::ModelShape;

/// Slowly-drifting "topic" so routing has temporal locality the n-gram
/// predictor can exploit (spec §9 — MoE routing temporal locality).
pub fn topic_of(position: u32) -> u32 { position / 16 }

/// Deterministic true expert routing for a (layer, position). Pure function:
/// identical inputs => identical experts. No RNG, no clock.
pub fn true_experts(shape: &ModelShape, layer: u16, position: u32) -> Vec<u16> {
    if !shape.is_moe() { return Vec::new(); }
    let topic = topic_of(position);
    let mut seed = (layer as u64)
        .wrapping_mul(0x9E3779B1)
        .wrapping_add((topic as u64).wrapping_mul(0x85EBCA77));
    let mut out = Vec::with_capacity(shape.top_k as usize);
    while out.len() < shape.top_k as usize {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let e = (seed >> 33) as u16 % shape.n_experts;
        if !out.contains(&e) { out.push(e); }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn routing_is_deterministic() {
        let s = ModelShape::qwen3_30b_a3b();
        assert_eq!(true_experts(&s, 3, 100), true_experts(&s, 3, 100));
    }
    #[test]
    fn routing_has_temporal_locality() {
        let s = ModelShape::qwen3_30b_a3b();
        // Same topic window (positions 96..112 share topic 6) => identical experts.
        assert_eq!(true_experts(&s, 3, 96), true_experts(&s, 3, 111));
        // Different topic window => (almost surely) different.
        assert_ne!(true_experts(&s, 3, 96), true_experts(&s, 3, 200));
    }
    #[test]
    fn dense_shape_has_no_experts() {
        let mut s = ModelShape::qwen3_30b_a3b();
        s.n_experts = 0;
        assert!(true_experts(&s, 0, 0).is_empty());
    }
}
