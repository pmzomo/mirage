// SPDX-License-Identifier: Apache-2.0
//! Formal TokenTrace schema (spec §14).
use mirage_core::control::{ChainPhase, PrecisionTier};
use mirage_core::decision::ComputeDecision;
use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LatencyBreakdown {
    pub total_us: u32,
    pub prefetch_us: u32,
    pub compute_us: u32,
    pub stall_us: u32,
    pub cpu_us: u32,
    pub sample_us: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TokenTrace {
    pub schema_version: u16,
    pub model_id: u64,
    pub run_id: u128,
    pub token_id: u32,
    pub position: u32,
    pub chain_phase: ChainPhase,
    pub predicted_experts: Vec<u16>,
    pub actual_experts: Vec<u16>,
    pub precision_plan: Vec<PrecisionTier>,
    pub profiler_decision: ComputeDecision,
    pub latency: LatencyBreakdown,
    pub vram_usage_mb: u32,
    pub pcie_bytes: u64,
    pub kv_cache_bytes: u64,
    pub energy_mj: u32,
    pub logits_entropy: f32,
    pub logit_margin: f32,
    pub fallback_level: u8,
    pub deterministic: bool,
}

#[derive(Debug, PartialEq)]
pub enum TraceError { WrongSchema(u16), LatencyInconsistent }

/// Invariant I8: a trace must validate against its declared schema version.
pub fn validate(t: &TokenTrace) -> Result<(), TraceError> {
    if t.schema_version != SCHEMA_VERSION {
        return Err(TraceError::WrongSchema(t.schema_version));
    }
    let parts = t.latency.prefetch_us + t.latency.compute_us
        + t.latency.stall_us + t.latency.cpu_us + t.latency.sample_us;
    if parts > t.latency.total_us {
        return Err(TraceError::LatencyInconsistent);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mirage_core::control::DifficultyClass;

    fn sample() -> TokenTrace {
        TokenTrace {
            schema_version: SCHEMA_VERSION, model_id: 7, run_id: 1,
            token_id: 10, position: 3, chain_phase: ChainPhase::Early,
            predicted_experts: vec![1, 2], actual_experts: vec![1, 2],
            precision_plan: vec![PrecisionTier::Q4],
            profiler_decision: ComputeDecision {
                difficulty: DifficultyClass::Normal, target_depth: 48,
                precision_per_group: vec![PrecisionTier::Q4], head_budget: 32,
                burst: false, chain_phase: ChainPhase::Early,
            },
            latency: LatencyBreakdown { total_us: 100, prefetch_us: 10,
                compute_us: 70, stall_us: 5, cpu_us: 5, sample_us: 2 },
            vram_usage_mb: 11000, pcie_bytes: 1024, kv_cache_bytes: 2048,
            energy_mj: 50, logits_entropy: 0.4, logit_margin: 1.2,
            fallback_level: 0, deterministic: true,
        }
    }

    #[test]
    fn valid_trace_passes() { assert!(validate(&sample()).is_ok()); }

    #[test]
    fn wrong_schema_rejected() {
        let mut t = sample(); t.schema_version = 99;
        assert_eq!(validate(&t), Err(TraceError::WrongSchema(99)));
    }

    #[test]
    fn inconsistent_latency_rejected() {
        let mut t = sample(); t.latency.compute_us = 10_000;
        assert_eq!(validate(&t), Err(TraceError::LatencyInconsistent));
    }
}
