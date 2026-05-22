// SPDX-License-Identifier: Apache-2.0
use crate::control::{ChainPhase, TokenKind};
use crate::decision::ActualRouting;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TokenContext {
    pub token_id: u32,
    pub position: u32,
    pub recent_logit_margin: f32,
    pub recent_entropy: f32,
    pub token_kind: TokenKind,
    pub chain_phase: ChainPhase,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutingContext {
    pub position: u32,
    pub recent_experts: Vec<u16>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ExecTelemetry {
    pub prefetch_us: u32,
    pub compute_us: u32,
    pub stall_us: u32,
    pub cpu_us: u32,
    pub vram_mb: u32,
    pub pcie_bytes: u64,
    pub kv_bytes: u64,
    pub energy_mj: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecOutcome {
    pub actual: ActualRouting,
    pub telemetry: ExecTelemetry,
}
