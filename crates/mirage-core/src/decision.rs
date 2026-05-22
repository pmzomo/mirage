// SPDX-License-Identifier: Apache-2.0
use crate::control::{ChainPhase, DifficultyClass, PrecisionTier};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputeDecision {
    pub difficulty: DifficultyClass,
    pub target_depth: u16,
    pub precision_per_group: Vec<PrecisionTier>,
    pub head_budget: u16,
    pub burst: bool,
    pub chain_phase: ChainPhase,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RoutingPrediction {
    pub experts_per_moe_layer: Vec<Vec<u16>>,
    pub likely_exit_layer: u16,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ActualRouting {
    pub experts_per_moe_layer: Vec<Vec<u16>>,
    pub exit_layer: u16,
}
