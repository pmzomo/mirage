// SPDX-License-Identifier: Apache-2.0
//! mirage-bench — measures the moat. Plan-divergence scoring, telemetry
//! aggregation, and the reasoning-retention CI gate.

pub mod task;
pub mod tasks;
pub mod scorer;
pub mod oracle;
pub mod runner;
pub mod report;
pub mod render;
