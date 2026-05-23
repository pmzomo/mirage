// SPDX-License-Identifier: Apache-2.0
use crate::scorer::Score;
use mirage_telemetry::TokenTrace;
use serde::{Deserialize, Serialize};

/// Aggregate stats for one bench run. Latency is sim-time (microseconds
/// computed by the SimBackend latency model — NOT wall clock).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchReport {
    pub config_name: String,
    pub task_name: String,
    pub tokens: usize,
    pub sim_us_total: u64,
    pub sim_tok_per_s: f64,
    pub p50_total_us: u32,
    pub p99_total_us: u32,
    pub stall_fraction: f64,
    pub mean_vram_mb: f64,
    pub peak_vram_mb: u32,
    pub fallback_level_max: u8,
    pub plan_divergence: Option<Score>,
}

fn percentile(sorted: &[u32], p: f64) -> u32 {
    if sorted.is_empty() { return 0; }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

impl BenchReport {
    pub fn from_traces(
        config_name: &str,
        task_name: &str,
        traces: &[TokenTrace],
        divergence: Option<Score>,
    ) -> Self {
        let tokens = traces.len();
        let sim_us_total: u64 = traces.iter().map(|t| t.latency.total_us as u64).sum();
        let stall_us_total: u64 = traces.iter().map(|t| t.latency.stall_us as u64).sum();
        let sim_tok_per_s = if sim_us_total == 0 { 0.0 }
                            else { tokens as f64 * 1_000_000.0 / sim_us_total as f64 };
        let mut totals: Vec<u32> = traces.iter().map(|t| t.latency.total_us).collect();
        totals.sort_unstable();
        let peak_vram_mb = traces.iter().map(|t| t.vram_usage_mb).max().unwrap_or(0);
        let mean_vram_mb = if tokens == 0 { 0.0 }
                           else { traces.iter().map(|t| t.vram_usage_mb as f64).sum::<f64>()
                                  / tokens as f64 };
        let fallback_level_max = traces.iter().map(|t| t.fallback_level).max().unwrap_or(0);

        BenchReport {
            config_name: config_name.to_string(),
            task_name: task_name.to_string(),
            tokens,
            sim_us_total,
            sim_tok_per_s,
            p50_total_us: percentile(&totals, 0.50),
            p99_total_us: percentile(&totals, 0.99),
            stall_fraction: if sim_us_total == 0 { 0.0 }
                            else { stall_us_total as f64 / sim_us_total as f64 },
            mean_vram_mb,
            peak_vram_mb,
            fallback_level_max,
            plan_divergence: divergence,
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("BenchReport serializes")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mirage_telemetry::tests_support::sample_trace;
    #[test]
    fn empty_traces_produce_zero_report() {
        let r = BenchReport::from_traces("c", "t", &[], None);
        assert_eq!(r.tokens, 0);
        assert_eq!(r.sim_tok_per_s, 0.0);
        assert_eq!(r.p50_total_us, 0);
    }
    #[test]
    fn report_aggregates_total_us_correctly() {
        let xs: Vec<_> = (0..10).map(sample_trace).collect();
        let r = BenchReport::from_traces("c", "t", &xs, None);
        assert_eq!(r.tokens, 10);
        assert_eq!(r.sim_us_total, 10 * 100); // every sample trace has total_us=100
    }
    #[test]
    fn report_round_trips_through_json() {
        let xs: Vec<_> = (0..3).map(sample_trace).collect();
        let r = BenchReport::from_traces("c", "t", &xs, None);
        let s = r.to_json();
        let back: BenchReport = serde_json::from_str(&s).unwrap();
        assert_eq!(r, back);
    }
}
