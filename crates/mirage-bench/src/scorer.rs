// SPDX-License-Identifier: Apache-2.0
use mirage_telemetry::TokenTrace;

/// Output of a scoring pass. `value` is in [0, 1] where 0 = identical
/// and 1 = total divergence. `units` describes what was measured.
#[derive(Clone, Debug, PartialEq)]
pub struct Score {
    pub units: &'static str,
    pub value: f64,
    pub samples: usize,
}

/// A `Scorer` compares two trace runs of the same input. Phase 1 ships
/// `PlanDivergence`; when the CUDA backend lands, a text-accuracy
/// `Scorer` plugs into the same trait without touching downstream code.
pub trait Scorer {
    fn name(&self) -> &'static str;
    fn score(&self, oracle: &[TokenTrace], adaptive: &[TokenTrace]) -> Score;
}

pub struct PlanDivergence;

impl Scorer for PlanDivergence {
    fn name(&self) -> &'static str { "plan-divergence" }

    fn score(&self, oracle: &[TokenTrace], adaptive: &[TokenTrace]) -> Score {
        let n = oracle.len().min(adaptive.len());
        if n == 0 {
            return Score { units: "fraction", value: 0.0, samples: 0 };
        }
        let mut mismatches = 0usize;
        for (o, a) in oracle.iter().zip(adaptive.iter()).take(n) {
            // A token's plan diverges if difficulty class, target depth,
            // or precision plan differ. (Burst + chain_phase are upstream
            // of the decision so a sim-oracle / adaptive split rarely
            // changes them — they're informational.)
            let diff = o.profiler_decision.difficulty != a.profiler_decision.difficulty
                || o.profiler_decision.target_depth != a.profiler_decision.target_depth
                || o.precision_plan != a.precision_plan;
            if diff { mismatches += 1; }
        }
        Score {
            units: "fraction",
            value: mismatches as f64 / n as f64,
            samples: n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mirage_telemetry::tests_support::sample_trace;
    #[test]
    fn identical_traces_score_zero() {
        let xs: Vec<_> = (0..10).map(sample_trace).collect();
        let s = PlanDivergence.score(&xs, &xs);
        assert_eq!(s.value, 0.0);
        assert_eq!(s.samples, 10);
    }
    #[test]
    fn fully_diverged_scores_one() {
        let oracle: Vec<_> = (0..4).map(sample_trace).collect();
        let mut adaptive = oracle.clone();
        for t in &mut adaptive {
            t.profiler_decision.target_depth = 1; // every token diverges
        }
        let s = PlanDivergence.score(&oracle, &adaptive);
        assert_eq!(s.value, 1.0);
    }
    #[test]
    fn empty_inputs_score_zero_samples() {
        let s = PlanDivergence.score(&[], &[]);
        assert_eq!(s.samples, 0);
    }
}
