// SPDX-License-Identifier: Apache-2.0
use crate::task::{BenchSample, SampleStep, Task};
use mirage_core::control::{ChainPhase, TokenKind};

pub struct GsmToy;

impl Task for GsmToy {
    fn name(&self) -> &'static str { "gsm-toy" }

    fn samples(&self) -> Vec<BenchSample> {
        (0..5).map(|i| BenchSample {
            name: SAMPLE_NAMES[i as usize],
            steps: build_steps(i, 64),
        }).collect()
    }
}

const SAMPLE_NAMES: &[&str] = &[
    "gsm-toy-01", "gsm-toy-02", "gsm-toy-03", "gsm-toy-04", "gsm-toy-05",
];

fn build_steps(seed: u32, len: u32) -> Vec<SampleStep> {
    (0..len).map(|p| {
        // Three phases per sample so the scheduler sees realistic kind/phase mix.
        let kind = match p % 8 {
            0 | 1 => TokenKind::Content,
            2     => TokenKind::CommonWord,
            3 | 4 => TokenKind::ReasoningMarker,
            5     => TokenKind::Formatting,
            6     => TokenKind::Punctuation,
            _     => TokenKind::Content,
        };
        let phase = if p < len / 4 { ChainPhase::Early }
                    else if p < 3 * len / 4 { ChainPhase::Middle }
                    else { ChainPhase::Ending };
        // Logit margin oscillates between high (easy tokens) and low (reasoning),
        // entropy follows the inverse pattern. Deterministic per (seed, p).
        let osc = ((seed.wrapping_mul(2654435761) ^ p.wrapping_mul(40503)) % 1000) as f32 / 1000.0;
        let margin = if matches!(kind, TokenKind::ReasoningMarker) { 0.3 + osc } else { 2.0 + osc * 3.0 };
        let entropy = if matches!(kind, TokenKind::ReasoningMarker) { 0.6 + osc * 0.3 } else { 0.1 + osc * 0.1 };
        SampleStep {
            token_id: seed * 1000 + p,
            token_kind: kind,
            chain_phase: phase,
            recent_logit_margin: margin,
            recent_entropy: entropy,
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gsm_toy_emits_five_samples_of_sixty_four_tokens() {
        let s = GsmToy.samples();
        assert_eq!(s.len(), 5);
        assert!(s.iter().all(|x| x.steps.len() == 64));
    }
    #[test]
    fn samples_contain_reasoning_markers() {
        for sample in GsmToy.samples() {
            let n = sample.steps.iter()
                .filter(|s| matches!(s.token_kind, TokenKind::ReasoningMarker))
                .count();
            assert!(n > 0, "every sample must exercise the reasoning path");
        }
    }
    #[test]
    fn samples_are_deterministic() {
        assert_eq!(GsmToy.samples(), GsmToy.samples());
    }
}
