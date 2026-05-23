// SPDX-License-Identifier: Apache-2.0
use mirage_core::control::{ChainPhase, TokenKind};

/// One step in a bench sample. Mirrors what a real decoder would consume
/// per token but doesn't require a model: token_kind + chain_phase are
/// the only inputs the scheduler reads.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleStep {
    pub token_id: u32,
    pub token_kind: TokenKind,
    pub chain_phase: ChainPhase,
    pub recent_logit_margin: f32,
    pub recent_entropy: f32,
}

/// A single bench input: a sequence of steps fed to the scheduler.
/// `name` identifies the sample in the report.
#[derive(Clone, Debug, PartialEq)]
pub struct BenchSample {
    pub name: &'static str,
    pub steps: Vec<SampleStep>,
}

pub trait Task {
    fn name(&self) -> &'static str;
    fn samples(&self) -> Vec<BenchSample>;
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Empty;
    impl Task for Empty {
        fn name(&self) -> &'static str { "empty" }
        fn samples(&self) -> Vec<BenchSample> { vec![] }
    }
    #[test]
    fn task_trait_is_object_safe() {
        let t: Box<dyn Task> = Box::new(Empty);
        assert_eq!(t.name(), "empty");
        assert!(t.samples().is_empty());
    }
}
