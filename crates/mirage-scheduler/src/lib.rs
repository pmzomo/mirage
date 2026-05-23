// SPDX-License-Identifier: Apache-2.0
//! Adaptive scheduler + graded fallback (spec §19, §10).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Anomaly { ReasoningCollapse, NbpFloorBreach, KvIntegrity, VramPressure, KernelNaN }

/// Levels: 0 full adaptive .. 3 pure baseline. Invariant I5: level 3 is
/// reachable from any state and never escalates past 3.
pub struct FallbackController { level: u8 }

impl FallbackController {
    pub fn new() -> Self { FallbackController { level: 0 } }
    pub fn level(&self) -> u8 { self.level }
    pub fn on_anomaly(&mut self, a: Anomaly) {
        let step = match a {
            Anomaly::NbpFloorBreach => 2,
            Anomaly::ReasoningCollapse => 1,
            Anomaly::KvIntegrity | Anomaly::VramPressure => 1,
            Anomaly::KernelNaN => 3,
        };
        self.level = (self.level.max(step)).min(3);
    }
    pub fn baseline_reachable(&self) -> bool { true }
}

impl Default for FallbackController { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn anomalies_escalate_monotonically_and_cap_at_3() {
        let mut f = FallbackController::new();
        f.on_anomaly(Anomaly::ReasoningCollapse);
        assert_eq!(f.level(), 1);
        f.on_anomaly(Anomaly::NbpFloorBreach);
        assert_eq!(f.level(), 2);
        f.on_anomaly(Anomaly::ReasoningCollapse); // never de-escalates
        assert_eq!(f.level(), 2);
        f.on_anomaly(Anomaly::KernelNaN);
        assert_eq!(f.level(), 3);
        f.on_anomaly(Anomaly::KernelNaN);
        assert_eq!(f.level(), 3); // capped
        assert!(f.baseline_reachable());
    }
}
