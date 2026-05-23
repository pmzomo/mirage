// SPDX-License-Identifier: Apache-2.0
use crate::report::BenchReport;
use std::fmt::Write;

pub fn to_markdown(reports: &[BenchReport]) -> String {
    let mut out = String::new();
    writeln!(out, "# mirage-bench report\n").ok();
    writeln!(out, "| config | task | tokens | sim tok/s | p50 µs | p99 µs | stall | mean VRAM MB | peak VRAM MB | fb-max | divergence |").ok();
    writeln!(
        out,
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    .ok();
    for r in reports {
        let div = r
            .plan_divergence
            .as_ref()
            .map(|s| format!("{:.4} ({})", s.value, s.samples))
            .unwrap_or_else(|| "-".into());
        writeln!(
            out,
            "| {} | {} | {} | {:.1} | {} | {} | {:.3} | {:.1} | {} | {} | {} |",
            r.config_name,
            r.task_name,
            r.tokens,
            r.sim_tok_per_s,
            r.p50_total_us,
            r.p99_total_us,
            r.stall_fraction,
            r.mean_vram_mb,
            r.peak_vram_mb,
            r.fallback_level_max,
            div,
        )
        .ok();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use mirage_telemetry::tests_support::sample_trace;
    #[test]
    fn renders_a_single_row_per_report() {
        let xs: Vec<_> = (0..2).map(sample_trace).collect();
        let r = BenchReport::from_traces("c", "t", &xs, None);
        let md = to_markdown(&[r]);
        assert!(md.contains("c | t |"));
        assert_eq!(md.matches('\n').count(), 5); // title + blank + header + sep + row
    }
}
