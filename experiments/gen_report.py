"""Generate results/REPORT.md and LaTeX tables from results/numbers.json.

Usage: python experiments/gen_report.py
"""
from __future__ import annotations

import json
import os

R = os.path.join(os.path.dirname(__file__), "..", "results")
T1 = ("Random", "Greedy", "JSQ", "Erlang-C", "Whittle", "PSO", "GA", "DQN")
T2 = ("Lag-SIPP", "Whittle-F", "PSO-F", "PPO")


def fmt_p(p):
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def main_table_tex(res):
    M = res["methods"]
    def by_q(ms):
        return sorted([m for m in ms if m in M], key=lambda m: -M[m]["QASS"]["mean"])
    order = ["SDQ-Match"] + by_q(("SDQ-core",)) + by_q(T2) + by_q(T1)
    # decision times from the sequential, single-process timing run (equal conditions)
    tpath = os.path.join(R, "timing.json")
    timing = json.load(open(tpath))["main"] if os.path.exists(tpath) else {}
    lines = [r"\begin{tabular}{|l|c|c|c|c|c|c|}", r"\hline",
             r"\textbf{Method} & \textbf{QASS (\%)} & \textbf{SDMR (\%)} & \textbf{AQW} & "
             r"\textbf{P95} & \textbf{Abandon (\%)} & \textbf{ms/decision} \\ \thickhline"]
    for m in order:
        q = M[m]
        name = "SDQ-Match (ours)" if m == "SDQ-Match" else m
        ms = f"{timing[m]:.2f}" if m in timing else "--"
        lines.append(f"{name} & {q['QASS']['mean']:.2f} $\\pm$ {q['QASS']['std']:.2f} & "
                     f"{q['SDMR']['mean']:.2f} & {q['AQW']['mean']:.2f} & {q['P95']['mean']:.1f} & "
                     f"{q['AB']['mean']:.2f} & {ms} \\\\ \\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def sig_table_tex(res):
    C = res["compare"]
    lines = [r"\begin{tabular}{|l|c|c|c|c|c|}", r"\hline",
             r"\textbf{Baseline} & \textbf{Gain (\%)} & \textbf{95\% CI} & "
             r"\textbf{$p$ (Holm)} & \textbf{$d_z$} & \textbf{W/T/L} \\ \thickhline"]
    tiers = [sorted([x for x in T if x in C], key=lambda x: C[x]["rel_gain"]) for T in (T2, T1)]
    for m in tiers[0] + tiers[1]:
        c = C[m]
        lines.append(f"{m} & {100*c['rel_gain']:+.1f} & [{100*c['ci_lo']:+.1f}, {100*c['ci_hi']:+.1f}] & "
                     f"{fmt_p(c['p_holm'])} & {c['dz']:.2f} & {c['wins']}/{c['ties']}/{c['losses']} \\\\ \\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


LABELS = [
    ("main", r"Calibration ($\lambda_0=0.16$, $\tau=5$, $\bar P=45$)"),
    ("load_0.12", r"Load $\lambda_0=0.12$"), ("load_0.14", r"Load $\lambda_0=0.14$"),
    ("load_0.18", r"Load $\lambda_0=0.18$"), ("load_0.2", r"Load $\lambda_0=0.20$"),
    ("tau_2", r"Latency $\tau=2$"), ("tau_8", r"Latency $\tau=8$"), ("tau_10", r"Latency $\tau=10$"),
    ("pat_20", r"Patience $\bar P=20$"), ("pat_30", r"Patience $\bar P=30$"),
    ("pat_60", r"Patience $\bar P=60$"),
    ("noise_0.1", r"Forecast error $\epsilon=0.1$"), ("noise_0.2", r"Forecast error $\epsilon=0.2$"),
    ("noise_0.3", r"Forecast error $\epsilon=0.3$"), ("noise_0.5", r"Forecast error $\epsilon=0.5$"),
    ("profile_weekend", "Weekend profile"), ("profile_festival", "Festival profile"),
    ("mismatch_weekend", "Weekend, weekday history"),
    ("mismatch_festival", "Festival, weekday history"),
    ("dist_negbin", "Negative-binomial arrivals"), ("dist_lognormal", "Log-normal flight times"),
    ("dist_uniformpat", "Uniform patience"),
    ("scale_10", r"$S=10$, $N=20$"), ("scale_20", r"$S=20$, $N=40$"), ("scale_40", r"$S=40$, $N=80$"),
]


def scenario_table_tex(nums):
    lines = [r"\begin{tabular}{|l|c|l|c|l|c|}", r"\hline",
             r"\textbf{Scenario} & \textbf{SDQ-Match} & \textbf{Best tier 1} & \textbf{Gain (\%)} & "
             r"\textbf{Best tier 2} & \textbf{Gain (\%)} \\ \thickhline"]
    for tag, lab in LABELS:
        if tag not in nums:
            continue
        C, M = nums[tag]["compare"], nums[tag]["methods"]
        b1 = min([m for m in C if m in T1], key=lambda m: C[m]["rel_gain"])
        b2 = min([m for m in C if m in T2], key=lambda m: C[m]["rel_gain"])

        def star(c):
            return "" if c["p_holm"] < 0.05 else r"$^{\dagger}$"
        lines.append(f"{lab} & {M['SDQ-Match']['QASS']['mean']:.2f} & {b1} & "
                     f"{100*C[b1]['rel_gain']:+.1f}{star(C[b1])} & {b2} & "
                     f"{100*C[b2]['rel_gain']:+.1f}{star(C[b2])} " + r"\\ \hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def es_table_tex(nums):
    lines = [r"\begin{tabular}{|l|c|c|c|}", r"\hline",
             r"\textbf{Scenario} & \textbf{SDQ-core} & \textbf{SDQ-Match} & "
             r"\textbf{Difference (points), $p$} \\ \thickhline"]
    for tag, lab in LABELS:
        if tag not in nums or "SDQ-core" not in nums[tag]["ablation"]:
            continue
        a = nums[tag]["ablation"]["SDQ-core"]
        M = nums[tag]["methods"]
        lines.append(f"{lab} & {M['SDQ-core']['QASS']['mean']:.2f} & "
                     f"{M['SDQ-Match']['QASS']['mean']:.2f} & "
                     f"{a['diff_mean']:+.2f}, {fmt_p(a['p'])} " + r"\\ \hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main():
    nums = json.load(open(os.path.join(R, "numbers.json")))
    out = ["# SDQ-Match benchmark report (generated)\n",
           "All numbers: paired common-random-number evaluation, seeds 200-229 "
           "(scale runs: 200-209). Gains are relative QASS gains of SDQ-Match; p values are "
           "two-sided Wilcoxon signed-rank, Holm-corrected over the baselines of a scenario.\n",
           "| scenario | SDQ-Match | strongest tier-1 (published, as specified) | gain | p_Holm "
           "| strongest tier-2 (forecast-augmented) | gain | p_Holm |",
           "|---|---|---|---|---|---|---|---|"]
    for tag in sorted(nums):
        res = nums[tag]
        C, M = res["compare"], res["methods"]
        t1 = [m for m in C if m in T1]
        t2 = [m for m in C if m in T2]
        b1 = min(t1, key=lambda m: C[m]["rel_gain"]) if t1 else None
        b2 = min(t2, key=lambda m: C[m]["rel_gain"]) if t2 else None
        cell = lambda b: (f"{b} ({M[b]['QASS']['mean']:.2f}) | {100*C[b]['rel_gain']:+.1f}% "
                          f"[{100*C[b]['ci_lo']:+.1f},{100*C[b]['ci_hi']:+.1f}] | "
                          f"{fmt_p(C[b]['p_holm'])}") if b else "- | - | -"
        out.append(f"| {tag} | {M['SDQ-Match']['QASS']['mean']:.2f} | {cell(b1)} | {cell(b2)} |")
    if "main" in nums:
        res = nums["main"]
        out += ["\n## Main benchmark: ablations (full minus variant, QASS points)\n"]
        for m, c in res["ablation"].items():
            out.append(f"- {m}: {c['diff_mean']:+.2f} ({100*c['rel_gain']:+.2f}%), p={fmt_p(c['p'])}")
        os.makedirs(os.path.join(R, "tables"), exist_ok=True)
        open(os.path.join(R, "tables", "main.tex"), "w").write(main_table_tex(res))
        open(os.path.join(R, "tables", "significance.tex"), "w").write(sig_table_tex(res))
    os.makedirs(os.path.join(R, "tables"), exist_ok=True)
    open(os.path.join(R, "tables", "scenarios.tex"), "w").write(scenario_table_tex(nums))
    open(os.path.join(R, "tables", "es_variant.tex"), "w").write(es_table_tex(nums))
    open(os.path.join(R, "REPORT.md"), "w", encoding="utf8").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
