"""Generate every number, table and figure used in the manuscript from results/ (no hand-copied values).

Writes paper/generated/numbers.tex (\\newcommand macros), paper/generated/tab_*.tex and paper/figures/*.pdf.
Macros for results that do not exist are not defined, so LaTeX fails loudly instead of printing a stale value.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

from common import ROOT, RESULTS

GEN = ROOT / "paper" / "generated"
FIG = ROOT / "paper" / "figures"
GEN.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
macros = {}


def M(name, value):
    macros[name] = value


def f(x, d=3):
    return f"{x:.{d}f}"


def load_ab(name):
    p = RESULTS / "expAB" / f"{name}.jsonl"
    if not p.exists() or p.stat().st_size == 0:
        return None
    d = pd.read_json(p, lines=True)
    if "eps" not in d:
        d["eps"] = np.nan
    return d


METHOD_ORDER = ["CEN", "LOC", "CNT", "FED"]


def summary(d):
    g = d.groupby(["dataset", "method", "eps"], dropna=False)
    s = g.agg(auc=("auc", "mean"), auc_sd=("auc", "std"), brier=("brier", "mean"), ll=("logloss", "mean"),
              b_rmse=("b_rmse", "mean"), b_rmse_sd=("b_rmse", "std"), rho=("b_spearman", "mean"),
              auc_min=("auc_org_min", "mean"), n=("seed", "nunique")).reset_index()
    if "expert_spearman" in d:
        s = s.merge(g.expert_spearman.mean().reset_index(), on=["dataset", "method", "eps"], how="left")
    return s


def nonprivate_table(s, datasets, fname, caption, label, ref_note):
    lines = [r"\begin{widetable}", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\small",
             r"\begin{tabular}{llcccc}", r"\toprule",
             r"Dataset & Method & AUC & Brier & b-RMSE & Spearman $\rho$ \\", r"\midrule"]
    for ds in datasets:
        sub = s[(s.dataset == ds) & s.eps.isna()]
        for m in METHOD_ORDER:
            r = sub[sub.method == m]
            if r.empty:
                continue
            r = r.iloc[0]
            lines.append(f"{ds if m == 'CEN' else ''} & {m} & {f(r.auc, 4)}$\\pm${f(r.auc_sd, 4)} & {f(r.brier, 4)} & "
                         f"{f(r.b_rmse)} & {f(r.rho)} \\\\")
        lines.append(r"\midrule" if ds != datasets[-1] else r"\bottomrule")
    lines += [r"\end{tabular}", rf"\par\smallskip\footnotesize {ref_note}", r"\end{widetable}"]
    (GEN / fname).write_text("\n".join(lines) + "\n")


def main():
    syn, asst, jun = load_ab("syn"), load_ab("assist09"), load_ab("junyi")
    frames = [x for x in (syn, asst, jun) if x is not None]
    allr = pd.concat(frames, ignore_index=True)
    s = summary(allr)
    s.to_csv(RESULTS / "expAB" / "summary.csv", index=False)
    M("NSeeds", str(allr.seed.nunique()))
    fed = allr[allr.method == "FED"]
    M("FedMaxDiff", f"{fed.max_abs_diff_vs_CEN.max():.1e}".replace("e-", r"\times10^{-") + "}")
    M("FedMaxDiff", "$" + macros["FedMaxDiff"] + "$")

    # dataset statistics
    if asst is not None:
        a0 = asst.iloc[0]
        M("AssistNSchools", str(int(a0.K))); M("AssistUsers", f"{int(a0.n_users):,}"); M("AssistItems", f"{int(a0.n_items):,}")
        M("AssistResp", f"{int(a0.n_train + a0.n_test):,}")
    hp = json.loads((RESULTS / "hparams_frozen.json").read_text())["DP-FED"]
    M("HpT", str(hp["rounds"])); M("HpC", str(hp["clip"])); M("HpCn", str(hp["clip_count"])); M("HpLr", str(hp["lr"]))

    # Per-school accuracy of ASSIST09 after protocol filters (running example)
    sys.path.insert(0, str(ROOT / "src"))
    from lfchain.data.datasets import load_assist09
    d = load_assist09()
    accs = [float(np.mean(np.concatenate([tr.y, te.y]))) for tr, te in zip(d.train, d.test)]
    M("AssistAccMin", f(min(accs), 2)); M("AssistAccMax", f(max(accs), 2))
    cov = [len(np.unique(tr.i)) / d.n_items for tr in d.train]
    M("AssistCovMedian", f"{100 * np.median(cov):.0f}"); M("AssistCovMin", f"{100 * min(cov):.0f}"); M("AssistCovMax", f"{100 * max(cov):.0f}")

    if jun is not None and len(jun):
        j0 = jun.iloc[0]
        M("JunyiNCities", str(int(j0.K))); M("JunyiUsers", f"{int(j0.n_users):,}"); M("JunyiItems", f"{int(j0.n_items):,}")
        M("JunyiResp", f"{int(j0.n_train + j0.n_test):,}"); M("JunyiTrainRowsM", f(j0.n_train / 1e6, 1))
    # raw size of Junyi log
    try:
        man = json.loads((RESULTS / "expD" / "manifest_pipeline.json").read_text())
        M("JunyiRawRowsM", f(man["n_records"] / 1e6, 1))
    except FileNotFoundError:
        cache = ROOT / "data" / "cache" / "junyi_log.parquet"
        import pyarrow.parquet as pq
        M("JunyiRawRowsM", f(pq.ParquetFile(cache).metadata.num_rows / 1e6, 1))

    for ds, key in (("SYN-K10-nonIID", "SynTenNon"), ("ASSIST09", "Assist"), ("Junyi", "Junyi")):
        r = allr[allr.dataset == ds]
        if len(r):
            M(f"{key}RespPerItem", f"{(r.n_train / r.n_items).mean():,.0f}")
            M(f"{key}RespPerUser", f"{(r.n_train / r.n_users).mean():,.0f}")
    jr = allr[allr.dataset == "Junyi"]
    if len(jr):
        M("JunyiFEDSec", f"{jr[jr.method == 'FED'].seconds.mean():.0f}")
        M("JunyiCENSec", f"{jr[jr.method == 'CEN'].seconds.mean():.0f}")
        M("JunyiDPFEDSec", f"{jr[jr.method == 'DP-FED'].seconds.mean():.0f}")
        M("JunyiFEDRounds", f"{jr[jr.method == 'FED'].rounds.mean():.0f}")
    syn_ds = ["SYN-K3-IID", "SYN-K3-nonIID", "SYN-K10-IID", "SYN-K10-nonIID"]
    nonprivate_table(s, syn_ds, "tab_syn.tex",
                     "Synthetic data (D3): calibration utility without privacy, mean over seeds ($\\pm$ s.d. for AUC). b-RMSE and $\\rho$ are against the true difficulties.",
                     "tab:syn", "FED equals CEN by Proposition~\\ref{prop:exact}; CEN requires pooling raw data.")
    real = [x for x in ("ASSIST09", "Junyi") if x in set(s.dataset)]
    nonprivate_table(s, real, "tab_real.tex",
                     "Real multi-institution logs: test-set prediction and agreement with the centralised reference, mean over seeds ($\\pm$ s.d. for AUC).",
                     "tab:real", "b-RMSE and $\\rho$ are measured against CEN, which is a reference, not ground truth; CEN and FED therefore show 0 and 1.")
    for ds in s.dataset.unique():
        key = {"SYN-K3-IID": "SynThreeIid", "SYN-K3-nonIID": "SynThreeNon", "SYN-K10-IID": "SynTenIid",
               "SYN-K10-nonIID": "SynTenNon", "ASSIST09": "Assist", "Junyi": "Junyi"}[ds]
        for m in METHOD_ORDER:
            r = s[(s.dataset == ds) & (s.method == m) & s.eps.isna()]
            if len(r):
                r = r.iloc[0]
                M(f"{key}{m}Auc", f(r.auc, 4)); M(f"{key}{m}AucMin", f(r.auc_min, 4)); M(f"{key}{m}Rmse", f(r.b_rmse)); M(f"{key}{m}Rho", f(r.rho))
                if "expert_spearman" in r and not pd.isna(r.expert_spearman):
                    M(f"{key}{m}Expert", f(r.expert_spearman))
        for m in ("DP-FED", "DP-CNT"):
            for e in (0.5, 1, 2, 4, 8):
                r = s[(s.dataset == ds) & (s.method == m) & (s.eps == e)]
                if len(r):
                    r = r.iloc[0]
                    tag = {0.5: "Half", 1: "One", 2: "Two", 4: "Four", 8: "Eight"}[e]
                    mm = m.replace("-", "")
                    M(f"{key}{mm}{tag}Auc", f(r.auc, 4)); M(f"{key}{mm}{tag}Rmse", f(r.b_rmse)); M(f"{key}{mm}{tag}Rho", f(r.rho))
                    if "expert_spearman" in r and not pd.isna(r.expert_spearman):
                        M(f"{key}{mm}{tag}Expert", f(r.expert_spearman))
    privacy_figure(s)
    ablation_table()
    ledger_assets(allr)

    c2 = RESULTS / "expC" / "c2_enforcement_models.json"
    if c2.exists():
        c = json.loads(c2.read_text())
        M("CTwoTasks", str(c["n_tasks"])); M("CTwoEvents", str(c["n_mid_task_events"]))
        M("CTwoAuthRounds", f"{c['authorised_round_participations']:,}")
        M("CTwoViolLogin", f"{c['models']['login_only_rbac']['unauthorized_round_participations']:,}")
        M("CTwoViolJoin", f"{c['models']['join_time_check']['unauthorized_round_participations']:,}")
        M("CTwoViolRound", f"{c['models']['per_round_reeval']['unauthorized_round_participations']:,}")
        M("CTwoFalseDenyRound", str(c['models']['per_round_reeval']['false_denials']))
    c1 = RESULTS / "expC" / "c1_differential.json"
    if c1.exists():
        c = json.loads(c1.read_text())
        M("COneN", f"{c['n_requests']:,}"); M("COneAgree", f"{100 * c['agreement']:.2f}")
        M("COneDisagree", str(c["n_disagree"])); M("COneAllowRate", f"{100 * c['allow_rate']:.1f}")
        M("COneTasks", str(c["n_tasks"])); M("COneGrants", str(c["n_grants"])); M("COneRevoked", str(c["n_revoked"]))
        M("COneNReasons", str(len(c["reason_counts"])))
        M("COnePdpUs", f(c["pdp_us_per_decision"], 1))
        M("COneChainSingleMs", f(c["chain_single_check_ms"]["p50"], 1)); M("COneChainSingleP", f(c["chain_single_check_ms"]["p95"], 1))
        M("COneChainBatchMs", f(c["chain_batch200_ms"]["mean"], 0))
        if "boundary" in c:
            bd = c["boundary"]
            M("COneBndN", f"{bd['n_requests']:,}"); M("COneBndAgree", f"{100 * bd['agreement']:.2f}")
            M("COneBndDisagree", str(bd["n_disagree"]))
            for code, mac in (("GRANT_REVOKED", "COneBndRevoked"), ("GRANT_EXPIRED", "COneBndExpired"),
                              ("PURPOSE_MISMATCH", "COneBndPurpose"), ("ALLOW", "COneBndAllow")):
                M(mac, f"{bd['reason_counts'].get(code, 0):,}")
        if "negative_write_tests" in c:
            nt = c["negative_write_tests"]
            M("COneNegN", str(len(nt))); M("COneNegPass", str(sum(v["pass"] for v in nt.values())))
            reason_table(c)
        if "revocation_latency" in c:
            rl = c["revocation_latency"]
            M("RevokeN", str(rl["n"])); M("RevokeMean", f(rl["effect_s_mean"], 2)); M("RevokePFifty", f(rl["effect_s_p50"], 2))
            M("RevokePNinetyFive", f(rl["effect_s_p95"], 2)); M("RevokeMax", f(rl["effect_s_max"], 2))
            M("RevokeCommitMean", f(rl["commit_s_mean"], 2))
    c3 = RESULTS / "expC" / "c3_tamper.json"
    if c3.exists():
        c = json.loads(c3.read_text())
        M("CThreeN", f"{c['n_artifacts']:,}"); M("CThreeDetected", f"{c['tampered_detected']:,}"); M("CThreeClean", f"{c['untampered_verified']:,}")
        M("CThreeVerifyMs", f(c["chain_verify_ms"]["mean"], 1)); M("CThreeProofLen", str(c["proof_len"]))
        hs = c["hash_ms_by_size"]
        M("CThreeHashSmall", f(hs[min(hs, key=int)], 3)); M("CThreeHashLarge", f(hs[max(hs, key=int)], 1))
    env = json.loads((RESULTS / "env.json").read_text())
    M("HostCPU", env["cpu"].replace("(R)", "").replace("(TM)", "").replace("  ", " "))
    M("HostRAM", str(int(int(env["ram_bytes"]) / 2**30)))
    M("FabricVersion", "2.5.14")
    try:
        bench = json.loads((RESULTS / "expD" / "bench_summary.json").read_text())
        fb = bench["fabric"]
        M("FabricBatchTimeout", fb["BatchTimeout"].strip()); M("FabricMaxMsg", fb["MaxMessageCount"].strip())
    except FileNotFoundError:
        import re
        cfg = (ROOT / "fabric/fabric-samples/test-network/configtx/configtx.yaml").read_text()
        M("FabricBatchTimeout", re.search(r"^\s*BatchTimeout:\s*(.+)$", cfg, re.M).group(1).strip())
        M("FabricMaxMsg", re.search(r"^\s*MaxMessageCount:\s*(.+)$", cfg, re.M).group(1).strip())

    exploratory_assets()
    import re
    bad = [k for k in macros if not re.fullmatch(r"[A-Za-z]+", k)]
    assert not bad, f"invalid macro names: {bad}"
    out = ["% Auto-generated by experiments/make_paper_assets.py -- do not edit"]
    for k, v in sorted(macros.items()):
        out.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (GEN / "numbers.tex").write_text("\n".join(out) + "\n")
    print(f"wrote {len(macros)} macros")


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def exploratory_assets():
    from scipy.stats import wilcoxon
    tag = {0.5: "Half", 1: "One", 2: "Two", 4: "Four", 8: "Eight"}
    # ---- E: trusted-aggregator baseline ----
    ej, es = RESULTS / "expE" / "junyi.jsonl", RESULTS / "expE" / "syn.jsonl"
    if es.exists() and ej.exists() and ej.stat().st_size:
        e = pd.concat([pd.read_json(es, lines=True), pd.read_json(ej, lines=True)], ignore_index=True)
        # The evaluation matches the frozen runs, so DP-FED must reproduce them exactly on the shared seeds.
        ab = pd.concat([pd.read_json(RESULTS / "expAB" / "syn.jsonl", lines=True), pd.read_json(RESULTS / "expAB" / "junyi.jsonl", lines=True)])
        chk = e[e.method == "DP-FED"].merge(ab[ab.method == "DP-FED"], on=["dataset", "seed", "eps"], suffixes=("", "_ab"))
        assert len(chk) and (chk.auc - chk.auc_ab).abs().max() < 1e-9, "experiment E does not reproduce the frozen DP-FED runs"
        g = e.groupby(["dataset", "method", "eps"])[["auc", "b_rmse"]].mean().reset_index()
        piv = g.pivot_table(index=["dataset", "eps"], columns="method", values=["auc", "b_rmse"])
        ratio = piv["b_rmse"]["DP-FED"] / piv["b_rmse"]["DP-FED-SECAGG"]
        use = ratio.loc[["SYN-K10-nonIID", "Junyi"]]
        M("ExRatioMin", f(use.min(), 1)); M("ExRatioMax", f(use.max(), 1))
        M("ExRatioMinWhere", {"SYN-K10-nonIID": "synthetic", "Junyi": "Junyi"}[use.idxmin()[0]] + f", $\\varepsilon={use.idxmin()[1]:g}$")
        M("ExRatioMaxWhere", {"SYN-K10-nonIID": "synthetic", "Junyi": "Junyi"}[use.idxmax()[0]] + f", $\\varepsilon={use.idxmax()[1]:g}$")
        M("ExSeedsJunyi", str(e[e.dataset == "Junyi"].seed.nunique()))
        for ds, key in (("Junyi", "Junyi"), ("SYN-K10-nonIID", "Syn")):
            for eps, tg in tag.items():
                M(f"Ex{key}Fed{tg}Rmse", f(piv["b_rmse"]["DP-FED"].loc[(ds, eps)]))
                M(f"Ex{key}Sec{tg}Rmse", f(piv["b_rmse"]["DP-FED-SECAGG"].loc[(ds, eps)]))
                M(f"Ex{key}Fed{tg}Auc", f(piv["auc"]["DP-FED"].loc[(ds, eps)], 4))
                M(f"Ex{key}Sec{tg}Auc", f(piv["auc"]["DP-FED-SECAGG"].loc[(ds, eps)], 4))
                M(f"Ex{key}Ratio{tg}", f(ratio.loc[(ds, eps)], 1))
        lines = [r"\begin{widetable}", r"\centering",
                 r"\caption{Trusted-aggregator baseline versus DP-FED (exploratory). The baseline adds one noise draw of standard deviation $\sigma C$ per round to the aggregated gradient, and likewise for the counts, that is $\sqrt{K}$ times less total noise than per-institution releases at the same $(\varepsilon,\delta)$. The factor is DP-FED RMSE divided by baseline RMSE. AUC uses the same evaluation as Table~\ref{tab:real}. Mean over \NSeeds{} seeds on synthetic non-IID data with $K=10$ and \ExSeedsJunyi{} seeds on Junyi.}",
                 r"\label{tab:fedirtdp}", r"\footnotesize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{l rrrcc rrrcc}", r"\toprule",
                 r"& \multicolumn{5}{c}{Synthetic non-IID, $K=10$} & \multicolumn{5}{c}{Junyi} \\", r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
                 r"$\varepsilon$ & \multicolumn{2}{c}{RMSE} & factor & \multicolumn{2}{c}{AUC} & \multicolumn{2}{c}{RMSE} & factor & \multicolumn{2}{c}{AUC} \\",
                 r"& DP-FED & baseline & & DP-FED & baseline & DP-FED & baseline & & DP-FED & baseline \\", r"\midrule"]
        for eps in tag:
            row = [f"{eps:g}"]
            for ds in ("SYN-K10-nonIID", "Junyi"):
                row += [f(piv["b_rmse"]["DP-FED"].loc[(ds, eps)]), f(piv["b_rmse"]["DP-FED-SECAGG"].loc[(ds, eps)]), f(ratio.loc[(ds, eps)], 2),
                        f(piv["auc"]["DP-FED"].loc[(ds, eps)], 4), f(piv["auc"]["DP-FED-SECAGG"].loc[(ds, eps)], 4)]
            lines.append(" & ".join(row) + r" \\")
        lines = [x for x in lines]
        lines[5] = r"\begin{tabular}{l rrr rr rrr rr}"
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{widetable}"]
        (GEN / "tab_fedirtdp.tex").write_text("\n".join(lines) + "\n")
    # ---- F: clipping ----
    for name, key in (("syn", "Syn"), ("junyi", "Junyi")):
        cp = RESULTS / "expF" / f"clip_rates_{name}.json"
        fp = RESULTS / "expF" / f"{name}.jsonl"
        if not (cp.exists() and fp.exists() and fp.stat().st_size):
            continue
        c = json.loads(cp.read_text())
        d = pd.read_json(fp, lines=True)
        M(f"F{key}ClipCntFourPct", f"{100 * c['count_clip_fraction']['4.0']:.1f}")
        M(f"F{key}ClipCntEightPct", f"{100 * c['count_clip_fraction']['8.0']:.1f}")
        grid = d.groupby(["C", "Cn"])[["grad_clip_fraction", "b_rmse", "auc"]].mean()
        M(f"F{key}GradClipHalfPct", f"{100 * grid.loc[(0.5, 4.0), 'grad_clip_fraction']:.1f}")
        M(f"F{key}GradClipTwoPct", f"{100 * grid.grad_clip_fraction.xs(2.0, level='C').mean():.1f}")
        M(f"F{key}SensSeeds", str(d.seed.nunique()))
        M(f"F{key}SensRmseMin", f(grid.b_rmse.min())); M(f"F{key}SensRmseMax", f(grid.b_rmse.max()))
        M(f"F{key}SensFrozenRmse", f(grid.loc[(0.5, 4.0), "b_rmse"]))
        M(f"F{key}SensAucMin", f(grid.auc.min(), 4)); M(f"F{key}SensAucMax", f(grid.auc.max(), 4))
        best, worst = grid.b_rmse.idxmin(), grid.b_rmse.idxmax()
        M(f"F{key}SensBest", f"$C={best[0]:g}$, $C_n={best[1]:g}$"); M(f"F{key}SensWorst", f"$C={worst[0]:g}$, $C_n={worst[1]:g}$")
        M(f"F{key}SensWorstPenalty", f"{100 * (grid.b_rmse.max() / grid.loc[(0.5, 4.0), 'b_rmse'] - 1):.0f}")
    # ---- G: shared scale ----
    gs = RESULTS / "expG" / "syn.jsonl"
    if gs.exists():
        g = pd.read_json(gs, lines=True)
        M("GSynLOON", str(len(g))); M("GSynFedWins", str(int((g.auc_all_fed > g.auc_all_loc).sum())))
        M("GSynFedAuc", f(g.auc_all_fed.mean(), 4)); M("GSynLocAuc", f(g.auc_all_loc.mean(), 4)); M("GSynCntAuc", f(g.auc_all_cnt.mean(), 4))
        M("GSynFedLocDiff", f"{(g.auc_all_fed - g.auc_all_loc).mean():+.4f}")
        M("GSynFedLocP", sci(wilcoxon(g.auc_all_fed, g.auc_all_loc).pvalue))
    gu = RESULTS / "expG" / "syn_unseen.jsonl"
    if gu.exists():
        u = pd.read_json(gu, lines=True)
        M("GUnseenN", str(len(u))); M("GUnseenFedWins", str(int((u.auc_unseen_fed > u.auc_unseen_loc).sum())))
        M("GUnseenFedAuc", f(u.auc_unseen_fed.mean(), 4)); M("GUnseenLocAuc", f(u.auc_unseen_loc.mean(), 4)); M("GUnseenCntAuc", f(u.auc_unseen_cnt.mean(), 4))
        M("GUnseenFedRmse", f(u.rmse_unseen_fed.mean())); M("GUnseenLocRmse", f(u.rmse_unseen_loc.mean())); M("GUnseenCntRmse", f(u.rmse_unseen_cnt.mean()))
        M("GUnseenP", sci(wilcoxon(u.auc_unseen_fed, u.auc_unseen_loc).pvalue))
        M("GUnseenCoveragePct", f"{100 * u.n_withheld_seen_elsewhere.sum() / u.n_withheld_items.sum():.0f}")
        M("GUnseenItems", str(int(u.n_withheld_items.iloc[0]))); M("GUnseenTestMean", f"{u.n_test_on_withheld.mean():.0f}")
    st = RESULTS / "expG" / "seed_tests.json"
    if st.exists():
        s = json.loads(st.read_text())["FED_vs_LOC"]
        M("GJunyiFedLocDiff", f"{s['mean_diff']:+.4f}"); M("GJunyiFedLocP", f"{s['wilcoxon_p']:.4f}"); M("GJunyiFedLocSd", sci(s["sd_diff"]))
    gl = RESULTS / "expG" / "junyi_loo.jsonl"
    if gl.exists() and gl.stat().st_size:
        # CNT and LOC from junyi_loo; FED from the zero-start fits, which use nothing of the held-out city
        cold = pd.read_json(RESULTS / "expG" / "junyi_loo_cold.jsonl", lines=True)
        d = pd.read_json(gl, lines=True).drop(columns=["auc_all_fed", "fed_rounds"]).merge(
            cold[["seed", "held_out_org", "auc_all_fed", "fed_rounds"]], on=["seed", "held_out_org"], validate="one_to_one")
        M("GLooFedRoundsMax", str(int(d.fed_rounds.max())))
        M("GLooSeeds", str(d.seed.nunique())); M("GLooCities", str(d.held_out_org.nunique()))
        M("GLooFedAuc", f(d.auc_all_fed.mean(), 4)); M("GLooLocAuc", f(d.auc_all_loc.mean(), 4)); M("GLooCntAuc", f(d.auc_all_cnt.mean(), 4))
        M("GLooUnseenMax", f"{int(d.n_test_unseen_locally.max()):,}")
        city = d.groupby("held_out_org")[["auc_all_fed", "auc_all_loc", "auc_all_cnt"]].mean()
        rng = np.random.default_rng(20261018)
        for other, tagn in (("loc", "Loc"), ("cnt", "Cnt")):
            diff = (city.auc_all_fed - city[f"auc_all_{other}"]).to_numpy()
            boots = [diff[rng.integers(len(diff), size=len(diff))].mean() for _ in range(10000)]
            M(f"GLooFed{tagn}Diff", f"{diff.mean():+.4f}")
            M(f"GLooFed{tagn}Lo", f"{np.percentile(boots, 2.5):+.4f}"); M(f"GLooFed{tagn}Hi", f"{np.percentile(boots, 97.5):+.4f}")
            M(f"GLooFed{tagn}P", f"{wilcoxon(city.auc_all_fed, city[f'auc_all_{other}']).pvalue:.4f}")
            M(f"GLooFed{tagn}Wins", str(int((diff > 0).sum())))
    # ---- H: skill aggregation ----
    hp = RESULTS / "expH" / "assist09_skill.jsonl"
    if hp.exists():
        h = pd.read_json(hp, lines=True)
        M("HSkills", str(int(h.n_skills.iloc[0]))); M("HRespPerSkill", f"{h.resp_per_skill.iloc[0]:,.0f}")
        cen = h[h.method == "CEN"]; M("HCenAuc", f(cen.auc.mean(), 4))
        for eps, tg in tag.items():
            r = h[(h.method == "DP-FED") & (h.eps == eps)]
            M(f"HDp{tg}Auc", f(r.auc.mean(), 4)); M(f"HDp{tg}Rho", f(r.b_spearman.mean()))
    # ---- C2 end to end and forged timestamps ----
    ep = RESULTS / "expC" / "c2_e2e.json"
    if ep.exists():
        runs = json.loads(ep.read_text())["runs"]
        ten = [r for r in runs if r["K"] == 10 and not r["race"]]
        race = [r for r in runs if r["race"]]
        three = [r for r in runs if r["K"] == 3]
        one = ten[1]
        M("EtoeRuns", str(len(ten))); M("EtoeRevokeRounds", ", ".join(str(r["revoke_round"]) for r in ten[:-1]) + f" and {ten[-1]['revoke_round']}")
        M("EtoeRounds", str(one["rounds"]))
        M("EtoeCommitMean", f(np.mean([r["revocation_commit_s"] for r in ten]), 2))
        assert all(r["rounds_to_effect"] == 0 for r in ten)
        M("EtoeExampleRound", str(one["revoke_round"])); M("EtoeExampleCommit", f(one["revocation_commit_s"], 2)); M("EtoeExampleExcluded", str(one["excluded_from_round"]))
        M("EtoeExampleRecords", str(one["anchored_batch_records"])); M("EtoeExampleDenied", str(one["n_denied"]))
        M("EtoeExampleFedBefore", f(one["fed_rmse_uninterrupted"])); M("EtoeExampleFedAfter", f(one["fed_rmse_with_revocation"])); M("EtoeExampleFedChange", f(one["fed_max_abs_change"]))
        M("EtoeExampleDpBefore", f(one["dp_rmse_uninterrupted"])); M("EtoeExampleDpAfter", f(one["dp_rmse_with_revocation"]))
        tot = {k: sum(r["audit"][k] for r in runs) for k in runs[0]["audit"]}
        M("EtoeAuditRecords", f"{tot['records']:,}"); M("EtoeAuditProofs", f"{tot['proofs_verified']:,}"); M("EtoeAuditValid", f"{tot['transactions_valid']:,}")
        M("EtoeAuditMeta", f"{tot['block_and_time_match']:,}"); M("EtoeAuditDecisions", f"{tot['decisions_reproduced']:,}")
        M("EtoeGateBefore", f(np.mean([r["gate_per_round_s_before"] for r in ten if r["gate_per_round_s_before"]]), 2))
        M("EtoeGateAfter", f(np.mean([r["gate_per_round_s_after"] for r in ten if r["gate_per_round_s_after"]]), 2))
        M("EtoeGateTotal", f(np.mean([r["gate_total_s"] for r in ten]), 0))
        M("EtoeAdmitPFifty", f(np.mean([r["admission_latency_s"]["p50"] for r in ten]), 2)); M("EtoeAdmitPNinetyFive", f(np.mean([r["admission_latency_s"]["p95"] for r in ten]), 2))
        M("EtoeOffline", f(np.mean([r["offline_s"] for r in ten]), 1)); M("EtoeTask", f(np.mean([r["task_s"] for r in ten]), 0))
        M("EtoeThreeGate", f(np.mean([r["gate_per_round_s_mean"] for r in three]), 2)); M("EtoeThreeTotal", f(np.mean([r["gate_total_s"] for r in three]), 0))
        M("EtoeTxPerTask", f"{np.mean([r['n_committed_records'] for r in ten]):.0f}")
        if race:
            o = race[0]["victim_round_outcome_at_revocation"]
            M("EtoeRaceRound", str(race[0]["revoke_round"])); M("EtoeRaceExcluded", str(race[0]["excluded_from_round"]))
            M("EtoeRaceReason", {"NOT_COMMITTED": "was invalidated at commit", "GRANT_REVOKED": "was denied", "ALLOW": "was admitted, having been ordered first"}[o["reason"]])
    tp = RESULTS / "expC" / "tx_timestamp_test.json"
    if tp.exists():
        c = json.loads(tp.read_text())
        ok = [r["skew_s"] for r in c["rows"] if r["endorsed"]]; no = [r["skew_s"] for r in c["rows"] if not r["endorsed"]]
        assert all(abs(s) <= c["tolerance_s"] for s in ok) and all(abs(s) > c["tolerance_s"] for s in no)
        M("TsTolerance", str(c["tolerance_s"])); M("TsAcceptedMax", str(max(abs(s) for s in ok))); M("TsRejectedMin", str(min(abs(s) for s in no)))
        M("TsN", str(len(c["rows"]))); M("TsRejected", str(len(no)))


def reason_table(c):
    names = {"foreign_msp_registers_dataset": "Dataset version registered by another member (A7)",
             "duplicate_dataset_version": "Duplicate dataset version (A6)",
             "foreign_msp_revokes_grant": "Grant revoked by another member (A7)",
             "foreign_msp_grants_for_other_institution": "Grant issued for another member's institution (A7)",
             "double_revoke": "Second revocation of a grant",
             "replayed_grant_id": "Replayed grant identifier (A6)",
             "illegal_transition_completed_to_running": "Illegal task transition completed$\\to$running",
             "new_task_not_draft": "New task created outside draft",
             "calibration_on_unfinished_task": "Calibration recorded for an unfinished task",
             "malformed_digest": "Malformed digest",
             "replayed_anchor_id": "Replayed anchor identifier (A6)",
             "foreign_msp_requests_admission": "Round admission requested for another member's institution (A7)"}
    lines = [r"\begin{widetable}", r"\centering",
             r"\caption{Write-side negative tests on Fabric: each forbidden ledger write and the chaincode error observed.}",
             r"\label{tab:negwrites}", r"\small", r"\begin{tabular}{p{0.5\textwidth}p{0.3\textwidth}c}", r"\toprule",
             r"Attempted write & Observed error & Rejected \\", r"\midrule"]
    for k, v in c["negative_write_tests"].items():
        lines.append(f"{names.get(k, k)} & \\texttt{{{v['observed'].replace('_', chr(92) + '_')}}} & {'yes' if v['pass'] else 'no'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{widetable}"]
    (GEN / "tab_negwrites.tex").write_text("\n".join(lines) + "\n")
    rc = c["reason_counts"]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Decision outcomes of the \COneN{} random requests in C1 (identical for the backend PDP and the chaincode).}",
             r"\label{tab:reasons}", r"\small", r"\begin{tabular}{lr}", r"\toprule", r"Reason code & Requests \\", r"\midrule"]
    for k, v in sorted(rc.items(), key=lambda kv: -kv[1]):
        lines.append(f"\\texttt{{{k.replace('_', chr(92) + '_')}}} & {v:,} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (GEN / "tab_reasons.tex").write_text("\n".join(lines) + "\n")


def ledger_assets(allr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    man_p = RESULTS / "expD" / "manifest_pipeline.json"
    if man_p.exists():
        m = json.loads(man_p.read_text())
        M("ManN", f"{m['n_records']:,}"); M("ManInst", str(m["n_institutions"])); M("ManRecBytes", str(m["record_bytes"]))
        M("ManLeafRate", f"{m['leaf_hash_records_per_s'] / 1e3:,.0f}"); M("ManTreeRate", f"{m['tree_records_per_s'] / 1e3:,.0f}")
        M("ManLeafSec", f"{m['leaf_hash_s']:.0f}"); M("ManTreeSec", f"{m['tree_build_s']:.0f}")
        M("ManDepth", str(m["largest_depth"])); M("ManProofBytes", f"{m['proof_bytes']:,}"); M("ManVerifyUs", f"{m['proof_verify_us']:.0f}")
        M("ManLargest", m["largest_institution"]); M("ManLargestN", f"{m['institutions'][m['largest_institution']]['n_records']:,}")
        lat = [c["latency_ms"] for c in m["chain"].values()]
        if lat:
            M("ManRegMs", f"{np.mean(lat):,.0f}")
    b_p = RESULTS / "expD" / "bench_summary.json"
    if not b_p.exists():
        return
    bench = json.loads(b_p.read_text())
    runs = pd.DataFrame(bench["runs"])
    runs["B"] = runs.batch
    peer_cpu = runs.resources.apply(lambda r: r.get("peer0.org1.example.com", {}).get("cpu_mean", np.nan))
    runs["peer_cpu"] = peer_cpu
    g = runs.groupby("B").agg(tx=("tx_per_s", "mean"), tx_sd=("tx_per_s", "std"), rec=("records_per_s", "mean"),
                              rec_sd=("records_per_s", "std"), p50=("lat_p50_ms", "mean"), p95=("lat_p95_ms", "mean"),
                              err=("error_rate", "max"), bpr=("ledger_bytes_per_record", "mean"), bpt=("ledger_bytes_per_tx", "mean"), cpu=("peer_cpu", "mean"),
                              reps=("rep", "nunique")).reset_index()
    g.to_csv(RESULTS / "expD" / "bench_table.csv", index=False)
    M("BenchDur", str(bench["duration_s"])); M("BenchWarm", str(bench["warmup_s"])); M("BenchConc", str(bench["concurrency"]))
    M("BenchReps", str(bench["reps"]))
    tagname = {1: "Rec", 10: "Ten", 100: "Hundred", 500: "FiveHundred", 1000: "Thousand"}
    for _, r in g.iterrows():
        k = tagname[int(r.B)]
        M(f"Bench{k}Tx", f"{r.tx:,.1f}"); M(f"Bench{k}Rec", f"{r.rec:,.0f}"); M(f"Bench{k}PFifty", f"{r.p50:,.0f}")
        M(f"Bench{k}PNinetyFive", f"{r.p95:,.0f}"); M(f"Bench{k}Bpr", f"{r.bpr:,.1f}" if not np.isnan(r.bpr) else "--")
        M(f"Bench{k}Err", f"{100 * r.err:.2f}"); M(f"Bench{k}Bpt", f"{r.bpt:,.0f}")
    M("BenchMaxErr", f"{100 * g.err.max():.2f}")
    rec1 = g[g.B == 1].rec.iloc[0]; recK = g[g.B == 1000].rec.iloc[0]
    M("BenchSpeedup", f"{recK / rec1:,.0f}")
    if man_p.exists():
        n = json.loads(man_p.read_text())["n_records"]
        M("ExtrapRecHours", f"{n / rec1 / 3600:,.1f}"); M("ExtrapBatchMin", f"{n / recK / 60:,.1f}")
    # BlockFL-style comparison: bytes the FED messages of one Junyi task would occupy if written on chain
    jr = allr[(allr.dataset == "Junyi") & (allr.method == "FED")]
    if len(jr):
        M("JunyiFEDBytesMB", f"{jr.bytes_up.mean() / 2**20:,.0f}")
    lines = [r"\begin{widetable}", r"\centering",
             rf"\caption{{Anchoring on Fabric: per-record transactions versus Merkle batches of $B$ records (closed loop, {bench['concurrency']} clients, {bench['duration_s']}~s measurement after {bench['warmup_s']}~s warm-up, mean of {bench['reps']} repetitions). Ledger bytes are the growth of the peer's block store per committed record.}}",
             r"\label{tab:bench}", r"\small", r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"$B$ & tx/s & records/s & p50 (ms) & p95 (ms) & bytes/record & peer CPU (\%) \\", r"\midrule"]
    for _, r in g.iterrows():
        lines.append(f"{'1 (per record)' if r.B == 1 else int(r.B)} & {r.tx:,.1f}$\\pm${r.tx_sd:,.1f} & {r.rec:,.0f} & {r.p50:,.0f} & {r.p95:,.0f} & "
                     f"{r.bpr:,.1f} & {r.cpu:,.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{widetable}"]
    (GEN / "tab_bench.tex").write_text("\n".join(lines) + "\n")
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.2))
    ax[0].errorbar(g.B, g.rec, yerr=g.rec_sd, marker="o", ms=3, lw=1.2, color="#1f5a96", capsize=2)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("records per transaction $B$"); ax[0].set_ylabel("anchored records/s")
    ax[1].plot(g.B, g.p50, marker="o", ms=3, lw=1.2, color="#1f5a96", label="p50")
    ax[1].plot(g.B, g.p95, marker="s", ms=3, lw=1.2, color="#c0602a", label="p95")
    ax[1].set_xscale("log"); ax[1].set_xlabel("records per transaction $B$"); ax[1].set_ylabel("submit-to-commit latency (ms)")
    ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "anchoring.pdf"); plt.close(fig)


def ablation_table():
    files = [RESULTS / "expB_ablation" / f"{w}.jsonl" for w in ("syn", "assist09", "junyi")]
    if not all(f.exists() and f.stat().st_size for f in files):
        return
    d = pd.concat([pd.read_json(f, lines=True) for f in files])
    g = d.groupby(["dataset", "variant", "eps"]).agg(auc=("auc", "mean"), rmse=("b_rmse", "mean")).reset_index()
    names = {"full": "full (v2)", "no_pseudo": "without pseudo-count", "no_damping": "without damping", "neither": "neither"}
    lines = [r"\begin{widetable}", r"\centering",
             r"\caption{Exploratory ablation of DP-FED (not part of the frozen protocol): test AUC at three privacy levels and difficulty RMSE at $\varepsilon=2$ (against truth for synthetic data, against CEN for real logs), mean over seeds, frozen hyper-parameters.}",
             r"\label{tab:ablation}", r"\small", r"\begin{tabular}{llcccc}", r"\toprule",
             r"Dataset & Variant & AUC $\varepsilon{=}1$ & AUC $\varepsilon{=}2$ & AUC $\varepsilon{=}8$ & RMSE $\varepsilon{=}2$ \\", r"\midrule"]
    order = ["SYN-K10-nonIID", "ASSIST09", "Junyi"]
    for ds in order:
        for k, v in enumerate(("full", "no_pseudo", "no_damping", "neither")):
            r = g[(g.dataset == ds) & (g.variant == v)].set_index("eps")
            lines.append(f"{ds if k == 0 else ''} & {names[v]} & {r.auc[1]:.4f} & {r.auc[2]:.4f} & {r.auc[8]:.4f} & {r.rmse[2]:.3f} \\\\")
        lines.append(r"\midrule" if ds != order[-1] else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{widetable}"]
    (GEN / "tab_ablation.tex").write_text("\n".join(lines) + "\n")
    key = {"SYN-K10-nonIID": "SynTenNon", "ASSIST09": "Assist", "Junyi": "Junyi"}
    tag = {1: "One", 2: "Two", 8: "Eight"}
    vk = {"full": "Full", "no_pseudo": "NoPseudo", "no_damping": "NoDamp", "neither": "Neither"}
    for _, r in g.iterrows():
        M(f"Abl{key[r.dataset]}{vk[r.variant]}{tag[int(r.eps)]}Auc", f"{r.auc:.4f}")
        M(f"Abl{key[r.dataset]}{vk[r.variant]}{tag[int(r.eps)]}Rmse", f"{r.rmse:.2f}")


def privacy_figure(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
    dsets = [d for d in ["SYN-K10-nonIID", "ASSIST09", "Junyi"] if d in set(s.dataset)]
    fig, axes = plt.subplots(2, len(dsets), figsize=(2.45 * len(dsets), 3.9), squeeze=False, sharex=True)
    colors = {"DP-FED": "#1f5a96", "DP-CNT": "#c0602a"}
    for col, ds in enumerate(dsets):
        sub = s[s.dataset == ds]
        for row, (metric, sd) in enumerate((("auc", "auc_sd"), ("rho", None))):
            ax = axes[row][col]
            for m in ("DP-FED", "DP-CNT"):
                r = sub[sub.method == m].sort_values("eps")
                ax.errorbar(r.eps, r[metric], yerr=r[sd] if sd else None, marker={"DP-FED": "o", "DP-CNT": "s"}[m], ms=3, lw=1.2, capsize=2,
                            color=colors[m], label=m)
            for m, ls in (("FED", "-"), ("CNT", "--"), ("LOC", ":")):
                r = sub[(sub.method == m) & sub.eps.isna()]
                if len(r):
                    ax.axhline(r[metric].iloc[0], color="0.35", ls=ls, lw=0.9, label=m + " (non-private)")
            ax.set_xscale("log", base=2); ax.set_xticks([0.5, 1, 2, 4, 8]); ax.set_xticklabels(["0.5", "1", "2", "4", "8"])
            if row == 0:
                ax.set_title(ds)
            else:
                ax.set_xlabel(r"$\varepsilon$ ($\delta=10^{-5}$)")
    axes[0][0].set_ylabel("test AUC")
    axes[1][0].set_ylabel("Spearman $\\rho$ of $b$\n(vs. truth / vs. CEN)")
    axes[1][-1].legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "privacy_utility.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
