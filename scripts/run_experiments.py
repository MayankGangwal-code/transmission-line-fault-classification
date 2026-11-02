"""Generate the data, run every method, write the report.

Steps
-----
1. simulate the events: a training set and a test set on the 220 kV / 100 km
   reference system with randomised operating points, plus a test set on an
   *unseen* system (longer line, different source strengths, line constants
   changed by 15 %)
2. relay processing at 1 cycle after the disturbance detector trips ->
   feature table; rule-based classifiers; ML classifiers (fit on train only)
3. decision-time study (0.25 - 2 cycles), measurement-noise study (clean -> 20 dB)
   and the unseen-system test
4. fault location at 3 cycles: reactance method, Takagi method, ML regressor
5. the public Kaggle "Electrical fault detection and classification" dataset:
   random split versus time-block split (why the usual 99.9 % is an illusion)
6. figures, reports/results.md, reports/metrics.json, models/*.joblib,
   data/*.csv (features and event parameters), data/waveforms_sample.npz

Usage:  python scripts/run_experiments.py        (about 2 minutes)
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tlf import CLASSES, __version__  # noqa: E402
from tlf.classical import RULES, apply_rule  # noqa: E402
from tlf.evaluate import confusion, per_class, summarise  # noqa: E402
from tlf.features import analyse_record, feature_names  # noqa: E402
from tlf.locate import reactance_distance_km, takagi_distance_km  # noqa: E402
from tlf.models import RANDOM_STATE, build_classifiers, build_locator, decode, encode  # noqa: E402
from tlf.plots import (  # noqa: E402
    plot_bars,
    plot_confusion,
    plot_feature_importance,
    plot_lines,
    plot_location_errors,
    plot_sequence_angle_map,
    plot_waveforms,
)
from tlf.simulate import FaultSpec, LineParams, SystemParams, add_noise, random_spec, random_system, simulate  # noqa: E402

DATA, REPORTS, FIG, MODELS = ROOT / "data", ROOT / "reports", ROOT / "reports" / "figures", ROOT / "models"
N_TRAIN_PER_CLASS, N_TEST_PER_CLASS, N_UNSEEN_PER_CLASS = 600, 300, 200
TRAIN_SNR_DB = 40.0
DECISION_CYCLES = 1.0  # classification window
LOCATE_CYCLES = 3.0  # fault-location window (after the DC offset has decayed)
DECISION_TIMES = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
SNR_LEVELS = [np.inf, 40.0, 35.0, 30.0, 25.0, 20.0]

UNSEEN_SYSTEM = SystemParams(
    s_sc_mva=8000.0, xr_s=15.0, r_sc_mva=1500.0, xr_r=6.0,
    line=LineParams(length_km=150.0, R1=0.07 * 1.15, L1=1.27e-3 * 0.85, C1=9.0e-9 * 1.1, R0=0.25 * 0.85, L0=3.8e-3 * 1.15, C0=5.5e-9),
)


def md_table(df: pd.DataFrame, fmt="{:.3f}", index_name="") -> str:
    cols = [index_name] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx in df.index:
        cells = [str(idx)]
        for c in df.columns:  # df.at keeps the column dtype (iterrows would upcast integer columns to float)
            v = df.at[idx, c]
            cells.append("" if (isinstance(v, (float, np.floating)) and np.isnan(v)) else (fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pct(x) -> str:
    return f"{100 * x:.1f} %"


# --------------------------------------------------------------------------- #
# data generation
# --------------------------------------------------------------------------- #
def generate(n_per_class: int, seed: int, base: SystemParams | None = None, spread: float = 0.3):
    """Balanced set of events; every event has its own random operating point."""
    rng = np.random.default_rng(seed)
    recs, rows = [], []
    for k in range(n_per_class * len(CLASSES)):
        kind = CLASSES[k % len(CLASSES)]
        sysp = random_system(rng, base, spread)
        spec = random_spec(rng, kind, sysp)
        recs.append(simulate(spec, sysp))
        rows.append({"event": k, "label": kind, **asdict(spec), "delta_deg": sysp.delta_deg, "s_sc_mva": sysp.s_sc_mva,
                     "r_sc_mva": sysp.r_sc_mva, "e_s_pu": sysp.e_s_pu, "e_r_pu": sysp.e_r_pu, "line_km": sysp.line.length_km,
                     "rf_total": spec.r_arc + (spec.r_ground if (kind.endswith("G") or (kind == "ABC" and spec.three_phase_grounded)) else 0.0)})
    return recs, pd.DataFrame(rows)


def features_of(recs, snr_db: float | None, decision_cycles: float, seed: int = 0) -> pd.DataFrame:
    rows = []
    for k, r in enumerate(recs):
        rr = add_noise(r, snr_db, seed=seed * 100000 + k)  # ratio errors always, noise if snr is finite
        rows.append(analyse_record(rr, decision_cycles).features)
    return pd.DataFrame(rows)


def locate_all(recs, labels, snr_db: float | None, seed: int = 1):
    """Reactance and Takagi distances at LOCATE_CYCLES (loop chosen by ``labels``: the true class for the
    training set, the relay's predicted class for the test set), plus the 3-cycle feature table for the ML locator."""
    rows, feats = [], []
    for k, (r, lab) in enumerate(zip(recs, labels)):
        rr = add_noise(r, snr_db, seed=seed * 100000 + k)
        an = analyse_record(rr, LOCATE_CYCLES)
        rows.append({"reactance": reactance_distance_km(an, lab, r.system), "takagi": takagi_distance_km(an, lab, r.system)})
        feats.append(an.features)
    return pd.DataFrame(rows), pd.DataFrame(feats)


def main() -> None:
    t_start = time.time()
    for d in (DATA, FIG, MODELS):
        d.mkdir(parents=True, exist_ok=True)
    names = feature_names()

    # ------------------------------------------------------------------ #
    # 1. simulate
    # ------------------------------------------------------------------ #
    print("1. simulating events")
    t0 = time.time()
    train_recs, train_meta = generate(N_TRAIN_PER_CLASS, seed=1)
    test_recs, test_meta = generate(N_TEST_PER_CLASS, seed=2)
    unseen_recs, unseen_meta = generate(N_UNSEEN_PER_CLASS, seed=3, base=UNSEEN_SYSTEM, spread=0.2)
    print(f"   {len(train_recs)} train, {len(test_recs)} test, {len(unseen_recs)} unseen-system events in {time.time() - t0:.0f} s")
    y_train, y_test, y_unseen = train_meta["label"].to_numpy(), test_meta["label"].to_numpy(), unseen_meta["label"].to_numpy()

    # a small sample of waveforms for the figures and the app
    sample_idx = [int(np.where(y_test == c)[0][0]) for c in CLASSES]
    np.savez_compressed(
        DATA / "waveforms_sample.npz",
        t=test_recs[0].t, v=np.stack([test_recs[i].v for i in sample_idx]), i=np.stack([test_recs[i].i for i in sample_idx]),
        labels=np.array([y_test[i] for i in sample_idx]), fault_index=np.array([test_recs[i].fault_index for i in sample_idx]),
        distance_km=np.array([test_recs[i].spec.distance_km for i in sample_idx]), rf_total=test_meta.loc[sample_idx, "rf_total"].to_numpy(),
    )

    # ------------------------------------------------------------------ #
    # 2. features at 1 cycle, rules, ML
    # ------------------------------------------------------------------ #
    print("2. relay features at 1 cycle and classifiers")
    X_train = features_of(train_recs, TRAIN_SNR_DB, DECISION_CYCLES, seed=1)
    X_test = features_of(test_recs, TRAIN_SNR_DB, DECISION_CYCLES, seed=2)
    X_unseen = features_of(unseen_recs, TRAIN_SNR_DB, DECISION_CYCLES, seed=3)
    for name, X, meta in [("train", X_train, train_meta), ("test", X_test, test_meta), ("unseen", X_unseen, unseen_meta)]:
        pd.concat([meta, X], axis=1).to_csv(DATA / f"events_{name}.csv", index=False)

    preds_test: dict[str, np.ndarray] = {}
    for name, rule in RULES.items():
        preds_test[name] = apply_rule(rule, X_test)
    clfs = build_classifiers()
    fit_s = {}
    for name, pipe in clfs.items():
        t0 = time.time()
        pipe.fit(X_train, encode(y_train))
        fit_s[name] = time.time() - t0
        preds_test[name] = decode(pipe.predict(X_test))
    results = pd.DataFrame({name: summarise(y_test, p) for name, p in preds_test.items()}).T
    for name in results.index:
        print(f"   {name:28s} accuracy {results.loc[name, 'accuracy']:.4f}   macro-F1 {results.loc[name, 'macro_f1']:.4f}   false trips {int(results.loc[name, 'false_trips'])}")
    ml_names = list(clfs)
    # the model used for the follow-up studies is chosen on the TRAINING set (5-fold cross-validated
    # macro-F1), never on the test set; ties go to the first model in the list
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_f1 = {name: float(cross_val_score(build_classifiers()[name], X_train, encode(y_train), cv=cv, scoring="f1_macro").mean()) for name in ml_names}
    best_name = max(ml_names, key=lambda k: round(cv_f1[k], 4))
    best = clfs[best_name]
    print("   5-fold CV macro-F1 on the training set: " + "  ".join(f"{k}={v:.4f}" for k, v in cv_f1.items()))
    print(f"   best ML classifier (by training-set CV): {best_name}")
    joblib.dump({"models": clfs, "best": best_name, "feature_names": names, "package_version": __version__}, MODELS / "classifiers.joblib")

    # ------------------------------------------------------------------ #
    # 3a. decision time
    # ------------------------------------------------------------------ #
    print("3a. accuracy versus decision time")
    dt_rows = {}
    for tau in DECISION_TIMES:
        Xtr_t = X_train if tau == DECISION_CYCLES else features_of(train_recs, TRAIN_SNR_DB, tau, seed=1)
        Xte_t = X_test if tau == DECISION_CYCLES else features_of(test_recs, TRAIN_SNR_DB, tau, seed=2)
        row = {name: summarise(y_test, apply_rule(rule, Xte_t))["accuracy"] for name, rule in RULES.items()}
        pipe = build_classifiers()[best_name]
        pipe.fit(Xtr_t, encode(y_train))
        row[f"{best_name} (trained at this time)"] = summarise(y_test, decode(pipe.predict(Xte_t)))["accuracy"]
        dt_rows[tau] = row
        print(f"   {tau:4.2f} cycles: " + "  ".join(f"{k.split(' (')[0]}={v:.3f}" for k, v in row.items()))
    decision_table = pd.DataFrame(dt_rows).T
    decision_table.index.name = "cycles after trigger"

    # ------------------------------------------------------------------ #
    # 3b. noise
    # ------------------------------------------------------------------ #
    print("3b. accuracy versus measurement noise (models trained at 40 dB)")
    snr_rows = {}
    for snr in SNR_LEVELS:
        Xte_s = X_test if snr == TRAIN_SNR_DB else features_of(test_recs, snr, DECISION_CYCLES, seed=2)
        row = {name: summarise(y_test, apply_rule(rule, Xte_s))["accuracy"] for name, rule in RULES.items()}
        for name in ml_names:
            row[name] = summarise(y_test, decode(clfs[name].predict(Xte_s)))["accuracy"]
        snr_rows["clean" if not np.isfinite(snr) else f"{snr:.0f} dB"] = row
        print(f"   SNR {list(snr_rows)[-1]:6s}: " + "  ".join(f"{k[:14]}={v:.3f}" for k, v in row.items()))
    noise_table = pd.DataFrame(snr_rows).T
    noise_table.index.name = "SNR"

    # ------------------------------------------------------------------ #
    # 3c. unseen system
    # ------------------------------------------------------------------ #
    print("3c. unseen system (150 km line, different sources, line constants changed by 15 %)")
    preds_unseen = {name: apply_rule(rule, X_unseen) for name, rule in RULES.items()}
    for name in ml_names:
        preds_unseen[name] = decode(clfs[name].predict(X_unseen))
    unseen_results = pd.DataFrame({name: summarise(y_unseen, p) for name, p in preds_unseen.items()}).T
    for name in unseen_results.index:
        print(f"   {name:28s} accuracy {unseen_results.loc[name, 'accuracy']:.4f}")

    # ------------------------------------------------------------------ #
    # 4. fault location (3-cycle window)
    # ------------------------------------------------------------------ #
    print("4. fault location")
    fault_tr = y_train != "none"
    fault_te = y_test != "none"
    loc_tr, F3_tr = locate_all(train_recs, y_train, TRAIN_SNR_DB, seed=4)
    loc_te, F3_te = locate_all(test_recs, preds_test[best_name], TRAIN_SNR_DB, seed=5)  # loop of the *predicted* class
    onehot = lambda labels: pd.get_dummies(pd.Categorical(labels, categories=CLASSES)).astype(float)  # noqa: E731
    L_tr = pd.concat([F3_tr, onehot(y_train).reset_index(drop=True)], axis=1)[fault_tr]
    L_te = pd.concat([F3_te, onehot(preds_test[best_name]).reset_index(drop=True)], axis=1)  # the ML locator uses the *predicted* class
    locator = build_locator()
    locator.fit(L_tr, train_meta.loc[fault_tr, "distance_km"])
    loc_te["ml"] = locator.predict(L_te)
    loc_te.loc[~fault_te, "ml"] = np.nan
    joblib.dump({"locator": locator, "columns": list(L_tr.columns)}, MODELS / "locator.joblib")
    d_true = test_meta["distance_km"].to_numpy()
    loc_df = pd.DataFrame({"rf": test_meta["rf_total"], "label": y_test, "d": d_true})
    for m in ("reactance", "takagi", "ml"):
        loc_df[f"{m}_err"] = np.abs(loc_te[m].to_numpy() - d_true)
    loc_df = loc_df[fault_te]
    loc_summary = pd.DataFrame({m: {"mae_km": loc_df[f"{m}_err"].mean(), "median_km": loc_df[f"{m}_err"].median(), "p90_km": loc_df[f"{m}_err"].quantile(0.9), "max_km": loc_df[f"{m}_err"].max()} for m in ("reactance", "takagi", "ml")}).T
    loc_summary.index = ["Reactance method", "Takagi method", "ML regressor (gradient boosting)"]
    for name in loc_summary.index:
        print(f"   {name:34s} MAE {loc_summary.loc[name, 'mae_km']:.2f} km  median {loc_summary.loc[name, 'median_km']:.2f} km  90th pct {loc_summary.loc[name, 'p90_km']:.2f} km")
    low_rf = loc_df[loc_df.rf < 0.5]
    loc_by_rf = loc_df.groupby(pd.cut(loc_df["rf"], [0, 0.5, 2, 5, 10, 20, 60]), observed=False)[["reactance_err", "takagi_err", "ml_err"]].median()
    loc_by_rf.index = [str(i) for i in loc_by_rf.index]
    loc_by_rf.columns = ["Reactance", "Takagi", "ML"]

    # ------------------------------------------------------------------ #
    # 5. Kaggle benchmark
    # ------------------------------------------------------------------ #
    print("5. public Kaggle dataset benchmark")
    kaggle = None
    kfile = DATA / "external" / "classData.csv"
    if kfile.exists():
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        kd = pd.read_csv(kfile)
        kd["cls"] = kd[["A", "B", "C", "G"]].astype(int).astype(str).agg("".join, axis=1)
        Xk, yk = kd[["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]].to_numpy(), kd["cls"].to_numpy()
        Xa, Xb, ya, yb = train_test_split(Xk, yk, test_size=0.3, random_state=0, stratify=yk)
        rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(Xa, ya)
        acc_random = float(np.mean(rf.predict(Xb) == yb))
        # time-block split: the file is six contiguous runs (one simulation per class); train on the
        # first 70 % of every run, test on the last 30 %
        run = (kd["cls"] != kd["cls"].shift()).cumsum()
        pos = kd.groupby(run).cumcount() / kd.groupby(run)["cls"].transform("size")
        tr_mask = (pos < 0.7).to_numpy()
        rf2 = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(Xk[tr_mask], yk[tr_mask])
        acc_block = float(np.mean(rf2.predict(Xk[~tr_mask]) == yk[~tr_mask]))
        # sampling is irregular (variable-step solver, no time column): estimate samples per cycle per run from zero crossings
        spc = {}
        for r_id, g in kd.groupby(run):
            zc = np.where(np.diff(np.sign(g["Va"].to_numpy())) > 0)[0]
            spc[str(g["cls"].iloc[0])] = float(np.median(np.diff(zc))) if len(zc) > 2 else float("nan")
        spc = {k: (None if not np.isfinite(v) else v) for k, v in spc.items()}  # strict JSON: no NaN
        kaggle = {"n_rows": int(len(kd)), "n_classes": int(kd["cls"].nunique()), "n_runs": int(run.nunique()), "random_split_accuracy": acc_random,
                  "time_block_split_accuracy": acc_block, "samples_per_cycle_by_run": spc}
        print(f"   {len(kd)} rows, {kd['cls'].nunique()} classes in {run.nunique()} contiguous runs: random split {acc_random:.4f}, time-block split {acc_block:.4f}")
    else:
        print("   data/external/classData.csv not found - skipped")

    # ------------------------------------------------------------------ #
    # 6. figures
    # ------------------------------------------------------------------ #
    print("6. figures")
    ex = simulate(FaultSpec(kind="AG", distance_km=60, r_arc=0.5, r_ground=5.0, inception_deg=90.0), SystemParams(), post_cycles=4)
    ex = add_noise(ex, TRAIN_SNR_DB, seed=7)
    an_ex = analyse_record(ex, DECISION_CYCLES)
    plot_waveforms(ex, "Phase-A-to-ground fault at 60 km, Rf = 5.5 Ω, inception at the voltage zero crossing (maximum DC offset)", FIG / "waveforms_AG.png",
                   trigger=an_ex.trigger_index, decision_end=int(an_ex.trigger_index + DECISION_CYCLES * ex.system.samples_per_cycle))
    ex2 = add_noise(simulate(FaultSpec(kind="BCG", distance_km=25, r_arc=0.2, r_ground=1.0, inception_deg=70.0), SystemParams(), post_cycles=4), TRAIN_SNR_DB, seed=8)
    an_ex2 = analyse_record(ex2, DECISION_CYCLES)
    plot_waveforms(ex2, "B-C-to-ground fault at 25 km, Rf = 1.2 Ω", FIG / "waveforms_BCG.png", trigger=an_ex2.trigger_index,
                   decision_end=int(an_ex2.trigger_index + DECISION_CYCLES * ex2.system.samples_per_cycle))
    plot_sequence_angle_map(X_test, y_test, FIG / "sequence_angle_map.png")
    plot_confusion(confusion(y_test, preds_test["Sequence-angle relay logic"]), FIG / "confusion_sequence_rule.png", title="Sequence-angle relay logic — test set")
    plot_confusion(confusion(y_test, preds_test[best_name]), FIG / f"confusion_best_ml.png", title=f"{best_name} — test set")
    plot_bars(results["accuracy"], FIG / "accuracy_all_methods.png", title=f"Accuracy on the {len(y_test)} test events (11 classes, 1 cycle after trigger)", highlight=best_name)
    plot_lines(decision_table, FIG / "accuracy_vs_decision_time.png", title="Accuracy versus decision time after the disturbance detector trips", xlabel="cycles after trigger (1 cycle = 20 ms)", ylim=(0.4, 1.02))
    # noise_table rows are already ordered clean -> 20 dB (SNR_LEVELS order): more noise to the right
    plot_lines(noise_table, FIG / "accuracy_vs_noise.png", title="Accuracy versus measurement noise (SNR), models trained at 40 dB", xlabel="signal-to-noise ratio (less noise on the left)", ylim=(0.4, 1.02))
    plot_bars(unseen_results["accuracy"], FIG / "accuracy_unseen_system.png", title="Unseen system: 150 km line, other sources, line constants ±15 %", highlight=best_name)
    plot_location_errors(loc_df, ["reactance", "takagi", "ml"], FIG / "location_error_vs_rf.png")
    # the best model is (almost) perfect on the test set, so permutation importance is ~0 for every
    # feature (the features are redundant); use the Random Forest's impurity-based importance instead
    rf = clfs["Random Forest"].named_steps["clf"]
    imp = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)
    plot_feature_importance(imp, FIG / "feature_importance.png", title="Random Forest feature importance (mean decrease in impurity)")

    # ------------------------------------------------------------------ #
    # 7. metrics + report
    # ------------------------------------------------------------------ #
    cm_seq = confusion(y_test, preds_test["Sequence-angle relay logic"])
    cm_best = confusion(y_test, preds_test[best_name])
    pc_best = per_class(y_test, preds_test[best_name])
    pc_seq = per_class(y_test, preds_test["Sequence-angle relay logic"])
    hi_rf = test_meta["rf_total"] > 20
    hi_rf_acc = {name: float(np.mean(preds_test[name][hi_rf.to_numpy()] == y_test[hi_rf.to_numpy()])) for name in preds_test}
    summary = {
        "package_version": __version__, "n_train": int(len(y_train)), "n_test": int(len(y_test)), "n_unseen": int(len(y_unseen)),
        "n_features": len(names), "features": names, "decision_cycles": DECISION_CYCLES, "locate_cycles": LOCATE_CYCLES, "train_snr_db": TRAIN_SNR_DB,
        "best_ml": best_name, "cv_macro_f1_train": {k: round(v, 4) for k, v in cv_f1.items()},
        "test": results.round(4).to_dict(orient="index"), "unseen_system": unseen_results.round(4).to_dict(orient="index"),
        "decision_time": decision_table.round(4).to_dict(orient="index"), "noise": noise_table.round(4).to_dict(orient="index"),
        "high_resistance_accuracy_rf_gt_20ohm": {k: round(v, 4) for k, v in hi_rf_acc.items()}, "n_high_resistance": int(hi_rf.sum()),
        "location": loc_summary.round(3).to_dict(orient="index"), "location_by_rf_median_km": loc_by_rf.round(3).to_dict(orient="index"),
        "location_bolted_rf_lt_0p5": {m: round(float(low_rf[f"{m}_err"].median()), 3) for m in ("reactance", "takagi", "ml")},
        "feature_importance_top10": imp.head(10).round(4).to_dict(), "kaggle": kaggle, "fit_seconds": {k: round(v, 1) for k, v in fit_s.items()},
        "per_class_best": pc_best.to_dict(orient="index"), "per_class_sequence_rule": pc_seq.to_dict(orient="index"),
        "runtime_s": round(time.time() - t_start, 1),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2, default=float, allow_nan=False))

    res_show = results[["accuracy", "macro_f1", "group_accuracy", "detection_accuracy", "missed_faults", "false_trips"]].copy()
    res_show["accuracy"], res_show["macro_f1"], res_show["group_accuracy"], res_show["detection_accuracy"] = (
        [pct(v) for v in results.accuracy], [f"{v:.3f}" for v in results.macro_f1], [pct(v) for v in results.group_accuracy], [pct(v) for v in results.detection_accuracy])
    res_show.columns = ["accuracy (11 classes)", "macro-F1", "fault-type accuracy (LG/LL/LLG/LLL/none)", "fault vs no-fault", "missed faults", "false trips"]
    res_show["missed faults"], res_show["false trips"] = res_show["missed faults"].astype(int), res_show["false trips"].astype(int)
    pc_best, pc_seq = pc_best.copy(), pc_seq.copy()
    pc_best["support"], pc_seq["support"] = pc_best["support"].astype(int), pc_seq["support"].astype(int)
    unseen_show = pd.DataFrame({"accuracy": [pct(v) for v in unseen_results.accuracy], "macro-F1": [f"{v:.3f}" for v in unseen_results.macro_f1],
                                "false trips": unseen_results.false_trips.astype(int)}, index=unseen_results.index)
    dt_show = decision_table.apply(lambda s: s.map(pct))
    nt_show = noise_table.apply(lambda s: s.map(pct))
    hi_show = pd.DataFrame({"accuracy": [pct(v) for v in hi_rf_acc.values()]}, index=list(hi_rf_acc))
    loc_show = loc_summary.copy().round(2)
    loc_show.columns = ["MAE (km)", "median (km)", "90th percentile (km)", "max (km)"]
    kaggle_txt = ""
    if kaggle:
        kaggle_txt = f"""## 6. The public Kaggle dataset — what it can and cannot show

The widely used Kaggle set *"Electrical fault detection and classification"* ({kaggle['n_rows']} rows,
{kaggle['n_classes']} classes) is a Simulink export in which every row is **one time sample** and the file is
{kaggle['n_runs']} contiguous runs — a single simulation per class, at one location and one fault resistance, with
no time column and an irregular (variable-step) sample spacing. That rules out relay-style processing (no
window can be defined) and any test of generalisation. Classifying the raw samples one at a time with a
Random Forest gives **{pct(kaggle['random_split_accuracy'])}** with a random 70/30 split and **{pct(kaggle['time_block_split_accuracy'])}**
when trained on the first 70 % of every run and tested on the last 30 %: an instantaneous sample is simply an
ambiguous description of a fault (every waveform passes through zero). A relay works on a window — which is
exactly what this project does, on events it can vary.

"""
    report = f"""# Results

Generated by `scripts/run_experiments.py` (random_state = {RANDOM_STATE}, runtime {summary['runtime_s']} s).

**Data.** {len(y_train)} training and {len(y_test)} test events simulated on a 220 kV, 100 km, 50 Hz line between two
sources ({N_TRAIN_PER_CLASS} / {N_TEST_PER_CLASS} per class, 11 classes: no fault or balanced load switching, AG, BG, CG, AB, BC, CA, ABG,
BCG, CAG, ABC). Every event has its own random fault location (2–98 km), arc resistance (0.01–5 Ω), ground
resistance (0.1–50 Ω), inception angle (0–360°), load angle (5–30°), source strengths (±30 %) and source
voltages. A third set of {len(y_unseen)} events uses an **unseen system** (150 km line, line constants changed by 15 %,
different source strengths). The relay sees the signals through instrument transformers with random
per-phase ratio errors of ±2 % and with white measurement noise at 40 dB SNR, sampled at {int(SystemParams().fs)} Hz
({SystemParams().samples_per_cycle} samples per cycle).

**Relay processing.** Delta-current disturbance detector → full-cycle DFT phasors of the window that ends
{DECISION_CYCLES:g} cycle after the trigger → symmetrical components, superimposed quantities, loop impedances
({len(names)} features). Fault location uses the window ending {LOCATE_CYCLES:g} cycles after the trigger, when the DC
offset has decayed.

## 1. Classification, 1 cycle after the trigger ({len(y_test)} test events)

{md_table(res_show, index_name="Method")}

![acc](figures/accuracy_all_methods.png)

Best ML classifier — chosen by 5-fold cross-validated macro-F1 on the *training* set (ties go to the first in
the list), used for the decision-time study, the confusion matrix below and the ML locator's class input:
**{best_name}** ({", ".join(f"{k} {v:.4f}" for k, v in cv_f1.items())}).

### High-resistance faults (total fault resistance > 20 Ω, n = {int(hi_rf.sum())})

{md_table(hi_show, index_name="Method")}

### Confusion matrices

![cm1](figures/confusion_sequence_rule.png)

![cm2](figures/confusion_best_ml.png)

### Per-class precision / recall / F1 — {best_name}

{md_table(pc_best, index_name="Class")}

### Per-class precision / recall / F1 — sequence-angle relay logic

{md_table(pc_seq, index_name="Class")}

## 2. How fast can the decision be made?

Accuracy when the decision window ends the given number of cycles after the disturbance detector trips
(a window shorter than one cycle still contains pre-fault samples). Rules are fixed; the ML model is
retrained for each decision time.

{md_table(dt_show, index_name="cycles after trigger")}

![dt](figures/accuracy_vs_decision_time.png)

## 3. Measurement noise

Models and thresholds unchanged (trained / set at 40 dB); the test waveforms get white noise at the given SNR.

{md_table(nt_show, index_name="SNR")}

![noise](figures/accuracy_vs_noise.png)

## 4. Unseen system ({len(y_unseen)} events: 150 km line, line constants ±15 %, other source strengths)

{md_table(unseen_show, index_name="Method")}

![unseen](figures/accuracy_unseen_system.png)

## 5. Fault location ({int(fault_te.sum())} faulted test events, window ending {LOCATE_CYCLES:g} cycles after the trigger)

All three locators use the fault class predicted by {best_name} (the relay's own classification) to choose
the loop; the ML regressor is trained on the faulted training events with their true class.

{md_table(loc_show, fmt="{:.2f}", index_name="Method")}

Median |error| in km by total fault resistance:

{md_table(loc_by_rf, fmt="{:.2f}", index_name="fault resistance (Ω)")}

![loc](figures/location_error_vs_rf.png)

The reactance method is exact for bolted faults (median {low_rf['reactance_err'].median():.2f} km below 0.5 Ω) and loses accuracy as the
fault resistance grows, because the current fed from the remote end makes the voltage drop across the
fault resistance look partly reactive. Takagi's method corrects most of that with the superimposed
current; the ML regressor learns the residual for *this* system.

{kaggle_txt}## {7 if kaggle else 6}. Which features matter? (Random Forest, mean decrease in impurity)

![fi](figures/feature_importance.png)

## {8 if kaggle else 7}. The relay's decision space

The relay's phase selector is the angle between the negative-sequence current and the superimposed
positive-sequence current, ∠(I2/ΔI1): six sectors, one per faulted phase or phase pair (AG 0°, BG +120°,
CG −120°, BC/BCG 180°, AB/ABG +60°, CA/CAG −60°); residual current then separates LL from LLG. The left
panel also shows the classical ground selector ∠(I2/I0), which works for single-phase faults (AG 0°,
BG −120°, CG +120°) but drifts for double-line-to-ground faults with ground resistance — the reason the
rule keys on ∠(I2/ΔI1). Every test event, coloured by its true class:

![map](figures/sequence_angle_map.png)

## Example waveforms

![w1](figures/waveforms_AG.png)

![w2](figures/waveforms_BCG.png)
"""
    (REPORTS / "results.md").write_text(report)
    print(f"\nwrote {REPORTS / 'results.md'}, {REPORTS / 'metrics.json'}, models/, data/ and {len(list(FIG.glob('*.png')))} figures")
    print("\naccuracy on the test set:")
    print(results["accuracy"].round(4).to_string())
    print(f"\ndone in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
