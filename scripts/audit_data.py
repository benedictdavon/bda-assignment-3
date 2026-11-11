"""Read-only dataset diagnostics; no predictive model fitting or submissions.

Run from Assignment 3: python scripts/audit_data.py
Outputs are deterministic apart from environment metadata.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "reports/data_audit"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, test, sample = [pd.read_csv(RAW / n) for n in
                           ("train.csv", "test.csv", "sample_submission.csv")]
    features = [c for c in test if c != "id"]
    assert list(train) == [*list(test), "quality"]
    assert list(sample) == ["id", "quality"]
    assert sample.id.equals(test.id)
    y = train.quality
    assert y.notna().all() and np.isfinite(y).all()
    assert ((y == np.floor(y)) & y.between(0, 10)).all()
    counts = y.astype(int).value_counts().sort_index().rename("count").to_frame()
    counts["fraction"] = counts["count"] / len(train)
    counts.to_csv(OUT / "target_distribution.csv", index_label="quality")
    schemas = []
    summary = {}
    for name, d in [("train", train), ("test", test), ("sample_submission", sample)]:
        schemas.extend({"dataset": name, "column": c, "dtype": str(d[c].dtype),
                        "missing": int(d[c].isna().sum()), "unique": d[c].nunique()}
                       for c in d)
        summary[name] = {"shape": list(d.shape), "id_min": int(d.id.min()),
                         "id_max": int(d.id.max()), "id_duplicates": int(d.id.duplicated().sum()),
                         "ids_contiguous_sorted": bool(np.array_equal(d.id, np.arange(d.id.min(), d.id.min()+len(d))))}
        if name != "sample_submission":
            d[features].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T.to_csv(OUT / f"{name}_feature_summary.csv")
            summary[name]["duplicate_feature_excess_rows"] = int(d.duplicated(features).sum())
            summary[name]["duplicate_feature_affected_rows"] = int(d.duplicated(features,keep=False).sum())
            summary[name]["nonfinite_features"] = int((~np.isfinite(d[features])).sum().sum())
            summary[name]["negative_features"] = {c:int((d[c]<0).sum()) for c in features}
            summary[name]["free_sulfur_gt_total"] = int((d['free sulfur dioxide']>d['total sulfur dioxide']).sum())
    pd.DataFrame(schemas).to_csv(OUT / "schema.csv", index=False)
    groups = train.groupby(features, dropna=False).quality.agg(["size","nunique","min","max"])
    conflicts = groups[groups["nunique"]>1]
    conflicts.to_csv(OUT / "conflicting_duplicates.csv")
    summary["conflicting_duplicate_groups"] = len(conflicts)
    summary["conflicting_duplicate_rows"] = int(conflicts["size"].sum())
    cross = test.merge(train, on=features, how="inner", suffixes=("_test","_train"))
    cross.to_csv(OUT / "train_test_exact_matches.csv",index=False)
    summary["train_test_exact_pairs"] = len(cross)
    summary["test_rows_exact_match_train"] = int(cross.id_test.nunique())
    train[features+["quality"]].corr(method="spearman").to_csv(OUT / "spearman_correlations.csv")
    train[features+["quality"]].corr().to_csv(OUT / "pearson_correlations.csv")
    drift = []
    for c in features:
        a,b=train[c],test[c]
        ks=ks_2samp(a,b)
        q1,q3=a.quantile([.25,.75]);iqr=q3-q1
        drift.append({"feature":c,"ks_statistic":ks.statistic,"ks_pvalue_unadjusted":ks.pvalue,
                      "standardized_mean_difference":(b.mean()-a.mean())/a.std(),
                      "test_outside_train_range":int(((b<a.min())|(b>a.max())).sum()),
                      "train_iqr_outliers":int(((a<q1-1.5*iqr)|(a>q3+1.5*iqr)).sum()),
                      "target_spearman":spearmanr(a,y).statistic,
                      "id_spearman":spearmanr(train.id,a).statistic})
    pd.DataFrame(drift).to_csv(OUT / "drift_and_outliers.csv",index=False)
    bins = pd.qcut(train.id,10,labels=False)
    pd.crosstab(bins,y,normalize="index").to_csv(OUT / "id_decile_target_fractions.csv")
    summary["id_target_spearman"] = float(spearmanr(train.id,y).statistic)
    # IQR distances are diagnostics only, fitted on train features without labels.
    scale = (train[features].quantile(.75)-train[features].quantile(.25)).replace(0,1)
    a = train[features].to_numpy()/scale.to_numpy()
    b = test[features].to_numpy()/scale.to_numpy()
    nn = NearestNeighbors(n_neighbors=2,algorithm="kd_tree",n_jobs=1).fit(a)
    d,i=nn.kneighbors(a)
    dt,it=nn.kneighbors(b,n_neighbors=1)
    pd.DataFrame({"id":train.id,"nearest_train_id":train.id.to_numpy()[i[:,1]],
                  "iqr_euclidean_distance":d[:,1],"absolute_quality_gap":np.abs(y.to_numpy()-y.to_numpy()[i[:,1]])}).to_csv(OUT/"train_nearest_neighbors.csv",index=False)
    pd.DataFrame({"id":test.id,"nearest_train_id":train.id.to_numpy()[it[:,0]],
                  "iqr_euclidean_distance":dt[:,0]}).to_csv(OUT/"test_nearest_neighbors.csv",index=False)
    for name,dist in [("train_to_other_train",d[:,1]),("test_to_train",dt[:,0])]:
        summary[name] = {"distance_quantiles":dict(zip(["0","1%","5%","50%","95%","100%"],np.quantile(dist,[0,.01,.05,.5,.95,1]).tolist())),
                         "rows_distance_le_0.05":int((dist<=.05).sum()),"rows_distance_le_0.1":int((dist<=.1).sum())}
    summary["train_near_le_0.1_label_gap_mean"] = float(np.abs(y.to_numpy()-y.to_numpy()[i[:,1]])[d[:,1]<=.1].mean()) if (d[:,1]<=.1).any() else None
    manifest=[]
    for f in [ROOT/"BigData_Assignment 3.pdf",ROOT/"nycu-big-data-2026-fall-assignment-3.zip",*sorted(RAW.glob("*.csv"))]:
        manifest.append({"path":f.relative_to(ROOT).as_posix(),"bytes":f.stat().st_size,"sha256":hashlib.sha256(f.read_bytes()).hexdigest()})
    (OUT/"input_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    env={"python":sys.version,"executable":sys.executable,"platform":platform.platform(),"logical_cpu_count":os.cpu_count()}
    for pkg in ["numpy","pandas","scipy","scikit-learn","lightgbm","catboost","xgboost","torch"]:
        try: env[pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError: env[pkg]="not installed"
    (OUT/"environment.json").write_text(json.dumps(env,indent=2),encoding="utf-8")
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    print(counts.to_string())


if __name__ == "__main__":
    main()
