"""Independent count-matrix QWK and contract checks, not model experiments."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import cohen_kappa_score


def qwk(y_true, y_pred):
    a,b=np.asarray(y_true),np.asarray(y_pred)
    if a.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError("nonempty equal-length 1D arrays required")
    for x in (a,b):
        if not (np.isfinite(x).all() and (x == np.floor(x)).all() and ((0<=x)&(x<=10)).all()):
            raise ValueError("integer labels in 0..10 required")
    a,b=a.astype(int),b.astype(int)
    o=np.zeros((11,11));np.add.at(o,(a,b),1)
    e=np.outer(o.sum(1),o.sum(0))/len(a)
    w=(np.arange(11)[:,None]-np.arange(11)[None,:])**2/100
    den=(w*e).sum()
    return float(1-(w*o).sum()/den) if den else float("nan")


def main():
    cases=[([3,4,5,6,7,8],[3,4,5,6,7,8]),([3,4,5,6,7,8],[8,7,6,5,4,3]),
           ([3,4,5,6,7,8],[5]*6),([3,5,8],[3,8,5]),([4,4,4],[5,5,5])]
    rng=np.random.default_rng(20261111)
    cases.extend((rng.integers(0,11,100).tolist(),rng.integers(0,11,100).tolist()) for _ in range(100))
    gaps=[]
    moment_gaps=[]
    for a,b in cases:
        expected=cohen_kappa_score(a,b,labels=list(range(11)),weights="quadratic")
        actual=qwk(a,b);assert np.isclose(expected,actual,atol=1e-12)
        gaps.append(abs(expected-actual))
        av,bv=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
        den=av.var()+bv.var()+(av.mean()-bv.mean())**2
        moment=1-np.mean((av-bv)**2)/den
        assert np.isclose(actual,moment,atol=1e-12)
        moment_gaps.append(abs(actual-moment))
    assert np.isnan(qwk([5,5],[5,5]))
    rejected=0
    for a,b in [([],[]),([5],[5.1]),([5],[11]),([5],[float('nan')]),([5],[4,5])]:
        try:qwk(a,b)
        except ValueError:rejected+=1
    assert rejected==5
    result={"matrix_vs_sklearn_cases":len(cases),"max_absolute_difference":max(gaps),
            "moment_identity_cases":len(cases),"moment_identity_max_difference":max(moment_gaps),
            "invalid_contract_cases_rejected":rejected,"degenerate_constant_identical":"NaN",
            "competition_server_parity":"not verified; server scorer inaccessible"}
    out=Path(__file__).resolve().parents[1]/"reports/data_audit/metric_checks.json"
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
