"""Freeze split membership only. No model fitting and no holdout prediction scores."""
from pathlib import Path
import hashlib
import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT=Path(__file__).resolve().parents[1]


def main():
    train_path=ROOT/'data/raw/train.csv'
    d=pd.read_csv(train_path)
    features=[c for c in d if c not in ['id','quality']]
    assert not d.duplicated(features).any(), 'Choose grouped splits for duplicates'
    dev,holdout=train_test_split(d.index,test_size=.2,stratify=d.quality,random_state=20261111)
    rows=d[['id']].copy();rows['partition']='development';rows.loc[holdout,'partition']='confirmation'
    for seed in [20261111,20261112]:
        col=f'fold_seed_{seed}';rows[col]=-1
        splitter=StratifiedKFold(5,shuffle=True,random_state=seed)
        for fold,(_,val) in enumerate(splitter.split(dev,d.loc[dev,'quality'])):
            rows.loc[dev[val],col]=fold
        assert (rows.loc[holdout,col]==-1).all()
        assert set(rows.loc[dev,col])==set(range(5))
    out=ROOT/'configs';out.mkdir(exist_ok=True)
    path=out/'split_membership_v1.csv'
    content=rows.to_csv(index=False).encode()
    if path.exists() and path.read_bytes()!=content:
        raise RuntimeError('Refuse to change existing v1 split; create version v2')
    path.write_bytes(content)
    manifest={'version':'v1','train_sha256':hashlib.sha256(train_path.read_bytes()).hexdigest(),
              'membership_sha256':hashlib.sha256(content).hexdigest(),
              'development_rows':len(dev),'confirmation_rows':len(holdout),
              'holdout_seed':20261111,'cv_seeds':[20261111,20261112],'folds':5,
              'group_policy':'No exact duplicates; no near pairs at IQR Euclidean distance <= 0.1. IID stratification assumed.',
              'holdout_status':'Descriptive full-data audit performed before reservation. No predictive performance evaluated.'}
    (out/'split_manifest_v1.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
