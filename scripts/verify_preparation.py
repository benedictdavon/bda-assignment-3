"""Verify local preparation contracts and Markdown link targets; no predictive fits."""
from pathlib import Path
from urllib.parse import unquote
import hashlib
import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REQUIRED={
    'docs/agents/ASSIGNMENT_SPEC.md','docs/agents/DATASET_CARD.md',
    'docs/agents/METRIC_AND_VALIDATION.md','docs/agents/COMPETITION_RULES.md',
    'docs/agents/WORKFLOW_AND_CONVENTIONS.md','docs/research/README.md',
    'docs/research/SOURCE_REGISTRY.md','docs/research/LITERATURE_REVIEW.md',
    'docs/research/KAGGLE_AND_COMMUNITY_INSIGHTS.md','docs/research/MODELING_STRATEGIES.md',
    'docs/research/QWK_OPTIMIZATION.md','docs/research/VALIDATION_AND_LEAKAGE.md',
    'docs/research/EXPLAINABILITY.md','docs/research/RESEARCH_SYNTHESIS.md',
    'docs/research/PAPERS_TO_DOWNLOAD.md','docs/EXPERIMENT_PLAN.md','docs/EXPERIMENT_LOG.md'}


def main():
    receipt_path=ROOT/'reports/preparation_verification.json'
    missing=[n for n in sorted(REQUIRED) if not (ROOT/n).is_file()]
    assert not missing,missing
    errors=[];links=0;words=0
    for path in sorted(ROOT.rglob('*.md')):
        text=path.read_text(encoding='utf-8')
        assert text.strip(),path
        words+=len(text.split())
        for target in re.findall(r'!?\[[^\]]*\]\(([^\s]+)\)',text):
            if re.match(r'^[a-zA-Z]+:',target):continue
            local=unquote(target.split('#')[0]);links+=1
            resolved=(path.parent/local).resolve()
            # The log links to this check's receipt, which is created after checks pass.
            if local and not resolved.exists() and resolved!=receipt_path.resolve():
                errors.append(f'{path.relative_to(ROOT)} -> {target}')
    assert not errors,errors
    inputs=json.loads((ROOT/'reports/data_audit/input_manifest.json').read_text())
    for item in inputs:
        assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256'],item
    materials_path=ROOT/'reports/materials_review/manifest.json'
    materials_verified=None
    if materials_path.is_file():
        materials=json.loads(materials_path.read_text(encoding='utf-8'))
        for item in materials:
            assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256'],item
        materials_verified=len(materials)
    split_path=ROOT/'configs/split_membership_v1.csv'
    manifest=json.loads((ROOT/'configs/split_manifest_v1.json').read_text())
    assert hashlib.sha256(split_path.read_bytes()).hexdigest()==manifest['membership_sha256']
    train=pd.read_csv(ROOT/'data/raw/train.csv');test=pd.read_csv(ROOT/'data/raw/test.csv')
    sample=pd.read_csv(ROOT/'data/raw/sample_submission.csv');splits=pd.read_csv(split_path)
    assert splits.id.equals(train.id) and splits.id.is_unique
    assert sample.id.equals(test.id) and list(sample)==['id','quality']
    assert set(train.id).isdisjoint(test.id)
    assert splits.partition.value_counts().to_dict()=={'development':12000,'confirmation':3000}
    for seed in manifest['cv_seeds']:
        col=f'fold_seed_{seed}'
        assert (splits.loc[splits.partition=='confirmation',col]==-1).all()
        assert splits.loc[splits.partition=='development',col].value_counts().to_dict()==dict.fromkeys(range(5),2400)
    metric=json.loads((ROOT/'reports/data_audit/metric_checks.json').read_text())
    assert metric['matrix_vs_sklearn_cases']==105 and metric['max_absolute_difference']<1e-12
    assert metric['moment_identity_cases']==105 and metric['moment_identity_max_difference']<1e-12
    # Hash scripts/docs/configs so this non-Git handoff has a recoverable code version.
    file_hashes=[]
    for folder in ['scripts','docs','configs']:
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                file_hashes.append({'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
    result={'checked_at':datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
            'required_markdown_files':len(REQUIRED),'all_required_files_present':True,
            'markdown_words_including_readme':words,'relative_links_checked':links,'broken_relative_links':errors,
            'input_hashes_match':True,'split_coverage_and_hash_match':True,'submission_sample_ids_match':True,
            'supplied_material_hashes_verified':materials_verified,
            'metric_checks_match':True,'checks_scope':'local preparation only; remote URL accessibility and scientific judgments require recorded review',
            'predictive_models_trained':False,'kaggle_submissions_made':False,'file_hashes':file_hashes}
    receipt_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert receipt_path.is_file()
    print(json.dumps({k:v for k,v in result.items() if k!='file_hashes'},indent=2))


if __name__=='__main__':main()
