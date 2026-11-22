"""Audit CPU reproductible des réponses et CSV déjà collectés ; aucun réseau."""
from __future__ import annotations

import collections
import csv
import hashlib
import io
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "sources/branche-c2"))
import analyze as A
import features as F


def records(name):
    return [json.loads(line) for line in evidence["files"][name]["text"].splitlines() if line.strip()]


def key(r):
    return r["endpoint"], r["id"], r["repeat"]


def integrity(rows):
    keys = collections.Counter(key(r) for r in rows)
    seen = collections.defaultdict(set)
    prompts = collections.defaultdict(set)
    for r in rows:
        seen[key(r)].add(r["response"])
        prompts[r["id"]].add(r["prompt"])
    counts = collections.Counter(e for e, _, _ in keys)
    return dict(lines=len(rows), unique=len(keys), duplicate_lines=len(rows)-len(keys),
                conflicting_duplicate_keys=sum(len(v)>1 for v in seen.values()),
                prompt_ids=len(prompts), prompt_id_collisions=[p for p,v in prompts.items() if len(v)>1],
                unique_per_endpoint=dict(sorted(counts.items())),
                empty_unique=sum(not r["response"].strip() for r in {key(r):r for r in rows}.values()),
                repeats=dict(collections.Counter(str(r[2]) for r in keys)))


def partition(rows):
    fam, model = collections.defaultdict(list), collections.defaultdict(list)
    for ep, f, m in rows:
        fam[f].append(ep)
        model[f+":"+m].append(ep)
    return dict(families=dict(fam), models=dict(model),
                same_family_pairs=sum(len(v)*(len(v)-1)//2 for v in fam.values()),
                same_model_pairs=sum(len(v)*(len(v)-1)//2 for v in model.values()))


def confusion(truth, rows):
    pred={e:(f,m) for e,f,m in rows}
    c=collections.Counter()
    wrong=[]
    for a,b in itertools.combinations(sorted(truth),2):
        t,g=truth[a]==truth[b], pred[a]==pred[b]
        c["TP" if t and g else "FN" if t else "FP" if g else "TN"]+=1
        if t!=g:wrong.append([a,b,"fusion erronée" if g else "séparation erronée"])
    return dict(counts=dict(c), wrong_pairs=wrong, **A.score(truth,rows))


def concise(res):
    return dict(t_family=res["t_family"],t_model=res["t_model"],partition=partition(res["rows"]),
                metrics=confusion(truth,res["rows"]))


evidence=json.loads((BASE/"remote_evidence.json").read_text())
truth=A.read_truth(BASE/"sources/branche-c2/kit/practice_answers.csv")
names=[n for n in evidence["files"] if n.endswith(".jsonl")]
result={"data_integrity":{n:integrity(records(n)) for n in names}}

# L'identifiant repeat est local à une collecte ; on ne fusionne pas les deux journaux ref.
ours=records("c2run/ref_answers.jsonl")
minimal=records("SLZ/model_equality/output/ref_pilot_minimal.jsonl")
sophie=records("SLZ/model_equality/reference_answers.jsonl")
result["cross_corpus"]={}
for an,ar,bn,br in [("ours",ours,"minimal",minimal),("sophie",sophie,"minimal",minimal),("ours",ours,"sophie",sophie)]:
    aa={key(r):r for r in ar};bb={key(r):r for r in br};common=aa.keys()&bb.keys()
    result["cross_corpus"][an+"_vs_"+bn]=dict(common_keys=len(common),
        conflicting_response_keys=sum(aa[k]["response"]!=bb[k]["response"] for k in common),
        conflicting_prompt_keys=sum(aa[k]["prompt"]!=bb[k]["prompt"] for k in common))

result["saved_partitions"]={}
for n,f in evidence["files"].items():
    if not n.endswith(".csv") or "_dist" in n or "prompts" in n or "practice_answers" in n:continue
    rows=[tuple(r[:3]) for r in csv.reader(io.StringIO(f["text"])) if r and r[0]!="endpoint"]
    d=partition(rows);d["sha256"]=f["sha256"]
    if rows[0][0].startswith("ref-"):d["metrics"]=confusion(truth,rows)
    result["saved_partitions"][n]=d

g=A.group(ours,"ref-")
g=A.subsample(g,max_reps=4)
groups={"pilote1_32_prompts":{p:d for p,d in g.items() if not p.startswith("w")},
        "pilote2_12_prompts":{p:d for p,d in g.items() if p.startswith("w")},
        "ensemble_44_prompts":g,
        "minimal_5_prompts":A.group(minimal,"ref-")}
result["offline_replays"]={}
for name,gg in groups.items():
    print("Analyse",name,flush=True)
    d={"prompt_count":len(gg),"responses_per_endpoint":A.answers_per_endpoint(gg),"kernels":{}}
    for kernel in ("hamming","ngram"):
        for stat in ("corr","mmd"):
            res=A.analyze(gg,kernel,stat,truth=truth,calibrate_on_truth=True)
            fixed=A.analyze(gg,kernel,stat,t_family=10,truth=truth,calibrate_on_truth=True)
            auto=A.analyze(gg,kernel,stat)
            d["kernels"][kernel+"_"+stat]={"auto":concise(auto),"calibrated_same_data":concise(res),
                  "calibrated_known_one_family":concise(fixed)}
    ev=F.evaluate(gg,truth,top=15)
    d["feature_validation"]=ev
    fd=F.feature_distances(F.prepare(gg),sorted(truth))
    ranked=F.rank_features(fd,sorted(truth),truth)
    d["feature_auc"]={"n_features":len(ranked),"n_auc_ge_09":sum(x[1]>=.9 for x in ranked),
         "n_auc_ge_08":sum(x[1]>=.8 for x in ranked),"best_auc":max(x[1] for x in ranked),
         "top10":sorted(ranked,key=lambda x:-x[1])[:10]}
    result["offline_replays"][name]=d
    print(name, "validation",[(x["mcc_sel"],x["mcc_eval"]) if x else None for x in ev["folds"]],flush=True)

# Contrôle de la règle de longueur déjà proposée, sans optimiser le seuil ni les prompts.
pids=[r["id"] for r in csv.DictReader(io.StringIO(evidence["files"]["c2run/ep/prompts_ep_1b.csv"]["text"]))]
result["fixed_length_rule"]={}
for name,gg in [("ref_15_prompts",{p:g[p] for p in pids}),
                ("ep_5_prompts",A.group(records("c2run/ep/ep_answers.jsonl"),"ep-"))]:
    gg=A.strip_constant_affixes(gg)
    eps=sorted({e for d in gg.values() for e in d})
    llama={"ep-05","ep-09","ep-10","ep-11","ep-13","ep-15","ep-16","ep-19","ep-27","ep-29","ep-31"}
    families={e:("P" if e.startswith("ref") else "L" if e in llama else "Q") for e in eps}
    values={}
    for rep in [None,0,1,2,3]:
        accum=collections.defaultdict(list)
        for pid,by in gg.items():
            for fam in set(families.values()):
                members=[e for e in eps if families[e]==fam]
                lens={e:float(np.mean([len(t) for t in by.get(e,[]) if t.strip()])) if rep is None else
                      (len(by[e][rep]) if e in by and len(by[e])>rep else np.nan) for e in members}
                med=np.nanmedian(list(lens.values()))
                for e,v in lens.items():
                    if np.isfinite(v) and med>0 and v>0:accum[e].append(np.log(v/med))
        if accum:values["all" if rep is None else str(rep)]={e:float(np.exp(np.mean(v))) for e,v in accum.items()}
    # Les familles et les prompts restent les mêmes pour toutes les répétitions.
    d={"ratios":values,"warning":"seuil déjà ajusté aux ref ; pas une validation sur de nouveaux modèles"}
    if name.startswith("ref"):
        for split in [(0,1),(2,3)]:
            ratios={e:[] for e in eps}
            for pid,by in gg.items():
                lens={e:np.mean([len(by[e][i]) for i in split]) for e in eps}
                med=np.median(list(lens.values()))
                for e,v in lens.items():ratios[e].append(np.log(max(v,1)/max(med,1)))
            isol=[e for e,v in ratios.items() if np.exp(np.mean(v))>1.25]
            rows=[(e,"P",e if e in isol else "S") for e in eps]
            d[str(split)]={"isolated":isol,"metrics":confusion(truth,rows)}
    result["fixed_length_rule"][name]=d

# Le support de paires est connu même lorsque les vérités ep restent cachées.
epfiles={n:d for n,d in result["saved_partitions"].items() if "/ep_first_" in n or n.endswith("c2_lump_v1.csv")}
result["ep_pair_changes"]={}
for a,b in itertools.combinations(epfiles,2):
    def labels(n):
        rr=csv.DictReader(io.StringIO(evidence["files"][n]["text"]))
        return {r["endpoint"]:(r["family"],r["model"]) for r in rr}
    aa,bb=labels(a),labels(b)
    fam=model=0
    for e,f in itertools.combinations(sorted(aa),2):
        fam+=(aa[e][0]==aa[f][0])!=(bb[e][0]==bb[f][0])
        model+=(aa[e]==aa[f])!=(bb[e]==bb[f])
    result["ep_pair_changes"][Path(a).name+"_vs_"+Path(b).name]={"family_pair_changes":fam,"model_pair_changes":model}

html=evidence["leaderboard_html"]
result["leaderboard"]={"collected_at":evidence["at"],"source":"http://34.147.229.55/leaderboard_page","slz":[]}
for tr in re.findall(r'<tr[^>]*data-team="SLZ".*?</tr>',html,re.S):
    task=re.search(r'data-task="([^"]+)"',tr)
    at=re.search(r'data-at="([^"]+)"',tr)
    if task and at:
        score=re.search(r'class="s" title="([^"]+)"',tr)
        rank=re.search(r'class="rank">(\d+)',tr)
        result["leaderboard"]["slz"].append({"task":task[1],"score":float(score[1]),"at":at[1],"rank":int(rank[1])})

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.ndarray):return clean(x.tolist())
    return x

(BASE/"audit_results.json").write_text(json.dumps(clean(result),ensure_ascii=False,indent=2))
print("Résultats :",BASE/"audit_results.json",flush=True)
