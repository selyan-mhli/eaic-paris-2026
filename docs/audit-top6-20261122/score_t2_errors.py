"""Diagnostic sur huit consignes à réponse vérifiable déjà posées aux neuf références."""

import csv
import itertools
import json
import re
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BANKS = [('m03', 'm07', 'c02', 'c11'), ('c01', 'm04', 'c10', 'c04')]
PI = ('14159265358979323846264338327950288419716939937510'
      '58209749445923078164062862089986280348253421170679'
      '82148086513282306647093844609550582231725359408128'
      '48111745028410270193852110555964462294895493038196')
PRIMES = [n for n in range(2, 282) if all(n % d for d in range(2, int(n**.5) + 1))]
FIB = [1, 1]
while len(FIB) < 30:
    FIB.append(sum(FIB[-2:]))


def best_prefix(seq, target):
    best = 0
    for start, value in enumerate(seq):
        if value == target[0]:
            size = 0
            for a, b in zip(seq[start:], target):
                if a != b:
                    break
                size += 1
            best = max(best, size)
    return best / len(target)


def grade(pid, raw):
    s = raw.lower()
    if pid == 'm07':
        return float(bool(re.search(r'(?<!\d)153[, ]?892(?!\d)', s)))
    if pid == 'c04':
        return float('deadbeef' in s)
    if pid == 'c10':
        correct = set()
        for left, right, result in re.findall(r'(\d+)\s*[×x*]\s*(\d+)\s*=\s*(\d+)', s):
            a, b, result = map(int, (left, right, result))
            if 17 in (a, b) and a*b == result and 1 <= result//17 <= 20:
                correct.add(result)
        return len(correct)/20
    if pid == 'c02':
        return best_prefix(re.sub(r'[^a-z]', '', s), 'zyxwvutsrqponmlkjihgfedcba')
    if pid == 'm04':
        digits = re.sub(r'\s+', '', s)
        return max((best_prefix(run, PI) for run in re.findall(r'\d+', digits)), default=0.)
    nums = [int(n) for n in re.findall(r'\d+', s) if len(n) < 100]
    if pid == 'm03':
        return best_prefix(nums, PRIMES)
    if pid == 'c01':
        return max(best_prefix(nums, FIB), best_prefix(nums, [0, *FIB[:29]]))
    if pid == 'c11':
        return best_prefix(nums, [2**i for i in range(6, 46)])
    raise KeyError(pid)


def mcc(y, pred):
    if np.all(pred == pred[0]):
        return 0.
    return float(np.corrcoef(y, pred)[0, 1])


def main():
    truth = {r['endpoint']: r['model'] for r in csv.DictReader((ROOT/'c2/kit/practice_answers.csv').open())}
    endpoints = sorted(truth)
    corpus = ROOT/'docs/task2-work-20261121/corpus.jsonl'
    records = {}
    for line in corpus.read_text().splitlines():
        r = json.loads(line)
        if r['campaign'] == 'pilot1' and r['attempt'] == 0 and r['pid'] in BANKS[0] + BANKS[1]:
            k = r['endpoint'], r['pid'], r['repeat']
            assert k not in records
            records[k] = grade(r['pid'], r['response'])
    assert len(records) == 9*8*4 and 487*316 == 153892 and len(PRIMES) == 60
    pair_indices = list(itertools.combinations(range(len(endpoints)), 2))
    same = np.array([truth[endpoints[i]] == truth[endpoints[j]] for i, j in pair_indices])
    banks = []
    for pids in BANKS:
        profile = np.array([[np.mean([records[ep, pid, r] for r in range(4)]) for pid in pids] for ep in endpoints])
        distance = pdist(profile, metric='cityblock') / len(pids)
        tree = linkage(distance, method='complete')
        ranks = rankdata(-distance)
        def auc(mask):
            n = mask.sum()
            return float((ranks[mask].sum()-n*(n+1)/2)/(n*(len(mask)-n)))
        observed = auc(same)
        null = []
        for group in itertools.combinations(range(len(endpoints)), 5):
            null.append(auc(np.array([i in group and j in group for i,j in pair_indices])))
        banks.append(dict(pids=pids, tree=tree, auc=observed, permutation_p=float(np.mean(np.array(null) >= observed))))
    def evaluate(tree, t):
        part = fcluster(tree, t, criterion='distance')
        return mcc(same, np.array([part[i] == part[j] for i,j in pair_indices]))
    output = []
    for source, held in [(0,1),(1,0)]:
        b = banks[source]
        # En cas d'égalité, retenir la plus petite distance de fusion.
        cuts = np.unique(np.r_[-1., b['tree'][:,2]])
        cutoff = max(cuts, key=lambda t: evaluate(b['tree'], t))
        output.append(dict(selection=source+1, evaluation=held+1, threshold=cutoff,
                           mcc_fit=evaluate(b['tree'], cutoff), mcc_other=evaluate(banks[held]['tree'],cutoff)))
    result = dict(responses=len(records), pairs=len(same), same_pairs=int(same.sum()),
                  banks=[{k:v for k,v in b.items() if k!='tree'} for b in banks], transfer=output)
    (HERE/'comparaison-t2.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
