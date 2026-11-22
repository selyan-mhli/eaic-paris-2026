"""Filtre corrigé, hors ligne, sans remplacer le code d'une autre session.

Sépare l'égalité réelle des simples décalages robustes au contexte. Le contrôle
tous schémas est une option conservatrice nouvelle, pas une règle historique.
"""
import argparse
import json
from pathlib import Path


def valid(r):
    return (r['mass'] >= .8 and min(r['mass_ctx']) >= .8
            and abs(r['d0']) <= .5
            and max(abs(v-r['d0']) for v in r['d_ctx']) < .3
            and max(abs(v-r['d0']) for v in r['d_fp8'].values()) >= .25)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--search',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    data=json.loads(a.search.read_text())
    rows=data['all']
    selected=[r for r in rows if valid(r)]
    selected.sort(key=lambda r:abs(r['d0']))
    coherent=[]
    for r in selected:
        shifts=[v-r['d0'] for v in r['d_fp8'].values()]
        if min(abs(v) for v in shifts)>=.25 and (all(v>0 for v in shifts) or all(v<0 for v in shifts)):
            coherent.append(r)
    for r in selected:r.setdefault('arg_fp8',None)
    data['ties_robust']=selected
    # Les anciennes bascules ne sont pas automatiquement sûres au contexte.
    data['flips']=[]
    data['filter_audit']=dict(actual_gap_bound=.5,selected=len(selected),
        coherent_all_schemes=len(coherent),status='RECHERCHE SEULEMENT ; pas de validation refs')
    a.out.write_text(json.dumps(data,ensure_ascii=False,indent=1)+'\n')
    print(json.dumps(data['filter_audit'],ensure_ascii=False))


if __name__=='__main__':main()
