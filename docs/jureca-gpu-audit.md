# Audit de l'usage GPU sur JURECA — équipe SLZ

Date des mesures : 2026-11-21, 16:38-16:50 CET. Lecture seule : aucun job annulé, modifié ou relancé.
Méthode : `squeue`, `sacct`, `scontrol show job/res`, `sinfo`, et `srun --jobid=<id> --overlap` (accepté)
pour `nvidia-smi` et `top` sur les nœuds en cours. Seconde lecture : Codex (gpt-6.1-sol, lecture seule).

## Verdict

**Non optimal.** Chaque job `j1` demande `--gres=gpu:1` mais la partition `dc-gpu` alloue et facture
le nœud entier (4 A100, 256 threads). Un seul GPU sert, et il est peu chargé : un seul cœur CPU travaille.
Le correctif le plus rentable consiste à lancer **4 shards par nœud, un par GPU**. Il libère environ 70 %
des nœuds occupés sans modifier le code Python.

## 1. Mesures

### Allocation (mesuré)

`scontrol show job 15901232` (représentatif des 14 jobs `j1`) :

| | Demandé (ReqTRES) | Alloué (AllocTRES) |
|---|---|---|
| CPU | 32 | 256 (nœud entier) |
| GPU | 1 | 4 |
| Mémoire | 515 000 Mo | nœud entier |
| billing | 16 | 128 |

`OverSubscribe=NO` : le nœud ne peut pas être partagé avec un autre job.

### Jobs au moment de la mesure

| Job(s) | Utilisateur | Script / commande | Demandé | Alloué | Utilisé (mesuré) | Verdict |
|---|---|---|---|---|---|---|
| 14 × `j1` (15901196-201, 15901227-234) | mouhali1 | `work_c1/j1_run.sbatch` → `box_search.py --shard k --nshards 6/8` | 1 GPU, 32 CPU | 4 GPU, 256 CPU, 1 nœud chacun | GPU0 seul : 13,7-18,5 Go/40 Go, activité moyenne 14 % (voir ci-dessous) ; GPU1-3 : 0 %, 0 Mo ; 1 process Python à ~100 % d'un cœur, load ≈ 1 ; RAM 24/503 Go | **Non optimal** : 3 GPU sur 4 inactifs, CPU à moins de 1 % |
| `j1` terminés (15901137, 15901150, 15901151, 15901153) | mouhali1 | `j1_heats.sbatch` → `real_heats.py` | 1 GPU | 1 nœud entier | 1-12 min ; logs : 0,23-1,31 s par image | Non optimal (même schéma) ; 2 jobs d'essai d'une minute ont mobilisé un nœud entier |
| `j1s` (15901315 FAILED en 18 s, 15901339, 15901340) | mouhali1 | `work_seg/run.sbatch` (`--ntasks-per-node=4 --gres=gpu:4`) | 4 GPU, 4 tâches | 4 GPU | essais de 1-1,5 min, non échantillonnés | Bonne forme d'allocation ; voir le risque de partage des données au § 3 |
| `j3v` (15901389), et avant `j3u`/`j3s`/`j3` | zhang95 | `unlearning/*_job.sh` → `unlearn_v2.py`, `soup_search.py`, `audit.py`, `unlearn.py` | 1 GPU, 8 CPU | 4 GPU, 256 CPU, billing 128 (au lieu de 4) | non mesurable (`srun --overlap` refusé sur le job d'un autre utilisateur) | **Non optimal** : même schéma, voir § 1 bis |
| — | pelet1 | `model_equality/` (requêtes HTTP vers le serveur, `PARALLEL = 4`) | — | — | aucun job, aucun log dans `output/` | **Correct** : la tâche 2 n'a pas besoin de GPU, il ne faut pas réserver de nœud pour elle |

Activité GPU0, échantillon de 8 s à 1 Hz par job (`nvidia-smi -l 1`) :

| Moyenne GPU0 | Jobs |
|---|---|
| 4-5 % | 15901196, 197, 201, 228, 229, 230, 231, 232, 233 (9 jobs) |
| 14 % | 15901199 |
| 26-45 % | 15901198, 200, 227, 234 |

Moyenne des 14 jobs : **14 % sur GPU0**, pic à 71 %, soit environ **3,6 % de la capacité GPU des nœuds**
alloués. Il s'agit de l'activité GPU (`utilization.gpu`), pas d'un pourcentage de FLOPS.

### Comptabilité de la journée (sacct, mouhali1, depuis minuit)

- 22 jobs : 14 RUNNING, 6 COMPLETED, 1 FAILED, 1 CANCELLED.
- 2,76 nœud-heures, **11,05 GPU-heures facturées**. GPU réellement sollicités : 1 sur 4 par nœud.
- Attente en file (Submit → Start), n = 22 : médiane 413 s, p90 472 s, maximum 513 s.
- Démarrage : environ 20 s entre le début du batch et le début du step `srun` (15901137, 15901153, 15901339).
- Aucun job bloqué en CONFIGURING côté SLZ au moment de la mesure. Un job d'une autre équipe l'était.

### 1 bis. Tâche 3 (zhang95) et tâche 2 (pelet1), mesurés à 16:52

`sacct` ne montre pas les jobs des autres comptes depuis `mouhali1`, et `srun --jobid=15901389 --overlap`
renvoie `Access/permission denied`. Sources utilisées : `squeue`, `scontrol show job`, les scripts et les
logs de `unlearning/output/`.

- **zhang95, `j3v` 15901389** (`unlearn_v2_job.sh`) : ReqTRES `cpu=8, gres/gpu=1, billing=4` ;
  AllocTRES `cpu=256, gres/gpu=4, billing=128`. Comme pour T1 : un nœud entier pour un GPU.
- Les quatre scripts de zhang95 (`audit_job.sh`, `soup_job.sh`, `unlearn_job.sh`, `unlearn_v2_job.sh`) ont tous
  `--gres=gpu:1 --cpus-per-task=8`. Jobs précédents : 15901166, 15901181, 15901265 (`j3u`,
  16:36:53-16:38:08), 15901355 (`j3s`, 16:43:09-16:45:00). Ce sont des jobs courts, lancés en série.
- `unlearn_v2.py` : fine-tuning de ViT-B/16 en AMP fp16, batch 64, 20 epochs, avec une augmentation
  **image par image dans une boucle Python** (`_augment_then_preprocess`, `_random_erase_uint8`).
  Log mesuré : étape 160 atteinte environ 3 min après le départ, soit environ 1 étape/s.
  Estimation non mesurée : sur une A100, ViT-B/16 en fp16 avec batch 64 (élève plus modèle d'origine)
  devrait tenir plusieurs étapes/s, donc l'augmentation sur CPU limite probablement le débit.
- `soup_search.py` et les checkpoints `ckpt_ent04/08/12/16.pt` montrent un balayage d'hyperparamètres
  mené en série. Chaque variante est indépendante, ce qui en fait le cas idéal pour 4 variantes en parallèle, une par GPU.
- **pelet1** : aucun job en cours ni récent visible. `model_equality/query_reference.py` interroge le
  serveur en HTTP (`PARALLEL = 4`) et n'utilise pas de GPU. C'est correct : les requêtes peuvent partir
  du nœud de login ou du poste local, sans réserver de nœud.

Recommandations pour T3 :
1. Lancer **4 variantes par nœud** (lr, epochs, niveaux d'entropie) avec le même modèle de sbatch :
   `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`, une configuration par `$SLURM_LOCALID`. Gain : ×4 sur le
   balayage par nœud (estimation).
2. Faire l'augmentation **sur GPU et sur tout le batch** : charger `images` sur `cuda` une seule fois
   (`forget_images.npz`, 527 Mo), utiliser des transformations `torchvision.transforms.v2` sur le batch,
   et un random erasing vectorisé par masque. Gain non mesuré ; à vérifier avec `nvidia-smi` sur son propre job.
3. Préférer bf16 à fp16 sur A100 (plus de `GradScaler` à gérer). Gain de vitesse faible, mais plus stable (estimation).

### Réservation `cispahack` (mesuré)

- 30 nœuds, du 2026-11-21 11:00 au 2026-11-22 14:00, comptes `admin,training2666`, drapeaux `REPLACE_DOWN,MAGNETIC`.
- À 16:4x : 21 nœuds allocated, 8 reserved (libres dans la réservation), 1 completing.
- **mouhali1 occupe 14 nœuds sur 30 (≈ 47 %)**. Les autres équipes occupent 8 nœuds (dont 4 en `gpu:4`).
- Aucun job en attente faute de nœud : le seul PENDING (huang25) attend une `Dependency`.
  Nous ne bloquons donc personne pour l'instant, mais nous avions la moitié de la réservation pour un débit de 14 GPU.

### Données et modèles (mesuré)

- `release/` : 556 Mo (train 186 Mo, test 370 Mo, 1 020 images de test).
- `box_search.py` charge les 3 décodeurs une fois par process, avant la boucle (pas de rechargement par image).
- Logs `box_search` : StegaStamp 100-120 s par image recherchée, TrustMark 4-5 s, PixelSeal environ 20 s
  (temps cumulés affichés par le script, valeurs approximatives tirées des écarts).

## 2. Diagnostic

1. **Allocation : nœud entier pour un GPU (mesuré).** C'est le gaspillage principal : 42 A100 inactives
   sur 14 nœuds. L'en-tête de `common/jureca/main.sh` (`--gres=gpu:1`, commentaire « passer à gpu:4 si le code
   utilise les 4 A100 ») a servi de modèle à `j1_run.sbatch` et `j1_heats.sbatch`.
2. **Le GPU utilisé est peu chargé (mesuré) ; cause probable : boucle Python (hypothèse).** Un cœur à 100 %
   et une faible activité GPU orientent vers un goulot côté Python. Codex rappelle que le même signal est
   compatible avec des synchronisations, du code natif ou un repli CPU d'ONNX Runtime. Non tranché.
   Éléments du code qui vont dans ce sens (`box_search.py`) :
   - `accs_for` groupe par taille exacte (h, w) : la grille donne 13 500 boîtes en 120 appels (ratio 16)
     et 6 126 en 58 (TrustMark). Pendant `refine`, les candidats ont des tailles presque toutes distinctes,
     d'où des batches minuscules (calcul local de Codex, sans modèles).
   - Après chaque appel, `.cpu()` provoque une synchronisation ; les crops sont construits par `torch.cat`
     dans une boucle Python.
   - `refine` redécode des boîtes déjà évaluées : il n'y a pas de cache par boîte.
3. **StegaStamp tourne sur GPU (mesuré après coup, job 15901414).** Le log de chaque shard affiche
   `onnxruntime StegaStamp : ['CUDAExecutionProvider', 'CPUExecutionProvider']` : la session ORT a bien
   chargé CUDA, il n'y a pas de repli CPU. La lenteur de StegaStamp (100-120 s par image) vient donc plutôt
   des petits batches et du passage GPU → NumPy CPU → ORT à chaque batch (hypothèse). Ce log ne prouve pas
   que chaque opérateur passe par CUDA (Codex) ; un profil ORT le dirait.
4. **32 CPU par tâche : inutiles en l'état (mesuré : environ 1 cœur occupé).** Sans conséquence aujourd'hui,
   puisque le nœud entier est alloué de toute façon. Pour un packing, 8 cœurs physiques par shard suffisent (estimation).
5. **I/O : non limitant (estimation).** 556 Mo de PNG pour des recherches de plusieurs secondes à plusieurs
   minutes par image. `/dev/shm` n'apporterait rien de mesurable ; le décodage PNG resterait de toute façon.
6. **Pas de sauvegarde intermédiaire.** Le `.npz` n'est écrit qu'en fin de parcours : un job tué à
   `--time` perd tout son travail.

## 3. Recommandations classées par gain

| # | Action | Gain | Effort | Statut |
|---|---|---|---|---|
| 1 | **4 shards par nœud, un par GPU** (modèle ci-dessous) | Jusqu'à ×4 de débit par nœud. Les 14 shards tiendraient sur 4 nœuds au lieu de 14 (≈ 70 % de nœuds libérés pour nous ou les autres équipes). Ne raccourcit pas un shard (estimation) | Faible, aucun changement Python | Recommandé maintenant |
| 2 | Journaliser `sess.get_providers()` de StegaStamp dans le log du job | Pas de gain : CUDA est bien utilisé | Très faible | **Fait** (journalisé dans `box_search.py`, job 15901414) |
| 3 | Cache des scores par boîte dans `refine` (par image et décodeur) | Non chiffré (dépend du taux de réutilisation) | Faible | Recommandé |
| 4 | Décoder ensemble des crops de tailles différentes : `interpolate` par groupe de taille, puis tampon à résolution fixe décodé par lots (32 pour StegaStamp) | Potentiellement important pendant `refine` ; le facteur ×10 n'est pas démontré | Moyen | Vérifier l'égalité des logits et des meilleures boîtes. `roi_align` n'est **pas** équivalent au resize antialiasé, à éviter sans validation du score |
| 5 | ORT I/O binding : passer un tenseur CUDA NHWC contigu via `run_with_iobinding()`, sans passage par NumPy | Inconnu | Moyen | Après le n° 2 |
| 6 | Checkpoint `.npz` toutes les N images | Aucun gain de vitesse ; réduit la perte en cas de `--time` dépassé | Faible | Recommandé pour les longues relances |
| 7 | Essais `--limit` : lancer en `srun` interactif sur un nœud déjà alloué, ou les packer dans le même job | Évite de mobiliser un nœud entier pour une minute | Faible | Recommandé |
| 8 | `work_seg` : les shards passent par `SLURM_PROCID/SLURM_NTASKS` (`seg/feats.py:131`) | Correction fonctionnelle : quatre steps séparés auraient chacun rang 0/1 et dupliqueraient le travail. Le format actuel (un step, 4 tâches) est correct | — | À conserver tel quel |
| 9 | `seg/feats.py:114` : préparer les crops par blocs plutôt que tout garder en liste avant `cat` | Pic mémoire estimé à environ 15,5 Gio pour StegaStamp (calcul de Codex à partir du code) | Faible | Ne prouve pas la cause du FAILED de 15901315 |
| 10 | `/dev/shm`, préchargement RAM | Négligeable (estimation) | — | Pas prioritaire |

Pour les arrays : un élément par shard garde un nœud par shard. Utiliser un élément par **groupe de 4 shards**.
Conserver `--nshards` 6/8 pour retrouver les mêmes images : le découpage se fait par `i % nshards`.

### Modèle de sbatch corrigé (4 shards, un par GPU)

Ce modèle s'appuie sur le constat que les GPU ne sont pas isolés par cgroups (message du login).
Le masque est donc imposé par `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`, et chaque shard vérifie qu'il ne voit qu'un GPU.
Le code de sortie est non nul si au moins un shard échoue (`srun` renvoie le pire code ; `--kill-on-bad-exit=0`
laisse les autres shards finir). **Validé sur JURECA** (job 15901414, voir ci-dessous).

```bash
#!/bin/bash
# Usage (depuis le dossier de la tâche) :
#   sbatch work_c1/j1_pack.sbatch <split> <nshards> <premier_shard>
#   ex. test/8 : sbatch ... test 8 0   puis   sbatch ... test 8 4
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH -J j1
#SBATCH --output=output/%j.out
set -euo pipefail
SPLIT=${1:?split}; NSHARDS=${2:?nshards}; FIRST=${3:-0}
NTASKS=$(( NSHARDS - FIRST < 4 ? NSHARDS - FIRST : 4 ))
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /p/scratch/training2666/mouhali1/SLZ/Hackathon_Watermark_Localisation_Task
source .venv/bin/activate
export SPLIT NSHARDS FIRST
srun --ntasks="$NTASKS" --kill-on-bad-exit=0 --cpu-bind=cores \
     --output="output/%j_%t.out" bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID      # 1 GPU par tâche (pas d isolation cgroups)
  SHARD=$(( FIRST + SLURM_LOCALID ))
  python -c "import torch,sys; n=torch.cuda.device_count(); sys.exit(0 if n==1 else f\"{n} GPU visibles\")" || exit 3
  echo "shard $SHARD/$NSHARDS sur GPU $CUDA_VISIBLE_DEVICES ($(hostname))"
  exec python -u work_c1/box_search.py --split "$SPLIT" --shard "$SHARD" --nshards "$NSHARDS" \
       --out "output/boxes_${SPLIT}_${SHARD}.npz"'
```

Codex propose une variante plus défensive : quatre `srun --exclusive --exact --gres=none &` avec
`wait` par PID, des UUID GPU et un trap. Elle est utile si `$SLURM_LOCALID` ne correspond pas à l'index
physique ou si les steps se sérialisent. Elle n'a pas été nécessaire : la version simple fonctionne.

### Validation sur JURECA (job 15901414, 2026-11-21)

La session de la tâche 1 a lancé `j1_pack.sbatch` avec `train 6 0` et `--limit 2`.

- **Mesuré (`sacct`, logs) :** COMPLETED en 2 min 58 s, 4 GPU alloués sur le nœud jrc0374, un log par shard
  (`output/15901414_0.out` à `_3.out`), chaque shard k annonce « shard k/6 sur GPU k ».
- **Rapporté par la session de la tâche 1, non revérifié ici :** torch voit exactement 1 GPU par tâche,
  `nvidia-smi` montre des UUID distincts, et pour les shards 1 et 2 les boîtes et les scores sont identiques
  à l'ancien schéma 1 job = 1 GPU (39/39 et 54/54 images).
- **Ajouts de la session de la tâche 1 au modèle :** `$4` = préfixe de sortie (défaut `boxes_<split>`),
  `$5` = arguments en plus pour `box_search.py`.

La tâche 1 utilise ce sbatch pour ses prochains lancements ; les jobs déjà en cours restent sur l'ancien schéma.
`c3/unlearn_pack.sh` (tâche 3) reprend le même principe mais n'a pas encore tourné.

## 4. Non vérifié

- L'attribution de chaque opérateur ONNX à CUDA : la session déclare CUDA, aucun profil ORT n'a été fait.
- La part du temps prise par Python, les transferts et l'inférence : aucun profilage, seulement `top` et `nvidia-smi`.
- L'activité GPU réelle des jobs de zhang95 : `srun --overlap` est refusé sur le job d'un autre
  utilisateur, et `sacct` d'autrui n'est pas visible. Pour la mesurer, zhang95 doit lancer elle-même
  `srun --jobid=<id> --overlap nvidia-smi`.
- La cause du débit d'environ 1 étape/s de `unlearn_v2.py` (augmentation CPU) : c'est une hypothèse.
- La cause du FAILED de 15901315 (`j1s`).
- L'isolation GPU effective : les étapes `--overlap` voient les 4 GPU, ce qui est cohérent avec le message
  « Do not constrain GPU devices with Slurm ». Le changelog JURECA évoque une activation des cgroups GPU,
  et la configuration réelle de la réservation prime : le test `device_count()==1` du modèle sert de garde-fou.
- `c3/unlearn_pack.sh` n'a jamais tourné sur JURECA ; l'égalité des sorties pour les shards 0 et 3 du test 15901414.
- Mes sondes `srun --overlap ... nvidia-smi -l 1` apparaissent comme steps FAILED dans `sacct`
  (interrompues volontairement après 8 s). Elles n'ont pas affecté les jobs.
