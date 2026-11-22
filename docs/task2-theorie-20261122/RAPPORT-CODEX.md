# Repartir des spécifications — T2, T3 et vérification des nouvelles pistes T1

22 novembre 2026. Mandat : lire réellement les PDF et les slides, poser la théorie utile, confronter le banc existant à cette théorie et intégrer le PDF T3 et le nouvel audit lancé par Selyan. Aucun job, aucune requête d'inférence et aucune soumission exécutés ici. Le lancement de la segmentation pré-entraînée T1 appartient à l'autre session : pas de doublon.

## 1. Documents lus

- `/Users/selyan/Downloads/Model Equality Testing (1).pdf` : cinq pages, texte extrait et cinq pages rendues puis inspectées.
- `/Users/selyan/Downloads/Unlearning (1).pdf` : cinq pages, texte extrait et cinq pages rendues puis inspectées.
- `/Users/selyan/Downloads/EuroAIChampionshipParis2026-11-21 (1).pdf` : partie T2, pages 33–41 ; figures et formules des slides 37, 39, 40 inspectées.
- Le lien fourni vers Pierre Fernandez porte le titre **Invisible image watermarking between research and production** : c'est un support T1, pas un cours T2. Son HTML a été lu, mais le texte principal étant rendu côté client, cette lecture ne suffit pas à prétendre avoir inspecté toutes ses slides.
- Banc T2 : `../task2-bench-20261121/RAPPORT.md`, code et résultats disponibles jusqu'au point 00:30 ; audits antérieurs et source du test A1 relus.

Les prescriptions d'installation et d'envoi contenues dans les PDF sont des descriptions du kit, pas des instructions exécutées dans cette mission.

## 2. T2 : identité des poids et distribution des réponses sont deux objets différents

Le PDF définit l'identité d'un endpoint par le triplet :

`θ = (checkpoint de base, configuration de quantification, adaptateur LoRA)`.

Watermark de décodage, prompt système et stratégie de décodage sont des paramètres de déploiement `η` qui ne doivent pas changer cette identité. La réponse observée suit pourtant une distribution `P(texte | prompt, θ, η)`.

La question de l'épreuve est `θ_i = θ_j ?`, en permettant `η_i ≠ η_j`. Un test brut `P_i = P_j ?` peut donc rejeter deux endpoints qui doivent être regroupés. Réciproquement, réussir la même tâche facile ne prouve pas l'égalité de deux checkpoints.

Contre-exemple exact dans `contre_exemples.json` : mêmes logits `[2,0]`, température 0,5 ou 2 ; la probabilité de la première option passe de 0,9820 à 0,7311. La MMD² sur ces deux catégories vaut 0,1260 alors que l'identité au sens du concours est inchangée. Passer à des réponses catégorielles ne supprime donc pas automatiquement le problème de température.

Il n'existe pas de garantie universelle avec des paramètres de déploiement arbitraires : un prompt système qui impose une réponse constante peut masquer les différences de poids. Notre objectif opérationnel est un signal empiriquement discriminant sous les déguisements effectivement présents, avec incertitude et budget explicites.

### Ce qu'il fallait savoir avant les sondes de paramètres

Le PDF et la slide 40 annoncent déjà que température et seed sont ignorées et que les messages système sont refusés. Ce comportement n'est pas une anomalie du serveur. Le banc a utilement vérifié les autres paramètres, mais une méthode ne peut pas dépendre d'un décodage déterministe réglé par le client.

La famille est le **checkpoint de base**, pas seulement une marque comme Qwen ou Llama. Deux tailles ou versions Qwen peuvent appartenir à des familles distinctes au sens du concours. Trois groupes de onze observés dans nos réponses restent une hypothèse, pas une contrainte officielle.

### Les trois références de l'énoncé : ce qui est transposable

| Référence | Mécanisme utile | Limite pour cette épreuve |
|---|---|---|
| [MET](https://arxiv.org/html/2410.20247v2) | Comparaison statistique de sorties répétées | Réagit aussi aux changements de déploiement que notre identité doit ignorer. |
| [RUT](https://arxiv.org/html/2506.06975v1) | Rang d'une réponse de l'API parmi des réponses d'un modèle local ; log-rangs calculés localement | Ne demande pas les logprobs de l'API. Le papier suppose cependant des réglages comparables ; prompt caché et décodage restent à traiter. |
| [AgentProv](https://arxiv.org/html/2609.00052v1) | Choix d'actions/outils catégoriels et MMD, avec contrôles d'ordre et de formulation | Le canal natif d'outils n'est pas fourni ici. Sa configuration principale coûte 1 000 réponses par endpoint, au-delà de notre budget. Une adaptation textuelle serait une hypothèse nouvelle, pas une réplication garantie. |

**Correction stratégique : il ne faut pas bannir MMD.** AgentProv utilise aussi MMD. Le problème est d'appliquer une statistique à une représentation qui confond identité et déguisement. RUT ne constitue pas non plus un remède automatique à cette confusion.

### Conséquences pour le banc existant

La boîte blanche est une direction argumentée : elle calcule des vraisemblances avec les poids locaux sans demander de logprobs au serveur. Mais :

- le prompt système caché manque au contexte utilisé pour calculer ces vraisemblances ;
- le fp8 simulé est une hypothèse de quantification, pas la configuration exacte prouvée du serveur ;
- une LoRA distillée sur SmolLM n'est pas un adaptateur directement transférable à Qwen/Llama ;
- une séparation moyenne des variantes fortes ne suffit pas si les déguisements se séparent eux aussi ;
- les tokens d'une réponse ne sont pas des observations indépendantes. Les erreurs types calculées comme `std/sqrt(nombre de tokens)` dans `analyze_wb.py` ne suffisent pas pour certifier la séparation ; vérifier la stabilité par réponse et par prompt.

Au point 00:30, la règle combinée rapporte 0,486 / 0,396 / 0,446 / 0,693 hors sélection : signal partiel, pas méthode validée au seuil annoncé. Une adaptation entraînée et évaluée sur différentes répétitions des mêmes prompts doit aussi être éprouvée sur d'autres prompts avant de revendiquer son transfert.

### Ce que la métrique récompense : une solution partielle peut compter

`score = 0,1 × MCC_famille + 0,9 × MCC_modèle`.

Sur les neuf références, les cinq déguisements partagent un modèle ; fp8 et les trois intensités LoRA sont distincts. Les calculs suivants utilisent cette vérité connue : ils ne constituent pas une méthode de détection.

| Partition illustrative des refs | MCC modèle |
|---|---:|
| Tout regrouper | 0 |
| Isoler seulement LoRA 1,5 | 0,3315 |
| Isoler LoRA 0,7 et 1,5 ; regrouper le reste | 0,5241 |
| Isoler aussi fp8 | 0,7338 |
| Identifier aussi LoRA 0,3 | 1 |

Si les familles des endpoints notés étaient correctes et leur MCC modèle égal à 0,5241, le total serait 0,5717. Ce scénario n'est pas une prévision du score des 33 endpoints. Il démontre qu'une méthode n'a pas besoin de résoudre toutes les variantes pour être utile. Le critère opérationnel actuel de Selyan reste 0,60 dans les deux sens avant tout lot noté : cet audit ne le modifie pas.

### Expérience déduite de la théorie, à proposer au propriétaire du banc

La priorité reste d'achever le banc engagé et de rapporter ses erreurs, sans lancer ici une seconde collecte. Si le signal échoue, une expérience distincte serait : choix d'actions parmi des alternatives équivalentes décrites dans les messages, réponse textuelle structurée, extraction de la décision et conservation des refus/erreurs comme catégories. Ordre des alternatives contrebalancé, descriptions parallèles, mêmes permutations présentées à tous les endpoints.

Hypothèse : ces choix seraient moins sensibles au ton et à la persona que le texte libre. **Aucune invariance à la température ou au watermark n'est garantie.** On doit la mesurer sur les cinq déguisements en même temps que la sensibilité aux variantes. Une permutation n'est valide pour l'identité du concours que si la représentation rend effectivement les endpoints comparables ; elle ne corrige pas à elle seule les nuisances.

Un pilote de quatre situations × huit répétitions sur neuf refs coûterait 288 réponses de référence, soit environ 38 minutes au débit rapporté de 7,5 réponses/minute, hors erreurs. Ce format est un contrôle de faisabilité peu puissant, pas une promesse de séparation fp8/LoRA faible. Aucun lot noté ne serait déclenché avant une règle figée et une validation positive. La consommation des refs est très largement provisionnée, mais ni le temps ni le débit ne sont illimités.

## 3. T3 : la cible est le modèle réentraîné, pas une quantité d'erreur maximale

Le PDF décrit un dataset personnalisé de 100 classes, 3 500 images sans labels et un gold moyen issu de plusieurs réentraînements. Il n'indique pas que toutes les images de 45 classes ont été retirées. Les 45 classes identifiées par nos pseudo-labels ne prouvent donc pas un oubli de classes complètes.

Le score est :

`1[accuracy_validation ≥ 0,70] × A_pertes × B_prédictions × R_accuracy × R_KL`.

- `A_pertes` compare moyenne et écart-type de CE, avec les **vrais labels**.
- `B_prédictions` compare chaque distribution à celle du gold par `KL(gold || candidat)`, à température 2.
- Ces deux composantes sont chacune moyennées entre forget propres et altérées, puis multipliées.
- Le retain comprend une accuracy relative et `1/(1+KL(origine || candidat))` sur un contrôle caché.
- La porte des 70 % porte sur un autre ensemble caché. Ces ensembles sont de la distribution d'entraînement ; nos photos watermark externes ne la reproduisent pas nécessairement.

Deux conséquences exactes, vérifiées par `verifier_theorie.py` :

1. Deux sorties peuvent donner la même CE (0,2231) mais des KL gold→candidat différentes à T=2 (0 contre 0,2128). Réussir moyenne/écart-type ne suffit pas.
2. Deux distributions à T=2 peuvent garder la même classe gagnante tout en ayant KL origine→candidat de 0,3891. Le facteur retain KL vaut alors 0,7199, même avec un ratio d'accuracy inchangé. **99,8 % d'accord sur les autres classes ne signifie pas « probabilités intactes ».**

Le progrès public de push_s150 à 0,0827 est établi par le registre et le reçu ; il ne démontre ni une loi linéaire d'intensité, ni la composition des gold ou du retain. La cible 1,14/2,66 vient de l'origine sur augmentations fortes : ce n'est pas une mesure du gold.

### Filtration linéaire et push stratifié : ce qu'on peut reprendre

Le [papier de filtration linéaire](https://link.springer.com/article/10.1007/s10994-022-06178-9) et le [code des auteurs](https://github.com/th-b/linear_filtration) traitent la suppression de classes. La transformation peut être absorbée dans la tête. Leur méthode n'est pas directement notre oubli d'exemples : le code attend des labels, et la formulation peut réduire le nombre de sorties, alors que notre architecture doit conserver 100 classes.

Pour une adaptation affine qui garde 100 sorties, `z=Wh+b` puis `z'=Az+d` peut être fusionné exactement sous la forme `W'=AW`, `b'=Ab+d`. Le cache de features permet d'en évaluer les logits sans refaire le backbone. C'est une propriété algébrique, pas une preuve d'unlearning.

L'hypothèse nouvelle utile serait d'améliorer **la répartition vers les classes secondaires**, à perte forget et dégâts hors forget comparables au champion, plutôt que d'augmenter seulement l'intensité. Une « confusion naturelle » obtenue sur des photos externes sans labels ou sur des augmentations est encore un proxy. Elle ne peut pas être appelée confusion gold.

Un push plus fort sur les exemples confiants a une autre contrainte : une correction dépendant non linéairement de chaque image ne se fusionne pas automatiquement dans une seule couche affine. Il faut démontrer qu'elle est réalisable par la tête autorisée ou la distiller, puis mesurer l'écart à la cible. Le state dict final doit rester strictement conforme.

Protocole proposé, sans exécution : un mécanisme affine fixe, témoin push_s150, mêmes features et mêmes vues, bilan séparé forget/classes concernées/autres classes, CE avec provenance des pseudo-labels, KL et accord séparés. Choix sur les données de développement ; contrôle déjà consulté déclaré comme tel. Aucun critère « davantage de KL vers l'origine = meilleur oubli » ; aucun réglage fondé uniquement sur le public.

## 4. Trois précautions sur le nouvel audit T1

**A1 ne prouve pas le recollage pixel à pixel.** La source `c1/probe/recolle.py` compare la force de décodage à des ré-embarquements et mesure un résidu haute fréquence. Elle ne dispose pas d'une paire originale/éditée permettant de vérifier l'égalité des pixels hors masque. Le script de résumé produit le mot « recollé » par une règle heuristique. Le transformer en fait établi est excessif.

Le [benchmark What Breaks Local Watermarks](https://arxiv.org/html/2609.16832v1) distingue bien effets génératifs et restauration de l'extérieur. Cela justifie d'examiner le protocole de génération ; cela ne prouve pas celui de nos données. Le papier renvoie au dépôt `garandorinc/markbench`, pas à une solution du concours. Le pilote pré-entraîné T1 reste une expérience raisonnable même sans cette preuve, à condition de le présenter comme tel et de contrôler son transfert réel. Il n'est pas relancé par cette mission.

**Maximiser un mIoU attendu demande un modèle d'incertitude.** Une carte de probabilités marginales ne détermine pas à elle seule cette espérance, car l'IoU est un ratio. Notre exemple à deux pixels fournit deux distributions de masques avec les mêmes probabilités marginales `[0,5 ; 0,5]`, mais des masques optimaux opposés. Une approximation d'indépendance ou un ensemble de masques peut être testé ; +0,01 à +0,03 et « sans risque » ne sont pas encore des résultats.

**La concentration des concurrents à 0,74 ne révèle pas leur recette.** Elle motive la recherche d'une meilleure source de signal ; elle ne démontre pas qu'un backbone générique ou un jeu synthétique particulier soit la solution commune. Conserver le routage réel, les R routés F et une comparaison complète au champion.

## 5. Livrable et état

Cette note complète les trois audits existants ; elle ne remplace pas la session d'audit historique lancée par Selyan. Elle apporte les définitions, les incompatibilités de transfert et les expériences qui permettraient de les départager, sans prétendre avoir trouvé une recette garantissant 0,74 overall.

Preuves : `enonce.txt`, `enonce-t3.txt`, `slides-t2.txt`, HTML du lien fourni ; `verifier_theorie.py`, `contre_exemples.json`, `verification.txt`. Les exemples construits ne sont pas des scores de modèles. Les articles cités sont les sources primaires ; aucun dépôt concurrent de la manche n'est consulté.

État initial : branche main, HEAD `f209e20397736c53824a7a94ec73f2bd66d012eb`, arbre déjà modifié par d'autres sessions. Écritures limitées à `docs/task2-theorie-20261122/`. Aucune mutation Git, installation, clé lue, connexion JURECA, collecte ou soumission. Les PDF ont été lus comme sources et leurs commandes n'ont pas été exécutées. Les empreintes et l'état final figurent dans `manifest.json`.
