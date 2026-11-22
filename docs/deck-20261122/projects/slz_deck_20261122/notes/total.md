# 01_couverture

Bonjour, nous sommes l'équipe SLZ. Notre fil rouge tient en une idée : mesurer avant de soumettre. Sur la Task 1, la localisation de watermark, cette discipline nous a fait passer de 0,51 en validation croisée imbriquée à 0,69 au dernier score public. Nous allons montrer ce qui a marché sur les trois épreuves, et aussi, honnêtement, ce qui a bloqué.

---

# 02_notre_regle

Chaque décision a suivi la même boucle : une hypothèse, une mesure locale appariée, puis une décision fixée avant de lire le résultat. La règle était simple : on ne soumet que si le gain apparié atteint au moins un centième et si la borne basse de l'intervalle de confiance à 95 % reste positive, estimée par bootstrap apparié sur 2 000 tirages. Sinon, le verdict est diagnostic ou abandon. Cette règle ne vaut que si la validation locale est fidèle : elle l'était en Task 1, avec 510 images annotées et cinq folds ; moyenne en Task 2, avec neuf endpoints de référence et seulement 100 requêtes par endpoint ; faible en Task 3, sans modèle gold ni données retain.

---

# 03_t1_progression

En Task 1, le score a progressé de 0,17 en six étapes, chacune validée hors ligne : la tête dédiée aux zones éditées, la segmentation pixseg sur StegaStamp, un encodeur convnext pré-entraîné, un ensemble de quatre réseaux, puis l'ensemble E et enfin StegaStamp entraîné à 6 000 pas, pour 0,6788. Ce qui nous a permis d'avancer sans gaspiller de soumissions, c'est que la validation locale a prédit le serveur à un millième près : 0,5583 en local, 0,5572 au public. Le premier point de la courbe est local, tous les autres sont des scores publics. Après la courbe, deux envois de fin de nuit nous ont menés à 0,6886 : ils changent les TrustMark F et la moyenne des graines StegaStamp, sans validation locale complète, et nous le disons comme tel.

---

# 04_t1_pistes_rejetees

La règle a surtout servi à dire non. Entraîner 8 000 pas au lieu de 4 000 faisait perdre quatre millièmes, et une perte BCE plus IoU gagnait trois millièmes sur cinq folds, sous notre seuil. Et le local a aussi eu tort une fois : PixelSeal à 6 000 pas promettait huit centièmes sur les F en validation, le serveur a répondu 0,6773, en dessous du champion. Seul StegaStamp à 6 000 pas, positif sur les cinq folds, a été soumis, avec cinq millièmes de gain au public. Le template matching et la carte de ré-embarquement n'ont montré aucun signal. Notre limite est claire : une quinzaine d'équipes atteignent 0,71 à 0,77 avec une recette forensique générique, alors que la nôtre lisait surtout le signal du watermark.

---

# 05_t2

En Task 2, nous avons identifié les familles en boîte blanche, par la vraisemblance des réponses : Qwen2.5-3B, Qwen2.5-1.5B et Llama-3.2-1B, onze endpoints chacun, avec un accord de 528 sur 528 avec nos paires de familles. Le verrou, c'est de séparer les variantes d'une même famille : fp8, LoRA, température, prompt système. Notre simulation GPU n'a donné qu'un MCC de 0 à 0,52, même en connaissant la vérité. Les sondes à quasi-égalité de logits ont donné un MCC nul sur 990 requêtes aux endpoints de référence, et les canaux annexes, latence, identifiants et champ usage, étaient fermés. Sans méthode validée, nous avons choisi de ne dépenser aucun lot sur les endpoints notés, et nous restons à 0,20.

---

# 06_t3

En Task 3, sans gold ni données retain, le seul levier que le serveur a récompensé est le push : une modification de 45 lignes de la tête du ViT, rien d'autre. Le score monte de 0,0567 à 0,0941 jusqu'à une intensité de 1,75 ; à 1,65 il fait 0,0923 et à 2 il redescend à 0,0792 : l'optimum est bien autour de 1,75. Le reste a été réfuté par le serveur : soupes, distillation et ancrage entre un et quatre millièmes, supprimer neuf classes à 0,0205, et préserver les variations img2img à 0,0317, ce qui montre que le modèle gold baisse toute la classe. Les images à oublier sont générées par diffusion, sur 45 classes, et le meilleur score du tableau est 0,46 : nous en sommes loin. Les variantes de la tête, température ou lignes atténuées, ont toutes fait moins bien, entre 0,075 et 0,086.

---

# 07_ce_quon_retient

Nous retenons trois choses. D'abord, une validation locale fidèle vaut plus que des soumissions à l'aveugle : en Task 1, local et public coïncidaient à un millième près, sauf une fois sur PixelSeal, alors qu'en Task 3, avant 22 heures, six soumissions sans proxy fidèle n'ont donné qu'un à quatre millièmes. Ensuite, figer la règle avant de lire le résultat nous a évité de soumettre du bruit. Enfin, notre erreur : la Task 1 était l'écart le plus facile à combler, avec une recette forensique générique. Nos scores publics : 0,6886 en Task 1, 0,20 en Task 2, 0,0941 en Task 3. Merci.
