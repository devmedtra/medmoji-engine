# Journal des défauts

Chaque défaut de cette liste a coûté du temps, et chacun a laissé un témoin
derrière lui. On les garde parce qu'un défaut expliqué ne se reproduit pas.

Nuit du 29 au 30 août 2026.

---

## Le mur central : deux générations ne donnent pas le même corps

La première moitié de la nuit a été passée à extraire des calques de vêtements
depuis des personnages générés, puis à les poser sur un corps de base. Trois
tentatives, trois échecs, chacun corrigé au mauvais endroit.

```
recouvrement de deux silhouettes générées (IoU) :  82,0 %
à 55 % de hauteur : corps 676 px, habillé 908 px   → écart 232 px
à 85 % (jambes)   : corps 269 px, habillé 429 px   → écart 160 px, décalé de 61 px
```

**Aucun calque extrait de l'une ne peut couvrir l'autre.** Ce n'est pas un
réglage : c'est 18 % de silhouette qui ne coïncide pas.

Les symptômes traités un par un — liséré à l'épaule, ligne de raccord, corps de
base habillé — étaient tous la même cause. Le générateur **redessine** le
personnage au lieu de l'habiller.

**La sortie** : l'inpainting ne repeint que l'intérieur du masque, et on recolle
explicitement l'original ailleurs. Écart hors zone : **0 sur 255**.

---

## Le mur juridique

Les trois modèles de *virtual try-on* ouverts résolvent exactement ce problème —
ils habillent une personne en la préservant au pixel près.

| Modèle | Licence | Commercial |
|---|---|---|
| CatVTON | CC BY-NC-SA 4.0 | non |
| IDM-VTON | CC BY-NC-SA 4.0 | non — **les images produites aussi** |
| OOTDiffusion | CC BY-NC-SA 4.0 | non |

CatVTON a été installé et fonctionne (899 M paramètres, moins de 8 Go de VRAM).
**Le blocage n'est pas technique.**

> ⭐ Le fichier LICENSE a été lu *après* l'installation complète — torch CUDA,
> diffusers, une dizaine de gigaoctets. Le badge était pourtant dans la première
> ligne du README. **Lire la licence avant d'installer.**

---

## Les défauts de la Fabrique

### 1. Ratio de travail faux
Génération en 768×1024 (ratio 0,750) remontée en 1536×2752 (0,558) :
**+34,4 % d'étirement vertical**. Les « manches trop courtes » venaient de là,
pas du prompt. Aucun réglage de texte n'aurait corrigé ça.
*Trouvé par un audit externe, vérifié par le calcul.*

### 2. L'alpha final découpait le vêtement
Restituer l'alpha du corps **nu** à la fin découpe toute pièce plus large que
lui : la marge du masque était annulée à la dernière ligne. Aucun manteau ample
ne pouvait exister. L'alpha est désormais l'**union** corps ∪ vêtement.
*Silhouette : +8,0 % au lieu de −0 %.*

### 3. Le masque commençait dans le visage
Seuil de 17 % choisi au jugé. Les repères mesurés :
```
nez 19,9 %    bouche 22,7 %    MENTON 28,1 %
```
Le masque démarrait **au-dessus du nez**. Le col ne pouvait que monter sur le
visage — ce n'était pas le modèle, c'était l'endroit où on lui disait de peindre.

### 4. Le masque était un trapèze
On prenait le point le plus à gauche et le plus à droite de chaque ligne, puis on
remplissait **tout** entre les deux — donc le vide entre les bras et le torse.
Résultat : une cape rigide.

> **SDXL remplit exactement le masque qu'on lui donne.** Un masque trapézoïdal
> donne un vêtement trapézoïdal. Le masque doit épouser le corps.

### 5. La première correction du 4 n'a rien changé
Dilater la silhouette de 22 % de sa largeur (83 px) rebouchait tous les creux —
l'écart bras/torse mesure 5 px à l'aisselle et 115 px plus bas.

```
creux conservés dans le masque « corrigé » :  0 px
rayon utilisé : 83 px — 28 fois trop
```

Le code avait été corrigé sans vérifier que la correction produisait l'effet
voulu. **D'où le témoin qui compte les creux restants** : un masque sans creux
*est* un trapèze, quel que soit le code qui l'a produit.

### 6. L'ordre des passes teintait la peau
```
ombre de contact          →  la peau s'assombrit
détection par différence  →  cette peau devient « du tissu »
teinture                  →  le cou est colorié
```
**Toute passe qui modifie les pixels détruit une segmentation calculée après
elle.** Ordre figé : masque → teinture → ombre.
*17 205 px de peau cessent d'être traités comme du vêtement.*

### 7. La détection par seuil était fausse à plus de 50 %
Même dans le bon ordre, un seuil ne distingue pas l'ombre du menton sur le cou de
l'ombre à l'intérieur du col.

```
masque par seuil       805 583 px
masque sémantique      372 263 px
peau classée « vêtement »  433 789 px
```

Le masque vient désormais d'un segmenteur qui sait ce qu'**est** un vêtement.

### 8. La teinture sur masque approximatif — le test A/B qui innocente

Des plaques sombres sur l'épaule, un effet « sac poubelle » luisant sur la
manche bleue. Test A/B : même fonction de teinture, mêmes couleurs, seul le
masque change.

```
masque par différence   549 500 px
masque sémantique       534 045 px
  peau classée « tissu » par la différence :  16 449 px
  tissu oublié par la différence           :     994 px
```

Ces 16 449 pixels de transition — anti-crénelage, ombres douces de l'épaule —
ne font pas que salir les bords. Ils **décalent le 88ᵉ centile** qui sépare le
tissu des détails clairs : 63 498 px classés « clairs » au lieu de 57 758. Des
ombres du vêtement se retrouvent traitées comme des reflets à préserver, et
restent en gris luisant au milieu du bleu.

> ⭐ **La fonction de teinture est innocente.** Elle calculait juste sur des
> données fausses. Une hypothèse a d'ailleurs été écartée par la mesure :
> l'écrêtage arithmétique. Zéro valeur ne dépassait 255 avant le clip.

**Conséquence architecturale** : le masque sémantique devient obligatoire, et le
pipeline **refuse** au lieu d'avertir. Un avertissement laissait passer l'asset ;
seul un refus protège le catalogue.

### 9. La manche recouvrait la main
La borne de fin était fixée à 62 %. La courbe de largeur du bras :
```
50 → 59 %   142 → 95 px    l'avant-bras s'affine      → POIGNET à 59 %
60 → 63 %    95 → 128 px   ça s'élargit               → LA MAIN
64 → 69 %    79 → 59 px    5 segments au lieu de 3    → LES DOIGTS
```
À 62 %, la manche mordait sur la main.

> ⚠️ **La détection automatique du poignet était pire que la constante.** Elle
> cherchait le minimum global de largeur et tombait à 68,5 % — *entre deux
> doigts*. Elle aurait empiré les choses en ayant l'air plus rigoureuse. La
> constante mesurée une fois, avec la courbe qui la justifie inscrite au-dessus,
> vaut mieux qu'une détection qui se trompe en silence.

---

### 10. Les témoins vérifiaient ce que je soupçonnais

Med, 30 août : « j'ai pas l'impression que tu analyses ce que tu m'envoies,
parce que tu constaterais toi-même les problèmes ». Il avait raison — j'envoyais
des images sans les ouvrir, en me fiant aux chiffres.

Sur un pantalon cargo, deux témoins annonçaient :

```
écart hors zone touchée : 0/255      ← vrai
mains intactes : 3/255               ← vrai
```

Les deux mesures étaient exactes. Ce qu'elles ne voyaient pas, et qui sautait
aux yeux sur l'image :

- les mains **enfermées dans des moufles vertes** — les doigts préservés à
  l'intérieur d'un bloc de tissu, ce qui est pire que de ne rien protéger ;
- le **sous-vêtement effacé** — il était dans la zone de peinture, rien ne le
  surveillait ;
- le **torse redessiné**, abdominaux inventés ;
- le pantalon arrêté à mi-mollet.

> ⭐ **Un témoin ne doit pas vérifier une hypothèse, il doit vérifier une
> INVARIANCE.** « Tout ce qui n'est pas le vêtement doit être identique à
> l'original » ne présume rien de l'endroit du défaut. Ce qu'on n'a pas pensé à
> surveiller est exactement ce qui casse.

Le témoin d'invariance (`temoins_fabrique.py`) rejette désormais les trois
versions du cargo, y compris celles que les anciens déclaraient bonnes :

```
🔴 le vêtement empiète sur les mains : 72,7 % de la zone
🔴 torse    1 203 px modifiés
🔴 jambes     998 px modifiés
```

---

## Les défauts du Moteur

### La corpulence calculée à deux endroits
Le validateur ne lisait que les facteurs déclarés par la pièce ; le plan
appliquait les facteurs par défaut. Une pièce sans déclaration **passait le
contrôle de déformation puis se faisait étirer de 25 %**.

> Une règle appliquée à deux endroits n'est pas une règle, c'est deux occasions
> de diverger. Une seule fonction, utilisée par les deux.

*Trouvé par un témoin, pas par relecture.*

### L'ordre des clés changeait l'empreinte
`JSON.stringify` suit l'ordre d'insertion. Deux codes construisant le même
avatar dans un ordre différent produisaient deux empreintes — donc deux entrées
de cache pour une seule image. Le cache gonfle, le CDN sert deux fichiers
identiques, **et personne ne s'en aperçoit jamais**.

### L'Unicode n'était pas normalisé
« café » précomposé (U+00E9) et « café » décomposé (U+0065 U+0301) sont deux
suites d'octets pour le même mot à l'écran. Même conséquence, même invisibilité.
*Trouvé par un audit externe.*

---

## Les instruments cassés

Une catégorie à part : les mesures qui mentaient.

| Instrument | Symptôme | Cause |
|---|---|---|
| Détection du sujet | « 100 % au bord bas » sur une image complète | seuil à 247 alors que le fond blanc vaut 237–247 |
| Masque alpha | largeur constante du crâne aux pieds | `convert('RGBA')` fabrique un alpha opaque : tout devient « sujet » |
| Rapport du visage | 1,284 au lieu de 0,717 | `x` normalisé sur la largeur, `y` sur la hauteur — deux espaces mélangés |
| Dérive vidéo | valeur identique sur 10 images | le letterbox noir compté comme sujet |
| Débordement du corps | « déborde de 312 px » au cou | la tête comptée comme un débordement — le manteau n'est pas censé la couvrir |

> ⭐ **La même valeur exacte sur deux entrées différentes signifie que
> l'instrument est cassé, pas que le sujet est identique.** C'est le signal le
> plus fiable, et il a servi cinq fois cette nuit.

---

## Ce que ça enseigne

**Mesurer avant de corriger.** Cinq des huit défauts de la Fabrique ont d'abord
été traités au symptôme. Le seul qui ait tenu du premier coup est celui dont la
cause avait été mesurée avant d'écrire une ligne.

**Vérifier que la correction corrige.** Le masque en trapèze a été « corrigé »
sans que rien ne change dans le résultat, faute d'un témoin sur la sortie.

**Une détection qui se trompe est pire qu'une constante assumée.** La mesure
automatique du poignet paraissait plus rigoureuse et donnait un résultat pire.

**L'œil humain trouve ce que les chiffres ratent.** La cape, la bavure au cou, la
main recouverte : trois défauts vus d'un coup d'œil, dont deux que les témoins
déclaraient bons.

---

# 30 août 2026 — la déchirure du pantalon

## Le défaut

Le pantalon cargo sortait coupé en deux : une bande de peau nue en travers des
cuisses, et sous elle un second vêtement que le modèle terminait proprement,
ourlets compris. À `strength 0,88` c'est explicite — un short blanc à braguette
en haut, un cargo olive en bas, **deux vêtements différents**.

## Deux hypothèses, une réfutée par la mesure

Le conseil d'IA proposait :

- **A** — masque d'inpaint non connexe au genou, perdu au sous-échantillonnage
  latent (`vae_scale_factor = 8`).
- **B** — masque connexe, le débruiteur lâche la structure verticale à
  `strength 0,62`.

Balayage de `strength`, masque identique, mesure de la bande de peau nue *à
l'intérieur* de la silhouette :

| `strength` | bande de peau nue | lignes |
|---|---|---|
| 0,62 | 68,1 → 69,8 % de la hauteur | 43 |
| 0,75 | 68,1 → 69,8 % | 42 |
| 0,88 | 67,1 → 72,7 % | **132** |

Monter le `strength` **aggrave d'un facteur 3**. B est morte.

## La cause : ce n'était pas le genou

Le conseil objectait à sa propre hypothèse A : « si `membres` = bras/mains,
R = 40 n'atteint **pas** les genoux ». Exact — et c'est ce qui a fait chercher
au mauvais endroit pendant deux essais.

La bande est à **68-70 % de la hauteur**, soit la hauteur des **doigts**
(repères mesurés : menton 28,1 %, poignet 59 %, mains 61-63 %, doigts 64-69 %).
Les bras pendent le long du corps. Le disque d'exclusion R = 40 autour de chaque
main creuse une **tranchée horizontale** dans le pantalon des deux côtés, et le
modèle termine le vêtement au bord de la tranchée.

Distance de chaque pixel de peau nue (60-80 % de hauteur) au membre le plus proche :

| `strength` | peau nue | médiane | à moins de 40 px (= R) |
|---|---|---|---|
| 0,62 | 27 492 px | 26 px | **75,2 %** |
| 0,75 | 27 521 px | 26 px | **75,0 %** |
| 0,88 | 55 847 px | 59 px | 37,0 % |

## Le correctif : l'exclusion est anatomique, pas métrique

Une **moufle** est du tissu peint **dans le fond** autour de la main. Un
**pantalon** est du tissu peint **sur le corps**. Les deux se séparent sans
aucun paramètre à régler :

```python
# avant — un disque, qui ne sait pas ce qu'il mange
interdit = d < R
# après — la géométrie décide, l'anatomie tranche
interdit = (d < R) & (~corps | membres)
```

Conséquence voulue : le vêtement a le droit de passer **derrière** la main, et
le recollage des membres (`out[membres] = a_or[membres]`) la remet par-dessus.
C'est l'occlusion correcte — qu'un disque isotrope interdisait par construction.

Prédiction avant génération : 74,8 % des moufles de l'essai raté restent
exclues (elles sont dans le fond), 20 668 px de cuisse rendus au pantalon.

| critère | isotrope | anatomique |
|---|---|---|
| bande de peau nue | 50 lignes | **7 lignes** |
| vêtement **sur** les membres | — | 205 px / 128 841 (**0,16 %**) |
| éclats gris près des mains | 4 672 px | 3 865 px (**−17 %**) |
| ourlet | — | 93,1 % (chevilles) |
| corps hors vêtement modifié | — | 17 764 px (2,17 %), max 65/255 |

## 🔴 Trois instruments qui ont déclaré « rien à signaler »

Sur une image **visiblement** déchirée :

| Instrument | Verdict rendu | Pourquoi il ne voyait rien |
|---|---|---|
| Composantes connexes du masque | **1 seule**, sur les trois essais | les deux moitiés se rejoignent par les côtés |
| « lignes couvertes à plus de 25 % » | **31 sur 31** | seuil sous le niveau du défaut |
| `largeur_vêtement / largeur_corps` | **couverture 134 %**, « chevilles 145 % » | un pantalon déborde de la jambe nue |

> ⭐ **134 % de couverture est impossible.** La valeur absurde était là, dans la
> sortie, et elle est passée une fois avant de m'arrêter. Une grandeur bornée
> par construction doit être **assertée**, pas relue.

Le témoin qui marche est borné au corps, et plante s'il dépasse :

```python
prof = [(y, (vet & corps)[y].sum() / corps[y].sum() * 100) for y in ...]
assert max(v for _, v in prof) <= 100.001, 'instrument cassé'
```

Il attrape rétroactivement les trois essais ratés : 50, 68 et 277 lignes sous
50 % de couverture.

---

# 30 août 2026 (suite) — le pouce

## Ce que le conseil a tranché, et où il s'est trompé

Verdict soumis avec les mesures. Deux membres sur quatre ont répondu.

**Accord** : la déchirure horizontale a disparu, les mains sont libres, le
disque isotrope est refusé — il encode une contrainte d'occlusion fausse.

**Désaccord frontal** sur « le pantalon est-il un seul vêtement continu ? » :
l'un le voyait continu, l'autre décrivait une **bande verticale claire** de la
ceinture au bas-ventre — une braguette ouverte.

> ⭐ Aucun témoin ne pouvait les départager : **toute l'instrumentation était
> orientée en lignes** (`[y]`). Un défaut vertical est invisible à un profil
> ligne par ligne. Le témoin symétrique, en colonnes, sur la tranche de la
> taille, tranche : les colonnes creuses sont **latérales**, jamais médianes.
> Pas d'ouverture centrale — ce qui a été pris pour un slip est le rabat de
> braguette fermé.

**Les deux membres ont proposé, indépendamment, le même correctif pour les
éclats gris : ne garder que les composantes de vêtement ancrées au corps.**
Mesuré avant d'être écrit : **30 composantes, 30 ancrées, zéro orpheline.** Le
correctif ne tuait rien. Les éclats ne flottaient pas — ils étaient soudés.

## La vraie cause, trouvée en regardant

Zoom ×6 sur le plus gros éclat : ce n'est pas du tissu détaché, c'est une
**excroissance qui pousse du bout du pouce**, et le pouce lui-même est décoloré.
Superposition du masque `membres` sur la main : il couvre les quatre doigts et
la paume, **pas le pouce**.

Le masque avait été calculé par pure géométrie — segments latéraux quand une
ligne se divise en trois. Le pouce, accolé à la paume du côté interne, n'en
faisait pas partie. Il n'était donc jamais reposé après génération : SDXL le
repeignait, et faisait pousser un lambeau à son extrémité.

Les deux pouces s'identifient sans ambiguïté sur un corps de base qui ne bouge
jamais : composantes de `corps & ~membres`, 500 à 20 000 px, entre 58 et 72 %
de la hauteur, adjacentes au masque. **Exactement deux, symétriques, à 64,9 %
toutes les deux** — 2 497 px à droite, 2 288 px à gauche.

## Bilan des trois passes

| critère | isotrope | anatomique | + pouce |
|---|---|---|---|
| bande de peau nue | 50 lignes | 7 | **0** |
| corps hors vêtement modifié | — | 17 764 px | **8 161 px** |
| mains/bras modifiés | — | 15 158 px, max 65/255 | **5 437 px, max 52/255** |
| éclats gris près des mains | 4 839 px, 10 taches | 3 865 px, 12 | **1 602 px, 6** |
| couverture médiane à la taille | 84 % | 97 % | **100 %** |
| chevilles couvertes | — | — | **100 %** |

## 🔴 Un critère de pourtour ne sépare pas ce qu'un regard sépare

Avant de compléter le masque, j'ai cherché une règle générique : « la fraction
du pourtour d'une composante qui touche le masque des membres ». Mesuré : le
pouce **12,6 %**, le tronc **5,4 %**, les pieds **30 %** et **56 %**. Les
populations ne se séparent pas.

De même, la distance au membre ne sépare pas le tissu parasite du tissu
légitime dans le fond : à R = 100 px, on tue 51 % du parasite en mangeant 19 %
du légitime.

> ⭐ Quand deux populations ne se séparent pas, **il n'y a pas de seuil à
> trouver** — il y a un autre critère à chercher. Ici, le bon critère n'était
> pas une métrique du tout : c'était de savoir que le corps de base est **fixe**,
> donc que son masque de membres est un fichier qu'on complète une fois, pas une
> règle qu'on généralise.
