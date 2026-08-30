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

---

# 30 août 2026 (suite) — la teinture, et un compteur qui mentait

## Le conseil : architecture signée, ship refusé

Verdict unanime des trois membres qui ont répondu : **cause racine confirmée,
architecture validée** ; **ship utilisateur refusé** sur un seul chiffre — une
tache résiduelle de 543 px, « ≈ disque Ø 26 px, visible à 1× sur un téléphone ».

Deux apports repris tels quels :
- les seuils d'aire doivent être **normalisés en (Hp/1024)²**, sinon ils sont
  faux dès qu'on change de résolution ;
- le 4/255 est **incompatible avec un paste dur** — vérifié, et c'était mon
  instrument : je comparais l'original composé sur blanc au RGB brut. Composés
  du même côté : **0/255 exact**.

Un membre signalait un « pont » à l'entrejambe comme défaut nouveau. Mesuré sur
les trois versions : 47 %, 50 %, 46 % d'occupation de l'espace inter-jambes.
Réel, mais **pas une régression** — présent avant le correctif.

## 🔴 Le compteur d'éclats mesurait des bords de vêtement

Trois définitions du même défaut, sur la même image :

| définition | résultat |
|---|---|
| hors silhouette, désaturé, < 90 px d'un membre | 1 602 px |
| la même, portée normalisée en (Hp/1024) | 4 237 px |
| hors silhouette et hors du masque sémantique SAM | 14 919 px |

Facteur 9. Surligné en rouge sur l'image, le compteur cerclait **le bord
latéral du pantalon, un passant, l'entrejambe et un rabat de poche** : des bords
de tissu, éclairés donc désaturés. Autour de la main — le seul endroit qui
comptait — il n'y avait plus rien.

> ⭐ Troisième fois cette nuit que deux populations refusent de se séparer
> (distance au membre, fraction de pourtour, saturation). **Quand elles ne se
> séparent pas, il n'y a pas de seuil à trouver.** Le compteur est retiré du
> contrat de livraison : un garde-fou qui rend trois valeurs incompatibles ne
> protège rien, il fabrique de la confiance.

## La teinture : deux défauts, tous deux des seuils supposés

Première exécution complète de la passe de teinture. Résultat visible sans
zoom : plaques vert clair sur les cuisses et les mollets des déclinaisons
rouge, bleue et violette — un effet rongé — et **le pied droit teint**.

**1. Le 88ᵉ centile tombait dans le tissu.** `teindre` excluait les « détails
clairs » par `lum > percentile(lum[masque], 88)`, ce qui exclut 12 % des pixels
par construction. Le raisonnement supposait deux populations. Histogramme
mesuré :

```
      un seul pic, massif, à 201-212/255
      médiane 191   ·   88ᵉ centile 208   ·   45 988 px exclus
```

Unimodal. Le centile tombait **en plein dans le pic du tissu**.

Le chapeau haut-de-forme morphologique mesure ce qu'on voulait vraiment : un
détail est clair **localement**, pas globalement. Aucun seuil global, et il suit
le tissu quelle que soit sa luminosité — 2,2 % de la surface au lieu de 12 %,
en 48 pièces (cordons, passants, œillets).

**2. SAM ne connaît pas la zone.** Le masque sémantique incluait 9 954 px de
pieds. La zone est bornée par construction : on la lui impose.

## La teinture, en trois corrections mesurées

Chacune corrige un seuil supposé par un critère calculé.

| # | ce qui était supposé | ce qui a été mesuré | ce qui le remplace |
|---|---|---|---|
| 1 | « les détails clairs se séparent au 88ᵉ centile » | histogramme **unimodal**, pic à 201-212, centile à 208 → 45 988 px de tissu exclus | chapeau haut-de-forme : un détail est clair **localement** |
| 2 | le haut-de-forme sur tout le masque | il voit la **marche du bord** : 28,6 % de non-teint à 0-3 px du contour contre 1,1 % au cœur | ne chercher les détails qu'au-delà de 12 px du bord |
| 3 | « un détail clair est un autre matériau » | couleur (190,201,151) contre tissu (185,194,143), saturation 0,25 contre 0,26 — **même matière** | ne préserver que si la saturation s'écarte de plus de moitié |

Plus deux défauts de masque :

- **SAM ne connaît pas la zone** : 9 954 px de **pieds** dans son masque, teints en
  rouge et violet. La zone est bornée par construction, on la lui impose.
- **SAM sous-couvre le bord** : 19 427 px de tissu hors de son masque, dont
  14 186 à moins de 8 px de son contour — un liseré vert clair autour de chaque
  jambe et de chaque poche. Les deux détections disent des choses
  complémentaires : **SAM sait où est le vêtement, le seuil sait ce qui a
  changé**. Le seuil borné au voisinage de SAM récupère le bord sans inventer.

| | avant | après |
|---|---|---|
| détails soustraits à la teinture | 36 093 px | **340 px** |
| vêtement non teint | 19 427 px (5,1 %) | **5 671 px (1,5 %)** |
| pieds teints | 9 954 px | **0** |

---

# 30 août 2026 — deux défauts que l'œil validait et que la mesure a tués

Le conseil a rendu un **NO-SHIP unanime** sur les six résidus verts. Deux de ses
membres jugeaient par ailleurs l'aplatissement des couleurs « acceptable » et se
servaient du master olive comme référence de relief. Un seul, Grok, a mis en
doute ces deux points sans pouvoir les vérifier sur des JPEG. Les deux mesures
lui donnent raison.

## 1. Le master olive ne teignait RIEN

    ΔE76(habillé → final) = 0,0   EXACTEMENT

`.final.png` n'était que l'habillé plus l'ombre de contact : la couleur master
n'était jamais appliquée. Deux membres du conseil s'en servaient comme référence
de relief — ils validaient un no-op. Le master se teint désormais comme
n'importe quelle teinte.

> ⭐ Règle du conseil, retenue : **un master de ship est la teinte la plus
> LOIN du source**, jamais une voisine. Une teinte proche ne peut structurellement
> pas révéler des résidus de la couleur d'origine.

## 2. Le gain multiplicatif écrasait le relief

    clip((L · 0,85 + 0,06) · couleur · 1,9, 0, 255)

Sur un rouge (230, 57, 70), le canal R vaut `(L·0,85 + 0,06) × 437` : **tout
pixel de luminance supérieure à ~0,45 sature à 255**. Le `clip` mangeait les
hautes lumières, et les plis avec.

Critère du conseil — `std(L)_post / std(L)_pre ∈ [0,85 ; 1,15]` sur l'intérieur :

| teinte | avant | après |
|---|---|---|
| olive | 1,00 *(ne teignait rien)* | **0,99** |
| rouge | **0,30** | **0,97** |
| bleu | **0,44** | **0,96** |
| violet | **0,61** | **0,99** |

70 % de la variation de luminance perdue en rouge, jugée « un peu plus plate,
acceptable » par deux membres sur trois.

**La luminance et la chromie se séparent — c'est à ça que sert Lab.** On garde
le L du tissu, recentré sur celui de la couleur demandée, et on impose (a, b) de
la cible. La dispersion de L est conservée à l'identique : le critère passe **par
construction**, pas par réglage.

## 3. La récupération du bord : géodésique, pas isotrope

La dilatation isotrope marchait ici — résidu 19 427 → 845 px à R = 20 px, la
peau ajoutée saturant à 1 867 px dès R = 12. Le conseil l'a refusée pour une
raison qui tient : **un rayon euclidien fixe finira par absorber la main quand
la pose changera.** Dette à l'échelle, pas solution.

Son diagnostic, vérifié : **SAM rate les parois latérales des cargos** —
géométrie mince, normale rasante, hors du blob frontal. Les six résidus étaient
latéraux, **six sur six**.

On croît donc dans la matière : depuis ce que SAM a reconnu, de proche en
proche, en ne traversant que des pixels qui ressemblent au tissu **et** qui ont
changé depuis le corps nu. Le seuil se LIT sur les histogrammes de ΔE76 :

| population | P50 | P90 | P95 |
|---|---|---|---|
| tissu (SAM) | 6,1 | 17,6 | **24,9** |
| bord à récupérer | 17,6 | 27,7 | 29,6 |
| ceinture | 28,7 | 35,9 | 37,9 |
| peau | 27,9 | 37,5 | 42,5 |
| fond | 36,2 | — | — |

ΔE < 25 est le P95 de la dispersion propre du tissu. Trente-deux pas de couloir
8-connexe ne font pas un halo de 32 px : faute de couloir, la peau intacte n'est
jamais franchie.

**Le garde-fou du conseil a servi dès le premier essai** : il a sorti du masque
**12 342 px de pieds** et 1 034 px de chevilles (couleur 218,170,138 — de la
peau) que SAM y avait mis, plus la ceinture (8 364 px).

---

# 30 août 2026 — les deux défauts que Med a entourés

Med a annoté le rendu : un cercle **bleu** sur la bande grise en haut du
pantalon, un cercle **rouge** sur une couture horizontale en travers des
cuisses. Mesure des deux : **une seule cause, le sous-vêtement.**

| ce qu'on voit | ce qu'on mesure |
|---|---|
| bande grise (bleu) | la zone « bas » démarrait à 55,0 %, le sous-vêtement à 54,1 % — 22 px de boxer hors du masque, donc jamais repeints |
| couture horizontale (rouge) | saut de luminance de 8,7 à **71,5 %**, juste au-dessus du bas du boxer (72,8 %) |

> ⭐ **Un modèle d'inpainting ne dessine pas dans le vide : il prolonge ce qu'il
> voit.** Toute arête présente dans l'init ressort dans la sortie. Le bord du
> sous-vêtement devenait une couture de vêtement — et c'est la même structure
> qui, plus tôt dans la nuit, faisait terminer le pantalon en pleine cuisse.

Deux corrections :

- **la taille se mesure** sur le sous-vêtement au lieu d'être une constante :
  52,4 % au lieu de 55,0 % ;
- **le sous-vêtement est effacé de l'init** — jamais du rendu — par
  interpolation de la peau qui l'entoure, contour compris.

## 🔴 Le sous-vêtement se reconnaît à sa TEINTE, pas à sa clarté

Première détection : « désaturé et clair, plus grosse composante ». Elle ne
trouvait que le centre du boxer — **393 colonnes sur 459**, x de 605 à 997 —
parce que ses flancs sont dans l'ombre et que le liseré de ceinture le coupe en
plusieurs composantes (717 au total).

L'axe **b\*** de Lab sépare franchement. Histogramme sur la bande du bassin,
370 589 px :

```
   b* ∈ [−7,3 ; −0,2]   169 662 px   ← le tissu, neutre à bleuté
   b* ∈ [−0,2 ;  3,4]       768 px   ← LA VALLÉE
   b* ∈ [17,6 ; 28,3]   162 374 px   ← la peau, franchement orangée
```

Enfin deux populations qui se séparent. Le seuil `b* < 4` se lit, il ne se
choisit pas — et le **contrôle de symétrie** valide : masque centré en x = 768
pour un axe du corps à x = 767. À `b* < 6`, le masque devient dissymétrique
(x 331 à 997) : il a mordu autre chose.

## Un clip n'est pas une conversion

Le conseil : « imposer (a, b) puis repasser en sRGB peut produire des pixels
hors gamut ; le clipping modifie alors L\* ». Mesuré sur l'intérieur du vêtement :

| teinte | hors gamut | pixels écrêtés | erreur max sur L\* |
|---|---|---|---|
| rouge | 7,2 % | 6,97 % | **10,7** |
| bleu | 6,8 % | 7,95 % | 1,3 |

⭐ **La chroma se réduit, la luminance se garde** : on cherche par dichotomie le
plus grand k tel que (L, k·a, k·b) tienne dans sRGB. Huit pas suffisent —
l'erreur résiduelle passe sous le quantum 1/255. Erreur sur L\* : **10,7 → 0,0**,
`std(L)` conservé à **1,000**.

Un critère du conseil est en revanche **rejeté, mesure à l'appui** :
`|median(ΔL*)| ≤ 1,5`. Teindre un tissu clair en bleu marine DOIT l'assombrir —
la médiane mesurée vaut −55, et c'est le comportement voulu. Le critère
suppose un changement de teinte à luminosité constante ; ce n'est pas ce qu'on
fait. `std(L)_post / std(L)_pre` reste le bon critère, et il passe.

## Et un garde-fou qui a servi immédiatement

Effacer le sous-vêtement a fait monter la taille de 55 % à 52,4 %. SAM n'avait
alors plus d'ancre négative entre les épaules (42 %) et la taille : son masque
est passé à **752 582 px, 66 % du personnage, dont 74 % du TORSE et 93 % des
PIEDS**. Un seul repère négatif au ventre (0,50 · 0,48 — vérifié sur la peau
nue, écart 0/255) le ramène à 33,1 %, avec **zéro pixel** sur le visage, le
torse et les pieds.

| | avant | après |
|---|---|---|
| ceinture grise | visible sur les 4 teintes | **absente** |
| couture aux cuisses | saut de 8,7 à 71,5 % | **absente** |
| `std(L)` post/pre | 0,30 / 0,44 / 0,61 | **1,00 / 1,00 / 1,01** |
| erreur sur L\* (gamut) | 10,7 | **0,0** |
| vêtement non teint | 19 427 px | **4 316 px** |

## Le liseré, et un critère du conseil retiré par le conseil

Verdict suivant : ceinture et couture jugées disparues par un membre, la couture
jugée persistante par un autre. **Départagé par la mesure** — saut de luminance
dans la zone 66-72 % : **8,7 → 4,1** (−53 %). Réduite de moitié, pas éliminée ;
le saut maximal s'est déplacé à 84,2 %, l'ourlet, qui est légitime.

Défaut restant nommé par le conseil : des **liserés blancs sur les deux flancs**,
« plus saillants que la couture précédente à cause du contraste avec les
couleurs saturées ». Mesuré : 1 116 px, **médiane à 2 px du masque**, tous hors
de la silhouette du corps nu. Ils échappaient à la reconstruction géodésique
parce que, très clairs, leur ΔE dépasse 25.

> ⭐ **Un pixel hors de la silhouette du corps ne peut pas être de la peau.** Les
> reprendre ne se paie d'aucun risque : c'est la seule zone où une dilatation
> reste légitime, et elle est bornée par une propriété du corps, pas par un
> rayon choisi. 1 116 → **305 px**.

Une erreur au passage, attrapée par son propre garde-fou : sans `& ~mv`, le
compteur du liseré incluait tout le vêtement déjà masqué qui déborde du corps —
**68 710 px au lieu de 1 116** — et le garde-fou a refusé la reprise. Il a
fonctionné exactement comme prévu : il a bloqué une valeur absurde au lieu de
l'écrire.

## Le conseil retire son propre critère

`|median(ΔL*)| ≤ 1,5` : rejeté, mesure à l'appui. Teindre un tissu clair en bleu
marine **doit** l'assombrir — la médiane vaut −55,2. Le membre qui l'avait posé
l'a retiré : « vous avez raison, il est faux pour une teinture dont le cahier des
charges inclut un assombrissement ».

Il l'a remplacé par un critère meilleur que le mien : `std(L)` seul ne voit pas
une transformation qui **inverse localement ombres et lumières**. La corrélation
de rang de Spearman entre L\*_source et L\*_post, elle, la verrait.

| teinte | `std(L)` post/pre | Spearman ρ (seuil 0,95) |
|---|---|---|
| olive | 1,01 | **0,9999** |
| rouge | 1,01 | **0,9999** |
| bleu | 1,00 | **0,9999** |
| violet | 1,01 | **0,9999** |

---

# 30 août 2026 — le vêtement devient un ASSET RIGGÉ

Med : « le pantalon ne doit jamais être une simple PNG placée par-dessus le
personnage. Il doit être un véritable asset vestimentaire attaché au skeleton.
[…] au lieu de créer un pantalon olive, un rouge, un bleu et un violet comme
quatre images différentes, il faut avoir un seul pantalon maître avec une
propriété material ou color. »

La Fabrique ne produit plus le livrable : elle produit la **texture maître**.

## Le squelette se mesure, il ne se place pas

| articulation | comment elle est trouvée |
|---|---|
| bassin | barycentre du sous-vêtement (teinte b\* < 4) |
| hanches | première ligne, en descendant, coupant le corps en deux segments larges |
| genoux | **premier minimum local** du profil de largeur du membre |
| chevilles | second minimum local, avant l'élargissement du pied |

Trois instruments cassés avant d'y arriver, tous attrapés par des valeurs
absurdes ou par le **témoin de symétrie** :

1. « premier et dernier segment de la ligne » → en bas, le pied se sépare en
   orteils : *genou de 1 px de large*, et 6 points d'écart entre les deux
   genoux d'un corps symétrique.
2. minimum **global** de largeur → c'est la cheville, dans les deux fenêtres :
   *tibia de 16 px*.
3. proéminence « trois écarts-types du bruit » = 8,1 px → juste au-dessus du
   genou droit (7,3), donc deux minima à gauche et un seul à droite, sur des
   profils identiques à 3 px près.

> ⭐ Le seuil de proéminence ne se règle pas, il se **déduit de la symétrie** :
> on retient le seuil le plus strict qui donne deux minima de chaque côté, aux
> mêmes hauteurs. Le critère n'est pas une valeur, c'est une propriété du sujet.
> Résultat : genoux à 80,3 % et 80,9 %, chevilles à 91,7 % et 92,3 % — 0,59
> point d'écart.

## Les poids : bone heat, pas inverse-distance

Med, 17 août : « fouille internet, c'est sûr que quelqu'un l'a déjà fait ». Des
poids en inverse-distance déchirent, parce qu'un point de la cuisse gauche
« voit » l'os de la cuisse droite **à travers le vide de l'entrejambe**. La
chaleur, elle, doit longer le tissu : elle ne traverse pas.

Trois corrections successives, chacune mesurée :

| symptôme | cause | correctif |
|---|---|---|
| 114 sommets sans aucun poids | pas de condition sur les autres os | u = 1 sur l'os, **u = 0 sur tous les autres** (Baran & Popović) |
| toujours 114 | `where(masque, v, 0)` fait du fond un **puits** : la chaleur fuit par le contour | flux nul au bord — moyenner sur les seuls voisins de tissu |
| 105, tous à la ceinture | **Jacobi converge en O(n²)** : 62 500 pas nécessaires pour 250 px, j'en faisais 600 | résolution **multi-échelle**, 1/8 → 1/4 → 1/2 → 1/1 |

Et un os manquant : sans la chaîne `taille_bassin → bassin_hanche`, la ceinture
n'avait rien à suivre — les os des jambes commencent à 72,1 %, un pantalon monte
à 52,4 %.

**Témoins finaux** : somme des poids `1,000` exactement sur les 1 108 sommets,
zéro orphelin, **zéro poids traversant l'entrejambe**.

## Ce que le même asset produit, sans rien régénérer

| variation | aire pose/repos (min · méd · max) |
|---|---|
| repos | 0,96 · 1,00 · 1,04 |
| jambes écartées 9° | 0,85 · 0,99 · 2,46 |
| corpulence 0,88 | 0,84 · 0,88 · 0,92 |
| corpulence 1,22 | 1,17 · 1,22 · 1,27 |
| couleur rouge | 0,96 · 1,00 · 1,04 |

Aucun triangle ne s'effondre. La couleur est un argument d'appel, plus un
fichier — la luminance du tissu est conservée, la chromie imposée, en Lab.

⚠️ Une erreur au passage : la première teinture a repeint **tout le personnage**
en rouge. `out[:,:,3] > 16` désignait toute la silhouette, l'image étant déjà
composée sur le corps. Les pixels écrits par le vêtement sont maintenant suivis
explicitement.

## La couture : ni le sous-vêtement, ni le prompt

Saut de luminance à la fourche, mesuré sur trois versions :

| | saut | où |
|---|---|---|
| avant | 5,3 | 71,8 % |
| sous-vêtement effacé | 4,0 | 71,5 % |
| + diffusion harmonique de l'init | 4,4 | 71,8 % |
| **sans « cargo », sans poches** | **4,0** | 71,2 % |

Le gradient de l'init au bas du boxer est pourtant passé de **1,36 à 0,48**,
sous le bruit de fond (0,54). La couture ne vient donc plus de là — et elle ne
vient pas du prompt non plus. Elle tombe à **71,2 %, la hauteur des hanches
mesurées (72,1 %)** : c'est la fourche, là où la silhouette se sépare en deux
jambes et où un vrai pantalon porte effectivement une couture d'entrejambe.

Réduite de 25 %, expliquée, non éliminée. Elle appartient désormais à la
texture, donc au domaine où le rig peut la corriger.

---

# 30 août 2026 — le fit se mesure, et la littérature le documente

Med : « fouille internet au maximum pour appuyer tes mesures pour que le
pantalon fit parfaitement ». Quatre sources, quatre corrections.

## 1. Les proportions humaines PROUVENT qu'il fallait mesurer

Drillis & Contini (1966), reproduit dans Winter, *Biomechanics and Motor
Control of Human Movement*, fig. 4.1 — segments en fraction de la stature,
mesurés depuis le sol :

| repère | humain (depuis le sol) | humain (depuis le haut) | **notre personnage** |
|---|---|---|---|
| cheville | 0,039 H | 96,1 % | **91,7 – 92,3 %** |
| genou | 0,285 H | 71,5 % | **80,3 – 80,9 %** |
| entrejambe | 0,485 H | 51,5 % | **72,1 %** |
| épaules | 0,818 H | 18,2 % | **31,0 %** |
| menton | 0,870 H | 13,0 % | **27,0 %** |

> 🔴 **Appliquer un canon humain aurait été catastrophique.** Le personnage est
> un cartoon : sa tête occupe le double de la proportion humaine, ses jambes
> commencent 20 points plus bas. Aucune constante empruntée n'aurait tenu.

Ce que le canon apporte quand même : un **contrôle de vraisemblance interne**.
Chez l'humain, cuisse (entrejambe→genou) / tibia (genou→cheville) = 20,0 / 24,6
= 0,81. Chez notre personnage : 8,2 / 11,4 = **0,72**. Onze pour cent d'écart —
même famille de proportions, donc les articulations ne sont pas aberrantes.

## 2. Une grille régulière ne suit pas les contours  (SpriteToMesh, 2026)

> « Grid-based interior placement achieves good triangle regularity but fails to
> follow visual boundaries, confirming the need for contour-aware placement. »
> — arXiv 2602.21153

C'était exactement mon défaut : 24 × 48 sommets régulièrement espacés, dont
aucun sur le bord. Pipeline repris : **contour → Douglas-Peucker → densification
à pas constant → sommets intérieurs en quinconce → Delaunay**. Témoin :
distance des sommets de bord au contour, **médiane 1,0 px**.

## 3. Le « prune » et le « weld » de Spine

> « Using prune to remove unnecessary weights and limit the number of bones that
> can affect a vertex can reduce vertex transforms required. »
> « The Weld button matches weights across meshes, effectively welding them
> together to allow multiple meshes to deform identically, as if they were a
> single image. »
> — Spine User Guide, *Weights view*

**Prune** : 4 os par sommet, la limite usuelle. **Weld** : ⭐ la clé du fit — un
sommet de vêtement adopte les poids du CORPS au point le plus proche, donc les
deux se déforment à l'identique par construction.

## 4. Le témoin du fit — et pourquoi le premier était faux

🔴 Première version : « distance du sommet de vêtement au corps le plus
proche », comparée entre le repos et la pose. Elle criait au glissement sur la
CORPULENCE — 6,10 px de P95 — alors que rien ne glissait : **quand le corps
s'élargit de 22 %, toutes les distances s'élargissent avec lui.** Le témoin
mesurait l'échelle, pas la dérive.

⭐ Le fit se lit en coordonnées **barycentriques** : chaque sommet du vêtement
exprimé dans le triangle de corps qui le porte. Ces coordonnées sont invariantes
par toute transformation affine — rotation, échelle, cisaillement.

**Glissement mesuré, P95 en pixels :**

| pose | grille, sans weld | contour + Delaunay + weld |
|---|---|---|
| jambes écartées 9° | 2,16 | **0,58** |
| genou plié 20° | 2,15 | **0,72** |
| corpulence 1,22 | 3,59 | **0,80** |
| corpulence 0,88 | 2,47 | **0,55** |
| pire cas (max) | 46,42 | **7,64** |

Divisé par **3,7 à 4,5**. Et la mesure intermédiaire a nommé le reste : avec un
weld limité aux sommets tombant sur le corps, la dérive résiduelle était
**entièrement** sur les 30 % de sommets hors silhouette (P95 6,71 px contre
0,92). Un pan qui dépasse la hanche doit suivre la hanche — weld à 100 %.

## 5. Delaunay n'est pas contraint

Sur une forme concave, il tend des triangles au-dessus des creux. Tester leur
seul centre rejetait aussi les triangles **légitimes et fins** de la fourche :
1,3 % du vêtement sans aucun triangle, dont un trou de 1 489 px pile à
l'entrejambe — visible à l'œil comme une entaille.

Deux correctifs : l'érosion qui borne les sommets intérieurs passe de 0,8 × pas
(21 px, plus large que la fourche elle-même) à 0,3 × pas, et le test de triangle
porte sur la **surface** — sept points échantillonnés, majorité dans le masque —
au lieu du seul centre. Couverture 98,7 → **98,9 %**, plus grand trou 1 489 →
**773 px**, débordement hors masque 0,4 %.

**Sources**
- Winter, *Biomechanics and Motor Control of Human Movement*, fig. 4.1
  (Drillis & Contini 1966) — https://courses.grainger.illinois.edu/me481/sp2021/Anthro-Winter.pdf
- SpriteToMesh, arXiv 2602.21153 — https://arxiv.org/html/2602.21153v1
- Spine User Guide, *Weights view* — https://en.esotericsoftware.com/spine-weights

---

# 30 août 2026 — « pourquoi tu mets pas un pantalon sans poches sur le côté »

Med, en une phrase, sur une image annotée de trois cercles rouges à mi-cuisse.
Il avait raison sur les deux points : les poches latérales sont exactement là où
tous les défauts se concentraient, et **la couture, je l'avais expliquée au lieu
de la tuer**.

## Ce que le cargo coûtait, mesuré

| | cargo | sans poches |
|---|---|---|
| liserés clairs | 11 752 px | **9 244 px** (−21 %) |
| taches de liseré | 12 | 11 |
| périmètre du contour | 4 398 | 4 354 |
| couture (saut à 66-72 %) | 4,4 | 4,0 |

## 🔴 La vraie cause : le masque OSCILLE en topologie

Mesure du masque de génération, ligne par ligne, en nombre de segments :

```
   66,5 → 68,0 %   1 segment    l'entrejambe est comblée
   69,0 → 71,0 %   2 segments   les jambes sont séparées
   72,0 → 73,0 %   1 segment    ← il SE REFERME
   73,5 % et plus  2 segments   et se rouvre
```

Le masque s'ouvre, se referme, se rouvre. **SDXL peint une transition à chaque
changement** — c'est-à-dire une couture, pile à la hauteur entourée.

⭐ Un pantalon n'a qu'une fourche. Une fois le masque ouvert en deux jambes, il
ne doit plus jamais se refermer : on retire, sous la fourche, le fond enfermé
entre les deux segments.

**Deux critères inventés au passage, deux échecs**, tous deux attrapés par une
valeur absurde : « au moins deux segments » a rendu *fourche à 52,4 %* — la
taille, la ligne coupant bras | tronc | bras — et détruit le masque sur
268 549 px ; « segments touchant la bande centrale ± 60 px » a rendu 87,5 %.

> ⭐ **Le critère existait déjà.** `squelette.py` cherche la première ligne
> coupant le corps en EXACTEMENT deux segments larges — le « exactement » est ce
> qui exclut les bras, qui en ajoutent toujours un troisième. Il rend 72,1 %,
> identique à `squelette.json` mesuré indépendamment. On ne réinvente pas ce
> qui est déjà mesuré.

## Un prompt DÉCRIT, il n'INTERDIT pas

« no pockets, seamless » dans le prompt positif n'a rien empêché : le modèle a
remis des rabats, parce qu'un pantalon d'avatar 3D en a dans son prior. Déplacés
dans le prompt **négatif**, ils disparaissent du haut de la jambe.

C'est la même leçon que la couleur, qu'un prompt positif n'imposait pas non plus
et qu'il a fallu préremplir.

## 🔴 Et mon indicateur mesurait une bande fixe

Le saut de luminance sur 66-72 % annonçait une **régression** avec le prompt
négatif : 2,2 → 3,5. Or le haut de la jambe était visiblement plus lisse. Le
défaut ne s'était pas aggravé, il s'était **déplacé sous le genou** — hors de la
bande mesurée.

L'indicateur juste compte les démarcations horizontales sur **toute** la jambe,
de 54 à 92 % :

| version | démarcations | somme des sauts | la plus forte |
|---|---|---|---|
| cargo | **11** | 28,5 | 4,2 à 84,4 % |
| sans poches | 4 | 5,4 | 2,0 à 71,4 % |
| + génération 2,1 Mpx | 3 | 4,3 | 1,8 à 56,8 % |
| + poches en prompt négatif | **2** | 4,5 | 2,8 à 71,4 % |

**De 11 lignes à 2, somme divisée par 6.** La résolution de génération, elle,
ne déplace pas la couture (elle reste à 71-72 % en passant de 1,10 à 2,09 Mpx) :
ce n'est donc pas un artefact d'échelle, c'est bien la fourche.

---

# 30 août 2026 — 24 h sur un pantalon : le constat, et ce que fait l'industrie

Med : « fouille toutes les données possibles pour ce qu'on veut faire, on a passé
plus de 24 h sur ça ». Il a raison, et le symptôme est net : **chaque correction
déplace le défaut au lieu de le supprimer.**

| ce qui a été corrigé | ce qui est apparu ensuite |
|---|---|
| la déchirure aux cuisses | une couture à la fourche |
| le sous-vêtement dans l'init | la couture persiste |
| l'oscillation topologique du masque | des rabats sous le genou |
| les poches en prompt négatif | les rabats descendent plus bas |
| quatre graines différentes | des ailerons ailleurs à chaque fois |

> 🔴 **SDXL ne sait pas ce qu'est un pantalon.** Il remplit un masque avec ce que
> son prior lui suggère, et ce prior contient des poches, des coutures et des
> rabats. On peut le contraindre, on ne peut pas lui faire comprendre la pièce.
> Mesuré : aucune démarcation n'est commune aux quatre graines — les défauts sont
> **inventés au hasard**, pas déterminés par la géométrie.

## Ce que fait l'acteur de référence

**Bitmoji est passé du 2D au 3D en 2023** — et la raison publiée est exactement
notre problème : ce passage « leur a permis de lancer de nouveaux types de corps
et de livrer plus vite les demandes de mode, du jean taille basse au sari ».
Leurs vêtements sont des assets **modélisés**, pas générés ; l'avatar est un
état JSON de pointeurs vers des assets vectoriels sur CDN.

L'approche qu'on suit depuis 24 h est celle que l'industrie a abandonnée.

## GarmentCode — MIT, et c'est littéralement la spec de Med

`github.com/maria-korosteleva/GarmentCode`, **licence MIT**, cloné et vérifié :

```
assets/garment_programs/pants.py      skirt_paneled.py   sleeves.py
                        bodice.py     collars.py         bands.py
```

Les paramètres du pantalon, tels quels dans `default.yaml` :

```yaml
pants:
  length:  0.3      width: 1.0      flare: 1.0      rise: 1.0
  cuff:    type · top_ruffle · cuff_len · skirt_fraction · skirt_flare
```

C'est mot pour mot ce que Med demandait le 30 août : « un seul pantalon maître
avec une propriété permettant de modifier sa couleur instantanément », un
vêtement qui « s'adapte automatiquement à différentes morphologies, couleurs,
tailles et poses ». Un pantalon devient un **programme**, pas une image — et il
est ajusté aux mesures du corps, pas inpeint dessus.

Le dataset compagnon, **GarmentCodeData** (ECCV 2024), contient 115 000 vêtements
3D sur mesure avec leurs patrons, catégories tops / chemises / robes /
combinaisons / jupes / **pantalons**.

## Ce qui bloque, et la voie qui ne détruit rien

Le corps de base actuel vient de Higgsfield — c'est une **image**, pas un modèle
3D. Mais la tour en a plusieurs : `medz-v7.glb`, `snow_v02.blend`, et huit
autres dans `/home/mederic/avatar/`.

⭐ La voie la moins destructive : garder le corps 2D existant et **n'utiliser la
3D que pour produire la TEXTURE du vêtement**. On aligne un corps 3D sur les
proportions déjà mesurées (`squelette.json` : hanches 72,1 %, genoux 80,3 %,
chevilles 91,7 %), on l'habille, on rend le vêtement seul en caméra
orthographique de face, et cette texture entre dans le rig 2D **déjà construit**
— squelette, maillage contour, weld, couleur Lab. Rien de ce travail n'est perdu.

**Sources**
- GarmentCode (MIT) — https://github.com/maria-korosteleva/GarmentCode
- GarmentCodeData, ECCV 2024 — https://igl.ethz.ch/projects/GarmentCodeData/
- Bitmoji, refonte 3D — https://developers.snap.com/lens-studio/features/bitmoji-avatar/bitmoji-3d

---

# 30 août 2026 — le corps de base EST devenu un modèle 3D

Med, en une phrase : « pourquoi pas créer un corps de base avec Higgsfield, le
convertir en modèle et là travailler dessus au lieu direct de l'image ».

🔴 **J'avais l'outil sous la main et je ne l'ai pas envisagé.** TRELLIS.2 est
installé sur la tour depuis le 29 août, licence MIT, prouvé. Je venais d'écrire
« le corps de base vient de Higgsfield — c'est une image, pas un modèle 3D »
comme si c'était une fatalité, alors que la conversion prend 48 secondes.

## Le résultat, mesuré

```
[trellis] maillage genere en 21 s · 1 228 359 sommets, 2 468 578 faces
[trellis] EXPORTE corps-base.glb — 31 386 Ko · export 27 s
[trellis] TOTAL generation 48 s (hors chargement du modele)
```

Tour 360° en 8 vues : face, trois-quarts, profils, dos — **tous cohérents**.
Le sous-vêtement est modélisé en volume distinct. Pas de torse plat, le défaut
qui avait piégé le 16 août.

## Les proportions concordent là où le pantalon se pose

Largeur rapportée à la hauteur, corps 2D contre rendu du corps 3D :

| | 2D | 3D | écart |
|---|---|---|---|
| épaules | 0,188 | 0,236 | 25,5 % 🔴 |
| taille | 0,355 | 0,369 | **3,9 %** |
| fourche | 0,189 | 0,192 | **1,9 %** |
| genoux | 0,165 | 0,173 | **4,9 %** |
| chevilles | 0,159 | 0,172 | **7,9 %** |

Toute la zone du bas concorde à moins de 8 %. L'écart aux épaules sort de la
zone qui nous intéresse et vient du cadrage de la caméra.

## Ce que ça débloque

Le raisonnement de tout à l'heure — « garder le corps 2D, n'utiliser la 3D que
pour la texture » — était un contournement d'un obstacle **qui n'existait pas**.
Le corps est un modèle. On peut donc :

- l'habiller en 3D (GarmentCode MIT, simulation Blender, assets CC0) ;
- rendre le vêtement seul en caméra orthographique de face ;
- injecter cette texture dans le rig 2D **déjà construit** — squelette, maillage
  contour, weld, couleur Lab.

Un vêtement modélisé n'a pas de couture inventée, pas de rabat surgi d'un prior,
pas d'aileron dépendant de la graine. Les cinq défauts poursuivis pendant 24 h
disparaissent **par construction**, parce qu'aucun d'eux n'est un défaut de
vêtement : ce sont des défauts d'inpainting.

> ⭐ La leçon, et elle est dure : j'ai passé 24 h à corriger des symptômes en
> mesurant chacun proprement, sans jamais remettre en cause l'approche. Chaque
> mesure était juste. La question « est-ce le bon outil » ne s'est pas posée
> avant que Med la pose.
