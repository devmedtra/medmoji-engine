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
