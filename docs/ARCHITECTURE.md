# Architecture technique

Medmoji Avatar Engine — version 0.10, 30 août 2026.

---

## 1. Deux mondes qui ne se touchent jamais

La seule décision structurante du projet : séparer ce qui coûte cher et se
valide à l'œil, de ce qui doit être instantané et reproductible.

| | La Fabrique | Le Moteur |
|---|---|---|
| **Où** | hors ligne, machine à GPU | app et serveur |
| **Rôle** | transformer des générations IA en pièces validées | assembler des pièces existantes |
| **Nature** | non déterministe | déterministe |
| **Entrée** | un prompt, le personnage neutre | un Avatar JSON, le catalogue |
| **Sortie** | un PNG détouré au gabarit + son manifeste | une image en 256, 512 ou 1024 |

Cette séparation rend le rendu **testable** — on compare deux sorties au pixel
près — et le catalogue **auditable**, parce que rien n'y entre sans être passé
sous les yeux d'un humain.

---

## 2. Audit des moteurs existants

Licences vérifiées le 30 août 2026, avant d'écrire une ligne de squelette.

| Moteur | Licence runtime | React Native | Verdict |
|---|---|---|---|
| **Spine** | gratuit à intégrer, mais **chaque usager doit détenir sa propre licence Éditeur** | via runtime | écarté — inapplicable à une app grand public |
| **Rive** | MIT | officiel | à tester pour l'animation |
| **DragonBones** | libre | runtime JS | écarté — projet d'origine abandonné |
| **Live2D Cubism** | gratuit sous 20 M¥ de ventes (~180 k$), payant au-delà | SDK natif | écarté — le seuil sera franchi |
| **PixiJS** | MIT | via WebGL | équivalent à Skia, mais Skia est natif en RN |
| **React Native Skia** | MIT, Shopify | natif | **retenu** |

**La conclusion est nuancée.** Ces moteurs résolvent tous le même problème :
*un artiste rigge un personnage à la main dans un éditeur*. Éditeur, timeline,
format d'animation, outils de peinture de poids — c'est 90 % de leur valeur, et
90 % de ce qui ne sert pas ici.

Notre situation est inverse : **un seul rig, partagé par toutes les pièces**, et
des milliers de pièces générées automatiquement. On emprunte les *algorithmes*
— skinning par poids de sommets, IK analytique à deux segments — qui tiennent en
quelques centaines de lignes. Pas les runtimes.

---

## 3. La pile technique

| Couche | Choix | Pourquoi |
|---|---|---|
| Rendu app | React Native Skia | même moteur que Chrome, MIT, GPU, expose les maillages triangulés |
| Rendu serveur | CanvasKit (Skia en WASM) | le *même* Skia — un renderer différent par plateforme tue le déterminisme |
| Caméra du scan | `react-native-vision-camera` + frame processor | l'analyse tourne en natif, hors du fil JavaScript |
| Traits du visage | MediaPipe Face Landmarker | 478 points 3D et 52 blendshapes, 3,6 Mo |
| Format des pièces | PNG raster + maillage | le style est un rendu 3D : le vectoriser le détruit |
| Segmentation | segmenteur sémantique | ses catégories couvrent exactement les besoins |
| Génération | SDXL inpainting | CreativeML Open RAIL++-M, commercial autorisé |
| Stockage | R2 derrière CDN | égrès nul ; la bande passante explose avant le calcul |

---

## 4. Le squelette se rigge tout seul

Dans un studio, chaque vêtement est riggé à la main. À dix tenues c'est
faisable ; à mille, c'est un budget d'atelier.

Notre situation est différente : **toutes les pièces sortent du même personnage
neutre, dans la même pose, au même gabarit**. Le squelette est donc identique
pour toutes, et on le pose **une seule fois**.

> **La règle qui supprime le rigging manuel** — un pixel de calque hérite des
> poids d'os du pixel du corps de base situé à la même coordonnée. Le manteau ne
> connaît pas les os : il emprunte ceux de l'épaule qu'il recouvre.

C'est la transposition en 2D du *bone heat weighting*. Pour les pixels qui
débordent la silhouette — manche bouffante, col de fourrure — le poids se
propage depuis le pixel de base le plus proche.

```
PELVIS                    racine
├─ SPINE → CHEST → NECK → HEAD
│  ├─ L_SHOULDER → L_ELBOW → L_WRIST → L_HAND
│  └─ R_SHOULDER → R_ELBOW → R_WRIST → R_HAND
├─ L_HIP → L_KNEE → L_ANKLE → L_FOOT
└─ R_HIP → R_KNEE → R_ANKLE → R_FOOT
```

Chaque os porte une position de repos en **fraction du gabarit**, jamais en
pixels, pour survivre à un changement de résolution.

---

## 5. Les plans de profondeur

Numérotés, jamais implicites : l'ordre de chargement ne doit avoir aucun effet.

| Plan | Contenu |
|---|---|
| 02 | sac à dos, face arrière |
| 03 | doublure de veste |
| 05 | cheveux, calque arrière |
| 10 | corps, jambes — **nu**, jamais habillé |
| 11 | ombres de contact |
| 20 | chaussures |
| 30 | bas |
| 40 | haut |
| 50 | bras — plan variable |
| 60 | manteau |
| 70 | tête, cou — vient du scan |
| 80 | cheveux, masse |
| 85 | cheveux, mèches libres |
| 90 | lunettes, chapeau, accessoires |

**Une coiffure occupe trois plans**, pas un. Sans le plan arrière, des cheveux
longs passent devant le manteau ; sans les mèches libres, la coiffure a l'air
d'un casque moulé.

**Le plan des bras exige des règles.** Une main est devant une manche courte et
derrière une manche longue — c'est la seule information que la Fabrique ne peut
pas déduire seule.

---

## 6. L'Avatar JSON

```json
{
  "moteur": "medmoji_v1.0.0",
  "corps":  { "morphologie": "moyenne", "taille": 1.0, "teint": "t3" },
  "visage": {
    "origine": "scan", "scanId": "sc_9f31a2",
    "pieces": { "yeux": "eyes_12", "nez": "nose_04", "bouche": "mouth_09" },
    "morphs": { "ecart_yeux": 0.02, "largeur_machoire": -0.05 }
  },
  "cheveux": "hair_32",
  "tenue": { "manteau": "coat_parka_green_001", "bas": "pants_charcoal_002" },
  "accessoires": ["glasses_04"],
  "pose": "debout", "expression": "neutre", "style": "medmoji_v1"
}
```

### Le déterminisme, et sa limite exacte

La clé de cache est `sha256(avatar + version_moteur + taille)` sur un JSON
canonique : clés triées à toute profondeur, nombres arrondis à six décimales,
chaînes normalisées en NFC.

**Mais l'empreinte du plan ne prouve pas l'identité des pixels.** `sharp` et
Skia sont deux pipelines distincts — noyaux d'interpolation différents, gestion
de l'alpha prémultiplié différente, espaces colorimétriques divergents sur iOS.

Le contrat réaliste : **identité stricte du plan**, et **équivalence
perceptuelle** entre serveur et app — au plus 2 niveaux d'écart sur 99,9 % des
pixels.

---

## 7. La mémoire, ce qui cassera en premier

```
un calque en 1536×2752 décodé en RGBA :  16,12 Mio
   8 calques →  129 Mio   (258 Mio avec copie GPU)
  12 calques →  194 Mio   (387 Mio avec copie GPU)

budget réel, Android d'entrée de gamme : 200 à 400 Mo
```

Un seul avatar au gabarit plein peut faire tomber l'app. Et ce n'est pas
l'éditeur qui posera problème, c'est le premier écran affichant plusieurs
personnes.

| Affichage réel | Taille | Coût par calque |
|---|---|---|
| pion sur la carte | 48 px | 0,02 Mio |
| vignette de liste | 96 px | 0,06 Mio |
| avatar de profil | 256 px | 0,45 Mio |
| vestiaire plein écran | 900 px | 5,53 Mio |

**Le gabarit est une taille de source, jamais de runtime.**

### Trois niveaux de détail

- **Complet** — éditeur, export : tous les effets, un seul avatar à l'écran.
- **Aplati** — listes, clavier, widgets : effets précalculés et fondus dans une
  seule image. Zéro shader au défilement.
- **Miniature** — pions, badges : une image unique, aucune composition.

---

## 8. Le Rule Engine

Onze contrôles avant tout rendu. Un seul échec suffit à refuser, avec une erreur
structurée — jamais une image approximative.

| Contrôle | Ce qu'il refuse |
|---|---|
| Pièces existantes | un identifiant absent du catalogue |
| Version du moteur | un avatar demandant un moteur inconnu |
| Ancres présentes | une pièce dont l'ancre n'existe pas |
| Plans valides | deux pièces au même plan |
| Compatibilité | un chapeau et une coiffure qui s'excluent |
| Morphologie | une pièce non déclarée pour cette morphologie |
| Déformation | un étirement au-delà du maximum mesuré de la pièce |
| Style | une pièce `v2` dans un avatar `v1` |
| Occlusion | un cycle dans les relations devant/derrière |
| Couverture | un torse nu quand aucun haut n'est posé |
| Bornes | un morph de visage hors plage |

```json
{ "valide": false,
  "erreur": "COMBINAISON_INVALIDE",
  "details": "hat_cap_003 est incompatible avec hair_afro_11",
  "suggestion": "hat_beanie_007" }
```

Le champ `suggestion` permet à l'orchestrateur de se rattraper sans repasser par
un humain, et sans jamais laisser passer un avatar invalide.

---

## 9. La morphologie

Cinq morphologies sur **un seul squelette** : les os s'écartent, les vêtements
suivent parce qu'ils empruntent les poids du corps.

> 🔴 **Un facteur d'échelle uniforme est faux.** Élargir un hoodie de 25 % étire
> aussi le logo, les boutons et les coutures d'épaule.

La bonne méthode est le **découpage en neuf zones** : coins et épaules ne se
déforment jamais ; seules les bandes centrales s'étirent. Les facteurs se lisent
comme une signature — athlétique élargit les épaules et resserre la taille,
fort fait l'inverse.

**La densité des plis suit la tension du tissu** : tendu sur un corps fort, donc
plis estompés ; flottant sur un corps mince, donc plis creusés.

Chaque pièce déclare son `deformation_max`, mesuré à la fabrication en la
passant de 0,8× à 1,4×. Au-delà, le Rule Engine refuse plutôt que de livrer un
vêtement distordu.

---

## 10. Le rendu vivant

Ces effets vivent **au rendu**, jamais dans les pièces — d'où la possibilité
d'en ajouter un après coup et de le voir s'appliquer à tout le catalogue.

| Effet | Ce qu'il apporte |
|---|---|
| **Ombres de contact** | une ombre étroite sous la jonction — c'est ce qui *cloue* un calque sur un corps |
| **Report de couleur** | un hoodie rouge teinte légèrement le cou : le corps cesse d'être une image séparée de ce qu'il porte |
| **Relief simulé** | une carte de normales par pièce : le reflet glisse quand le téléphone s'incline |
| **Lumière de contour** | détache le personnage du fond — règle la lisibilité sur une carte chargée |
| **Ombres portées sur le visage** | la frange et la casquette projettent leur ombre sur la peau |
| **Auto-ombrage** | le bras devant le torse projette son ombre sur le t-shirt |
| **Peau vivante** | dégradé chaud sur les zones fines : la peau cesse de ressembler à du plastique |
| **Cavités** | narines, bouche, oreilles : un dégradé profond au lieu d'un aplat |
| **Aberration + grain** | élimine aussi l'effet d'escalier des dégradés sur mobile |

Les deux premiers répondent au risque principal de cette architecture — **un
vêtement posé qui a l'air posé**. Ils se dérivent du masque et de la couleur
dominante, donc s'appliquent à tout le catalogue sans travail par pièce.

### Hors ligne

| Niveau | Gain | Charge | Livraison |
|---|---|---|---|
| Procédural | respiration, clignement | < 1 % | mise à jour à distance |
| Capteurs | regard qui suit l'inclinaison | 1–2 % | build natif |
| MediaPipe local | l'avatar reproduit le visage | 5–10 % | build natif |

---

## 11. La teinture

Régénérer un vêtement dans dix couleurs gaspille dix générations. On en génère
**une seule, en gris neutre**, avec toute sa texture, puis on injecte la teinte
au rendu.

```
gain 0,85 · offset 0,06   (pas une gamma globale)
```

L'offset relève le point noir : sans lui, les zones déjà sombres tombent à zéro
et les plis disparaissent. Les **détails clairs** — cordons, fermeture, œillets
— sont exclus du filtre, séparés au 88ᵉ centile de luminance.

Un asset téléchargé, une infinité de coloris. **Le catalogue cesse d'être
multiplié par sa palette.**

Les motifs suivent la même logique : un carré de texture répétable découpé par
la silhouette, puis la carte d'ombres multipliée par-dessus — le motif paraît
imprimé dans la fibre au lieu d'être collé dessus.

---

## 12. Versionnement

- **Une pièce publiée ne change jamais.** Une correction crée la version `1.1` ;
  la `1.0` reste servie aux avatars qui la référencent.
- **La version du moteur est dans l'avatar.** `medmoji_v1.0.0` rend comme au
  premier jour, même quand `v2` existe.
- **Le style est un profil versionné.** Une pièce `v1` ne peut pas entrer dans un
  avatar `v2` — c'est le Rule Engine qui l'empêche, pas la discipline.
- **Migrer est un acte explicite.** On propose, en montrant les deux. Jamais dans
  le dos de l'usager.

---

## 13. Possession et rareté

Une base locale est le bon endroit pour **mettre en cache** le catalogue. Elle
n'est pas le bon endroit pour décider qui possède quoi.

> 🔴 Une base locale est un fichier que son propriétaire peut modifier. Si c'est
> elle qui décide de la possession, **tout le catalogue se débloque en une ligne
> de SQL**.

La garde vit dans la base serveur : lecture pour son propriétaire seulement,
aucune écriture cliente, et la route d'achat comme seule porte. Le local ne fait
que refléter ce que le serveur a accordé.

La **rareté**, elle, est une propriété de la pièce, donc du catalogue public :
elle vit dans le manifeste à côté du prix. Il n'y a rien à protéger là-dedans.

---

## 14. Ce que l'audit externe a corrigé

Un panel d'ingénierie multi-laboratoires a été interrogé à l'aveugle.

| Reproche | Vérifié | Changement |
|---|---|---|
| « un SHA du plan ne prouve pas un SHA du framebuffer » | fondé | identité du plan, équivalence perceptuelle des pixels |
| la mémoire cassera avant tout le reste | refait, exact | niveaux de détail, WebP, plusieurs résolutions |
| `JSON.stringify` n'est pas canonique | vrai défaut | normalisation NFC + témoin |
| le squelette unique fige tout | fondé, assumé | c'est le prix du raster |
| pas de garde-fou visuel automatisé | fondé | images de référence en intégration continue — à écrire |

Sur le pari raster : « acceptable si vous avez des morphologies ; mauvais si la
pose est unique et le corps figé ». Nous en avons cinq, donc il tient.

---

## 15. Les trois écarts assumés

**Le SVG ne convient pas au personnage.** Le style est un rendu 3D — fourrure,
dégradés, matière matelassée. Vectoriser ça produit des aplats : un autre
personnage, pas le même en plus léger. Le vectoriel reste pour les masques, les
chemins de découpe et l'interface.

**Pas de fond vert.** La recommandation classique du batch est d'imposer un fond
vert puis de le retirer. Essayé, et refusé : un liséré vert subsistait sur les
bords et dans les cheveux. La chaîne retenue est fond blanc + outil de détourage
dédié — vert résiduel passé de 137 à 0,1.

**Les vêtements ne se génèrent pas à plat.** Un vêtement sans corps dedans n'a ni
volume ni tombé. Celui qu'on extrait d'un personnage se repose avec un visage
altéré de 0 sur 255. *Sauf* comme entrée d'un modèle de try-on, où le flat lay
est au contraire ce qu'il faut.
