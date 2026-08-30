# Medmoji Avatar Engine

Moteur d'assemblage d'avatars 2D déterministe. L'IA choisit les pièces, le
moteur les assemble : **même configuration, même avatar, toujours.**

Medtra inc. — construit dans la nuit du 29 au 30 août 2026.

---

## L'idée en une page

Un avatar n'est pas une image, c'est une **description** : 400 octets de JSON que
n'importe quel moteur à la bonne version sait reconstituer à l'identique. Ça
change tout — un million d'usagers ne produit pas un million d'images à stocker,
et deux téléphones affichent exactement le même personnage.

Le projet se divise en deux mondes qui ne se touchent jamais.

**La Fabrique** (`fabrique/`, Python) tourne hors ligne, sur une machine à GPU.
Elle transforme des générations IA en pièces validées. Non déterministe par
nature — c'est pour ça qu'elle est isolée. Chaque pièce passe devant un humain
avant d'entrer au catalogue.

**Le Moteur** (`moteur/`, TypeScript) tourne dans l'app et sur le serveur. Il ne
génère rien, il assemble. Aucun appel réseau, aucun aléa, aucune latence.

---

## Ce que les mesures ont imposé

Chaque décision d'architecture vient d'un chiffre, pas d'une intuition.

| Mesure | Conséquence |
|---|---|
| Deux générations du même personnage se superposent à **82 %** (IoU) | On ne peut pas extraire un calque d'une génération et le poser sur une autre. Toute la première approche est morte là-dessus. |
| Poser un vêtement en calque altère le visage de **0/255** ; greffer une tête, de **25 à 81/255** | Le vêtement se pose, la tête ne se greffe pas. |
| Générer à l'achat coûte **375×** un catalogue précalculé à 1 M d'usagers | Rien ne se génère à la volée. Le catalogue est produit une fois et validé. |
| Un calque en 1536×2752 pèse **16,12 Mio** décodé | Le gabarit est une taille de *source*. Un pion sur une carte coûte 800× moins. |
| Le masque par seuil trouvait **433 789 px de peau** classés « vêtement » | La segmentation est sémantique, jamais par seuil. |

---

## La Fabrique

```bash
python3 fabrique/fabrique.py "a charcoal grey cotton hoodie" hoodie haut "74,78,84"
```

Six passes, dans un ordre qui n'est pas négociable :

1. **masque de zone** — épouse le corps, bornes mesurées (menton, poignet)
2. **génération** — SDXL inpainting, au ratio du canevas
3. **recollage** — l'original hors masque, pixel pour pixel
4. **masque du vêtement** — sémantique
5. **teinture** — un asset, une infinité de couleurs
6. **ombre de contact** — en dernier, toujours

### Pourquoi l'inpainting plutôt qu'une génération

Un générateur redessine le personnage à chaque appel. L'inpainting ne repeint
que l'intérieur du masque, et on recolle explicitement l'original ailleurs :
l'écart hors zone est de **0 sur 255**, par construction et non par confiance.

### Licences — ce point a écarté la solution évidente

| Brique | Licence | Commercial |
|---|---|---|
| SDXL inpainting | CreativeML Open RAIL++-M | oui |
| diffusers | Apache 2.0 | oui |
| CatVTON · IDM-VTON · OOTDiffusion | CC BY-NC-SA 4.0 | **non** |

Les trois modèles de *virtual try-on* ouverts habillent mieux et tournent sans
peine sur une RTX 4090. Ils sont inutilisables pour un produit payant — et pour
IDM-VTON, même les **images produites** sont non commerciales.

---

## Le Moteur

```bash
npx tsx moteur/temoins.ts        # 25 témoins
npx tsx moteur/bout-en-bout.ts   # Avatar JSON → image
```

Le cœur est **pur** : il ne connaît ni le réseau, ni le disque, ni aucun
fournisseur. Il prend un Avatar JSON et un catalogue en mémoire, et rend un
**plan de rendu** — la liste ordonnée des couches, leur échelle, leur décalage.
Des adaptateurs exécutent ce plan : `sharp` côté serveur, Skia côté app.

C'est ce découpage qui rend le déterminisme testable. Comparer deux images
dépend de la bibliothèque graphique ; comparer deux plans est exact.

```
moteur/
├─ modeles.ts      types, gabarit, plans de profondeur, corpulence
├─ validateur.ts   11 contrôles — il refuse, il ne corrige jamais
├─ plan.ts         Avatar JSON → plan de rendu
├─ empreinte.ts    canonicalisation + SHA-256
├─ rendu-node.ts   adaptateur sharp
└─ temoins.ts      25 tests
```

### La limite exacte du déterminisme

L'empreinte du plan est stricte. Les pixels ne le sont pas entre deux moteurs
graphiques différents : `sharp` et Skia ont des noyaux d'interpolation
distincts, une gestion de l'alpha prémultiplié différente, et des espaces
colorimétriques qui divergent sur iOS. Le contrat réaliste est **identité du
plan, équivalence perceptuelle des pixels**.

---

## Ce que les témoins protègent

Chaque contrôle existe parce qu'un défaut est passé sans lui.

- **Ordre des clés** — deux codes construisant le même avatar dans un ordre
  différent donnaient deux empreintes, donc deux entrées de cache pour une image.
- **Unicode** — « café » précomposé et décomposé sont deux suites d'octets pour
  le même mot. Invisible à jamais sans normalisation NFC.
- **Corpulence** — le validateur et le plan la calculaient différemment : une
  pièce sans déclaration passait le contrôle puis se faisait étirer de 25 %.
- **Creux du masque** — un masque sans creux *est* un trapèze, et SDXL remplit
  exactement ce qu'on lui donne. Sans ce témoin, une correction du code avait
  été écrite sans que rien ne change dans le résultat.

---

## État

| Composant | État |
|---|---|
| Moteur d'assemblage, 25 témoins | ✅ |
| Empreinte déterministe | ✅ |
| Fabrique — masque, génération, recollage | ✅ |
| Teinture et ombre de contact | ✅ |
| Squelette, poses, déformation | à faire |
| Face Engine sur 478 repères | validé, à écrire |

---

## Documentation

| Document | Contenu |
|---|---|
| [docs/VISION.md](docs/VISION.md) | la vision produit — le personnage, l'économie du salon, les moteurs, l'échelle |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | l'architecture technique en 15 sections, avec l'audit des moteurs existants |
| [docs/JOURNAL.md](docs/JOURNAL.md) | le journal des défauts — chacun avec sa mesure et le témoin qu'il a laissé |

Le journal mérite une mention particulière : il documente huit défauts de la
Fabrique et trois du Moteur, dont cinq d'abord traités au symptôme avant que la
cause ne soit mesurée. Il contient aussi la liste des **instruments cassés** —
les mesures qui mentaient, et comment on s'en aperçoit.

---

## Licence

Propriétaire — Medtra inc. Les briques tierces conservent la leur.
