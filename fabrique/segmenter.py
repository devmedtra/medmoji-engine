#!/usr/bin/env python3
"""SEGMENTER UNE PIÈCE — SAM piloté par des repères mesurés, jamais par un seuil.

Med, 30 août 2026 : « les coordonnées des chevilles forcent SAM à sélectionner
les chaussures ou les pieds ».

═══ POURQUOI CE FICHIER REMPLACE LA DÉTECTION PAR DIFFÉRENCE ═══

Le test A/B de la nuit, même teinture, même couleur, seul le masque change :

    masque par différence   549 500 px
    masque sémantique       534 045 px
      peau classée « tissu » par la différence :  16 449 px

Ces 16 449 pixels de transition — anti-crénelage, ombres douces de l'épaule —
ne salissent pas seulement les bords : ils DÉCALENT le 88ᵉ centile qui sépare le
tissu des détails clairs, si bien que des ombres du vêtement finissent traitées
comme des reflets à préserver. C'est l'effet « sac poubelle » sur la manche.

Un seuil ne saura jamais ce qu'EST un vêtement. SAM le sait.

═══ ET POURQUOI PAS LES PASSES DE RENDU 3D ═══

La méthode infaillible serait d'exporter un Cryptomatte et un Z-buffer depuis le
moteur 3D. 🔴 Il n'y en a pas : nos medmojis sont des IMAGES GÉNÉRÉES dans un
style 3D, pas des rendus d'une scène. L'apparence trompe — le conseil d'IA est
tombé dans le même piège en recommandant « habillez-le en 3D ».

═══ LICENCE, vérifiée le 30 août 2026 ═══

    SAM (v1, v2, v3)   Apache 2.0   code ET poids, commercial libre

Le jeu d'entraînement SA-1B est réservé à la recherche, ce qui n'affecte pas
l'usage des modèles entraînés. Vérifié AVANT installation — leçon de CatVTON,
installé en entier avant qu'on lise son CC BY-NC-SA.

⭐ Gain secondaire, et il compte : la segmentation quitte un service tiers pour
tourner en local. Aucune image ne sort de la machine.
"""
import sys

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

MODELE = '/home/mederic/medmoji-fabrique/modeles/sam_vit_b.pth'

# ── LES REPÈRES, MESURÉS SUR LE PERSONNAGE NEUTRE ──────────────────────────
# En fraction de la hauteur DU PERSONNAGE. Chaque valeur vient d'une mesure
# faite dans la nuit du 29 au 30 août, pas d'une estimation :
#     menton 28,1 % (MediaPipe)   poignet 59 % (minimum de largeur du bras)
#     la main s'élargit à 61-63 %, les doigts se séparent à 64 %
# ⭐ LA RÈGLE DES ÉTIQUETTES, donnée par Med le 30 août : « SAM interprète les
# points 0 comme des murs. Il cherche la frontière naturelle la plus proche qui
# sépare les points 1 des points 0. »
#
# D'où le principe de symétrie : ce qui est POSITIF pour une zone doit être
# NÉGATIF pour les zones voisines. Les genoux excluent le haut ; les épaules
# excluent le bas. Sans ça, SAM déborde d'un vêtement sur l'autre.
REPERES = {
    'haut':   {'positifs': [(0.50, 0.38), (0.35, 0.42), (0.65, 0.42),
                            (0.50, 0.50)],
               'negatifs': [(0.50, 0.20),                     # le visage
                            (0.28, 0.62), (0.72, 0.62),       # les mains
                            (0.42, 0.80), (0.58, 0.80)]},     # les GENOUX
    'bas':    {'positifs': [(0.50, 0.66), (0.42, 0.78), (0.58, 0.78)],
               'negatifs': [(0.35, 0.42), (0.65, 0.42),       # les ÉPAULES
                            (0.50, 0.97),                     # les pieds
                            (0.50, 0.72)]},                   # ⚠️ ENTREJAMBE :
                                                   # sans ce point, SAM relie
                                                   # les deux jambes et rend
                                                   # une jupe-culotte.
    # 🔴 LES ABSCISSES SE MESURENT AUSSI. Premier essai : points positifs à
    # x=0,43 et 0,57 — soit ENTRE les deux pieds, dans le vide. SAM a
    # sélectionné le fond : 258 % du personnage, dont 2 987 821 px hors du
    # corps. Le témoin l'a rejeté. Mesure sur le personnage neutre :
    #     à 96 % : deux segments de 199 px, centrés en 0,33 et 0,67
    #     à 98 % : deux segments de 227 px, centrés en 0,29 et 0,71
    # ⭐ LES BRAS — calculés UNE SEULE FOIS dans la vie du projet.
    # Med, 30 août 2026 : « puisque ton medmoji a toujours la même pose, tu
    # n'as besoin de faire tourner SAM qu'une seule fois ». Le masque est
    # ensuite un simple fichier, et la production ne fait plus qu'une
    # opération NumPy : image_finale[masque] = corps_nu[masque].
    'bras':   {'positifs': [(0.13, 0.50), (0.87, 0.50),      # avant-bras
                            (0.12, 0.60), (0.88, 0.60),      # poignets
                            (0.13, 0.65), (0.87, 0.65)],     # mains
               'negatifs': [(0.50, 0.45), (0.50, 0.55),      # le torse
                            (0.50, 0.75)]},                  # les hanches
    'pieds':  {'positifs': [(0.33, 0.96), (0.67, 0.96),
                            (0.34, 0.99), (0.66, 0.99)],
               'negatifs': [(0.50, 0.96),                     # ⚠️ l'espace
                                                   # ENTRE les pieds : sinon
                                                   # les deux fusionnent en un
                                                   # sabot.
                            (0.35, 0.82), (0.65, 0.82)]},     # les mollets
}


def bornes_du_personnage(im):
    a = np.asarray(im.convert('RGBA'))
    corps = a[:, :, 3] > 16
    ys = np.where(corps.any(1))[0]
    xs = np.where(corps.any(0))[0]
    return xs.min(), ys.min(), xs.max(), ys.max()


def points(im, zone):
    """Convertit les repères en pixels, relatifs à la BOÎTE DU PERSONNAGE.

    ⚠️ Jamais relatifs à l'image : un personnage recadré différemment
    déplacerait tous les points. La boîte se mesure à chaque appel.
    """
    x0, y0, x1, y1 = bornes_du_personnage(im)
    L, H = x1 - x0, y1 - y0
    r = REPERES[zone]
    pos = [(x0 + fx * L, y0 + fy * H) for fx, fy in r['positifs']]
    neg = [(x0 + fx * L, y0 + fy * H) for fx, fy in r['negatifs']]
    coords = np.array(pos + neg, dtype=float)
    labels = np.array([1] * len(pos) + [0] * len(neg))
    return coords, labels


def segmenter(chemin, zone='haut', dest=None):
    from segment_anything import sam_model_registry, SamPredictor

    im = Image.open(chemin)
    rgb = Image.new('RGB', im.size, (255, 255, 255))
    rgb.paste(im.convert('RGBA'), (0, 0), im.convert('RGBA'))

    sam = sam_model_registry['vit_b'](checkpoint=MODELE).to('cuda')
    pred = SamPredictor(sam)
    pred.set_image(np.asarray(rgb))

    coords, labels = points(im, zone)
    print(f'  {int(labels.sum())} points positifs, '
          f'{int((labels == 0).sum())} négatifs')

    masques, scores, _ = pred.predict(point_coords=coords, point_labels=labels,
                                      multimask_output=True)
    # SAM rend trois hypothèses ; on prend la mieux notée, puis on VÉRIFIE.
    i = int(np.argmax(scores))
    m = masques[i]
    print(f'  3 hypothèses, scores {np.round(scores, 3)} → retenue nº{i}')

    # ── TÉMOINS. Un masque « réussi » peut être absurde. ──
    a = np.asarray(im.convert('RGBA'))
    corps = a[:, :, 3] > 16
    part = m.sum() / max(1, corps.sum()) * 100
    hors = (m & ~corps).sum()
    print(f'  masque : {m.sum():,} px, soit {part:.1f} % du personnage')
    print(f'  déborde hors du corps : {hors:,} px')

    if zone == 'bras' and not (3 <= part <= 30):
        sys.exit(f'🔴 ABSURDE pour des bras : {part:.1f} % du personnage')
    if zone != 'bras' and not (5 <= part <= 70):
        sys.exit(f'🔴 ABSURDE : {part:.1f} % du personnage — masque rejeté')

    # ⚠️ ÉROSION D'UN PIXEL au contact de la peau. Sans elle, un liséré de la
    # couleur d'origine survit à la teinture sur tout le pourtour — le défaut
    # classique de cette étape.
    m = ndimage.binary_erosion(m, np.ones((3, 3)), iterations=1)
    m = ndimage.binary_closing(m, np.ones((5, 5)))
    # une seule composante : les îlots isolés sont du bruit
    lab, n = ndimage.label(m)
    if n > 1:
        t = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(t)))
        print(f'  {n} composantes → la plus grosse retenue')

    dest = dest or chemin.replace('.png', '.semantique.png')
    Image.fromarray((m * 255).astype(np.uint8)).save(dest)
    print(f'  ÉCRIT : {dest}')
    return m


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('usage : segmenter.py <image> [zone] [destination]')
    segmenter(sys.argv[1],
              sys.argv[2] if len(sys.argv) > 2 else 'haut',
              sys.argv[3] if len(sys.argv) > 3 else None)
