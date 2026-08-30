#!/usr/bin/env python3
"""LE SANDWICH — trancher un vêtement en calque avant et calque arrière.

Med, 30 août 2026 : « un sweat à capuche nécessite une couche pour le torse
(qui va sous la tête) et une couche pour la capuche (qui passe derrière et
par-dessus le cou) ».

═══ POURQUOI ÇA MARCHE SANS MOTEUR 3D ═══

Un vêtement à capuche n'est pas plat : sa capuche passe DERRIÈRE le cou, son
torse passe DEVANT. Un seul calque ne peut pas faire les deux — soit la capuche
recouvre le cou, soit elle disparaît sous lui.

On tranche donc le masque en deux à la ligne des épaules, et l'empilement final
devient :

    plan 03   capuche, col intérieur   ← DERRIÈRE le corps
    plan 10   corps nu, cou, tête
    plan 60   torse, manches, col avant ← DEVANT le corps

⭐ La coupure nette entre l'avant et l'arrière est INVISIBLE : elle est cachée
par le cou et la tête, qui se trouvent exactement dessus. C'est ce qui rend
l'illusion gratuite.

🔴 LA LIGNE DE COUPE SE MESURE. La fixer au jugé ferait apparaître la couture
au-dessus ou en dessous du cou selon le personnage. On la trouve là où la
largeur du corps SAUTE — le passage du cou aux épaules est la plus forte
variation de toute la silhouette.
"""
import sys

import numpy as np
from PIL import Image
from scipy import ndimage


def ligne_des_epaules(corps, h0, Hp):
    """La hauteur où le cou devient les épaules — mesurée, pas supposée.

    On cherche le plus grand SAUT de largeur dans la zone plausible (25 à 45 %
    du personnage). C'est le passage du cou, étroit, aux épaules, larges.
    """
    largeurs = []
    for pct in range(25, 46):
        y = h0 + int(Hp * pct / 100)
        xs = np.where(corps[y])[0]
        largeurs.append((pct, y, len(xs)))
    sauts = [(largeurs[i + 1][2] - largeurs[i][2], largeurs[i + 1][0], largeurs[i + 1][1])
             for i in range(len(largeurs) - 1)]
    saut, pct, y = max(sauts)
    print(f'  épaules mesurées à {pct} % (saut de largeur : +{saut} px)')
    return y


def trancher(chemin_habille, chemin_masque, chemin_corps, prefixe):
    hab = Image.open(chemin_habille).convert('RGBA')
    a = np.asarray(hab).astype(np.uint8)
    m = np.asarray(Image.open(chemin_masque).convert('L')) > 127
    corps = np.asarray(Image.open(chemin_corps).convert('RGBA'))[:, :, 3] > 16

    ys = np.where(corps.any(1))[0]
    h0, h1 = ys.min(), ys.max()
    y_coupe = ligne_des_epaules(corps, h0, h1 - h0)
    # ⚠️ On remonte un peu : le col avant doit rester entier du bon côté.
    y_coupe -= int((h1 - h0) * 0.01)

    avant, arriere = m.copy(), m.copy()
    avant[:y_coupe, :] = False
    arriere[y_coupe:, :] = False

    for nom, masque in (('avant', avant), ('arriere', arriere)):
        img = a.copy()
        img[~masque, 3] = 0
        Image.fromarray(img).save(f'{prefixe}.{nom}.png')
        print(f'  {nom:8} : {masque.sum():8,} px  → {prefixe}.{nom}.png')

    # ── TÉMOINS ──
    print(f'\n  ligne de coupe : y={y_coupe}')
    print(f'  total {m.sum():,} px = avant {avant.sum():,} + arrière {arriere.sum():,}'
          f'  → {"exact" if avant.sum() + arriere.sum() == m.sum() else "🔴 PERTE"}')
    # ⭐ La coupure doit être CACHÉE par le corps : à la hauteur de la coupe, le
    #    corps doit couvrir toute la largeur du vêtement, sinon la couture se voit.
    ligne_vet = np.where(m[y_coupe])[0]
    ligne_corps = np.where(corps[y_coupe])[0]
    if len(ligne_vet):
        deborde = ((ligne_vet < ligne_corps.min()) | (ligne_vet > ligne_corps.max())).sum()
        print(f'  à la coupe : vêtement {len(ligne_vet)} px, corps '
              f'{len(ligne_corps)} px, dépasse de {deborde} px  → '
              f'{"couture CACHÉE" if deborde == 0 else "🔴 couture VISIBLE sur les côtés"}')
    if arriere.sum() < 500:
        print('  ⚠️ calque arrière quasi vide : ce vêtement n\'a pas de capuche')


if __name__ == '__main__':
    if len(sys.argv) < 5:
        sys.exit('usage : sandwich.py <habille> <masque> <corps> <prefixe>')
    trancher(*sys.argv[1:5])
