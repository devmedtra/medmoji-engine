#!/usr/bin/env python3
"""LES TÉMOINS DE LA FABRIQUE — vérifier TOUT ce qui ne doit pas changer.

Med, 30 août 2026 : « j'ai pas l'impression que tu analyses ce que tu m'envoies,
parce que tu constaterais toi-même les problèmes ».

🔴 LE DÉFAUT DE MÉTHODE QUE CE FICHIER CORRIGE. Mes témoins mesuraient ce que je
SOUPÇONNAIS, jamais le reste. Sur le cargo, ils annonçaient :

    écart hors zone touchée : 0/255      → vrai, mais ne dit rien de l'intérieur
    mains intactes : 3/255               → vrai, et pourtant les mains étaient
                                           enfermées dans des MOUFLES vertes

Les deux mesures étaient exactes. Ce qu'elles ne voyaient pas :
  · le sous-vêtement EFFACÉ — il était dans la zone, rien ne le surveillait ;
  · des MOUFLES peintes autour des mains — les doigts étaient préservés à
    l'intérieur d'un bloc de tissu, ce qui est pire que de ne rien protéger ;
  · le torse REDESSINÉ, abdominaux inventés ;
  · le pantalon arrêté à mi-mollet.

⭐ LA RÈGLE QUI EN DÉCOULE. Un témoin ne doit pas vérifier une hypothèse, il
doit vérifier une INVARIANCE : « tout ce qui n'est pas le vêtement doit être
identique à l'original ». Ce qu'on n'a pas pensé à surveiller est précisément ce
qui casse.
"""
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

# Ce que chaque zone n'a PAS le droit de toucher, en fraction de la hauteur du
# personnage. Mesuré le 30 août : menton 28,1 %, poignet 59 %, mains 61-63 %,
# doigts 64-69 %.
INTERDITS = {
    'haut':  [('le visage', 0.00, 0.28), ('les mains', 0.58, 0.70),
              ('les jambes', 0.72, 1.00)],
    'bas':   [('le visage', 0.00, 0.28), ('le torse', 0.30, 0.50),
              ('les mains', 0.58, 0.70), ('les pieds', 0.94, 1.00)],
    'pieds': [('le visage', 0.00, 0.28), ('le torse', 0.30, 0.50),
              ('les cuisses', 0.55, 0.85)],
}


def verifier(chemin_base, chemin_produit, zone, masque_vetement=None,
             tolerance=4):
    """Rend (nb_defauts, rapport). Ne juge pas l'esthétique — seulement ce qui
    a changé sans y être autorisé."""
    base = np.asarray(Image.open(chemin_base).convert('RGBA')).astype(float)
    prod = np.asarray(Image.open(chemin_produit).convert('RGBA')).astype(float)
    if base.shape != prod.shape:
        return 1, [f'🔴 tailles différentes : {base.shape} vs {prod.shape}']

    corps = base[:, :, 3] > 250          # pixels pleins, hors frange
    ys = np.where(base[:, :, 3].any(1))[0]
    h0, h1 = ys.min(), ys.max()
    Hp = h1 - h0
    ecart = np.abs(base[:, :, :3] - prod[:, :, :3]).max(2)

    # Le vêtement : fourni, ou déduit de ce qui a changé.
    if masque_vetement is not None:
        vet = np.asarray(Image.open(masque_vetement).convert('L')) > 127
    else:
        vet = (ecart > 18) & (prod[:, :, 3] > 200)
        vet = ndimage.binary_opening(ndimage.binary_closing(vet, np.ones((9, 9))),
                                     np.ones((5, 5)))

    rapport, defauts = [], 0

    # ── 1. INVARIANCE DU CORPS ────────────────────────────────────────────
    # Tout pixel du corps NON recouvert par le vêtement doit être identique.
    # C'est le témoin qui manquait : il ne présume rien de l'endroit du défaut.
    libre = corps & ~vet
    abimes = libre & (ecart > tolerance)
    part = abimes.sum() / max(1, libre.sum()) * 100
    rapport.append(f'corps hors vêtement : {libre.sum():,} px, '
                   f'dont {abimes.sum():,} modifiés ({part:.2f} %)')
    if abimes.sum() > 200:
        defauts += 1
        # où, exactement ? l'information qui manquait le plus.
        for nom, d, f in [('visage', 0, .28), ('torse', .28, .55),
                          ('mains/bras', .55, .72), ('jambes', .72, .94),
                          ('pieds', .94, 1.)]:
            z = np.zeros(corps.shape, bool)
            z[h0 + int(Hp * d):h0 + int(Hp * f)] = True
            n = (abimes & z).sum()
            if n > 100:
                rapport.append(f'   🔴 {nom} : {n:,} px modifiés '
                               f'(max {ecart[abimes & z].max():.0f}/255)')

    # ── 2. ZONES INTERDITES ───────────────────────────────────────────────
    # Le vêtement n'a rien à faire sur le visage, les mains ou les pieds.
    # C'est ce témoin qui aurait attrapé les MOUFLES.
    for nom, d, f in INTERDITS.get(zone, []):
        z = np.zeros(corps.shape, bool)
        z[h0 + int(Hp * d):h0 + int(Hp * f)] = True
        n = (vet & z & corps).sum()
        surface = (z & corps).sum()
        if n > surface * 0.02:
            defauts += 1
            rapport.append(f'🔴 le vêtement empiète sur {nom} : {n:,} px '
                           f'({n/max(1,surface)*100:.1f} % de la zone)')

    # ── 3. COUVERTURE ATTENDUE ────────────────────────────────────────────
    # Un pantalon qui s'arrête à mi-mollet est un défaut, pas un style.
    if zone == 'bas':
        z = np.zeros(corps.shape, bool)
        z[h0 + int(Hp * .85):h0 + int(Hp * .92)] = True
        couvert = (vet & z).sum() / max(1, (z & corps).sum()) * 100
        rapport.append(f'chevilles couvertes : {couvert:.0f} %')
        if couvert < 40:
            defauts += 1
            rapport.append('🔴 le bas ne descend pas aux chevilles')

    # ── 4. SILHOUETTE ─────────────────────────────────────────────────────
    sil = (prod[:, :, 3] > 16).sum() / max(1, (base[:, :, 3] > 16).sum()) * 100 - 100
    rapport.append(f'silhouette : {sil:+.1f} %')
    if sil < -2:
        defauts += 1
        rapport.append('🔴 la silhouette a RÉTRÉCI — du corps a été effacé')

    return defauts, rapport


if __name__ == '__main__':
    if len(sys.argv) < 4:
        sys.exit('usage : temoins_fabrique.py <base> <produit> <zone> [masque]')
    n, r = verifier(sys.argv[1], sys.argv[2], sys.argv[3],
                    sys.argv[4] if len(sys.argv) > 4 else None)
    for l in r:
        print('  ' + l)
    print(f'\n  {"✓ AUCUN DÉFAUT" if n == 0 else f"🔴 {n} DÉFAUT(S)"}')
    sys.exit(0 if n == 0 else 1)
