#!/usr/bin/env python3
"""LE MAILLAGE DU VÊTEMENT — un mesh 2D riggé, pas une image posée dessus.

Med, 30 août 2026 : « le pantalon doit être créé comme un mesh indépendant,
composé de plusieurs zones reliées à ces bones. Chaque zone du mesh possède des
weights afin que le pantalon puisse se déformer naturellement lorsque le corps
change de morphologie ou de position. »

Ce fichier transforme la sortie de la Fabrique — une image de vêtement — en
l'asset que le Moteur consomme :

    sommets      position (fraction du gabarit) + coordonnées de texture
    triangles    la topologie, calculée une fois
    poids        l'influence de chaque os sur chaque sommet

───────────────────────────────────────────────────────────────────────────────
  🔴 LES POIDS NE SE CALCULENT PAS EN INVERSE-DISTANCE
───────────────────────────────────────────────────────────────────────────────
Med, 17 août 2026 : « fouille internet, c'est sûr que quelqu'un l'a déjà fait ».
Il avait raison : des poids en inverse-distance déchiraient la main, là où le
BONE HEAT WEIGHTING de Blender fait le travail proprement.

Le principe transposé en 2D : chaque os est une SOURCE DE CHALEUR, la chaleur
diffuse dans le vêtement, et le poids d'un os en un point est sa température
normalisée. La différence avec l'inverse-distance est décisive — la chaleur
CONTOURNE, elle ne traverse pas :

    inverse-distance   un point de la cuisse gauche « voit » l'os de la cuisse
                       droite à travers le vide de l'entrejambe, et le suit
    diffusion          la chaleur doit longer le tissu ; l'entrejambe étant
                       vide, elle ne passe pas, et les deux jambes restent
                       indépendantes

⭐ C'est la même équation Δu = 0 que celle qui efface le sous-vêtement dans
`fabrique.py`. Un seul solveur, deux usages.
"""
import json
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

# Combien de cellules sur la boîte du vêtement. Une grille plus fine ne rend pas
# la déformation plus juste : elle suit les MÊMES os. Elle coûte, en revanche.
GRILLE_X, GRILLE_Y = 24, 48


def charger_squelette(chemin):
    sq = json.load(open(chemin, encoding='utf-8'))
    return sq


def poids_par_diffusion(masque, os, forme, iterations=600):
    """Un champ de poids par os, obtenu par diffusion dans le tissu.

    Rend un tableau (n_os, H, W) normalisé pour que la somme des poids vaille 1
    en tout point du masque.
    """
    H, W = forme
    # 🔴 UNE DIFFUSION SANS CONDITION AU BORD DÉCROÎT VERS ZÉRO.
    # Première version : u = 1 sur l'os, u libre ailleurs. Loin de la source, la
    # chaleur devient numériquement nulle — 114 sommets orphelins mesurés, tous
    # en haut du vêtement (0 à 46 % de sa hauteur), c'est-à-dire la ceinture,
    # la plus éloignée de son os.
    #
    # ⭐ La formulation de Baran & Popović impose les DEUX conditions : u_i = 1
    # sur l'os i, et u_i = 0 sur TOUS LES AUTRES os. Le champ reste alors borné
    # entre 0 et 1, la somme est naturellement proche de 1 partout, et chaque os
    # ne perd son influence que là où un autre la prend.
    sources = []
    for o in os:
        x0, y0 = o['tete'][0] * W, o['tete'][1] * H
        x1, y1 = o['queue'][0] * W, o['queue'][1] * H
        n = max(2, int(np.hypot(x1 - x0, y1 - y0)))
        src = np.zeros((H, W), bool)
        for t in np.linspace(0, 1, n):
            xx, yy = int(x0 + (x1 - x0) * t), int(y0 + (y1 - y0) * t)
            if 0 <= yy < H and 0 <= xx < W:
                src[yy, xx] = True
        sources.append(ndimage.binary_dilation(src, np.ones((9, 9))) & masque)

    champs = np.zeros((len(os), H, W), np.float32)
    for i, src in enumerate(sources):
        if not src.any():
            continue
        autres = np.zeros((H, W), bool)
        for j, s2 in enumerate(sources):
            if j != i:
                autres |= s2
        autres &= ~src
        # 🔴 ET LE BORD DU VÊTEMENT NE DOIT PAS ÊTRE UN PUITS.
        # `np.where(masque, v, 0)` impose u = 0 sur tout le fond : la chaleur
        # FUIT par le contour du tissu. La condition juste au bord d'une pièce
        # de tissu est un FLUX NUL — on ne moyenne que sur les voisins qui sont
        # eux-mêmes du tissu. Le rapport de deux filtres uniformes le donne.
        #
        # 🔴 ET UNE RELAXATION DE JACOBI CONVERGE EN O(n²) ITÉRATIONS.
        # C'était la vraie cause des sommets orphelins, que trois correctifs
        # partiels n'avaient pas touchée : 114 puis 105, tous à la ceinture.
        # Pour porter la chaleur à 250 px de sa source il faut ~62 500 pas ;
        # j'en faisais 600. Le champ n'était pas faux, il n'était pas fini.
        #
        # ⭐ On résout d'abord au 1/8, où 250 px n'en font plus que 31, puis on
        # remonte le résultat comme point de départ de l'échelle suivante. La
        # même précision coûte alors quelques centaines de pas au lieu de
        # dizaines de milliers.
        u = None
        for div in (8, 4, 2, 1):
            mk = masque[::div, ::div]
            sk = src[::div, ::div]
            ak = autres[::div, ::div]
            if not sk.any():          # un os trop fin disparaît au 1/8
                sk = src[::div, ::div] | (ndimage.binary_dilation(
                    src, np.ones((div * 2 + 1,) * 2))[::div, ::div] & mk)
            mf = mk.astype(np.float32)
            den = np.maximum(ndimage.uniform_filter(mf, size=5), 1e-6)
            if u is None:
                u = sk.astype(np.float32)
            else:
                u = np.repeat(np.repeat(u, 2, 0), 2, 1)[:mk.shape[0], :mk.shape[1]]
                if u.shape != mk.shape:      # bord impair
                    z = np.zeros(mk.shape, np.float32)
                    z[:u.shape[0], :u.shape[1]] = u
                    u = z
            for _ in range(iterations):
                v = ndimage.uniform_filter(u * mf, size=5) / den
                u = np.where(sk, 1.0, np.where(ak, 0.0, np.where(mk, v, 0.0)))
        champs[i] = u
    somme = champs.sum(0)
    somme[somme < 1e-6] = 1.0
    return champs / somme[None, :, :]


def mailler(chemin_masque, chemin_squelette, dest):
    m = np.asarray(Image.open(chemin_masque).convert('L')) > 127
    H, W = m.shape
    sq = charger_squelette(chemin_squelette)
    os = sq['os']
    if sq['gabarit']['largeur'] != W or sq['gabarit']['hauteur'] != H:
        raise SystemExit(f'  🔴 le masque ({W}x{H}) et le squelette '
                         f'({sq["gabarit"]["largeur"]}x{sq["gabarit"]["hauteur"]}) '
                         f'ne sont pas au même gabarit')

    ys, xs = np.where(m)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    print(f'  vêtement : {m.sum():,} px, boîte {x1-x0+1}x{y1-y0+1}')

    print(f'  diffusion des poids pour {len(os)} os…')
    champs = poids_par_diffusion(m, os, (H, W))

    # la carte du plus proche pixel de tissu — calculée une fois
    proche = ndimage.distance_transform_edt(~m, return_indices=True)[1]

    # ── LA GRILLE ────────────────────────────────────────────────────────
    gx = np.linspace(x0, x1, GRILLE_X + 1)
    gy = np.linspace(y0, y1, GRILLE_Y + 1)
    idx = -np.ones((GRILLE_Y + 1, GRILLE_X + 1), int)
    sommets, poids = [], []
    for j, y in enumerate(gy):
        for i, x in enumerate(gx):
            yi, xi = int(round(y)), int(round(x))
            # un sommet n'existe que si le tissu est là, ou juste à côté :
            # les bords du vêtement doivent être portés par des sommets
            fen = m[max(0, yi - 12):yi + 13, max(0, xi - 12):xi + 13]
            if not fen.any():
                continue
            idx[j, i] = len(sommets)
            sommets.append([x / W, y / H])
            # ⚠️ Le poids se lit au pixel de TISSU le plus proche, jamais au
            # sommet lui-même : un sommet de bord tombe hors du masque, où la
            # diffusion vaut zéro. Première version : une transformée de
            # distance recalculée sur une fenêtre locale, avec des indices mal
            # ramenés — d'où des sommets à somme de poids NULLE (min 0,000).
            # La carte du plus proche tissu se calcule UNE FOIS, pour toute
            # l'image, et se lit ensuite en O(1).
            yy, xx = int(proche[0][yi, xi]), int(proche[1][yi, xi])
            poids.append(champs[:, yy, xx].tolist())

    # ── LES TRIANGLES ────────────────────────────────────────────────────
    tris = []
    for j in range(GRILLE_Y):
        for i in range(GRILLE_X):
            a, b = idx[j, i], idx[j, i + 1]
            c, d = idx[j + 1, i], idx[j + 1, i + 1]
            if a >= 0 and b >= 0 and c >= 0:
                tris.append([int(a), int(b), int(c)])
            if b >= 0 and d >= 0 and c >= 0:
                tris.append([int(b), int(d), int(c)])

    sortie = {
        'gabarit': sq['gabarit'],
        'squelette': chemin_squelette.split('/')[-1],
        'os': [o['nom'] for o in os],
        'sommets': [[round(v, 6) for v in s] for s in sommets],
        # ⭐ Les UV sont les positions elles-mêmes : la texture est produite au
        # gabarit, donc un sommet lit la texture là où il se trouve au repos.
        # Aucune projection à inventer, aucun dépliage à vérifier.
        'uv': [[round(v, 6) for v in s] for s in sommets],
        'triangles': tris,
        'poids': [[round(p, 4) for p in w] for w in poids],
    }
    json.dump(sortie, open(dest, 'w'), ensure_ascii=False)

    # ── TÉMOINS ──────────────────────────────────────────────────────────
    P = np.array(poids)
    print(f'  {len(sommets):,} sommets · {len(tris):,} triangles')
    nulles = np.where(P.sum(1) < 0.5)[0]
    print(f'  somme des poids : min {P.sum(1).min():.3f} · '
          f'max {P.sum(1).max():.3f} · {len(nulles)} sommet(s) orphelin(s)  '
          f'{"✓" if not len(nulles) else "🔴"}')
    if len(nulles):
        H_, W_ = m.shape
        hh = [(sommets[k][1] * H_ - y0) / (y1 - y0) * 100 for k in nulles]
        print(f'     hauteurs dans le vêtement : {min(hh):.0f} % à {max(hh):.0f} %, '
              f'médiane {np.median(hh):.0f} %')
    for i, o in enumerate(os):
        n = int((P[:, i] > 0.5).sum())
        print(f'     {o["nom"]:28} domine {n:4d} sommets '
              f'({n/max(1,len(sommets))*100:4.1f} %)')
    # ⭐ AUCUN OS D'UNE JAMBE NE DOIT PESER SUR L'AUTRE. C'est le contrôle qui
    # distingue la diffusion de l'inverse-distance : la chaleur ne traverse pas
    # le vide de l'entrejambe.
    # ⚠️ Les sommets sur l'axe même sont ignorés : leur côté est arbitraire.
    milieu = W / 2
    fautes = 0
    for i, o in enumerate(os):
        # ⚠️ Seuls les os de JAMBE sont testés. Ceux du bassin sont centraux :
        # `bassin_hanche_gauche` a légitimement de l'influence de l'autre côté
        # de l'axe, près de la fourche. Les inclure faisait crier le témoin
        # 117 fois sur un maillage correct.
        if not o['nom'].startswith(('hanche_genou', 'genou_cheville',
                                    'cheville_pied')):
            continue
        cote = 'gauche' if o['nom'].endswith('gauche') else 'droite'
        for k, s in enumerate(sommets):
            if P[k, i] < 0.2 or abs(s[0] * W - milieu) < 20:
                continue
            if (cote == 'gauche') != (s[0] * W < milieu):
                fautes += 1
    print(f'  poids traversant l\'entrejambe : {fautes} '
          f'{"✓" if fautes == 0 else "🔴 la diffusion a franchi le vide"}')
    print(f'  → {dest}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('usage : maillage.py <masque.png> [squelette.json] [sortie.json]')
    mailler(sys.argv[1],
            sys.argv[2] if len(sys.argv) > 2 else
            '/root/medtra-avatar/createur/squelette.json',
            sys.argv[3] if len(sys.argv) > 3 else
            '/root/medtra-avatar/createur/maillage-cargo.json')
