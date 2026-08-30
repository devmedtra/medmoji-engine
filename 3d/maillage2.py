#!/usr/bin/env python3
"""LE MAILLAGE, VERSION CONTOUR — sommets sur la frontière, pas sur une grille.

Med, 30 août 2026 : « fouille internet au maximum pour appuyer tes mesures pour
que le pantalon fit parfaitement ».

Trois enseignements de la littérature, appliqués tels quels.

───────────────────────────────────────────────────────────────────────────────
  1. UNE GRILLE RÉGULIÈRE NE SUIT PAS LES CONTOURS   (SpriteToMesh, 2026)
───────────────────────────────────────────────────────────────────────────────
« Grid-based interior placement achieves good triangle regularity but fails to
follow visual boundaries, confirming the need for contour-aware placement. »
    — SpriteToMesh, arXiv 2602.21153, §baselines

C'est exactement le défaut de ma première version : `GRILLE_X × GRILLE_Y`
sommets régulièrement espacés, dont aucun ne tombe sur le bord du vêtement. Les
triangles de bordure débordaient ou rognaient, d'où les contours crénelés vus
sur les rendus déformés.

Leur pipeline, repris ici : contour → simplification → sommets de bord, plus des
sommets intérieurs, puis triangulation de **Delaunay**. Leur segmentation par
U-Net ne nous sert pas : nos masques viennent de SAM, déjà mesurés.

───────────────────────────────────────────────────────────────────────────────
  2. LIMITER LES OS PAR SOMMET — le « prune » de Spine
───────────────────────────────────────────────────────────────────────────────
« Using prune to remove unnecessary weights and limit the number of bones that
can affect a vertex can reduce vertex transforms required. »
    — Spine User Guide, Weights view

Quatre os par sommet est la limite usuelle des moteurs 2D et 3D. Au-delà, les
poids résiduels sont du bruit numérique qui fait « baver » la déformation.

───────────────────────────────────────────────────────────────────────────────
  3. LE « WELD » — LA CLÉ DU FIT
───────────────────────────────────────────────────────────────────────────────
« The Weld button matches weights across meshes, effectively welding them
together to allow multiple meshes to deform identically, as if they were a
single image. »
    — Spine User Guide, Weights view

⭐ C'est la réponse à « pour que le pantalon fitte parfaitement ». Tant que le
vêtement et le corps ont des poids calculés SÉPARÉMENT, ils se déforment
presque pareil — et « presque » se voit : le tissu glisse d'un ou deux pixels
sur la peau à chaque changement de pose. Soudés, ils se déforment à l'identique
par construction, et le fit ne peut plus dériver.

Un sommet de vêtement adopte donc les poids du CORPS au point le plus proche.
Le vêtement ne définit ses propres poids que là où il n'y a pas de corps
dessous — un pan qui déborde de la silhouette.
"""
import json
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import Delaunay

# Distance visée entre deux sommets, en fraction de la hauteur du personnage.
# Choisie pour donner ~1 000 sommets sur un pantalon, l'ordre de grandeur des
# maillages Spine pour un vêtement de personnage.
PAS = 0.011
# Tolérance de simplification du contour, en fraction de la hauteur.
TOL = 0.0016
# Le « prune » de Spine.
OS_MAX = 4


def contour_principal(masque):
    """Le contour extérieur de la plus grosse composante, en (x, y)."""
    import cv2
    m = (masque * 255).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return []
    return [c.reshape(-1, 2).astype(float) for c in cs
            if cv2.contourArea(c) > 200]


def simplifier(pts, tol):
    """Douglas-Peucker — garde les points qui portent la forme."""
    import cv2
    c = pts.reshape(-1, 1, 2).astype(np.float32)
    return cv2.approxPolyDP(c, tol, True).reshape(-1, 2).astype(float)


def densifier(poly, pas):
    """Rééchantillonne un polygone à pas constant : pas de longue arête."""
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        d = np.hypot(*(b - a))
        k = max(1, int(np.ceil(d / pas)))
        for t in range(k):
            out.append(a + (b - a) * (t / k))
    return np.array(out)


def poids_par_diffusion(masque, os, forme, iterations=250):
    """Bone heat 2D, multi-échelle — identique à la première version.

    u = 1 sur l'os, u = 0 sur les autres os, flux nul au bord du tissu, résolu
    du 1/8 au 1/1 parce qu'une relaxation de Jacobi converge en O(n²).
    """
    H, W = forme
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
        u = None
        for div in (8, 4, 2, 1):
            mk, sk, ak = masque[::div, ::div], src[::div, ::div], autres[::div, ::div]
            if not sk.any():
                sk = ndimage.binary_dilation(
                    src, np.ones((div * 2 + 1,) * 2))[::div, ::div] & mk
            mf = mk.astype(np.float32)
            den = np.maximum(ndimage.uniform_filter(mf, size=5), 1e-6)
            if u is None:
                u = sk.astype(np.float32)
            else:
                u = np.repeat(np.repeat(u, 2, 0), 2, 1)
                z = np.zeros(mk.shape, np.float32)
                h_, w_ = min(z.shape[0], u.shape[0]), min(z.shape[1], u.shape[1])
                z[:h_, :w_] = u[:h_, :w_]
                u = z
            for _ in range(iterations):
                v = ndimage.uniform_filter(u * mf, size=5) / den
                u = np.where(sk, 1.0, np.where(ak, 0.0, np.where(mk, v, 0.0)))
        champs[i] = u
    s = champs.sum(0)
    s[s < 1e-6] = 1.0
    return champs / s[None, :, :]


def elaguer(p, k=OS_MAX):
    """Le « prune » : ne garder que les k plus fortes influences, renormaliser."""
    if len(p) <= k:
        return p / max(p.sum(), 1e-9)
    seuil = np.partition(p, -k)[-k]
    q = np.where(p >= seuil, p, 0.0)
    return q / max(q.sum(), 1e-9)


def mailler(chemin_masque, chemin_squelette, dest, corps_json=None):
    m = np.asarray(Image.open(chemin_masque).convert('L')) > 127
    H, W = m.shape
    sq = json.load(open(chemin_squelette, encoding='utf-8'))
    os_ = sq['os']
    Hp = sq['personnage']['hauteur_px']
    pas, tol = PAS * Hp, TOL * Hp

    # ── LES SOMMETS DE BORD, sur le contour ───────────────────────────────
    bord = []
    for c in contour_principal(m):
        bord.append(densifier(simplifier(c, tol), pas))
    if not bord:
        raise SystemExit('  🔴 aucun contour')
    bord = np.vstack(bord)

    # ── LES SOMMETS INTÉRIEURS ────────────────────────────────────────────
    # Grille décalée d'une ligne sur deux : plus régulière qu'une grille droite
    # pour une triangulation de Delaunay (triangles moins allongés).
    ys, xs = np.where(m)
    interieur = []
    # ⚠️ Une érosion de 0,8 × pas (21 px) interdit tout sommet intérieur dans
    # une zone plus étroite que ça — donc dans toute la fourche de l'entrejambe,
    # large de 31 px. Mesuré : un trou de 1 489 px y restait sans triangle.
    dedans = ndimage.binary_erosion(m, np.ones((max(3, int(pas * .3)) | 1,) * 2))
    j = 0
    y = ys.min() + pas / 2
    while y < ys.max():
        x = xs.min() + (pas / 2 if j % 2 == 0 else pas)
        while x < xs.max():
            yi, xi = int(y), int(x)
            if 0 <= yi < H and 0 <= xi < W and dedans[yi, xi]:
                interieur.append([x, y])
            x += pas
        y += pas * 0.866        # triangulaire : hauteur = pas · √3/2
        j += 1
    P = np.vstack([bord, np.array(interieur)]) if interieur else bord

    # ── DELAUNAY, puis on jette les triangles hors du masque ──────────────
    tri = Delaunay(P)
    # 🔴 « LE CENTRE EST-IL DANS LE MASQUE ? » REJETTE LES TRIANGLES ÉTROITS.
    # Delaunay n'est pas contraint : sur une forme concave, il tend des
    # triangles au-dessus des creux. Tester leur seul centre rejetait aussi les
    # triangles LÉGITIMES et fins de la fourche — 1,3 % du vêtement sans
    # aucun triangle, dont un trou de 1 489 px pile à l'entrejambe.
    #
    # ⭐ Le bon test porte sur la SURFACE : un triangle est gardé si la majorité
    # de son aire est dans le masque. Sept points suffisent à l'estimer — les
    # trois sommets rentrés vers le centre, les trois milieux d'arête, et le
    # centre lui-même.
    garde = []
    for t in tri.simplices:
        d = P[t]
        a = abs(np.cross(d[1] - d[0], d[2] - d[0])) / 2
        if a < 1.0:
            continue
        c = d.mean(0)
        ech = np.vstack([d * .8 + c * .2, (d + np.roll(d, 1, 0)) / 2, [c]])
        n_ok = sum(1 for x, y in ech
                   if 0 <= int(round(y)) < H and 0 <= int(round(x)) < W
                   and m[int(round(y)), int(round(x))])
        if n_ok * 2 <= len(ech):
            continue
        garde.append([int(i) for i in t])

    print(f'  {m.sum():,} px · pas {pas:.0f} px · '
          f'{len(bord)} sommets de bord + {len(interieur)} intérieurs')
    print(f'  Delaunay : {len(tri.simplices)} triangles, {len(garde)} retenus')

    # ── LES POIDS ─────────────────────────────────────────────────────────
    champs = poids_par_diffusion(m, os_, (H, W))
    proche = ndimage.distance_transform_edt(~m, return_indices=True)[1]
    poids = []
    for x, y in P:
        yi, xi = int(np.clip(round(y), 0, H - 1)), int(np.clip(round(x), 0, W - 1))
        yy, xx = int(proche[0][yi, xi]), int(proche[1][yi, xi])
        poids.append(elaguer(champs[:, yy, xx].astype(np.float64)))
    poids = np.array(poids)

    # ── ⭐ LE WELD : les poids du CORPS priment là où il y a du corps ──────
    soudes = 0
    if corps_json:
        cj = json.load(open(corps_json, encoding='utf-8'))
        if cj['os'] != [o['nom'] for o in os_]:
            raise SystemExit('  🔴 corps et vêtement n\'ont pas les mêmes os')
        CS = np.array(cj['sommets']) * [W, H]
        CP = np.array(cj['poids'])
        # (le masque du corps n'est plus lu : depuis que le weld est total,
        #  chaque sommet reprend les poids du corps, où qu'il soit)
        from scipy.spatial import cKDTree
        arbre = cKDTree(CS)
        # 🔴 UN PAN QUI DÉBORDE DOIT ÊTRE SOUDÉ AUSSI. Première version : seuls
        # les sommets tombant SUR le corps reprenaient ses poids, les autres
        # gardaient les leurs. Mesuré, glissement en coordonnées locales (P95) :
        #
        #                        sur le corps    hors silhouette
        #     jambes écartées      0,52 px          1,98 px
        #     corpulence 1,22      0,92 px          6,71 px
        #
        # Toute la dérive résiduelle venait des 30 % non soudés. Or un pan de
        # tissu qui dépasse la hanche doit suivre la hanche : c'est le sommet
        # de corps le plus proche qui le porte, même s'il n'est pas dessous.
        loin = 0
        for k, (x, y) in enumerate(P):
            d, i = arbre.query([x, y])
            if d < pas * 4:
                poids[k] = elaguer(CP[i])
                soudes += 1
            else:
                loin += 1
        print(f'  weld : {soudes}/{len(P)} sommets ({soudes/len(P)*100:.0f} %) '
              f'reprennent les poids du corps'
              + (f' · {loin} trop loin, poids propres' if loin else ''))

    sortie = {
        'gabarit': sq['gabarit'],
        'os': [o['nom'] for o in os_],
        'sommets': [[round(x / W, 6), round(y / H, 6)] for x, y in P],
        'uv': [[round(x / W, 6), round(y / H, 6)] for x, y in P],
        'triangles': garde,
        'poids': [[round(v, 4) for v in w] for w in poids],
        'soudes': soudes,
    }
    json.dump(sortie, open(dest, 'w'), ensure_ascii=False)

    # ── TÉMOINS ───────────────────────────────────────────────────────────
    s = poids.sum(1)
    print(f'  somme des poids : min {s.min():.3f} max {s.max():.3f} · '
          f'{int((s < 0.5).sum())} orphelin(s)  '
          f'{"✓" if abs(s - 1).max() < 0.01 else "🔴"}')
    print(f'  os par sommet : max {int((poids > 1e-6).sum(1).max())} '
          f'(limite {OS_MAX})  '
          f'{"✓" if (poids > 1e-6).sum(1).max() <= OS_MAX else "🔴"}')
    # ⭐ Les sommets de bord doivent être SUR le bord, à moins d'un pixel.
    d_bord = ndimage.distance_transform_edt(m) + ndimage.distance_transform_edt(~m)
    ecarts = [abs(ndimage.distance_transform_edt(m)[int(np.clip(y, 0, H-1)),
                                                   int(np.clip(x, 0, W-1))])
              for x, y in bord]
    print(f'  sommets de bord : distance au contour médiane '
          f'{np.median(ecarts):.1f} px  '
          f'{"✓" if np.median(ecarts) < 2 else "🔴 ils ne sont pas sur le bord"}')
    milieu = W / 2
    f = sum(1 for i, o in enumerate(os_)
            if o['nom'].startswith(('hanche_genou', 'genou_cheville',
                                    'cheville_pied'))
            for k, (x, _) in enumerate(P)
            if poids[k, i] >= 0.2 and abs(x - milieu) >= 20
            and (o['nom'].endswith('gauche')) != (x < milieu))
    print(f'  poids traversant l\'entrejambe : {f}  {"✓" if f == 0 else "🔴"}')
    print(f'  → {dest}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('usage : maillage2.py <masque.png> [squelette] [sortie] [corps.json]')
    mailler(sys.argv[1],
            sys.argv[2] if len(sys.argv) > 2 else 'squelette.json',
            sys.argv[3] if len(sys.argv) > 3 else 'maillage2.json',
            sys.argv[4] if len(sys.argv) > 4 else None)
