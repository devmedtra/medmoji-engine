#!/usr/bin/env python3
"""LA DÉFORMATION — le vêtement suit le squelette, et la couleur est un réglage.

Med, 30 août 2026 : « le même pantalon pourra automatiquement s'adapter à
différentes morphologies, couleurs, tailles et poses. »

C'est la preuve par l'image que le maillage tient : on plie un genou, on change
la corpulence, on change la couleur — et c'est toujours LE MÊME asset.

───────────────────────────────────────────────────────────────────────────────
  CE QUI SE PASSE ICI, DANS L'ORDRE
───────────────────────────────────────────────────────────────────────────────
    pose          une rotation par os, en degrés, autour de sa tête
    hiérarchie    chaque os hérite de la transformation de son parent
    skinning      un sommet est la moyenne PONDÉRÉE des positions que chaque
                  os lui donne — c'est le Linear Blend Skinning
    rendu         chaque triangle est une transformation affine de la texture

⭐ LA COULEUR N'EST PAS DANS L'ASSET. Une seule texture maître, et la teinte
s'applique au rendu en Lab — luminance du tissu conservée, chromie imposée.
Quatre couleurs ne font pas quatre fichiers : elles font quatre appels.

🔴 CE QU'ON NE FAIT PAS. Pas de dual-quaternion : en 2D et pour des angles de
genou plausibles (moins de 90°), le LBS ne produit pas l'effet « papier de
bonbon » qui justifie cette complication en 3D. Le témoin de surface le
vérifie — si une aire de triangle s'effondre, c'est mesuré, pas supposé.
"""
import json
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

_M = np.array([[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]])
_BL = np.array([.95047, 1.0, 1.08883])


def vers_lab(rvb):
    r = rvb / 255.0
    r = np.where(r > .04045, ((r + .055) / 1.055) ** 2.4, r / 12.92)
    xyz = (r @ _M.T) / _BL
    f = np.where(xyz > .008856, np.cbrt(np.maximum(xyz, 0)), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def _lin(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f > .206893, f ** 3, (f - 16 / 116) / 7.787) * _BL
    return xyz @ np.linalg.inv(_M).T


def vers_rvb(lab):
    """Avec ramenée dans le gamut : la chroma cède, la luminance tient."""
    r = _lin(lab)
    hors = ((r < 0) | (r > 1)).any(-1)
    if hors.any():
        bas, haut = np.zeros(lab.shape[:-1]), np.ones(lab.shape[:-1])
        for _ in range(8):
            k = (bas + haut) / 2
            e = lab.copy()
            e[..., 1] *= k
            e[..., 2] *= k
            ok = ~((_lin(e) < -1e-6) | (_lin(e) > 1 + 1e-6)).any(-1)
            bas, haut = np.where(ok, k, bas), np.where(ok, haut, k)
        lab = lab.copy()
        lab[..., 1] = np.where(hors, lab[..., 1] * bas, lab[..., 1])
        lab[..., 2] = np.where(hors, lab[..., 2] * bas, lab[..., 2])
        r = _lin(lab)
    r = np.where(r > .0031308, 1.055 * np.maximum(r, 0) ** (1 / 2.4) - .055, 12.92 * r)
    return np.clip(r, 0, 1) * 255


def transformations(sq, pose, corpulence=1.0):
    """Une matrice 2x3 par os, en pixels, hiérarchie comprise.

    `pose` : {nom_de_l_os: angle en degrés}. Positif = sens horaire à l'écran.
    `corpulence` : facteur d'échelle latérale autour de l'axe du corps.
    """
    W = sq['gabarit']['largeur']
    H = sq['gabarit']['hauteur']
    par_nom = {o['nom']: o for o in sq['os']}
    cache = {}

    def matrice(nom):
        if nom in cache:
            return cache[nom]
        o = par_nom[nom]
        parent = o.get('parent')
        M_p = matrice(parent) if parent and parent in par_nom else np.array(
            [[1., 0., 0.], [0., 1., 0.]])
        a = np.deg2rad(pose.get(nom, 0.0))
        px, py = o['tete'][0] * W, o['tete'][1] * H
        c, s = np.cos(a), np.sin(a)
        # rotation autour de la tête de l'os, exprimée en coordonnées du parent
        R = np.array([[c, -s, px - c * px + s * py],
                      [s, c, py - s * px - c * py]])
        M = np.array([[M_p[0, 0], M_p[0, 1], M_p[0, 2]],
                      [M_p[1, 0], M_p[1, 1], M_p[1, 2]],
                      [0., 0., 1.]]) @ np.vstack([R, [0., 0., 1.]])
        cache[nom] = M[:2]
        return cache[nom]

    mats = {nom: matrice(nom) for nom in par_nom}
    if corpulence != 1.0:
        cx = sq['racine']['position'][0] * W
        S = np.array([[corpulence, 0., cx * (1 - corpulence)], [0., 1., 0.]])
        for nom in mats:
            M = np.vstack([mats[nom], [0., 0., 1.]])
            mats[nom] = (np.vstack([S, [0., 0., 1.]]) @ M)[:2]
    return mats


def deformer(maillage, sq, pose, corpulence=1.0):
    """Les positions des sommets après skinning, en pixels."""
    W, H = sq['gabarit']['largeur'], sq['gabarit']['hauteur']
    mats = transformations(sq, pose, corpulence)
    noms = maillage['os']
    P = np.array(maillage['poids'], np.float64)          # (n, k)
    S = np.array(maillage['sommets'], np.float64) * [W, H]
    hom = np.hstack([S, np.ones((len(S), 1))])
    out = np.zeros_like(S)
    for j, nom in enumerate(noms):
        M = mats[nom]
        out += P[:, j:j + 1] * (hom @ M.T)
    return out


def rendre(maillage, sq, texture, positions, teinte=None, fond=None):
    """Rend le maillage déformé, triangle par triangle.

    Chaque triangle est une transformation AFFINE de son homologue au repos :
    trois points de départ, trois d'arrivée, une matrice exacte. Aucun
    échantillonnage à inventer.
    """
    W, H = sq['gabarit']['largeur'], sq['gabarit']['hauteur']
    tex = np.asarray(texture.convert('RGBA')).astype(np.float32)
    uv = np.array(maillage['uv'], np.float64) * [W, H]
    out = (np.zeros((H, W, 4), np.float32) if fond is None
           else np.asarray(fond.convert('RGBA')).astype(np.float32).copy())
    ecrit = np.zeros((H, W), bool)     # les pixels que le VÊTEMENT a écrits

    aires_repos, aires_pose = [], []
    for t in maillage['triangles']:
        src = uv[t]
        dst = positions[t]
        a_r = abs(np.cross(src[1] - src[0], src[2] - src[0])) / 2
        a_p = abs(np.cross(dst[1] - dst[0], dst[2] - dst[0])) / 2
        aires_repos.append(a_r)
        aires_pose.append(a_p)
        if a_p < 1e-6:
            continue
        # la boîte du triangle d'arrivée
        x0 = max(0, int(np.floor(dst[:, 0].min())))
        x1 = min(W, int(np.ceil(dst[:, 0].max())) + 1)
        y0 = max(0, int(np.floor(dst[:, 1].min())))
        y1 = min(H, int(np.ceil(dst[:, 1].max())) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        # coordonnées barycentriques : elles donnent le test d'appartenance ET
        # l'interpolation, en une seule fois
        v0, v1, v2 = dst[0], dst[1] - dst[0], dst[2] - dst[0]
        den = v1[0] * v2[1] - v2[0] * v1[1]
        if abs(den) < 1e-9:
            continue
        px, py = xx - v0[0], yy - v0[1]
        b1 = (px * v2[1] - py * v2[0]) / den
        b2 = (py * v1[0] - px * v1[1]) / den
        dedans = (b1 >= -1e-9) & (b2 >= -1e-9) & (b1 + b2 <= 1 + 1e-9)
        if not dedans.any():
            continue
        su = src[0, 0] + b1 * (src[1, 0] - src[0, 0]) + b2 * (src[2, 0] - src[0, 0])
        sv_ = src[0, 1] + b1 * (src[1, 1] - src[0, 1]) + b2 * (src[2, 1] - src[0, 1])
        ui = np.clip(np.round(su).astype(int), 0, W - 1)
        vi = np.clip(np.round(sv_).astype(int), 0, H - 1)
        ech = tex[vi, ui]
        cible = out[yy, xx]
        # on n'écrit que là où la texture est opaque ET dans le triangle
        pose_ok = dedans & (ech[..., 3] > 16)
        cible[pose_ok] = ech[pose_ok]
        out[yy, xx] = cible
        e = ecrit[y0:y1, x0:x1]
        e[pose_ok] = True
        ecrit[y0:y1, x0:x1] = e

    # 🔴 NE TEINDRE QUE LE VÊTEMENT. Première version : `out[:,:,3] > 16` —
    # or `out` est composé sur le CORPS, donc le masque valait « tout le
    # personnage ». Le rouge a repeint la peau, le visage et les pieds. Le
    # vêtement rendu est suivi explicitement, pixel par pixel.
    if teinte is not None:
        m = ecrit
        if m.any():
            lab = vers_lab(out[:, :, :3])
            cible = vers_lab(np.array(teinte, float).reshape(1, 1, 3))[0, 0]
            L = lab[:, :, 0][m]
            lab[:, :, 0][m] = np.clip(cible[0] + (L - L.mean()), 0, 100)
            lab[:, :, 1][m] = cible[1]
            lab[:, :, 2][m] = cible[2]
            out[:, :, :3][m] = vers_rvb(lab)[m]

    ar, ap = np.array(aires_repos), np.array(aires_pose)
    valides = ar > 1e-6
    ratio = ap[valides] / ar[valides]
    return (Image.fromarray(out.round().astype(np.uint8)),
            {'triangles': len(ar), 'aire_min': float(ratio.min()),
             'aire_max': float(ratio.max()), 'aire_med': float(np.median(ratio))})


if __name__ == '__main__':
    base = '/root/medtra-avatar/createur/'
    mail = json.load(open(base + 'maillage-cargo.json', encoding='utf-8'))
    sq = json.load(open(base + 'squelette.json', encoding='utf-8'))
    tex = Image.open(sys.argv[1] if len(sys.argv) > 1
                     else base + 'sortie/inpaint/sv.png')
    # la texture n'est que le vêtement : on la découpe au masque
    mq = np.asarray(Image.open(base + 'sortie/inpaint/sv.masque.png')
                    .convert('L')) > 127
    a = np.asarray(tex.convert('RGBA')).copy()
    a[~mq, 3] = 0
    tex = Image.fromarray(a)

    corps = Image.open(base + 'sortie/tenues/base.norm.png').convert('RGBA')
    for nom, pose, corp, teinte in [
            ('repos', {}, 1.0, None),
            # ⚠️ De face, faire tourner un GENOU ne se voit pas : le pli est
            # dans le plan sagittal. Ce qui se lit à l'écran, c'est l'écart des
            # jambes — donc une rotation à la HANCHE, dans le plan frontal.
            ('jambes-ecartees', {'hanche_genou_gauche': 9,
                                 'hanche_genou_droite': -9}, 1.0, None),
            ('forte', {}, 1.22, None),
            ('mince', {}, 0.88, None),
            ('rouge', {}, 1.0, (230, 57, 70)),
    ]:
        pos = deformer(mail, sq, pose, corp)
        img, t = rendre(mail, sq, tex, pos, teinte, fond=corps)
        img.save(f'{base}sortie/inpaint/rig-{nom}.png')
        print(f'  {nom:10} {t["triangles"]:5,} triangles · aire pose/repos '
              f'min {t["aire_min"]:.2f} méd {t["aire_med"]:.2f} '
              f'max {t["aire_max"]:.2f}  '
              f'{"✓" if t["aire_min"] > 0.05 else "🔴 un triangle s effondre"}')
