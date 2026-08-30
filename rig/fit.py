#!/usr/bin/env python3
"""LE TÉMOIN DU FIT — le vêtement glisse-t-il sur la peau ?

Med, 30 août 2026 : « fouille internet au maximum pour appuyer tes mesures pour
que le pantalon fit parfaitement ».

⭐ « Parfaitement » se mesure, et voici avec quoi. Au repos, chaque sommet du
vêtement est à une certaine distance du corps. Si le vêtement est correctement
soudé au corps, cette distance est INVARIANTE quelle que soit la pose ou la
corpulence : le tissu suit la peau au lieu de flotter par-dessus.

    glissement d'un sommet = |d_pose − d_repos|

C'est le critère que le « weld » de Spine rend nul par construction :

    « The Weld button matches weights across meshes, effectively welding them
      together to allow multiple meshes to deform identically, as if they were
      a single image. »
        — Spine User Guide, Weights view

🔴 Sans lui, le corps et le vêtement ont des poids calculés séparément. Ils se
déforment PRESQUE pareil — et « presque » se voit : le tissu rampe d'un ou deux
pixels sur la peau à chaque changement de pose, et le fit dérive.
"""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

from deformer import deformer


def glissement(mail_v, mail_c, sq, poses):
    """De combien le vêtement bouge-t-il PAR RAPPORT au corps, en local.

    🔴 UNE DISTANCE EUCLIDIENNE N'EST PAS UN TÉMOIN DE FIT. Première version :
    « distance du sommet de vêtement au corps le plus proche », comparée entre
    le repos et la pose. Elle criait au glissement sur la CORPULENCE — 6,10 px
    de P95 — alors que rien ne glissait : quand le corps s'élargit de 22 %,
    toutes les distances s'élargissent de 22 % avec lui. Le témoin mesurait
    l'échelle, pas la dérive.

    ⭐ Le fit se lit en coordonnées LOCALES. Chaque sommet du vêtement est
    exprimé en coordonnées barycentriques dans le triangle de corps qui le
    porte. Ces coordonnées sont invariantes par toute transformation affine —
    rotation, échelle, cisaillement. Si elles bougent, le tissu a vraiment
    rampé sur la peau ; si elles ne bougent pas, le fit tient, quelle que soit
    la déformation.
    """
    V0 = deformer(mail_v, sq, {}, 1.0)
    C0 = deformer(mail_c, sq, {}, 1.0)
    tris = np.array(mail_c['triangles'])
    centres = C0[tris].mean(1)
    arbre = cKDTree(centres)

    def bary(V, C):
        """Coordonnées barycentriques de chaque sommet dans « son » triangle."""
        t = tris[arbre.query(V0)[1]]          # le triangle est choisi AU REPOS
        a, b, c = C[t[:, 0]], C[t[:, 1]], C[t[:, 2]]
        v0, v1, v2 = b - a, c - a, V - a
        d00 = (v0 * v0).sum(1)
        d01 = (v0 * v1).sum(1)
        d11 = (v1 * v1).sum(1)
        d20 = (v2 * v0).sum(1)
        d21 = (v2 * v1).sum(1)
        den = np.where(np.abs(d00 * d11 - d01 * d01) < 1e-9, 1e-9,
                       d00 * d11 - d01 * d01)
        u = (d11 * d20 - d01 * d21) / den
        v = (d00 * d21 - d01 * d20) / den
        return np.stack([u, v], 1)

    B0 = bary(V0, C0)
    # ⚠️ Le glissement est rendu en PIXELS de l'état de repos, pour rester
    # lisible : une dérive barycentrique multipliée par la taille du triangle.
    tailles = np.sqrt(np.abs(np.cross(
        C0[tris[:, 1]] - C0[tris[:, 0]],
        C0[tris[:, 2]] - C0[tris[:, 0]])))[arbre.query(V0)[1]]
    lignes = []
    for nom, pose, corp in poses:
        V = deformer(mail_v, sq, pose, corp)
        C = deformer(mail_c, sq, pose, corp)
        g = np.linalg.norm(bary(V, C) - B0, axis=1) * tailles
        lignes.append((nom, float(g.mean()), float(np.percentile(g, 95)),
                       float(g.max())))
    return lignes


if __name__ == '__main__':
    base = '/root/medtra-avatar/createur/'
    sq = json.load(open(base + 'squelette.json', encoding='utf-8'))
    poses = [
        ('jambes écartées 9°', {'hanche_genou_gauche': 9,
                                'hanche_genou_droite': -9}, 1.0),
        ('genou plié 20°', {'genou_cheville_gauche': 20}, 1.0),
        ('corpulence 1,22', {}, 1.22),
        ('corpulence 0,88', {}, 0.88),
    ]
    for lib, mv, mc in [
            ('SANS weld  (grille régulière)', 'maillage-cargo.json',
             'maillage-corps.json'),
            ('AVEC weld  (contour + Delaunay)', 'maillage2-cargo.json',
             'maillage2-corps.json')]:
        try:
            v = json.load(open(base + mv, encoding='utf-8'))
            c = json.load(open(base + mc, encoding='utf-8'))
        except FileNotFoundError:
            continue
        print(f'\n  ── {lib} ──   {len(v["sommets"])} sommets de vêtement')
        print(f'  {"pose":>22} {"glissement moyen":>18} {"P95":>8} {"max":>8}')
        for nom, moy, p95, mx in glissement(v, c, sq, poses):
            etat = '✓' if p95 < 2.0 else '🔴'
            print(f'  {nom:>22} {moy:15.2f} px {p95:7.2f} {mx:8.2f}  {etat}')
