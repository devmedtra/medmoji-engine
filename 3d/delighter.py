#!/usr/bin/env python3
"""DÉLIGHTER L'ENTRÉE — retirer l'ombre que la reconstruction prendrait pour du relief.

Med, 30 août 2026 : « fix la ligne au genou ».

🔴 CE N'EST PAS UN DÉFAUT DE GÉOMÉTRIE, et trois tentatives de le traiter comme
tel ont échoué :

  · lissage général            le Shrinkwrap le défaisait
  · lissage local au genou     1 pixel changé sur 4,2 millions
  · genou exempté du Shrinkwrap  16,6 → 21,0, et une ligne NOUVELLE à la
                                 frontière du groupe

La mesure disait déjà pourquoi : le rayon de la jambe décroît régulièrement de
0,0413 à 0,0326, **sans saut**. Il n'y a pas d'arête à lisser.

───────────────────────────────────────────────────────────────────────────────
  CE QUE C'EST VRAIMENT
───────────────────────────────────────────────────────────────────────────────
« Light information should be extracted from the processed texture; otherwise
 you'll have baked-in lighting and shadows on your assets. »

Une reconstruction image → 3D lit une ombre comme du relief. Mesuré sur le
profil de luminance de la jambe, image d'entrée :

    79,5 %   +3,1    une bosse de lumière
    81,5 %   −4,9    LE CREUX D'OMBRE du genou
    82,0 %   −1,6

Huit unités sur 255 — 3 % — que TRELLIS rend en une démarcation de 10,5 dans le
maillage, huit fois l'amplitude d'origine (le corps 2D, lui, ne montre que 1,3).

⭐ On retire donc l'ombre AVANT la reconstruction. C'est le même geste que
l'effacement du sous-vêtement dans l'init de SDXL : on ne retouche pas un
rendu, on nettoie l'ENTRÉE d'un pipeline — et la sortie n'en devient que plus
fidèle au corps réel.

    python3 delighter.py entree.png sortie.png [debut%] [fin%]
"""
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

# La bande à traiter, en fraction de la hauteur du personnage. Elle encadre le
# genou mesuré à 80,3 % et la cheville à 91,7 % : on s'arrête bien avant.
DEBUT, FIN = 0.76, 0.86
# La portée du « fond » : la luminance lissée sur cette hauteur sert de
# référence. Assez large pour ignorer l'ombre du genou, assez étroite pour
# suivre le dégradé naturel de la jambe.
PORTEE = 0.055


def segments(ligne):
    x = np.where(ligne)[0]
    if not len(x):
        return []
    return [(int(g[0]), int(g[-1]))
            for g in np.split(x, np.where(np.diff(x) > 1)[0] + 1)]


def delighter(chemin, dest, debut=DEBUT, fin=FIN):
    im = Image.open(chemin).convert('RGBA')
    a = np.asarray(im).astype(np.float64).copy()
    corps = a[:, :, 3] > 250
    ys = np.where(a[:, :, 3] > 16)[0] if (a[:, :, 3] > 16).any() else None
    yy = np.where((a[:, :, 3] > 16).any(1))[0]
    h0, h1 = yy.min(), yy.max()
    Hp = h1 - h0
    portee = max(5, int(PORTEE * Hp))

    # ── LE PROFIL, PAR JAMBE ─────────────────────────────────────────────
    # ⚠️ Chaque jambe a son propre éclairage : les moyenner effacerait le
    # dégradé latéral et créerait une correction fausse d'un côté.
    y0, y1 = h0 + int(Hp * debut), h0 + int(Hp * fin)
    marge = portee
    lignes = {}
    for y in range(max(h0, y0 - marge), min(h1, y1 + marge)):
        s = [t for t in segments(corps[y]) if t[1] - t[0] > 15]
        if len(s) == 2:
            lignes[y] = s
    if not lignes:
        raise SystemExit('  🔴 aucune ligne à deux jambes dans la bande')

    lum = a[:, :, :3] @ [0.2126, 0.7152, 0.0722]
    total = 0.0
    n_px = 0
    for cote in (0, 1):
        ys_ = sorted(lignes)
        prof = np.array([lum[y, lignes[y][cote][0]:lignes[y][cote][1] + 1].mean()
                         for y in ys_])
        fond = ndimage.uniform_filter1d(prof, size=portee, mode='nearest')
        residu = prof - fond
        for k, y in enumerate(ys_):
            if not (y0 <= y <= y1):
                continue
            d, e = lignes[y][cote]
            # fondu aux bords de la bande : une correction à bord franc
            # produirait à son tour une démarcation
            t = (y - y0) / max(1, y1 - y0)
            att = np.sin(np.pi * t)
            corr = -residu[k] * att
            a[y, d:e + 1, :3] = np.clip(a[y, d:e + 1, :3] + corr, 0, 255)
            total += abs(corr) * (e - d + 1)
            n_px += e - d + 1

    Image.fromarray(a.round().astype(np.uint8)).save(dest)

    # ── TÉMOINS ──────────────────────────────────────────────────────────
    b = np.asarray(Image.open(dest).convert('RGBA')).astype(np.float64)
    hors = ~np.zeros(corps.shape, bool)
    hors[y0:y1 + 1] = False
    ecart_hors = np.abs(a[:, :, :3] - b[:, :, :3]).max()
    print(f'  bande traitée : {debut*100:.0f} → {fin*100:.0f} % '
          f'({n_px:,} px), correction moyenne {total/max(1,n_px):.2f}/255')
    orig = np.asarray(Image.open(chemin).convert('RGBA')).astype(np.float64)
    d_hors = np.abs(orig[:, :, :3] - b[:, :, :3]).max(2)
    m_hors = np.ones(corps.shape, bool)
    m_hors[y0:y1 + 1] = False
    print(f'  hors de la bande : {d_hors[m_hors].max():.0f}/255 '
          f'{"✓ intact" if d_hors[m_hors].max() < 1 else "🔴 MODIFIÉ"}')
    print(f'  → {dest}')


if __name__ == '__main__':
    delighter(sys.argv[1] if len(sys.argv) > 1
              else '/root/medtra-avatar/createur/sortie/tenues/base.norm.png',
              sys.argv[2] if len(sys.argv) > 2
              else '/root/medtra-avatar/createur/sortie/3d/corps-delighte.png',
              float(sys.argv[3]) if len(sys.argv) > 3 else DEBUT,
              float(sys.argv[4]) if len(sys.argv) > 4 else FIN)
