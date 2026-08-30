#!/usr/bin/env python3
# teinte.py — LA transformation de peau/couleur du créateur de medmoji.
#
# ⭐ La règle de Med (29 août 2026) : « tout doit s'assembler parfaitement même
# si la couleur de peau est pâle ou foncée. » Comment c'est garanti ICI :
# la MÊME fonction `teinter` s'applique à la base ET à chaque calque de trait ;
# le poids de teinte est une fonction du PIXEL (continue), donc au bord d'un
# calque — où le calque reproduit la base — les deux chemins donnent la même
# couleur. Témoin mesuré (zone nez, ton foncé) : teinter(composer) vs
# composer(teinter) → diff moyen 0,03 / max 1 sur 255. Zéro couture.
#
# 🔴 CE QUI A ÉCHOUÉ AVANT (ne pas y revenir) :
# - masque par distance chromatique étroite (0,10) : les hautes lumières du
#   front sortent du masque → taches claires sur peau foncée ;
# - masque binaire spatial par seuillage (dist<0,055 + fermeture 31 px) :
#   vitiligo — plaques non teintées partout ;
# - et un témoin qui mesurait une bande HORS de la zone du calque : 0,00 par
#   construction, il ne prouvait rien. Mesurer DANS la zone.
#
# ⭐ LES SEUILS SONT MESURÉS sur la sonde F (percentiles réels), pas devinés :
#   peau (front, highlight inclus) sat ≥ 0,31 · t-shirt méd 0,055 mais PLIS
#   jusqu'à 0,39 (d'où la coupe géométrique y<1930 pour la base) · poil de
#   sourcil val 0,24-0,45 (reflets 0,79) · fond sat 0. Pour une autre tête de
#   référence, RE-MESURER avant de toucher aux seuils.
#
# Modes par famille de calque :
#   peau-pure  nez, menton, détails-peau (delta) — ratio partout ;
#   oreilles   idem mais préserve la marge de fond blanc du cover ;
#   yeux       préserve sclère/iris gris/pupille/cils (le gris de l'iris se
#              teinte ENSUITE par la couleur d'yeux choisie, autre fonction) ;
#   sourcils   préserve le poil (teinte-cheveux indépendante ensuite) ;
#   bouche     préserve les dents ;
#   base       peau seule : coupe t-shirt par y, fond exclu, et dilatation du
#              poids (MaxFilter 19) pour que l'anti-aliasing du contour et le
#              rim-light suivent la peau voisine au lieu de rester clairs.
#
# L'appelant fournit l'ORIGINE du calque dans le repère de la tête : le poids
# « base » dépend de y (coupe t-shirt), le reste est purement pixel.

from PIL import Image, ImageFilter
import numpy as np


def _smooth(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def canon_de(base_rgb):
    """Ton canonique = médiane du front (zone mesurée sonde F)."""
    a = np.asarray(base_rgb.convert('RGB')).astype(np.float32)
    return np.median(a[500:620, 750:1050].reshape(-1, 3), axis=0)


def _poids(a, mode, origine):
    rgb = a[..., :3]
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-4), 0)
    val = mx
    fond = (val > 0.975) & (sat < 0.06)
    if mode == 'peau-pure':
        p = np.ones_like(val)
    elif mode == 'oreilles':
        p = 1 - _smooth(val, 0.96, 0.99) * _smooth(0.08 - sat, 0.02, 0.06)
    elif mode == 'yeux':
        p = _smooth(sat, 0.14, 0.26) * _smooth(val, 0.26, 0.36)
    elif mode == 'sourcils':
        p = _smooth(val, 0.48, 0.58)
    elif mode == 'bouche':
        p = _smooth(sat, 0.12, 0.22)
    elif mode == 'base':
        h, w = val.shape
        y = np.arange(h)[:, None].astype(np.float32) + origine[1]
        coupe = (1 - _smooth(y, 1900, 1930)) * np.ones((1, w), np.float32)
        p = _smooth(sat, 0.10, 0.18) * coupe
        pi = Image.fromarray((p * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(19))
        pd = np.asarray(pi).astype(np.float32) / 255.0
        p = np.maximum(p, pd * 0.92 * coupe)
        # ⚠️ Vu par Med (30 août, cercles rouges sur les coins d'yeux) : la
        # dilatation ci-dessus déborde DANS l'œil — sur les tons pâles, les
        # coins de sclère se délavent en triangles blancs crus. L'œil reprend
        # un poids nul : sclère/iris (désaturés clairs) et pupille/cils
        # (sombres) gardent leur couleur d'origine, exactement comme le mode
        # 'yeux' les préserve sur les calques.
        oeil = np.maximum(_smooth(0.14 - sat, 0.0, 0.05) * _smooth(val, 0.45, 0.60),
                          _smooth(0.30 - val, 0.0, 0.08))
        p = p * (1 - oeil)
    elif mode == 'corps':
        # ⭐ CORPS ENTIER — comme 'base', SANS la coupe verticale.
        # 🔴 Piege mesure le 30 aout 2026 : la coupe de 'base' (y 1900-1930) est
        # calibree pour un buste de 2400 px de haut ; appliquee a un personnage
        # plein-pied de 2752 px, elle annulait la teinte a partir des cuisses.
        # Temoin : ecart moyen t0 vs t7 de 27 sur les bras, mais 0,1 sur les
        # tibias et 0,2 sur les pieds — un t7 aux jambes beiges, et rien dans
        # aucun journal. Ici, aucune coupe : la SATURATION seule decide, ce qui
        # laisse le t-shirt gris et le short noir intacts (mesure : ecart < 1).
        p = _smooth(sat, 0.10, 0.18)
        pi = Image.fromarray((p * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(19))
        pd = np.asarray(pi).astype(np.float32) / 255.0
        p = np.maximum(p, pd * 0.92)
        oeil = np.maximum(_smooth(0.14 - sat, 0.0, 0.05) * _smooth(val, 0.45, 0.60),
                          _smooth(0.30 - val, 0.0, 0.08))
        p = p * (1 - oeil)
    else:
        raise ValueError(mode)
    return np.where(fond, 0, p)[..., None]


def teinter(im, canon, cible, mode, origine=(0, 0)):
    """Ratio multiplicatif en gamma-linéaire, pondéré par _poids(mode)."""
    a = np.asarray(im.convert('RGBA')).astype(np.float32) / 255.0
    p = _poids(a, mode, origine)
    lin = np.power(a[..., :3], 2.2)
    r = (np.power(np.asarray(cible, np.float32) / 255.0, 2.2)
         / np.maximum(np.power(np.asarray(canon, np.float32) / 255.0, 2.2), 1e-4))
    a[..., :3] = np.clip(np.power(np.clip(lin * (1 + p * (r - 1)), 0, 4), 1 / 2.2), 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8))
