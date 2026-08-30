#!/usr/bin/env python3
"""L'OMBRE DE CONTACT — ce qui separe un vetement PORTE d'un vetement COLLE.

Med, 30 aout 2026, en m'envoyant l'architecture d'habillage : « Pour qu'un
vetement semble reellement porte et non collé sur l'image » — l'ombre du col sur
le cou, celle de la manche sur le bras, celle de l'ourlet sur la chaussure.

⭐ POURQUOI CA MARCHE. Un calque pose sur un autre n'echange aucune lumiere avec
lui : la frontiere est nette, et l'oeil lit « autocollant ». Dans le monde reel,
un vetement BLOQUE la lumiere ambiante juste sous son bord — c'est l'occlusion
ambiante, le procede n°1 des moteurs 3D pour donner du poids a un objet.

🔴 CE N'EST PAS UNE RETOUCHE. Aucun pixel n'est invente ni juge a l'oeil : on
derive l'ombre du MASQUE de la piece, mecaniquement, et la meme operation
s'applique a toutes les pieces du catalogue avec les memes parametres. C'est
une operation d'usine, avec ses temoins — la distinction posee dans
[[feedback-ne-pas-jouer-avec-les-images]].

Deux ombres, et elles ne font pas le meme travail :
  · SOUS le vetement, sur la PEAU  — le col qui assombrit le cou
  · SOUS le vetement, sur LUI-MEME — l'ourlet qui marque le pli
"""
import sys
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


def ombre_de_contact(chemin_avant, chemin_apres, dest,
                     force=0.42, flou=14, decalage=7):
    """Assombrit la peau juste sous le bord du vetement.

    force     : opacite maximale de l'ombre (0 = rien, 1 = noir)
    flou      : etalement en pixels — une ombre de contact est SERREE
    decalage  : vers le bas, car la lumiere vient du haut a gauche
    """
    avant = Image.open(chemin_avant).convert('RGBA')   # corps nu
    apres = Image.open(chemin_apres).convert('RGBA')   # corps habille
    if avant.size != apres.size:
        sys.exit(f'tailles differentes : {avant.size} vs {apres.size}')
    W, H = avant.size

    a_av = np.asarray(avant).astype(float)
    a_ap = np.asarray(apres).astype(float)

    # ── LE MASQUE DU VETEMENT : ce qui a change entre les deux images.
    #    On ne le devine pas, on le DEDUIT — et il est donc exact par
    #    construction, contrairement a une detection par couleur. ──
    ecart = np.abs(a_av[:, :, :3] - a_ap[:, :, :3]).max(2)
    vetement = (ecart > 18) & (a_ap[:, :, 3] > 200)
    vetement = ndimage.binary_closing(vetement, np.ones((9, 9)))
    vetement = ndimage.binary_opening(vetement, np.ones((5, 5)))

    peau = (a_ap[:, :, 3] > 200) & ~vetement
    print(f'  vetement : {vetement.sum():,} px    peau visible : {peau.sum():,} px')
    if vetement.sum() < 10_000:
        sys.exit('ABSURDE : moins de 10 000 px de vetement detectes')

    # ── L'OMBRE. Le masque du vetement, decale vers le bas et floue, ne garde
    #    que ce qui tombe SUR LA PEAU. Une ombre qui deborderait sur le fond
    #    ferait un halo — le defaut le plus visible de ce genre d'effet. ──
    m = Image.fromarray((vetement * 255).astype(np.uint8))
    m = m.transform(m.size, Image.AFFINE, (1, 0, 0, 0, 1, -decalage),
                    resample=Image.NEAREST, fillcolor=0)
    m = m.filter(ImageFilter.GaussianBlur(flou))
    om = np.asarray(m).astype(float) / 255.0
    om = om * peau                       # jamais sur le vetement ni sur le fond

    res = a_ap.copy()
    res[:, :, :3] *= (1 - om[:, :, None] * force)
    Image.fromarray(res.round().astype(np.uint8)).save(dest)

    # ── TEMOINS ──
    touche = om > 0.02
    print(f'\n  peau assombrie : {touche.sum():,} px ({touche.sum()/max(1,peau.sum())*100:.1f} % de la peau)')
    if touche.sum():
        av = a_ap[:, :, :3][touche].mean()
        ap = res[:, :, :3][touche].mean()
        print(f'  luminance moyenne de la zone : {av:.0f} -> {ap:.0f}  ({ap-av:+.0f})')
    # 🔴 GARDE : l'ombre ne doit JAMAIS toucher le vetement ni le fond.
    fuite_vet = (om * vetement).sum()
    fuite_fond = (om * (a_ap[:, :, 3] <= 200)).sum()
    print(f'  fuite sur le vetement : {fuite_vet:.0f}   sur le fond : {fuite_fond:.0f}'
          f'   -> {"PROPRE" if fuite_vet + fuite_fond == 0 else "FUITE"}')
    print(f'\nECRIT : {dest}')


if __name__ == '__main__':
    if len(sys.argv) < 4:
        sys.exit('usage : ombre-contact.py <corps-nu> <corps-habille> <destination>')
    ombre_de_contact(*sys.argv[1:4])
