#!/usr/bin/env python3
"""REMETTRE UN PERSONNAGE GENERE EXACTEMENT AU GABARIT.

Med, 30 aout 2026 : « c'est quoi le plan pour bien assembler ? » — et, plus tot,
la regle qui commande tout : « H et F doivent etre cadres pareil, sinon la base
n'est pas bonne. »

🔴 LE PROBLEME, MESURE. Deux generations du MEME personnage ne sortent pas
cadrees pareil. Mesure du 30 aout, meme prompt, meme reference :

    reference   : sommet 9,1 %  pieds 95,2 %  largeur 56 %
    tenue hiver : sommet 7,2 %  pieds 99,2 %  largeur 82 %

Empiler des calques la-dessus est impossible : un chapeau pose pour le premier
cadrage flotte a 20 cm du crane sur le second. On ne DEMANDE donc pas au
generateur d'etre precis — on MESURE ce qu'il rend et on le replace.

⭐ CE N'EST PAS UNE RETOUCHE. On ne change aucun pixel du personnage : on le
met a l'echelle et on le recentre dans un canevas. C'est une operation d'usine,
identique pour tous les assets, avec temoins chiffres — la distinction posee
dans [[feedback-ne-pas-jouer-avec-les-images]].

🔴 L'INSTRUMENT AVANT LE SUJET. Le fond « blanc » de Higgsfield vaut 237 a 247,
jamais 255. Un seuil a 247 classe donc TOUT le fond comme du sujet : le premier
temoin ecrit ce soir rendait « 100 % au bord bas » sur la reference elle-meme,
qui est pourtant complete. Deux entrees differentes, la meme valeur = le seuil
est faux. Il est mesure ici, pas devine.
"""
import sys, os
import numpy as np
from PIL import Image

# ── LE GABARIT. Mesure sur la reference validee par Med, pas choisi. ──
CANEVAS = (1536, 2752)   # comme reference-neutre.png
SOMMET = 0.091           # le crane commence a 9,1 % de la hauteur
PIEDS = 0.952            # les pieds finissent a 95,2 %
TOLERANCE = 0.015        # 1,5 point de pourcentage


def masque_sujet(a: np.ndarray) -> np.ndarray:
    """Sujet = ce qui n'est ni clair ni neutre.

    ⚠️ Sur une image AVEC canal alpha (apres remove_background), l'alpha est la
    verite et on ne devine rien du tout.
    """
    # ⚠️ Un canal alpha PRESENT n'est pas un canal alpha UTILE. `convert('RGBA')`
    # en fabrique un, opaque partout, sur une image qui n'en avait pas : le lire
    # revient a declarer toute l'image « sujet », fond compris. Mesure du
    # 30 aout — la reference ressortait « largeur 86 % » au lieu de 56 %, et le
    # temoin ne l'a pas vu parce qu'il partageait le defaut. L'alpha ne compte
    # que s'il DECOUPE vraiment quelque chose.
    if a.shape[2] == 4 and a[:, :, 3].min() < 250:
        return a[:, :, 3] > 16
    clair = (a[:, :, :3] >= 225).all(2)
    neutre = (a[:, :, :3].max(2) - a[:, :, :3].min(2)) <= 14
    return ~(clair & neutre)


def boite(m: np.ndarray):
    ys, xs = np.where(m)
    if len(ys) == 0:
        raise SystemExit('AUCUN SUJET DETECTE — on ne normalise pas a l aveugle')
    return xs.min(), ys.min(), xs.max(), ys.max()


def mesurer(chemin: str, etiquette: str):
    im = Image.open(chemin)
    a = np.asarray(im.convert('RGBA')).astype(int)
    h, w = a.shape[:2]
    m = masque_sujet(a)
    x0, y0, x1, y1 = boite(m)
    bas = m[-1].sum() / w * 100
    print(f'  {etiquette:12} {w}x{h}  sommet {y0/h*100:5.1f} %  pieds {y1/h*100:5.1f} %  '
          f'largeur {(x1-x0)/w*100:3.0f} %  bord bas {bas:4.1f} %')
    return im, (x0, y0, x1, y1), (w, h)


def normaliser(source: str, dest: str):
    print(f'AVANT :')
    im, (x0, y0, x1, y1), (w, h) = mesurer(source, 'source')

    # hauteur visee, en pixels du canevas cible
    CW, CH = CANEVAS
    haut_cible = (PIEDS - SOMMET) * CH
    haut_actuelle = (y1 - y0 + 1)
    facteur = haut_cible / haut_actuelle

    # 🔴 GARDE-FOU : un facteur absurde arrete tout. Med, 17 aout : appliquer
    # 4,83 « pour reduire » sans s arreter sur l absurdite.
    if not (0.15 <= facteur <= 8.0):
        raise SystemExit(f'FACTEUR ABSURDE ({facteur:.2f}) — on ne livre pas ca')

    im = im.convert('RGBA')
    decoupe = im.crop((x0, y0, x1 + 1, y1 + 1))
    nl = max(1, int(round(decoupe.width * facteur)))
    nh = max(1, int(round(decoupe.height * facteur)))
    decoupe = decoupe.resize((nl, nh), Image.LANCZOS)

    if nl > CW:
        raise SystemExit(f'le personnage mis a l echelle ({nl} px) deborde du canevas ({CW} px)')

    canevas = Image.new('RGBA', CANEVAS, (0, 0, 0, 0))
    canevas.paste(decoupe, ((CW - nl) // 2, int(round(SOMMET * CH))), decoupe)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    canevas.save(dest)

    # ── TEMOIN : on RE-MESURE le resultat. Un traitement qui ne verifie pas sa
    #    propre sortie est indiscernable d un traitement casse. ──
    print(f'APRES :')
    _, (a0, b0, a1, b1), (rw, rh) = mesurer(dest, 'resultat')
    s, p = b0 / rh, b1 / rh
    ok_s = abs(s - SOMMET) <= TOLERANCE
    ok_p = abs(p - PIEDS) <= TOLERANCE
    centre = ((a0 + a1) / 2) / rw
    ok_c = abs(centre - 0.5) <= 0.02
    print(f'\n  sommet vise {SOMMET*100:.1f} %  obtenu {s*100:.1f} %  -> {"OK" if ok_s else "ECHEC"}')
    print(f'  pieds  vise {PIEDS*100:.1f} %  obtenu {p*100:.1f} %  -> {"OK" if ok_p else "ECHEC"}')
    print(f'  centre vise 50.0 %  obtenu {centre*100:.1f} %  -> {"OK" if ok_c else "ECHEC"}')
    if not (ok_s and ok_p and ok_c):
        sys.exit(1)
    print(f'\nECRIT : {dest}')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit('usage : normaliser.py <source> <destination>')
    normaliser(sys.argv[1], sys.argv[2])
