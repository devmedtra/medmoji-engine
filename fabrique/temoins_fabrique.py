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
# 🔴 UNE BANDE HORIZONTALE N'EST PAS UNE ZONE ANATOMIQUE.
# Premiere version : « les mains » etait la bande 58-70 % de la hauteur. Or a
# ces hauteurs se trouvent les mains ET les hanches et le haut des cuisses, ou
# le pantalon doit evidemment etre. Le temoin comptait donc le pantalon comme
# un empietement, et rendait 73,8 % puis 73,0 % sur deux images DIFFERENTES.
# Deux valeurs quasi identiques sur deux entrees differentes : l instrument
# etait casse, pas le sujet.
INTERDITS = {
    'haut':  [('le visage', 0.00, 0.28), ('les jambes', 0.72, 1.00)],
    'bas':   [('le visage', 0.00, 0.28), ('le torse', 0.30, 0.50),
              ('les pieds', 0.94, 1.00)],
    'pieds': [('le visage', 0.00, 0.28), ('le torse', 0.30, 0.50),
              ('les cuisses', 0.55, 0.85)],
}

# Les membres se prennent dans leur masque GEOMETRIQUE exact, jamais dans une
# tranche de hauteur. Calcule une fois, le corps de base ne bougeant jamais.
MASQUE_MEMBRES = '/root/medtra-avatar/createur/masque-membres.png'


def verifier(chemin_base, chemin_produit, zone, masque_vetement=None,
             tolerance=4):
    """Rend (nb_defauts, rapport). Ne juge pas l'esthétique — seulement ce qui
    a changé sans y être autorisé."""
    base = np.asarray(Image.open(chemin_base).convert('RGBA')).astype(float)
    prod = np.asarray(Image.open(chemin_produit).convert('RGBA')).astype(float)
    if base.shape != prod.shape:
        return 1, [f'🔴 tailles différentes : {base.shape} vs {prod.shape}']

    # 🔴 COMPARER DEUX IMAGES DANS LE MÊME ESPACE. La fabrique travaille sur
    # l'original COMPOSÉ SUR BLANC ; ce témoin lisait le RGB brut, non composé.
    # Sur les pixels d'alpha 251-254 — la frange d'anti-crénelage, 125 784 px
    # ici — les deux diffèrent mécaniquement de 4/255. Le témoin rapportait
    # donc un écart de 4 sur des pixels recopiés à l'IDENTIQUE, à un cheveu de
    # sa propre tolérance (4). Composés du même côté : 0/255, exactement.
    def _sur_blanc(x):
        a = (x[:, :, 3:4] / 255.0)
        return x[:, :, :3] * a + 255.0 * (1 - a)
    base = np.dstack([_sur_blanc(base), base[:, :, 3]])
    prod = np.dstack([_sur_blanc(prod), prod[:, :, 3]])

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

    # ── 2bis. LES MEMBRES, DANS LEUR FORME EXACTE ─────────────────────────
    # C est ce controle qui attrape les MOUFLES : du tissu peint AUTOUR des
    # mains, hors d elles, donc invisible a un temoin qui ne regarde que les
    # pixels des mains eux-memes.
    import os as _os
    if _os.path.exists(MASQUE_MEMBRES):
        membres = np.asarray(Image.open(MASQUE_MEMBRES).convert('L')) > 127
        membres = membres & corps
        proche = ndimage.binary_dilation(membres, np.ones((15, 15))) & ~membres
        sur = int((vet & membres).sum())
        autour = int((vet & proche).sum())
        rapport.append(f'membres : {membres.sum():,} px, vetement dessus '
                       f'{sur:,}, autour {autour:,}')
        if sur > membres.sum() * 0.03:
            defauts += 1
            rapport.append(f'🔴 le vetement RECOUVRE les membres : '
                           f'{sur/max(1,membres.sum())*100:.1f} %')
        if autour > proche.sum() * 0.25:
            defauts += 1
            rapport.append(f'🔴 le vetement CERNE les membres (moufles) : '
                           f'{autour/max(1,proche.sum())*100:.1f} % du pourtour')

    # ── 2ter. LE VÊTEMENT NE DOIT PAS ÊTRE DÉCHIRÉ ────────────────────────
    # 🔴 CE TÉMOIN A MANQUÉ TROIS FOIS. Le pantalon était visiblement coupé en
    # deux, et deux instruments successifs ont répondu « rien à signaler » :
    #
    #   · composantes connexes du masque → 1 seule, dans les trois essais ;
    #   · « lignes couvertes à plus de 25 % » → 31 sur 31.
    #
    # Le premier ne voit rien parce que les deux moitiés se rejoignent par les
    # côtés. Le second parce qu'il divisait la LARGEUR DU VÊTEMENT par celle du
    # corps : un pantalon déborde de la jambe nue, d'où une « couverture » de
    # 134 % — valeur absurde qui aurait dû m'arrêter net.
    #
    # ⭐ Ce qui se mesure ici est la seule chose que l'œil voyait : une bande
    # de PEAU NUE à l'intérieur de la silhouette, sous la taille. Le ratio est
    # borné au corps, donc majoré par 100 % — l'assertion le garantit.
    if zone == 'bas':
        d0, f0 = 0.56, 0.92
        prof = [(y, int((vet & corps)[y].sum()) / int(corps[y].sum()) * 100)
                for y in range(h0 + int(Hp * d0), h0 + int(Hp * f0))
                if corps[y].sum() > 20]
        if prof:
            pire = max(v for _, v in prof)
            assert pire <= 100.001, f'instrument cassé : couverture {pire:.0f} %'
            nues = [y for y, v in prof if v < 50]
            rapport.append(f'jambes : couverture médiane '
                           f'{np.median([v for _, v in prof]):.0f} %, '
                           f'{len(nues)} ligne(s) sous 50 %')
            if len(nues) > 12:      # ~0,5 % de la hauteur : au-delà, ça se voit
                defauts += 1
                a, b = (nues[0] - h0) / Hp * 100, (nues[-1] - h0) / Hp * 100
                rapport.append(f'🔴 le vêtement est DÉCHIRÉ : bande de peau nue '
                               f'de {a:.1f} % à {b:.1f} % ({len(nues)} lignes)')

    # ── 2quater. LE VÊTEMENT DOIT ÊTRE FERMÉ ──────────────────────────────
    # 🔴 TOUTE L'INSTRUMENTATION ÉTAIT ORIENTÉE EN LIGNES. Le conseil d'IA,
    # 30 août : les deux membres consultés ont donné des verdicts opposés sur
    # « le pantalon est-il continu ? ». L'un voyait un vêtement d'un seul
    # tenant, l'autre une bande verticale claire de la ceinture au bas-ventre —
    # une braguette ouverte. Aucun témoin ne pouvait les départager : un défaut
    # VERTICAL est invisible à un profil calculé ligne par ligne.
    #
    # ⭐ Le témoin symétrique, en colonnes, sur la tranche de la taille.
    if zone == 'bas':
        d1, f1 = 0.56, 0.70
        y0, y1 = h0 + int(Hp * d1), h0 + int(Hp * f1)
        colv = (vet & corps)[y0:y1].sum(0)
        colc = corps[y0:y1].sum(0)
        xs = np.where(colc > (y1 - y0) * 0.5)[0]     # colonnes vraiment dans le corps
        if len(xs) > 20:
            p = colv[xs] / colc[xs] * 100
            assert p.max() <= 100.001, f'instrument cassé : colonne {p.max():.0f} %'
            creuses = int((p < 50).sum())
            rapport.append(f'taille : {len(xs)} colonnes, médiane '
                           f'{np.median(p):.0f} %, {creuses} sous 50 %')
            # une ouverture MÉDIANE (braguette) plutôt qu'un bord : au centre
            centre = xs[(xs > np.percentile(xs, 35)) & (xs < np.percentile(xs, 65))]
            if len(centre):
                pc = colv[centre] / colc[centre] * 100
                if (pc < 50).sum() > len(centre) * 0.15:
                    defauts += 1
                    rapport.append(f'🔴 le vêtement est OUVERT au centre : '
                                   f'{int((pc < 50).sum())}/{len(centre)} colonnes '
                                   f'médianes sous 50 % (braguette / entrebâillement)')

    # ── 3. COUVERTURE ATTENDUE ────────────────────────────────────────────
    # Un pantalon qui s'arrête à mi-mollet est un défaut, pas un style.
    if zone == 'bas':
        z = np.zeros(corps.shape, bool)
        z[h0 + int(Hp * .85):h0 + int(Hp * .92)] = True
        # ⚠️ borné au corps : un pantalon est plus large que la jambe nue, donc
        # (vet & z) / (z & corps) dépasse 100 % — c'est le bug qui rendait
        # « chevilles couvertes : 145 % ».
        couvert = (vet & corps & z).sum() / max(1, (z & corps).sum()) * 100
        rapport.append(f'chevilles couvertes : {couvert:.0f} %')
        if couvert < 40:
            defauts += 1
            rapport.append('🔴 le bas ne descend pas aux chevilles')

    # ── 3bis. LE CONTRAT DE LIVRAISON ─────────────────────────────────────
    # 🔴 « ASSEZ PROPRE » N'EST PAS UN CRITÈRE. Le conseil d'IA, 30 août, a
    # validé l'architecture à l'unanimité et refusé le ship à l'unanimité —
    # sur un seul chiffre : une tache de 543 px, « ≈ disque Ø 26 px, visible à
    # 1× sur un téléphone 6 pouces ». Les seuils ci-dessous sont les leurs, pas
    # les miens. Ils remplacent mon jugement par un contrat.
    #
    # ⭐ ET ILS SE NORMALISENT. Une aire en pixels absolus ne veut rien dire
    # d'une résolution à l'autre : le même défaut perceptuel vaut 80 px à
    # 1024 de haut et 578 px à 2752. Les seuils sont donc exprimés pour un
    # personnage de 1024 px et mis à l'échelle en (Hp/1024)².
    # 🔴 LE COMPTEUR D'ÉCLATS PARASITES A ÉTÉ RETIRÉ DE CE CONTRAT.
    # Trois définitions successives du même défaut, sur la MÊME image :
    #
    #     « hors silhouette, désaturé, à moins de 90 px d'un membre »   1 602 px
    #     la même, portée normalisée en (Hp/1024)                       4 237 px
    #     « hors silhouette et hors du masque sémantique de SAM »       14 919 px
    #
    # Facteur 9 entre le plus bas et le plus haut. Surligné en rouge sur
    # l'image, le compteur cerclait le bord latéral du pantalon, un passant,
    # l'entrejambe et un rabat de poche : des BORDS DE VÊTEMENT, éclairés donc
    # désaturés, que le critère de saturation ne distingue pas d'un parasite.
    # Autour de la main — le seul endroit qui comptait — il n'y avait plus rien.
    #
    # ⭐ Comme pour la distance au membre et la fraction de pourtour : les deux
    # populations ne se séparent pas, donc IL N'Y A PAS DE SEUIL À TROUVER.
    # Un garde-fou qui rend trois valeurs incompatibles pour un même défaut ne
    # protège rien — il fabrique de la confiance. On ne garde que ce qui mesure
    # une invariance vérifiable.
    ech = (Hp / 1024.0) ** 2
    if _os.path.exists(MASQUE_MEMBRES):
        mem = (np.asarray(Image.open(MASQUE_MEMBRES).convert('L')) > 127) & corps
        sur_mem = int((vet & mem).sum())
        rapport.append(f'── contrat de livraison (seuil du conseil, ×{ech:.1f} '
                       f'pour un personnage de {Hp} px) ──')
        ok = sur_mem < 50 * ech
        rapport.append(f'   {"✓" if ok else "🔴"} {"vêtement SUR les membres":28} '
                       f'{sur_mem:8,}  seuil {50 * ech:8,.0f}')
        if not ok:
            defauts += 1

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
