#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  MEDMOJI — LA FABRIQUE DE VÊTEMENTS
  Habiller un corps de base par inpainting, sans jamais le redessiner.
  Med + Claude, nuit du 29 au 30 août 2026.
═══════════════════════════════════════════════════════════════════════════════

USAGE
    python3 fabrique.py "<description du vêtement>" <nom> <zone> <r,g,b>

    zone : haut | bas | pieds
    ex.  python3 fabrique.py "a charcoal grey cotton hoodie with long sleeves" \
                             hoodie haut "74,78,84"

CE QUE ÇA PRODUIT
    <nom>.png          le personnage habillé, fond transparent
    <nom>.masque.png   le masque du vêtement — indispensable à la teinture
    <nom>.<couleur>.png une déclinaison par teinte demandée

───────────────────────────────────────────────────────────────────────────────
  POURQUOI L'INPAINTING ET PAS UNE GÉNÉRATION
───────────────────────────────────────────────────────────────────────────────
Un générateur d'images REDESSINE le personnage à chaque appel. Mesuré : deux
générations du même personnage ne se superposent qu'à 82 % (IoU), et le corps
déborde de 160 à 232 px aux jambes. Aucun calque extrait de l'une ne couvre
l'autre — c'est ce qui a fait échouer toute la première partie de la nuit.

L'inpainting ne repeint QUE l'intérieur du masque. Et on recolle explicitement
l'original ailleurs : l'écart hors zone est de 0 sur 255, par construction et
non par confiance.

  LICENCES, vérifiées le 30 août 2026 — ce point a écarté la solution évidente
    SDXL inpainting   CreativeML Open RAIL++-M   commercial AUTORISÉ
    diffusers 0.31    Apache 2.0
    CatVTON / IDM-VTON / OOTDiffusion : CC BY-NC-SA 4.0 → NON COMMERCIAL.
    Ils habillent pourtant mieux, et CatVTON tourne sur cette machine. Le
    blocage est juridique. Pour IDM-VTON, même les IMAGES produites sont NC.

───────────────────────────────────────────────────────────────────────────────
  LES SIX DÉFAUTS CORRIGÉS, ET CE QUI LES A RÉVÉLÉS
───────────────────────────────────────────────────────────────────────────────
 1. RATIO DE TRAVAIL FAUX          trouvé par le conseil d'IA, vérifié
    Génération en 768x1024 (ratio 0,750) remontée en 1536x2752 (0,558) :
    +34,4 % d'étirement vertical. Les « manches trop courtes » venaient de là,
    pas du prompt. Aucun réglage de texte n'aurait corrigé ça.

 2. ALPHA FINAL QUI DÉCOUPE        trouvé par le conseil, vérifié dans le code
    Restituer l'alpha du corps NU à la fin découpe tout vêtement plus large que
    lui : la marge du masque était annulée à la dernière ligne. Aucun manteau
    ample ne pouvait exister. L'alpha est maintenant l'UNION corps ∪ vêtement.

 3. MASQUE DANS LE VISAGE          trouvé en MESURANT au lieu de deviner
    Seuil de 17 % choisi au jugé. MediaPipe donne les vrais repères :
        nez 19,9 %   bouche 22,7 %   MENTON 28,1 %
    Le masque commençait au-dessus du nez : le col ne pouvait que monter sur le
    visage. Le seuil se mesure désormais à chaque appel.

 4. MASQUE EN TRAPÈZE              vu par Med — « le vêtement est trop large »
    Je prenais le point le plus à gauche et le plus à droite de chaque ligne
    puis je remplissais TOUT entre les deux — donc le vide entre les bras et le
    torse. SDXL remplit exactement le masque qu'on lui donne : un masque
    trapézoïdal donne une cape.

 5. RAYON DE DILATATION 28× TROP GRAND   première correction ratée
    Corriger le point 4 par une dilatation de 22 % de la largeur du corps
    (83 px) rebouchait tous les creux — l'écart bras/torse mesure 5 px à
    l'aisselle et 115 px plus bas. Le masque restait plein, à ZÉRO creux.
    Le rayon se calibre maintenant sur la médiane des creux mesurés.
    ⭐ Un témoin compte les creux restants : sans lui, j'avais « corrigé » le
    code sans vérifier que la correction produisait l'effet voulu.

 6. ORDRE DES PASSES               vu par Med — bavure grise au cou
    L'ombre de contact assombrit la PEAU ; une détection de vêtement par
    différence calculée après elle classe cette peau comme du tissu, et la
    teinture la colore. Mesuré : 17 205 px de peau teints à tort.
    L'ordre est figé : masque → teinture → ombre. Jamais autrement.

 ⭐ ET LA DÉTECTION PAR SEUIL EST MORTE. Même dans le bon ordre, un seuil ne
    distingue pas l'ombre du menton sur le cou de l'ombre dans le col : sur
    805 583 px détectés, 433 789 n'étaient PAS du vêtement — plus de la moitié.
    Le masque vient d'un segmenteur sémantique, qui sait ce qu'EST un vêtement.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

# ── Ce qui dépend de la machine ────────────────────────────────────────────
BASE = os.environ.get('MEDMOJI_BASE', '/home/mederic/medmoji-fabrique/base.norm.png')
SORTIE = os.environ.get('MEDMOJI_SORTIE', '/home/mederic/medmoji-fabrique')
MODELE_VISAGE = f'{SORTIE}/face_landmarker.task'
MODELE_SDXL = 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1'
# Le masque des bras, calculé une fois par géométrie pure — le corps de base
# ne bouge jamais, donc c'est un fichier et non un calcul.
MASQUE_BRAS = f'{SORTIE}/masque-membres.png'
# Portée de l'exclusion douce autour des bras et des mains. Se règle par
# mesure : le témoin compte le pourtour des membres cerné par du tissu.
# Rayon d'exclusion autour des membres, en pixels. Calculé sur la distribution
# des distances, jamais choisi : les moufles sont à moins de 40 px des membres,
# le vêtement légitime commence à 48 px.
RAYON_MEMBRES = int(os.environ.get('MEDMOJI_RAYON', '40'))

# ── Le gabarit, mesuré sur le personnage neutre validé ─────────────────────
ZONES = {
    # (début, fin) en fraction de la hauteur DU PERSONNAGE, pas de l'image.
    # Le début du haut est MESURÉ au menton ; la fin est le POIGNET, mesuré
    # une fois sur ce personnage et figé — la détection automatique du poignet
    # tombait sur un minimum entre deux doigts (68,5 %), donc pire que la
    # constante. On mesure, on vérifie, puis on fige.
    #
    #   58 %  bras 96 px    59 %  bras  95 px   ← POIGNET, minimum
    #   60 %  bras 108 px   62 %  bras 126 px   ← la MAIN s'élargit
    #   64 %  bras  79 px, 5 segments           ← les DOIGTS
    #
    # 0,62 mordait sur la main. 0,59 s'arrête au poignet, main dégagée.
    'haut': (None, 0.59),
    'bas': (0.55, 0.93),
    'pieds': (0.88, 1.00),
}
GRAINE = 20260830


# ═══════════════════════════════════════════════════════════════════════════
#  1. LE MASQUE — la pièce la plus délicate de toute la chaîne
# ═══════════════════════════════════════════════════════════════════════════

def sur_blanc(im):
    """Une image RGBA convertie en RGB rend le transparent NOIR. Un premier
    essai sortait un personnage sur fond noir pour cette seule raison."""
    f = Image.new('RGB', im.size, (255, 255, 255))
    f.paste(im, (0, 0), im)
    return f


def menton(im, h0, Hp):
    """La fraction de hauteur où se trouve le menton — MESURÉE.

    Repli à 30 % si le détecteur échoue : sous le menton mesuré sur ce
    personnage (28,1 %), donc sans risque d'entrer dans le visage.
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        o = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODELE_VISAGE),
            num_faces=1)
        with vision.FaceLandmarker.create_from_options(o) as det:
            r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=np.asarray(im.convert('RGB'))))
        if not r.face_landmarks:
            print('  ⚠️ aucun visage détecté — repli à 30 %')
            return 0.30
        f = (r.face_landmarks[0][152].y * im.size[1] - h0) / Hp   # 152 = menton
        print(f'  menton mesuré : {f*100:.1f} % du personnage')
        return f
    except Exception as e:
        print(f'  ⚠️ mesure impossible ({type(e).__name__}) — repli à 30 %')
        return 0.30


def poignet(corps, h0, Hp):
    """La fraction de hauteur où se trouve le poignet — MESURÉE.

    🔴 POURQUOI ELLE NE PEUT PAS ÊTRE UNE CONSTANTE. Med, 30 août : « au niveau
    des poignets et des mains, c'est la zone de transition la plus critique ; la
    main doit rester 100 % dégagée ». Ma borne fixe de 62 % couvrait le poignet
    ET le début de la main. Mesure sur le personnage :

        45 % : bras 143 px      59 % : bras  95 px   ← POIGNET, minimum
        61 % : bras 120 px      63 % : bras 128 px   ← la MAIN s'élargit

    Le poignet est donc un MINIMUM LOCAL de la largeur du bras, entre le coude
    et la main. On le cherche, on ne le suppose pas. La manche s'arrête là, et
    la main reste sur le calque du corps nu.
    """
    largeurs = []
    for pct in range(40, 75):
        y = h0 + int(Hp * pct / 100)
        xs = np.where(corps[y])[0]
        if len(xs) < 2:
            continue
        c = np.where(np.diff(xs) > 1)[0]
        if len(c) < 2:                      # bras encore soudés au torse
            continue
        largeurs.append((pct, len(np.split(xs, c + 1)[0])))
    if len(largeurs) < 5:
        print('  ⚠️ poignet non mesurable — repli à 58 %')
        return 0.58
    pcts = [p for p, _ in largeurs]
    vals = [v for _, v in largeurs]
    i = int(np.argmin(vals))
    f = pcts[i] / 100.0
    print(f'  poignet mesuré : {f*100:.0f} % (bras {vals[i]} px, '
          f'contre {max(vals)} px au plus large)')
    return f


def masque_zone(im, zone='haut'):
    """La zone où le modèle a le droit de peindre.

    🔴 ELLE ÉPOUSE LE CORPS, elle ne remplit pas sa boîte englobante. Voir les
    défauts 4 et 5 en tête de fichier — c'est ici que se jouait la « cape ».
    """
    a = np.asarray(im.convert('RGBA'))
    corps = a[:, :, 3] > 16
    ys = np.where(corps.any(1))[0]
    h0, h1 = ys.min(), ys.max()
    Hp = h1 - h0

    d, f = ZONES[zone]
    if d is None:
        d = menton(im, h0, Hp) + 0.015          # sous le menton
    if f is None:
        f = poignet(corps, h0, Hp) - 0.005      # au-dessus du poignet
    if zone == 'bas':
        # ⭐ LA TAILLE SE MESURE SUR LE SOUS-VÊTEMENT, elle ne se fixe pas.
        # 0,55 était une constante ; le sous-vêtement commence à 0,541. Les
        # 22 px d'écart restaient gris en haut du pantalon — la bande que Med a
        # entourée en bleu le 30 août — faute d'être dans le masque.
        sv = sous_vetement(im)
        if sv.any():
            y = int(np.where(sv.any(1))[0].min())
            avant_d = d
            d = min(d, (y - h0) / Hp - 0.01)
            print(f'  taille mesurée sur le sous-vêtement : {d*100:.1f} % '
                  f'(constante : {avant_d*100:.1f} %)')
    haut, bas = h0 + int(Hp * d), h0 + int(Hp * f)

    # Le rayon se calibre sur les CREUX du corps, jamais sur sa largeur.
    # Sens physique : une manche remplit l'aisselle (creux étroit), mais le
    # vêtement ne doit pas franchir l'espace ouvert sous l'avant-bras.
    creux = []
    for y in range(haut, bas):
        xs = np.where(corps[y])[0]
        if len(xs) < 2:
            continue
        gaps = np.diff(xs) - 1
        creux.extend(gaps[gaps > 0])
    rayon = 25 if not creux else min(25, max(8, int(np.median(creux) / 2)))

    bande = np.zeros(corps.shape, bool)
    bande[haut:bas] = True
    zone_corps = corps & bande
    m = ndimage.binary_dilation(zone_corps,
                                ndimage.generate_binary_structure(2, 2),
                                iterations=rayon) & bande

    # ── 🔴 LA TOPOLOGIE DU MASQUE NE DOIT PAS OSCILLER ────────────────────
    # Med, 30 août 2026, a entouré trois fois la même couture horizontale en
    # travers des cuisses. Je l'avais expliquée — « c'est la fourche » — au lieu
    # de la tuer. Mesure du masque, ligne par ligne, en nombre de segments :
    #
    #     66,5 → 68,0 %   1 segment    l'entrejambe est comblée
    #     69,0 → 71,0 %   2 segments   les jambes sont séparées
    #     72,0 → 73,0 %   1 segment    ← il SE REFERME
    #     73,5 % et plus  2 segments   et se rouvre
    #
    # Le masque s'ouvre, se referme, se rouvre. SDXL peint une transition à
    # chaque changement — c'est-à-dire une couture, exactement à la hauteur
    # entourée. Ni le sous-vêtement (gradient de l'init ramené de 1,36 à 0,48)
    # ni le prompt (4,0 sans « cargo » contre 4,4 avec) n'en étaient la cause.
    #
    # ⭐ Un pantalon n'a qu'UNE fourche. Une fois le masque ouvert en deux
    # jambes, il ne doit plus jamais se refermer : on impose la monotonie, en
    # retirant sous la fourche le fond enfermé entre les deux segments. La
    # transition devient unique et franche au lieu d'osciller.
    def _segments(ligne):
        xs = np.where(ligne)[0]
        if not len(xs):
            return []
        return [(int(g[0]), int(g[-1]))
                for g in np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)]

    # 🔴 LES BRAS SONT AUSSI DES SEGMENTS. Deux critères inventés au passage,
    # deux échecs, tous deux attrapés par une valeur absurde :
    #   · « au moins deux segments » → fourche à 52,4 %, la TAILLE (la ligne
    #     coupe bras gauche | tronc | bras droit) ; masque détruit sur 268 549 px
    #   · « segments touchant la bande centrale ± 60 px » → 87,5 %, les jambes
    #     s'étant écartées au-delà de la bande
    #
    # ⭐ LE CRITÈRE EXISTAIT DÉJÀ, dans `squelette.py` : la première ligne, en
    # descendant, qui coupe le corps en EXACTEMENT deux segments larges. Le
    # « exactement » est ce qui exclut les bras, qui en ajoutent toujours un
    # troisième. Il rend 72,1 %, identique à la valeur de `squelette.json`
    # mesurée indépendamment. On ne réinvente pas ce qui est déjà mesuré.
    def _larges(y):
        return [s for s in _segments(corps[y]) if s[1] - s[0] > 20]

    y_fourche = None
    for y in range(haut, bas):
        if len(_larges(y)) == 2:
            y_fourche = y
            break
    if y_fourche is not None:
        refermees = 0
        for y in range(y_fourche, bas):
            sc = _larges(y)
            if len(sc) != 2:
                continue
            # le vide du CORPS entre les deux jambes : le masque n'a rien à y
            # faire sous la fourche, quelle que soit sa dilatation
            # ⚠️ Nom distinct : `creux` désigne déjà, plus haut, la liste des
            # écarts du corps qui sert à calibrer le rayon de dilatation.
            # L'écraser par un slice faisait planter `np.median(creux)`.
            entrejambe = slice(sc[0][1] + 1, sc[-1][0])
            if entrejambe.stop > entrejambe.start and m[y, entrejambe].any():
                refermees += int(m[y, entrejambe].sum())
                m[y, entrejambe] = False
        print(f'  fourche à {(y_fourche - h0) / Hp * 100:.1f} % · '
              f'entrejambe rouverte sur {refermees:,} px')

    # ⭐ LE TÉMOIN QUI MANQUAIT : un masque sans creux EST un trapèze.
    restants = sum(int(np.ptp(np.where(m[y])[0]) + 1 - len(np.where(m[y])[0]))
                   for y in range(haut, bas) if m[y].any())
    # 🔴 EXCLUSION DOUCE DES MEMBRES — la vraie cause des moufles.
    # Audit du 30 août : « ce n'est pas un problème de protection des mains,
    # c'est un problème de FORME DU MASQUE. La dilatation isotrope remplit
    # l'espace AUTOUR des mains ; SDXL lit cet îlot comme du tissu. »
    #
    # Le recollage a posteriori ne peut rien y faire : les pixels parasites
    # sont HORS des mains, donc hors du booléen. Il faut que le modèle n'ait
    # jamais le droit de peindre là — mais sans bord dur, qu'il négocie mal.
    # D'où une exclusion GAUSSIENNE : le masque devient flottant près des
    # membres au lieu d'y avoir une frontière anguleuse.
    if os.path.exists(MASQUE_BRAS):
        membres = np.asarray(Image.open(MASQUE_BRAS).convert('L')) > 127
        # 🔴 EXCLUSION BINAIRE PAR DISTANCE, PAS UN FLOU.
        #
        # Deux erreurs corrigées ici, toutes deux mesurées le 30 août 2026 :
        #
        # 1. UN FLOU GAUSSIEN NE DILATE PAS. Un flou puis un seuil à 0,5 laisse
        #    le niveau 0,5 sur le contour d'origine — noyau symétrique. On
        #    achetait un lissage de coins, pas une zone d'exclusion. Effet
        #    mesuré : moufles de 19 361 à 17 783 px, soit −8 %.
        #
        # 2. SDXL BINARISE LE MASQUE DE TOUTE FAÇON. Vérifié, pas supposé :
        #       mask_processor.do_binarize = True   (diffusers 0.31)
        #       vae_scale_factor = 8
        #    Tout flottant est seuillé à 0,5 puis sous-échantillonné ×8. Une
        #    rampe de 15 px devient 1,9 px latent. Le masque flottant n'a
        #    jamais existé du point de vue du modèle.
        #
        # LE RAYON SE CALCULE, IL NE SE BALAIE PAS. Distribution mesurée des
        # distances au masque des membres :
        #       moufles          30 % du tissu à moins de 40 px
        #       jambes (légitime) minimum 48 px
        #    Les deux populations sont nettement séparées. R = 40 px les
        #    distingue sans manger le pantalon.
        #
        # 🔴 MAIS UNE EXCLUSION ISOTROPE DÉCHIRE LE PANTALON. Mesuré le
        # 30 août sur trois strengths : une bande de peau nue à 68-70 % de la
        # hauteur, dont 75 % des pixels à moins de 40 px d'un membre. Ce n'est
        # PAS le genou (l'hypothèse qu'on suivait) : c'est la hauteur des
        # DOIGTS, qui pendent le long des cuisses. Le disque R=40 autour de
        # chaque main creuse une tranchée horizontale de part et d'autre, et
        # le modèle « termine » le vêtement au bord de la tranchée. Monter le
        # strength aggrave : la bande passe de 43 lignes (0,62) à 132 (0,88).
        #
        # ⭐ LA SÉPARATION EST ANATOMIQUE, PAS MÉTRIQUE. Une moufle est du
        # tissu peint dans le FOND autour de la main ; un pantalon est du
        # tissu peint SUR le corps. Les deux se distinguent sans paramètre :
        #
        #       interdit = proche d'un membre ET (hors du corps OU sur le membre)
        #
        # Mesure de la règle : 74,8 % des moufles de l'essai raté restent
        # exclues (elles sont dans le fond), et 20 668 px de cuisse sont rendus
        # au pantalon. Le vêtement a désormais le droit de passer DERRIÈRE la
        # main — le recollage des membres la remet par-dessus, ce qui est
        # exactement l'occlusion voulue et qu'un disque isotrope interdisait.
        d = ndimage.distance_transform_edt(~membres)
        interdit = (d < RAYON_MEMBRES) & (~corps | membres)
        avant = int(m.sum())
        m = m & ~interdit
        print(f'  exclusion anatomique à {RAYON_MEMBRES} px des membres : '
              f'{avant - int(m.sum()):,} px retirés '
              f'({int((m & (d < RAYON_MEMBRES)).sum()):,} px de corps préservés)')
    m_img = Image.fromarray((m * 255).astype(np.uint8))

    print(f'  masque y={haut}-{bas}, dilatation {rayon} px '
          f'(creux médian du corps {int(np.median(creux)) if creux else 0} px)')
    print(f'  creux conservés : {restants:,} px  → '
          f'{"ÉPOUSE" if restants > 500 else "🔴 TRAPÈZE, le vêtement débordera"}')
    if restants <= 500:
        print("  ⚠️ le masque risque de sceller l'espace sous les bras")
    return m_img


# ═══════════════════════════════════════════════════════════════════════════
#  2. LA GÉNÉRATION
# ═══════════════════════════════════════════════════════════════════════════

def format_travail(W, H, cible=None):
    """Dimensions de génération qui RESPECTENT le ratio du canevas.

    SDXL travaille autour du million de pixels. Générer dans un autre ratio
    puis remonter étire l'image — c'est le défaut 1.
    """
    # ⭐ La surface de travail se règle : à 1,10 Mpx, une cellule latente vaut
    # 15,7 px du canevas final ; à 2,09 Mpx elle n'en vaut plus que 11,4. Tout
    # défaut qui suivrait cette échelle est un artefact de résolution, pas une
    # propriété du vêtement — c'est ce que ce réglage permet de trancher.
    cible = cible or int(os.environ.get('MEDMOJI_CIBLE', '1100000'))
    r = W / H
    h = (cible / r) ** 0.5
    return int(round(h * r / 8) * 8), int(round(h / 8) * 8)


def preremplir_aplat(orig, mq, couleur):
    """Aplat bruité de la couleur cible AVANT l'inpainting.

    🔴 Sans lui, le modèle part de la PEAU qu'il voit et harmonise avec elle :
    deux essais ont rendu un hoodie couleur chair puis rouge-brun, malgré un
    guidage à 10 et un prompt disant « clairement plus foncé que la peau ».
    Un prompt DÉCRIT une couleur, il ne l'IMPOSE pas.

    Le bruit répond à une réserve du conseil : un aplat uni est un prior de
    teinte mais pas de plis, et le modèle peut y voir un jersey plat.
    """
    a = np.asarray(orig).astype(np.float32).copy()
    m = np.asarray(mq) > 127
    bruit = np.random.default_rng(GRAINE).normal(0, 9, size=(m.sum(), 3))
    a[m] = np.clip(np.array(couleur, np.float32) + bruit, 0, 255)
    return Image.fromarray(a.astype(np.uint8))


def preremplir(orig, mq, couleur):
    """Décaler la TEINTE des pixels d'origine, au lieu de peindre un aplat.

    ⭐ Amélioration issue de l'audit du 30 août 2026. Un aplat uni force le
    modèle à réinventer toute la radiance : ombres, dégradés, occlusion. En
    déplaçant seulement la teinte et en conservant saturation et valeur, on lui
    donne l'éclairage 3D DÉJÀ PRÉSENT sur le corps — les volumes de la cuisse,
    l'ombre sous le genou, la lumière venant du haut à gauche.

    Le modèle n'a plus qu'à habiller une forme déjà éclairée.
    """
    import colorsys
    a = np.asarray(orig).astype(np.float32).copy()
    m = np.asarray(mq) > 127
    if not m.any():
        return Image.fromarray(a.astype(np.uint8))

    r, v, b = [x / 255.0 for x in couleur]
    h_cible, _, _ = colorsys.rgb_to_hsv(r, v, b)
    _, s_cible, _ = colorsys.rgb_to_hsv(r, v, b)

    px = a[m] / 255.0
    mx = px.max(1); mn = px.min(1)
    val = mx                                   # la VALEUR est conservée
    # on impose teinte et saturation de la cible, on garde la luminosité locale
    out = np.empty_like(px)
    for i in range(len(px)):
        out[i] = colorsys.hsv_to_rgb(h_cible, s_cible, float(val[i]))
    bruit = np.random.default_rng(GRAINE).normal(0, 8 / 255.0, size=out.shape)
    a[m] = np.clip((out + bruit) * 255.0, 0, 255)
    return Image.fromarray(a.astype(np.uint8))


def sous_vetement(im):
    """Le sous-vêtement du corps de base — mesuré, jamais supposé.

    🔴 IL EST LA CAUSE DE DEUX DÉFAUTS À LA FOIS, tous deux vus par Med sur le
    rendu du 30 août et confirmés par la mesure :

      · une BANDE GRISE en haut du pantalon. La zone « bas » démarrait à 55,0 %
        de la hauteur, le sous-vêtement à 54,1 % : 22 px de boxer que
        l'inpainting ne pouvait pas repeindre, donc 799 px de gris qui restent.

      · une COUTURE HORIZONTALE en travers des cuisses. Saut de luminance de
        8,7 mesuré à 71,5 %, juste au-dessus du bas du boxer (72,8 %). SDXL
        voit ce bord dans l'image d'entrée et le REPRODUIT comme une couture de
        vêtement. C'est la même structure qui, plus tôt, faisait terminer le
        pantalon en pleine cuisse.

    ⭐ Un modèle d'inpainting ne dessine pas dans le vide : il prolonge ce qu'il
    voit. Toute arête présente dans l'init ressort dans la sortie. Le
    sous-vêtement doit donc DISPARAÎTRE de l'init — la zone est de toute façon
    recouverte par le vêtement.

    ⭐ IL SE RECONNAÎT À SA TEINTE, PAS À SA CLARTÉ. Une première version
    prenait « désaturé et clair, plus grosse composante » : elle ne trouvait que
    le centre du boxer (x 605 à 997, soit 393 colonnes), parce que ses flancs
    sont dans l'ombre — trop sombres pour un seuil de clarté — et parce que le
    liseré de la ceinture le coupe en plusieurs composantes.

    L'axe b* de Lab (jaune ↔ bleu) sépare les deux populations franchement.
    Histogramme mesuré sur la bande du bassin, 370 589 px :

        b* ∈ [−7,3 ; −0,2]   169 662 px   ← le tissu, neutre à bleuté
        b* ∈ [−0,2 ;  3,4]       768 px   ← LA VALLÉE
        b* ∈ [17,6 ; 28,3]   162 374 px   ← la peau, franchement orangée

    Deux modes, une vallée nette : le seuil se lit, il ne se choisit pas.
    b* < 4 rend un masque symétrique autour de l'axe du corps (x 538 à 997,
    centre 767,5) — contrôle qui attrape un seuil trop lâche : à b* < 6 le
    masque devient x 331 à 997, dissymétrique, donc il a mordu autre chose.
    """
    a = np.asarray(im.convert('RGBA'))
    corps = a[:, :, 3] > 250
    ys = np.where((a[:, :, 3] > 16).any(1))[0]
    if not len(ys):
        return np.zeros(corps.shape, bool)
    h0, Hp = ys.min(), ys.max() - ys.min()
    bande = np.zeros(corps.shape, bool)
    bande[h0 + int(Hp * .50):h0 + int(Hp * .78)] = True
    b = _vers_lab(np.asarray(sur_blanc(im)).astype(float))[..., 2]
    m = corps & bande & (b < 4)
    lab, n = ndimage.label(m)
    if not n:
        return np.zeros(corps.shape, bool)
    t = ndimage.sum(m, lab, range(1, n + 1))
    return np.isin(lab, [i + 1 for i in range(n) if t[i] > 1500])


def effacer_sous_vetement(orig, src):
    """Remplace le sous-vêtement par la peau qui l'entoure, dans l'init.

    Pas une retouche du rendu : l'image ainsi modifiée ne sert QU'À nourrir le
    modèle, et la zone est intégralement recouverte par le vêtement généré.
    """
    sv = sous_vetement(src)
    if not sv.any():
        return orig, 0
    corps = np.asarray(src.convert('RGBA'))[:, :, 3] > 250
    # ⚠️ LE CONTOUR AUSSI. Le liseré de la ceinture et l'ourlet des cuisses
    # sont plus foncés que le tissu, donc hors du seuil b* < 4 — et ce sont
    # précisément les ARÊTES que le modèle recopie en couture. On les avale,
    # borné au corps : quelques pixels de peau remplacés par de la peau voisine
    # ne changent rien, et la zone est de toute façon recouverte.
    # Rayon MESURÉ sur le corps nu : sous le bord du sous-vêtement, la peau
    # porte son OMBRE — luminance 151 contre 160 juste en dessous, sur 4 à 5 px.
    # Cette ombre est une arête franche que le modèle recopie tout autant que le
    # tissu lui-même ; une dilatation de 3 px la laissait dehors. 8 px couvrent
    # l'ombre (5 px) et la frange d'anti-crénelage (2 px) avec une marge.
    sv = ndimage.binary_dilation(sv, np.ones((17, 17))) & corps
    a = np.asarray(orig).astype(float).copy()
    # La peau juste au-dessus et juste en dessous, colonne par colonne : le
    # dégradé du corps est conservé, aucune couleur n'est inventée.
    # 🔴 UNE INTERPOLATION LINÉAIRE LAISSE UNE CASSURE DE PENTE.
    # Première version : par colonne, dégradé droit entre la peau du dessus et
    # celle du dessous. Les VALEURS se raccordaient, mais pas les PENTES — la
    # peau réelle a un modelé, le dégradé est plat. Gradient vertical mesuré sur
    # l'init, au bas du sous-vêtement effacé :
    #
    #       71,0 %  0,08      72,5 %  0,51
    #       71,5 %  0,14      73,0 %  1,36   ← la cassure
    #       72,0 %  0,17      73,5 %  0,48
    #
    # Un membre du conseil l'avait vue sur l'image : « l'interpolation de peau
    # présente un raccord anguleux au bas du boxer ; le modèle s'est accroché à
    # cette discontinuité de gradient ». C'est la couture qui survivait.
    #
    # ⭐ La diffusion harmonique n'a pas ce défaut : elle résout Δu = 0 dans la
    # zone, donc la surface obtenue est celle de moindre énergie qui prolonge le
    # bord — sans extremum interne, sans cassure. Amorcée au plus proche voisin
    # valide, relaxée sur la seule boîte englobante du masque.
    ys, xs = np.where(sv)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    marge = 24
    y0, x0 = max(0, y0 - marge), max(0, x0 - marge)
    y1, x1 = min(a.shape[0], y1 + marge), min(a.shape[1], x1 + marge)
    boite = a[y0:y1, x0:x1].copy()
    trou = sv[y0:y1, x0:x1]
    # ⚠️ N'amorcer QUE dans le trou : `boite = boite[ind]` remplaçait toute la
    # boîte par son plus proche voisin, y compris les pixels valides — 72/255
    # d'écart mesuré hors du masque, sur des pixels à recopier à l'identique.
    ind = ndimage.distance_transform_edt(trou, return_indices=True)[1]
    boite[trou] = boite[ind[0], ind[1]][trou]        # amorce : plus proche bord
    for _ in range(400):
        lisse = ndimage.uniform_filter(boite, size=(5, 5, 1))
        boite[trou] = lisse[trou]                    # le bord reste fixe
    a[y0:y1, x0:x1] = boite
    return Image.fromarray(a.round().astype(np.uint8)), int(sv.sum())


def generer(orig, mq, description, couleur):
    import torch
    from diffusers import AutoPipelineForInpainting

    W, H = orig.size
    GEN = format_travail(W, H)
    print(f'  génération {GEN[0]}x{GEN[1]} (ratio {GEN[0]/GEN[1]:.3f}) '
          f'pour un canevas {W}x{H} (ratio {W/H:.3f})')

    pipe = AutoPipelineForInpainting.from_pretrained(
        MODELE_SDXL, torch_dtype=torch.float16, variant='fp16').to('cuda')
    pipe.set_progress_bar_config(disable=True)

    return pipe(
        # ⭐ L'ORDRE DES MOTS DICTE L'ATTENTION DU MODÈLE. Med, 30 août 2026 :
        # la matière et le volume AVANT le type de vêtement et la couleur.
        #     à éviter  : « a green cargo pants, 3d style »
        #     à écrire  : « thick heavy canvas, baggy loose fit, deep folds,
        #                   bulky pockets, olive green, 3d render »
        # Sans ça, le modèle moule le vêtement sur la peau au lieu de lui
        # donner une épaisseur — l'effet « peinture corporelle ».
        prompt=(f'{description}. Solid garment clearly darker than skin. '
                '3d cartoon character clothing, matte fabric with visible folds, '
                'soft studio lighting from upper left, white background'),
        # 🔴 LES TERMES QUI COLLENT LE TISSU À LA PEAU SONT BANNIS. Combinés à
        # une contrainte de géométrie, « tight », « fitted » ou « stretchy »
        # garantissent l'effet d'emballage sous vide.
        # 🔴 UN PROMPT DÉCRIT, IL N'INTERDIT PAS. « no pockets, seamless » dans
        # le prompt positif n'a rien empêché : le modèle a remis des rabats
        # latéraux à mi-cuisse sur les deux jambes, parce qu'un pantalon
        # d'avatar 3D en a dans son prior. Med les a entourés trois fois.
        # Ce qui doit disparaître se met dans le prompt NÉGATIF, où il pèse
        # vraiment — c'est la même leçon que la couleur, qu'un prompt positif
        # n'imposait pas non plus et qu'il a fallu préremplir.
        negative_prompt=(os.environ.get('MEDMOJI_NEG_SUP', '') +
                         'skin coloured clothing, beige, flesh tone garment, nude, '
                         'tight, fitted, skinny, slim fit, stretchy, spandex, '
                         'leggings, shrink wrap, body paint, '
                         'photorealistic, flat vector, black outline, text, watermark'),
        image=preremplir(orig, mq, couleur).resize(GEN, Image.LANCZOS),
        mask_image=mq.resize(GEN, Image.NEAREST),
        # ⭐ RÉGLAGES CALIBRÉS PAR L'AUDIT du 30 août 2026.
        #   strength 0,85 → 0,62 : à 0,85 l'init est oubliée vers le pas 15,
        #     ce qui explique À LA FOIS le pré-remplissage sans effet ET le
        #     pantalon arrêté à mi-mollet. À 0,62 on conserve 38 % de l'init,
        #     donc la couleur et le volume de départ survivent.
        #   guidance 8 → 7 : guidage haut + strength haute font surinterpréter
        #     le prior « vêtement = manches », d'où les moufles.
        num_inference_steps=40, guidance_scale=7.0, strength=float(os.environ.get('MEDMOJI_STRENGTH','0.62')),
        generator=torch.Generator('cuda').manual_seed(GRAINE),
    ).images[0].resize((W, H), Image.LANCZOS)


def recoller_membres(fusion, orig, corps, haut, bas, alpha_src):
    """Repose les bras et les mains d'origine PAR-DESSUS le vêtement généré.

    🔴 POURQUOI ON NE LES EXCLUT PLUS DU MASQUE. Med, 30 août 2026, devant des
    doigts tranchés : « quand le modèle vient buter contre une limite stricte
    sans marge de transition, il panique — il crée des artefacts, bave, ou coupe
    les phalanges ».

    Découper les bras du masque produisait des frontières anguleuses autour des
    doigts, que l'inpainting ne sait pas négocier. On laisse donc le modèle
    peindre librement, quitte à ce qu'il couvre les mains, puis on remet les
    membres d'origine par-dessus. Ils n'ont jamais été touchés, donc ils sont
    intacts à 100 %.

    ⭐ C'est le z-index appliqué au pipeline de génération : ce qui doit rester
    intact ne se protège pas en amont, il se repose en aval.
    """
    # ── LE MASQUE STATIQUE, calculé une seule fois ──
    # Le corps de base ne bouge jamais : ce masque est un fichier, pas un
    # calcul. S'il est absent, on le dérive à la volée par la même géométrie.
    if os.path.exists(MASQUE_BRAS):
        membres = np.asarray(Image.open(MASQUE_BRAS).convert('L')) > 127
    else:
        membres = np.zeros(corps.shape, bool)
        for y in range(corps.shape[0]):
            xs = np.where(corps[y])[0]
            if len(xs) < 2:
                continue
            c = np.where(np.diff(xs) > 1)[0]
            segs = np.split(xs, c + 1)
            if len(segs) >= 3:                 # bras | tronc | bras
                for seg in (segs[0], segs[-1]):
                    membres[y, seg] = True
        membres = ndimage.binary_closing(membres, np.ones((3, 3)))

    # 🔴 SEULS LES PIXELS PLEINEMENT OPAQUES SONT PROTÉGÉS.
    # Mesure du 30 août : 99 % des pixels encore abîmés étaient au BORD du
    # masque, avec un alpha de 110/255 — la frange d'anti-crénelage du
    # détourage. Ces pixels n'appartiennent ni au bras ni au fond : composés
    # sur blanc ils valent « bras + blanc », alors qu'ils devraient valoir
    # « bras + vêtement ». Les recopier tels quels injecte du blanc au pourtour.
    #
    # ⭐ Ce n'était donc ni le fondu gaussien (162) ni le booléen strict (208) :
    # les deux recopiaient la même frange fausse. On la laisse se composer
    # naturellement et on ne protège que l'intérieur plein.
    membres = membres & (alpha_src > 250)
    zone = np.zeros(corps.shape, bool)
    zone[haut:bas] = True
    membres = membres & zone
    if not membres.any():
        return fusion, 0

    # 🔴 REMPLACEMENT BOOLÉEN STRICT, SANS FONDU.
    # Med, 30 août 2026 : « en voulant adoucir le raccord, le masque s'est
    # transformé en zone de mélange, laissant le vêtement contaminer le bord de
    # la peau ». Mesuré : un fondu gaussien de 2 px laissait un écart de
    # 162/255 sur les mains. Chaque pixel est désormais soit 100 % d'origine,
    # soit 100 % généré — aucun compromis sur les bords.
    #
    # ⭐ Le fondu était justifié pour le RECOLLAGE DU VÊTEMENT, où les deux
    # images se ressemblent. Il ne l'est pas ici : le bras d'origine et le
    # vêtement généré n'ont rien en commun, et les mélanger ne peut que salir.
    out = fusion.copy()
    a_or = np.asarray(orig).astype(np.uint8)
    out[membres] = a_or[membres]
    return out, int(membres.sum())


def recoller(orig, genere, mq, src):
    """Hors du masque, l'original PIXEL POUR PIXEL.

    🔴 La préservation ne s'espère pas du modèle, elle s'impose. Le premier
    essai mesurait 141/255 d'écart hors masque — non parce que le modèle avait
    débordé, mais parce que je remontais l'image ENTIÈRE, ce qui rééchantillonne
    chaque pixel.

    L'alpha final est l'UNION du corps et du vêtement : rendre celui du corps nu
    découperait toute pièce plus large que lui (défaut 2).
    """
    doux = mq.filter(ImageFilter.GaussianBlur(6))
    a_or = np.asarray(orig).astype(np.float32)
    a_ge = np.asarray(genere).astype(np.float32)
    al = (np.asarray(doux).astype(np.float32) / 255.0)[:, :, None]
    fusion = (a_or * (1 - al) + a_ge * al).round().astype(np.uint8)

    # ── LES MEMBRES REPOSÉS PAR-DESSUS, avant de calculer l'alpha ──
    corps = np.asarray(src)[:, :, 3] > 16
    ys = np.where(np.asarray(mq).any(1))[0]
    if len(ys):
        fusion, n = recoller_membres(fusion, orig, corps, ys.min(), ys.max(),
                                     np.asarray(src)[:, :, 3])
        if n:
            print(f'  membres reposés intacts : {n:,} px')

    m = np.asarray(mq) > 127
    alpha = (np.asarray(src)[:, :, 3] > 16) | (m & (fusion.min(2) < 244))
    alpha = ndimage.binary_closing(alpha, np.ones((5, 5)))

    im = Image.fromarray(fusion).convert('RGBA')
    im.putalpha(Image.fromarray((alpha * 255).astype(np.uint8)))

    ecart = np.abs(a_or - fusion.astype(np.float32)).max(2)
    hors = ~(np.asarray(doux) > 0)
    print(f'  écart hors zone touchée : {ecart[hors].max():.0f}/255 → '
          f'{"INTACT" if ecart[hors].max() == 0 else "🔴 MODIFIÉ"}')
    return im, fusion, alpha


# ═══════════════════════════════════════════════════════════════════════════
#  3. LE MASQUE DU VÊTEMENT — sémantique, jamais par seuil
# ═══════════════════════════════════════════════════════════════════════════

def masque_vetement(chemin_habille, chemin_masque_semantique=None,
                    corps=None, habille=None, autoriser_repli=False):
    """Le masque exact du vêtement produit, pour la teinture et les calques.

    🔴 UN SEUIL NE SAIT PAS CE QU'EST UN VÊTEMENT. Mesuré le 30 août : une
    détection par différence trouvait 805 583 px, dont 433 789 n'étaient PAS du
    vêtement — plus de la moitié. L'ombre du menton sur le cou a la même
    signature que l'ombre dans le col ; aucun seuil ne les sépare.

    On passe donc par un segmenteur sémantique (catégories Upper Clothes /
    Coat / Lower Clothes / Shoe). Le repli par différence n'existe que pour ne
    pas bloquer la chaîne, et il est signalé bruyamment.
    """
    if chemin_masque_semantique and os.path.exists(chemin_masque_semantique):
        m = np.asarray(Image.open(chemin_masque_semantique).convert('L')) > 127
        print(f'  masque sémantique : {m.sum():,} px')
        return m

    # 🔴 FAIL-FAST : on REFUSE plutôt que de produire un asset abîmé.
    #
    # Med, 30 août 2026, en voyant les plaques sombres sur l'épaule : « on ne
    # peut rien tirer de la détection par seuil ». Le test A/B l'a confirmé —
    # même teinture, même couleur, seul le masque change :
    #
    #     masque par différence   549 500 px
    #     masque sémantique       534 045 px
    #       peau classée « tissu » par la différence : 16 449 px
    #
    # Ces 16 449 pixels de transition — anti-crénelage, ombres douces — ne font
    # pas que salir les bords. Ils DÉCALENT le 88ᵉ centile qui sépare le tissu
    # des détails clairs : 63 498 px « clairs » au lieu de 57 758, donc des
    # ombres du vêtement classées « détail à préserver » et laissées en gris
    # luisant au milieu du bleu. L'effet « sac poubelle ».
    #
    # ⭐ La fonction de teinture est innocente : elle calculait juste sur des
    # données fausses. Un avertissement laissait passer l'asset ; seul un refus
    # protège le catalogue.
    if not autoriser_repli:
        sys.exit(
            f'🔴 masque sémantique absent : {chemin_masque_semantique}\n'
            '   Passer la pièce au segmenteur (Upper Clothes / Coat / Lower\n'
            '   Clothes / Shoe) AVANT de teindre. La détection par seuil\n'
            '   classe 16 449 px de peau comme du tissu et abîme le rendu.\n'
            '   Pour un essai hors production : autoriser_repli=True.')

    print('  ⚠️ REPLI PAR DIFFÉRENCE — hors production uniquement')
    ec = np.abs(corps[:, :, :3] - habille[:, :, :3]).max(2)
    m = (ec > 18) & (habille[:, :, 3] > 200)
    m = ndimage.binary_opening(ndimage.binary_closing(m, np.ones((9, 9))),
                               np.ones((5, 5)))
    lab, n = ndimage.label(m)
    if n > 1:                      # ne garder que la plus grosse composante
        t = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(t)))
    return m


# ═══════════════════════════════════════════════════════════════════════════
#  4. TEINTURE — un asset, une infinité de couleurs
# ═══════════════════════════════════════════════════════════════════════════

GAIN, OFFSET = 0.85, 0.06

# ── Lab : le seul espace où la luminance se sépare de la couleur ───────────
# Écrits ici plutôt qu'importés : la Fabrique ne dépend que de numpy, PIL et
# scipy, et ces deux conversions tiennent en dix lignes chacune.
_M_RVB_XYZ = np.array([[.4124, .3576, .1805],
                       [.2126, .7152, .0722],
                       [.0193, .1192, .9505]])
_BLANC = np.array([.95047, 1.0, 1.08883])


def _vers_lab(rvb):
    r = rvb / 255.0
    r = np.where(r > .04045, ((r + .055) / 1.055) ** 2.4, r / 12.92)
    xyz = (r @ _M_RVB_XYZ.T) / _BLANC
    f = np.where(xyz > .008856, np.cbrt(np.maximum(xyz, 0)), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def _lin(lab):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f > .206893, f ** 3, (f - 16 / 116) / 7.787) * _BLANC
    return xyz @ np.linalg.inv(_M_RVB_XYZ).T


def _vers_rvb(lab, mapper=True):
    """Lab → sRGB. Avec `mapper`, la couleur hors gamut est RAMENÉE, pas écrêtée.

    🔴 UN CLIP N'EST PAS UNE CONVERSION. Le conseil d'IA, 30 août : « imposer
    (a, b) puis repasser en sRGB peut produire des pixels hors gamut ; le
    clipping RGB modifie alors L* et réduit localement la texture. » Mesuré sur
    la version précédente, à l'intérieur du vêtement :

        rouge   7,2 % hors gamut, 6,97 % de pixels écrêtés
        bleu    6,8 % hors gamut, 7,95 % de pixels écrêtés

    Sept pour cent des pixels perdaient leur luminance au profit d'un bord de
    cube. ⭐ La chroma se réduit, la luminance se garde : on cherche le plus
    grand facteur k tel que (L, k·a, k·b) tienne dans sRGB. Huit pas de
    dichotomie suffisent — l'erreur résiduelle est sous le quantum 1/255.
    """
    r = _lin(lab)
    if mapper:
        hors = ((r < 0) | (r > 1)).any(-1)
        if hors.any():
            bas = np.zeros(lab.shape[:-1]); haut = np.ones(lab.shape[:-1])
            for _ in range(8):
                k = (bas + haut) / 2
                essai = lab.copy()
                essai[..., 1] *= k
                essai[..., 2] *= k
                dedans = ~((_lin(essai) < -1e-6) | (_lin(essai) > 1 + 1e-6)).any(-1)
                bas = np.where(dedans, k, bas)
                haut = np.where(dedans, haut, k)
            lab = lab.copy()
            lab[..., 1] = np.where(hors, lab[..., 1] * bas, lab[..., 1])
            lab[..., 2] = np.where(hors, lab[..., 2] * bas, lab[..., 2])
            r = _lin(lab)
    r = np.where(r > .0031308, 1.055 * np.maximum(r, 0) ** (1 / 2.4) - .055, 12.92 * r)
    return np.clip(r, 0, 1) * 255

def teindre(habille, masque_vet, couleur):
    """Recolore le tissu en préservant plis, coutures et détails clairs.

    ⚠️ GAIN ET OFFSET, pas une gamma. Med, 30 août : une gamma globale
    « bouche les noirs, les plis perdent leur texture ». L'offset relève le
    point noir juste assez pour que le relief y survive.

    Les DÉTAILS CLAIRS — cordons, fermeture, œillets — sont exclus : ils sont
    nettement plus lumineux que le tissu, donc séparables au 88ᵉ centile.
    """
    a = np.asarray(habille).astype(float)
    lum = a[:, :, :3] @ [0.2126, 0.7152, 0.0722]
    # 🔴 UN CENTILE GLOBAL N'EST PAS UN DÉTECTEUR DE DÉTAILS.
    # Première version : `lum > percentile(lum[masque], 88)`. Elle exclut 12 %
    # des pixels de la teinture PAR CONSTRUCTION, quel que soit le vêtement.
    # Le raisonnement supposait deux populations séparées — un tissu sombre et
    # des détails clairs. Histogramme mesuré sous le masque, cargo vert clair :
    #
    #     un seul pic, massif, à 201-212/255 ; médiane 191 ; 88ᵉ centile 208
    #
    # Le centile tombe EN PLEIN DANS LE PIC DU TISSU : 45 988 px de tissu
    # ordinaire échappaient à la teinture. D'où les plaques vert clair sur les
    # cuisses et les mollets des déclinaisons rouge, bleue et violette — un
    # effet rongé visible sans zoom.
    #
    # ⭐ Un détail est une structure claire LOCALEMENT, pas globalement. Le
    # chapeau haut-de-forme morphologique mesure exactement ça : l'écart d'un
    # pixel à l'ouverture de son voisinage. Aucun seuil global, et il suit le
    # tissu quelle que soit sa luminosité. Mesuré ici : 2,2 % de la surface au
    # lieu de 12 %, en 48 pièces — cordons, passants, œillets.
    #
    # 🔴 ET IL NE SE CHERCHE PAS SUR LE BORD. `lum * masque_vet` vaut zéro hors
    # du masque : le haut-de-forme voit cette MARCHE et la lit comme un détail
    # clair. Taux de pixels laissés non teints, par distance au bord :
    #
    #     0-3 px  28,6 %      6-12 px   5,5 %      cœur (>25 px)  1,1 %
    #     3-6 px  21,2 %     12-25 px   3,2 %
    #
    # D'où le liseré vert clair qui cernait chaque jambe, chaque poche et
    # l'entrejambe sur les déclinaisons. On ne cherche les détails que là où le
    # taux retombe à moins de 4 fois celui du cœur — au-delà de 12 px du bord.
    # La couronne, elle, est du tissu et se teint comme tel.
    coeur = ndimage.binary_erosion(masque_vet, np.ones((25, 25)))
    details = ndimage.binary_opening(
        (ndimage.white_tophat(lum * masque_vet, size=15) > 18) & coeur,
        np.ones((3, 3)))

    # 🔴 ET UN DÉTAIL N'EST UN DÉTAIL QUE S'IL EST D'UNE AUTRE MATIÈRE.
    # Ce qui restait après les deux corrections précédentes : 4 186 px, visibles
    # comme des traits vert clair sur les poches et les genoux des déclinaisons
    # rouge, bleue et violette. Mesuré :
    #
    #     couleur du tissu   (185, 194, 143)   saturation 0,26
    #     couleur « détails » (190, 201, 151)   saturation 0,25   écart 7/255
    #
    # Même matière. C'était du RELIEF — des plis et des coutures du tissu — que
    # le haut-de-forme détecte parce qu'il est clair, et que la couleur seule
    # sépare correctement. Sur un cargo monochrome, une couture est de la
    # couleur du cargo ; sur un jean, un cordon blanc a une saturation nulle
    # face à un denim saturé, et lui doit survivre.
    #
    # ⭐ Le critère se lit sur la matière, pas sur la luminance : une pièce
    # n'est préservée que si sa saturation s'écarte de celle du tissu de plus
    # de la moitié. Aucun réglage à trouver — la comparaison est relative.
    if details.any():
        ref = masque_vet & ~details
        col = a[:, :, :3]
        s = lambda m: float(((col[m].mean(0).max() - col[m].mean(0).min())
                             / max(col[m].mean(0).max(), 1)))
        s_tissu = s(ref)
        lab, n = ndimage.label(details)
        for i in range(1, n + 1):
            p = lab == i
            if abs(s(p) - s_tissu) <= 0.5 * max(s_tissu, 0.02):
                details[p] = False          # même matière → c'est du tissu
    tissu = masque_vet & ~details

    # 🔴 LE GAIN MULTIPLICATIF ÉCRASAIT LE RELIEF. Première version :
    #     clip((L * 0,85 + 0,06) * couleur * 1,9, 0, 255)
    # Sur un rouge (230, 57, 70), le canal R vaut (L·0,85 + 0,06) × 437 : tout
    # pixel de luminance supérieure à ~0,45 sature à 255. Le `clip` mange donc
    # toutes les hautes lumières, et avec elles les plis.
    #
    # Critère du conseil d'IA, 30 août : `std(L)_post / std(L)_pre ∈ [0,85 ;
    # 1,15]` sur l'intérieur du vêtement. Mesuré sur cette version :
    #
    #     olive 1,00 (elle ne teignait rien)   rouge 0,30   bleu 0,44   violet 0,61
    #
    # 70 % de la variation de luminance perdue en rouge. Deux membres du conseil
    # jugeaient l'aplatissement « acceptable » à l'œil ; la mesure dit non.
    #
    # ⭐ LA LUMINANCE ET LA CHROMIE SE SÉPARENT — c'est à ça que sert Lab. On
    # garde le L du tissu, recentré sur celui de la couleur demandée, et on
    # impose (a, b) de la cible. La dispersion de L est conservée à
    # l'identique : le critère passe PAR CONSTRUCTION, pas par réglage.
    lab = _vers_lab(a[:, :, :3])
    cible = _vers_lab(np.array(couleur, float).reshape(1, 1, 3))[0, 0]
    Lt = lab[:, :, 0][tissu]
    neuf = lab.copy()
    neuf[:, :, 0][tissu] = np.clip(cible[0] + (Lt - Lt.mean()), 0, 100)
    neuf[:, :, 1][tissu] = cible[1]
    neuf[:, :, 2][tissu] = cible[2]
    out = a.copy()
    out[:, :, :3][tissu] = _vers_rvb(neuf)[tissu]
    return Image.fromarray(out.round().astype(np.uint8)), tissu, details


# ═══════════════════════════════════════════════════════════════════════════
#  5. OMBRE DE CONTACT — ce qui sépare un vêtement porté d'un autocollant
# ═══════════════════════════════════════════════════════════════════════════

def ombre_contact(habille, masque_vet, force=0.42, flou=14, decalage=7):
    """Assombrit la peau juste sous le bord du vêtement.

    ⚠️ TOUJOURS EN DERNIER. Elle modifie la peau ; toute segmentation calculée
    après elle prendrait cette peau pour du tissu (défaut 6).

    L'ombre ne doit jamais toucher le vêtement ni le fond — un halo est le
    défaut le plus visible de cet effet, d'où le témoin de fuite.
    """
    a = np.asarray(habille).astype(float)
    peau = (a[:, :, 3] > 200) & ~masque_vet

    m = Image.fromarray((masque_vet * 255).astype(np.uint8))
    m = m.transform(m.size, Image.AFFINE, (1, 0, 0, 0, 1, -decalage),
                    resample=Image.NEAREST, fillcolor=0)
    om = (np.asarray(m.filter(ImageFilter.GaussianBlur(flou))).astype(float) / 255.0) * peau

    out = a.copy()
    out[:, :, :3] *= (1 - om[:, :, None] * force)

    fuite = (om * masque_vet).sum() + (om * (a[:, :, 3] <= 200)).sum()
    touche = (om > 0.02).sum()
    print(f'  peau assombrie : {touche:,} px   fuite : {fuite:.0f} → '
          f'{"PROPRE" if fuite == 0 else "🔴 HALO"}')
    return Image.fromarray(out.round().astype(np.uint8))


# ═══════════════════════════════════════════════════════════════════════════
#  L'ENCHAÎNEMENT — l'ordre n'est pas négociable
# ═══════════════════════════════════════════════════════════════════════════

def fabriquer(description, nom, zone='haut', couleur=(74, 78, 84),
              teintes=None, masque_semantique=None, autoriser_repli=False):
    src = Image.open(BASE)
    orig = sur_blanc(src)

    print(f'\n── {nom} ──')
    mq = masque_zone(src, zone)
    mq.save(f'{SORTIE}/{nom}.zone.png')

    # ⭐ L'INIT NE DOIT CONTENIR AUCUNE ARÊTE QU'ON NE VEUT PAS VOIR RESSORTIR.
    # Le sous-vêtement est effacé de l'image qui nourrit le modèle — jamais du
    # rendu, et la zone est entièrement recouverte par le vêtement généré.
    init, n_sv = effacer_sous_vetement(orig, src)
    if n_sv:
        print(f'  sous-vêtement effacé de l\'init : {n_sv:,} px')
    genere = generer(init, mq, description, couleur)
    habille, fusion, alpha = recoller(orig, genere, mq, src)
    habille.save(f'{SORTIE}/{nom}.png')

    corps_px = (np.asarray(src)[:, :, 3] > 16).sum()
    print(f'  silhouette : {corps_px:,} → {alpha.sum():,} px '
          f'({alpha.sum()/corps_px*100-100:+.1f} %)')

    # ⚠️ Le masque du vêtement se calcule AVANT toute ombre.
    mv = masque_vetement(f'{SORTIE}/{nom}.png', masque_semantique,
                         np.asarray(orig.convert('RGBA')).astype(float),
                         np.asarray(habille).astype(float),
                         autoriser_repli=autoriser_repli)
    # 🔴 LE MASQUE SÉMANTIQUE NE CONNAÎT PAS LA ZONE. SAM segmente « le
    # vêtement du bas » et y inclut ce qui lui ressemble : mesuré, 9 954 px de
    # PIEDS, qui se retrouvaient teints en rouge, bleu et violet. La zone, elle,
    # est déjà bornée par construction — on l'impose au masque, avec la marge
    # que le vêtement a le droit de prendre au-delà du corps.
    # 🔴 SAM SOUS-COUVRE LE BORD DU VÊTEMENT. Mesuré sur le cargo : 19 427 px
    # de tissu hors du masque sémantique — 14 186 d'entre eux à moins de 8 px
    # de son contour. Non teints, ils formaient un liseré vert clair autour de
    # chaque jambe et de chaque poche sur les déclinaisons rouge, bleue et
    # violette. Le rognage par la zone n'y était pour rien : 0 px.
    #
    # ⭐ Les deux détections disent des choses différentes et complémentaires :
    # SAM sait OÙ est le vêtement, le seuil sait CE QUI a changé. Leur
    # intersection — le seuil, borné au voisinage immédiat de SAM — récupère le
    # bord sans jamais inventer : un pixel n'est repris que s'il a changé ET
    # qu'il touche ce que SAM a reconnu comme du tissu.
    # ⚠️ `habille.convert('RGB')` rend le transparent NOIR, alors que `orig` est
    # composé sur BLANC : sur la frange du contour, l'écart vaut 176/255 sans
    # qu'un seul pixel ait bougé. Les deux images se composent du même côté.
    ecart = np.abs(np.asarray(orig.convert('RGB')).astype(float)
                   - np.asarray(sur_blanc(habille)).astype(float)).max(2)
    change = (ecart > 18) & (np.asarray(habille)[:, :, 3] > 200)

    # 🔴 PAS UN RAYON PLUS GRAND — UNE RECONSTRUCTION GÉODÉSIQUE.
    # Une dilatation isotrope récupérait bien le bord ici (résidu 19 427 →
    # 845 px à R=20), mais le conseil d'IA l'a refusée pour une raison qui
    # tient : un rayon euclidien fixe finit par absorber la main, la peau ou le
    # fond dès que la pose change. C'est une dette à l'échelle, pas une
    # solution. Diagnostic du même conseil, vérifié sur les six résidus :
    # SAM rate les PAROIS LATÉRALES des cargos — géométrie mince, normale
    # rasante, hors du blob frontal. Mes six fragments étaient latéraux, 6 sur 6.
    #
    # ⭐ On croît donc DANS LA MATIÈRE, pas dans le vide : depuis ce que SAM a
    # reconnu, de proche en proche, en ne traversant que des pixels qui
    # ressemblent au tissu ET qui ont changé depuis le corps nu. Trente-deux pas
    # de couloir 8-connexe ne font pas un halo de 32 px : la peau intacte n'est
    # jamais franchie, faute de couloir.
    #
    # Le seuil se LIT sur les histogrammes de ΔE76 au tissu de référence, il ne
    # se choisit pas :
    #     tissu (SAM)      P50  6,1   P90 17,6   P95 24,9
    #     bord à récupérer P50 17,6   P90 27,7
    #     ceinture         P50 28,7        peau P50 27,9        fond 36,2
    # ΔE < 25 est le P95 de la dispersion propre du tissu : il englobe le tissu
    # et laisse dehors la ceinture, la peau et le fond.
    lab_h = _vers_lab(np.asarray(sur_blanc(habille)).astype(float))
    ref = np.median(lab_h[mv], axis=0)
    couloir = (np.sqrt(((lab_h - ref) ** 2).sum(2)) < 25) & change
    avant = int(mv.sum())
    for _ in range(32):
        suivant = ndimage.binary_dilation(mv, np.ones((3, 3))) & (couloir | mv)
        if int(suivant.sum()) == int(mv.sum()):
            break
        mv = suivant
    # Garde-fou du conseil : si la croissance a mordu la peau, on n'écrit pas.
    peau_nue = (np.asarray(src)[:, :, 3] > 250) & ~change
    morsure = int((mv & peau_nue).sum())
    print(f'  paroi latérale récupérée : +{int(mv.sum()) - avant:,} px '
          f'(géodésique, ΔE < 25) · peau mordue {morsure} px'
          f'{"" if morsure < 200 else "  🔴 ABANDON"}')
    if morsure >= 200:
        mv = mv & ~peau_nue

    # ⭐ LE LISERÉ DU BORD : HORS DE LA SILHOUETTE, DONC SANS RISQUE.
    # Le conseil, 30 août : « il reste des liserés blancs très lumineux sur les
    # deux flancs, plus saillants que la couture précédente à cause du contraste
    # avec les couleurs saturées ». Mesuré : 1 116 px, médiane à 2 px du masque,
    # tous en dehors de la silhouette du corps nu. Ils échappaient à la
    # reconstruction géodésique parce que, très clairs, leur ΔE dépasse 25.
    #
    # Un pixel hors de la silhouette du corps NE PEUT PAS être de la peau. Les
    # reprendre ne se paie d'aucun risque — c'est la seule zone où une
    # dilatation reste légitime, et elle est bornée par une propriété du corps,
    # pas par un rayon choisi.
    hors_corps = (np.asarray(habille)[:, :, 3] > 200) & ~(np.asarray(src)[:, :, 3] > 16)
    # ⚠️ `& ~mv` : sans lui, le compteur inclut tout le vêtement déjà masqué qui
    # déborde du corps — 68 710 px au lieu de 1 116, et le garde-fou ci-dessous
    # rejetait à tort une reprise parfaitement légitime.
    liseré = ndimage.binary_dilation(mv, np.ones((7, 7))) & hors_corps & ~mv
    if liseré.sum() > mv.sum() * 0.02:
        print(f'  ⚠️ liseré anormalement gros ({int(liseré.sum()):,} px) — ignoré')
    else:
        n0 = int(mv.sum())
        mv = mv | liseré
        print(f'  liseré du bord repris : +{int(mv.sum()) - n0:,} px (hors silhouette)')

    mv = mv & (np.asarray(mq.filter(ImageFilter.GaussianBlur(4))) > 40)
    Image.fromarray((mv * 255).astype(np.uint8)).save(f'{SORTIE}/{nom}.masque.png')

    # puis la teinture, puis seulement l'ombre
    for nom_t, c in (teintes or {}).items():
        teint, tissu, details = teindre(habille, mv, c)
        ombre_contact(teint, mv).save(f'{SORTIE}/{nom}.{nom_t}.png')
        print(f'  teinte {nom_t} : tissu {tissu.sum():,} px, '
              f'détails préservés {details.sum():,} px')

    # 🔴 LE MASTER N'ÉTAIT PAS TEINT. Mesuré : ΔE76(habillé → final) = 0,0
    # EXACTEMENT — le fichier `.final.png` n'était que l'habillé plus l'ombre.
    # Un membre du conseil l'avait soupçonné en regardant les images (« je ne
    # peux pas vérifier qu'olive a réellement teint ») ; les deux autres s'en
    # servaient comme référence de relief, donc validaient un no-op.
    # Le master se teint avec sa propre couleur, comme n'importe quelle teinte.
    master, _, _ = teindre(habille, mv, couleur)
    ombre_contact(master, mv).save(f'{SORTIE}/{nom}.final.png')

    json.dump({'id': nom, 'zone': zone, 'couleur_master': list(couleur),
               'description': description, 'graine': GRAINE,
               'silhouette_px': int(alpha.sum()),
               'vetement_px': int(mv.sum())},
              open(f'{SORTIE}/{nom}.json', 'w'), indent=2, ensure_ascii=False)
    print(f'  ÉCRIT : {SORTIE}/{nom}.final.png')


if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else 'a charcoal grey cotton hoodie'
    n = sys.argv[2] if len(sys.argv) > 2 else 'hoodie'
    z = sys.argv[3] if len(sys.argv) > 3 else 'haut'
    c = tuple(int(x) for x in sys.argv[4].split(',')) if len(sys.argv) > 4 else (74, 78, 84)
    fabriquer(d, n, z, c, teintes={
        'rouge': (230, 57, 70), 'bleu': (29, 53, 87),
        'vert': (45, 140, 90), 'violet': (122, 60, 190),
    }, masque_semantique=f'{SORTIE}/{n}.semantique.png')
