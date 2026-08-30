#!/usr/bin/env python3
"""MEDIAPIPE LIT-IL UN VISAGE DE DESSIN ANIME ?

C'est le test qui decide de la forme du Face Engine (section 09 de
l'architecture). MediaPipe Face Landmarker est entraine sur des visages REELS
et rend 478 points 3D plus 52 coefficients d'expression. Nos personnages ont
des yeux enormes, un nez minuscule et pas de texture de peau : rien ne garantit
qu'il les lise.

Deux issues, et elles n'ecrivent pas la meme architecture :
  · il lit -> le visage se decoupe en pieces ancrees sur un maillage, les
    parametres (ecart des yeux, largeur de machoire) sont MESURES sur la
    personne, et les 52 blendshapes donnent l'Expression Engine gratuitement ;
  · il ne lit pas -> le scan reste une tete d'un seul tenant, et les
    expressions doivent etre generees une par une.

⚠️ On ne conclut RIEN sans avoir regarde le rendu des points. Un detecteur qui
rend 478 points au hasard « fonctionne » du point de vue du code.
"""
import os, sys, urllib.request
import numpy as np
from PIL import Image, ImageDraw

MODELE = '/root/medtra-avatar/createur/face_landmarker.task'
URL = ('https://storage.googleapis.com/mediapipe-models/face_landmarker/'
       'face_landmarker/float16/1/face_landmarker.task')


def modele():
    if not os.path.exists(MODELE):
        print('telechargement du modele...')
        urllib.request.urlretrieve(URL, MODELE)
    print(f'modele : {os.path.getsize(MODELE)/1024/1024:.1f} Mo')
    return MODELE


def lire(chemin, etiquette):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=modele()),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
    )
    with vision.FaceLandmarker.create_from_options(opts) as det:
        im = Image.open(chemin).convert('RGB')
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(im))
        r = det.detect(img)

    print(f'\n─── {etiquette}  ({im.size[0]}x{im.size[1]})')
    if not r.face_landmarks:
        print('  AUCUN VISAGE DETECTE')
        return None
    pts = r.face_landmarks[0]
    print(f'  points detectes : {len(pts)}')

    # ── MESURES DE COHERENCE. Un detecteur peut rendre 478 points absurdes ;
    #    ces trois rapports disent s'ils forment un visage plausible. ──
    x = np.array([p.x for p in pts]); y = np.array([p.y for p in pts])
    # indices standards du maillage MediaPipe
    oeil_g, oeil_d = 33, 263          # coins externes des yeux
    menton, front = 152, 10
    nez = 1
    # 🔴 DEUX ESPACES A NE PAS MELANGER. MediaPipe normalise x sur la LARGEUR
    # et y sur la HAUTEUR de l image. Sur une image 9:16, comparer directement
    # un ecart en x a une hauteur en y gonfle le rapport d un facteur 1,78 — la
    # premiere version de ce script rendait 1,284 la ou la valeur est 0,72.
    # On repasse en PIXELS avant toute comparaison.
    W, H = im.size
    ecart_yeux = abs(x[oeil_g] - x[oeil_d]) * W
    hauteur = abs(y[menton] - y[front]) * H
    largeur_visage = (x.max() - x.min()) * W
    print(f'  etendue des points : x {x.min():.3f}-{x.max():.3f}  y {y.min():.3f}-{y.max():.3f}')
    print(f'  ecart des yeux : {ecart_yeux:.0f} px')
    print(f'  hauteur front-menton : {hauteur:.0f} px')
    print(f'  largeur du visage : {largeur_visage:.0f} px')
    print(f'  rapport yeux/hauteur : {ecart_yeux/hauteur:.3f}   '
          f'(un visage humain tourne autour de 0,42 a 0,55)')
    print(f'  rapport yeux/largeur : {ecart_yeux/largeur_visage:.3f}   '
          f'(un visage humain tourne autour de 0,80 a 0,90)')
    print(f'  nez entre les yeux : ' +
          ('oui' if min(x[oeil_g], x[oeil_d]) < x[nez] < max(x[oeil_g], x[oeil_d]) else 'NON — incoherent'))

    if r.face_blendshapes:
        bs = sorted(r.face_blendshapes[0], key=lambda b: -b.score)[:5]
        print(f'  blendshapes : {len(r.face_blendshapes[0])} coefficients')
        for b in bs:
            print(f'      {b.category_name:24} {b.score:.3f}')

    # ── LE RENDU. On DESSINE et on MONTRE : la mesure ne remplace pas l oeil. ──
    W, H = im.size
    v = im.copy(); d = ImageDraw.Draw(v)
    for p in pts:
        cx, cy = p.x * W, p.y * H
        d.ellipse([cx-1.5, cy-1.5, cx+1.5, cy+1.5], fill=(255, 60, 0))
    for i, coul in [(33,(0,160,255)), (263,(0,160,255)), (1,(0,220,80)),
                    (152,(255,220,0)), (10,(255,220,0))]:
        cx, cy = x[i]*W, y[i]*H
        d.ellipse([cx-7, cy-7, cx+7, cy+7], outline=coul, width=3)
    dest = f'/root/medtra-avatar/createur/sortie/visage-{etiquette}.png'
    # recadrer sur le visage pour que ce soit lisible
    bx0, bx1 = int(x.min()*W)-60, int(x.max()*W)+60
    by0, by1 = int(y.min()*H)-60, int(y.max()*H)+60
    v.crop((max(0,bx0), max(0,by0), min(W,bx1), min(H,by1))).save(dest)
    print(f'  rendu : {dest}')
    return pts


if __name__ == '__main__':
    cibles = [('/root/medtra-avatar/createur/reference-neutre.png', 'neutre'),
              ('/root/medtra-avatar/createur/sortie/tenues/hiver.brut.png', 'hiver')]
    for c, e in cibles:
        if os.path.exists(c):
            try:
                lire(c, e)
            except Exception as ex:
                print(f'\n─── {e} : ERREUR {type(ex).__name__} — {str(ex)[:200]}')
