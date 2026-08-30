#!/usr/bin/env python3
"""HABILLER EN 3D — le pantalon est une COQUE du corps, pas une image peinte.

Med, 30 août 2026 : « habille-le en 3D pis rends le pantalon ».

    blender -b --python habiller3d.py -- corps.glb sortie/

───────────────────────────────────────────────────────────────────────────────
  POURQUOI UNE COQUE, ET PAS UN PATRON
───────────────────────────────────────────────────────────────────────────────
Vingt-quatre heures d'inpainting ont produit, à chaque correction, un défaut
nouveau : déchirure, couture à la fourche, rabats sous le genou, ailerons qui
changent à chaque graine. Aucun n'est un défaut de vêtement — ce sont des
défauts de génération. Un modèle 2D remplit un masque avec ce que son prior lui
souffle ; il ne sait pas ce qu'est un pantalon.

⭐ Une coque ne peut PAS inventer. Elle est la surface du corps, décalée vers
l'extérieur : elle épouse la jambe par construction, elle n'a de couture que là
où on en met, et elle est identique d'une exécution à l'autre.

Les bornes ne sont pas choisies. Elles viennent de `squelette.json`, mesuré sur
le corps de base :

    taille      52,4 %   (haut du sous-vêtement, mesuré par sa teinte b* < 4)
    fourche     72,1 %   (première ligne à exactement deux segments larges)
    genoux      80,3 %   (premier minimum local de largeur du membre)
    chevilles   91,7 %   (second minimum local)

🔴 CE QUI DOIT M'ARRÊTER. Le maillage fait 1,2 million de sommets : toute
opération non décimée est un piège à mémoire. Et une hauteur en pourcentage
n'est PAS une coordonnée Z — la boîte englobante du modèle doit être mesurée
avant toute sélection, jamais supposée.
"""
import sys
import os

import bpy
import bmesh
from mathutils import Vector

ARGS = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
GLB = ARGS[0] if ARGS else 'corps-base.glb'
SORTIE = ARGS[1] if len(ARGS) > 1 else '.'

# Mesuré sur le corps de base — voir squelette.json. En fraction de la hauteur,
# comptée depuis le SOMMET du crâne.
TAILLE, CHEVILLE = 0.524, 0.917
# 🔴 UNE TRANCHE EN Z NE DISTINGUE PAS UNE JAMBE D'UNE MAIN. Premier essai :
# les mains, qui pendent à la hauteur des cuisses, se sont retrouvées prises
# dans le pantalon — deux blocs de tissu de part et d'autre du personnage.
#
# ⭐ L'histogramme des distances à l'axe, mesuré sur le modèle, a une VALLÉE
# franche (largeur du personnage = 0,382, axe en x = 0) :
#
#     à 55 %   corps jusqu'à 0,093 · RIEN de 0,093 à 0,124 · bras de 0,124 à 0,186
#     à 70 %   corps jusqu'à 0,110 · RIEN                  · bras au-delà
#     à 75 %   plus de bras du tout, jambes à 0,095 maximum
#
# Le seuil se lit dans la vallée, il ne se choisit pas.
ECART_MAX = 0.105
EPAISSEUR = 0.006          # en fraction de la hauteur du personnage
DECIMATION = 0.28          # 1,2 M de sommets ne servent à rien pour une coque
# ⚠️ 0,12 détruisait le maillage : le pantalon sortait en lambeaux.


def nettoyer():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for bloc in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for x in list(bloc):
            bloc.remove(x)


def charger(chemin):
    bpy.ops.import_scene.gltf(filepath=chemin)
    objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not objs:
        raise SystemExit('  🔴 aucun maillage dans le GLB')
    bpy.context.view_layer.objects.active = objs[0]
    for o in objs:
        o.select_set(True)
    if len(objs) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def boite(obj):
    """La boîte englobante EN COORDONNÉES MONDE — jamais supposée."""
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    return (Vector((min(p.x for p in pts), min(p.y for p in pts),
                    min(p.z for p in pts))),
            Vector((max(p.x for p in pts), max(p.y for p in pts),
                    max(p.z for p in pts))))


def main():
    nettoyer()
    corps = charger(GLB)
    corps.name = 'corps'
    lo, hi = boite(corps)
    H = hi.z - lo.z
    print(f'  corps : {len(corps.data.vertices):,} sommets · '
          f'boîte {hi.x-lo.x:.3f} x {hi.y-lo.y:.3f} x {H:.3f}')

    # Les hauteurs mesurées, converties en Z monde. 0 % = sommet du crâne.
    z_haut = hi.z - H * TAILLE
    z_bas = hi.z - H * CHEVILLE
    print(f'  pantalon : de z={z_bas:.3f} à z={z_haut:.3f} '
          f'({(CHEVILLE-TAILLE)*100:.1f} % de la hauteur)')

    # ── LA COQUE ─────────────────────────────────────────────────────────
    bpy.ops.object.select_all(action='DESELECT')
    corps.select_set(True)
    bpy.context.view_layer.objects.active = corps
    bpy.ops.object.duplicate()
    pantalon = bpy.context.view_layer.objects.active
    pantalon.name = 'pantalon'

    # Décimer AVANT tout le reste : 1,2 M de sommets pour une coque de
    # vêtement est un piège à mémoire, pas de la précision.
    d = pantalon.modifiers.new('dec', 'DECIMATE')
    d.ratio = DECIMATION
    bpy.ops.object.modifier_apply(modifier=d.name)
    # ⭐ SOUDER LES SOMMETS DOUBLES, AVANT TOUT LE RESTE. Un maillage issu d'un
    # scan porte des sommets dupliqués qui coupent artificiellement la surface :
    # c'est pour ça que la découpe rendait 858 composantes là où il y en a deux
    # ou trois. Et QuadriFlow « can't operate when two vertices are very close
    # together (up to a difference of 0.0001) ».
    bm = bmesh.new()
    bm.from_mesh(pantalon.data)
    n0 = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=H * 0.0008)
    bm.to_mesh(pantalon.data)
    bm.free()
    print(f'  décimé à {n0:,} puis soudé à {len(pantalon.data.vertices):,} sommets')

    # ── LA COUPE ─────────────────────────────────────────────────────────
    # 🔴 SUPPRIMER DES SOMMETS DONNE UN BORD EN ESCALIER. Premier essai : on
    # retirait tout sommet hors de la tranche, ce qui laisse les faces coupées
    # en dents de scie — le pantalon sortait en lambeaux, haut et bas.
    #
    # ⭐ `bisect_plane` coupe le maillage LE LONG D'UN PLAN : il crée les
    # arêtes exactement à la hauteur voulue, puis jette ce qui dépasse. Le bord
    # est net par construction, à n'importe quelle densité de maillage.
    mw = pantalon.matrix_world
    cx = (lo.x + hi.x) / 2
    for z, sens in ((z_haut, 1), (z_bas, -1)):
        bm = bmesh.new()
        bm.from_mesh(pantalon.data)
        bmesh.ops.bisect_plane(
            bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:],
            plane_co=(0, 0, z), plane_no=(0, 0, sens),
            clear_outer=True, clear_inner=False)
        bm.to_mesh(pantalon.data)
        bm.free()
    print(f'  après coupe : {len(pantalon.data.vertices):,} sommets')

    # ── LES BRAS, PAR COMPOSANTES ────────────────────────────────────────
    # Les mains pendent à la hauteur des cuisses : la tranche les emporte. Elles
    # forment cependant des composantes SÉPARÉES des jambes — on ne garde que
    # celles dont le centre est proche de l'axe du corps (vallée mesurée à
    # 0,105 ; jambes ≤ 0,095, bras ≥ 0,124).
    bm = bmesh.new()
    bm.from_mesh(pantalon.data)
    bm.verts.ensure_lookup_table()
    vus, groupes = set(), []
    for v0 in bm.verts:
        if v0.index in vus:
            continue
        pile, comp = [v0], []
        vus.add(v0.index)
        while pile:
            v = pile.pop()
            comp.append(v)
            for e in v.link_edges:
                o = e.other_vert(v)
                if o.index not in vus:
                    vus.add(o.index)
                    pile.append(o)
        groupes.append(comp)
    jetes = []
    gardes = 0
    for comp in groupes:
        xs = [abs((mw @ v.co).x - cx) for v in comp]
        if len(comp) < 40 or (sum(xs) / len(xs)) > ECART_MAX:
            jetes.extend(comp)
        else:
            gardes += 1
    if jetes:
        bmesh.ops.delete(bm, geom=jetes, context='VERTS')
    bm.to_mesh(pantalon.data)
    bm.free()
    print(f'  {len(groupes)} composantes · {gardes} gardée(s) · '
          f'{len(pantalon.data.vertices):,} sommets')

    if not len(pantalon.data.vertices):
        raise SystemExit('  🔴 tranche vide — les bornes ne tombent pas '
                         'dans la boîte du modèle')

    # ── REFERMER LA SURFACE ──────────────────────────────────────────────
    # 🔴 LE FILTRE PAR COMPOSANTES JETTE DES MORCEAUX DU PANTALON. Mesuré :
    # 858 composantes, 62 gardées — le maillage de TRELLIS est un patchwork, et
    # écarter les petites laisse des FISSURES dans le tissu, visibles comme des
    # traits clairs en travers de la cuisse.
    #
    # ⭐ Le remaillage par voxels ne rapièce pas : il reconstruit UNE surface
    # fermée à partir du volume occupé. Les fissures n'ont plus lieu d'être, et
    # la topologie devient régulière — ce dont le skinning a besoin ensuite.
    # ⚠️ VOXEL puis QUADRIFLOW, et pas l'un OU l'autre.
    # « Voxel Remesh is for hard surfaces and quick blocking […] creates blocky
    #  topology unsuitable for organic characters. » — comparatif des remailleurs
    # Le voxel REFERME (c'est ce qu'on veut ici : les fissures du scan) mais
    # laisse une topologie cubique. QuadriFlow, libre et intégré à Blender,
    # reconstruit ensuite des quads alignés sur le flux de la surface — celle
    # dont le skinning a besoin pour se déformer proprement.
    # 🔴 PAS DE VOXEL REMESH. Deux raisons documentées, l'une et l'autre
    # vérifiées sur l'essai précédent :
    #   · « Voxel Remesh […] creates blocky topology unsuitable for organic
    #     characters » — le pantalon est sorti en cubes, littéralement ;
    #   · « non-manifolds are practically inevitable when using it with Voxel
    #     Remesher » (Blender T70548) — d'où l'échec silencieux de QuadriFlow
    #     juste après, 25 550 sommets à l'entrée comme à la sortie.
    #
    # ⭐ QuadriFlow seul, mais à L'ÉCHELLE. Bug Blender #106883, « QuadriFlow
    # remesh does not work on tiny objects » : sur un objet de 0,39 unité il
    # rend une erreur de manifold sur un maillage pourtant propre. On agrandit,
    # on remaille, on remet — c'est le contournement officiel.
    n_av = len(pantalon.data.vertices)
    bpy.ops.object.select_all(action='DESELECT')
    pantalon.select_set(True)
    bpy.context.view_layer.objects.active = pantalon
    ECHELLE = 100.0
    pantalon.scale = (ECHELLE, ECHELLE, ECHELLE)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    try:
        bpy.ops.object.quadriflow_remesh(target_faces=8000)
        print(f'  quadriflow : {n_av:,} → {len(pantalon.data.vertices):,} sommets')
    except Exception as e:
        print(f'  🔴 quadriflow refuse : {type(e).__name__} — {e}')
    pantalon.scale = (1 / ECHELLE, 1 / ECHELLE, 1 / ECHELLE)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # ── L'ÉPAISSEUR ──────────────────────────────────────────────────────
    # ⭐ Le tissu est DEHORS : offset 1 pousse la coque vers l'extérieur, donc
    # le pantalon enveloppe la jambe au lieu de la traverser.
    s = pantalon.modifiers.new('sol', 'SOLIDIFY')
    s.thickness = H * EPAISSEUR
    s.offset = 1.0
    bpy.ops.object.modifier_apply(modifier=s.name)

    # ── CONTRE LE CLIPPING : SHRINKWRAP À OFFSET ─────────────────────────
    # La méthode établie du domaine, que je n'avais pas cherchée : « use a
    # Shrinkwrap modifier with a small positive offset (0.001-0.002 units) »
    # pour empêcher un vêtement ajusté de traverser le corps. C'est exactement
    # la peau qui perçait au genou sur l'essai précédent, et que j'allais
    # « corriger » au jugé.
    sw = pantalon.modifiers.new('sw', 'SHRINKWRAP')
    sw.target = corps
    sw.wrap_method = 'NEAREST_SURFACEPOINT'
    sw.offset = H * 0.0035
    bpy.ops.object.modifier_apply(modifier=sw.name)

    # un lissage léger : la décimation laisse des arêtes dures
    sm = pantalon.modifiers.new('lis', 'SMOOTH')
    sm.factor = 0.5
    sm.iterations = 3
    bpy.ops.object.modifier_apply(modifier=sm.name)
    bpy.ops.object.shade_smooth()

    # ── LE MATÉRIAU — la couleur est un réglage, pas une texture ─────────
    mat = bpy.data.materials.new('tissu')
    mat.use_nodes = True
    p = mat.node_tree.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = (0.106, 0.128, 0.062, 1)  # olive
    p.inputs['Roughness'].default_value = 0.92
    if 'Specular IOR Level' in p.inputs:
        p.inputs['Specular IOR Level'].default_value = 0.15
    pantalon.data.materials.clear()
    pantalon.data.materials.append(mat)

    # ── LA CAMÉRA : ORTHOGRAPHIQUE, de face ──────────────────────────────
    # 🔴 Une caméra en perspective déforme : les jambes du bas paraîtraient
    # plus étroites que le haut, et la texture ne se superposerait plus au
    # corps 2D. L'orthographique conserve les proportions exactement.
    cam_d = bpy.data.cameras.new('cam')
    cam_d.type = 'ORTHO'
    cam_d.ortho_scale = max(hi.x - lo.x, H) * 1.05
    cam = bpy.data.objects.new('cam', cam_d)
    bpy.context.scene.collection.objects.link(cam)
    centre = (lo + hi) / 2
    cam.location = (centre.x, lo.y - H * 2, centre.z)
    cam.rotation_euler = (1.5708, 0, 0)
    bpy.context.scene.camera = cam

    # Éclairage doux et frontal : on veut la FORME, pas un clair-obscur
    for pos, e in (((0, -H * 2, H), 3.0), ((-H, -H * 2, H * .6), 1.5),
                   ((H, -H * 2, H * .6), 1.5)):
        ld = bpy.data.lights.new('l', 'AREA')
        ld.energy = e * H * H * 40
        ld.size = H
        lo_ = bpy.data.objects.new('l', ld)
        lo_.location = pos
        lo_.rotation_euler = (1.2, 0, 0)
        bpy.context.scene.collection.objects.link(lo_)

    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_EEVEE_NEXT'
    sc.render.film_transparent = True
    sc.render.resolution_x = 1536
    sc.render.resolution_y = 2752
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'RGBA'

    os.makedirs(SORTIE, exist_ok=True)
    # 1. le pantalon SEUL — c'est la texture qui entrera dans le rig 2D
    corps.hide_render = True
    sc.render.filepath = f'{SORTIE}/pantalon3d.png'
    bpy.ops.render.render(write_still=True)
    # 2. porté, pour juger le fit
    corps.hide_render = False
    sc.render.filepath = f'{SORTIE}/porte3d.png'
    bpy.ops.render.render(write_still=True)
    print(f'  ÉCRIT {SORTIE}/pantalon3d.png et porte3d.png')


main()
