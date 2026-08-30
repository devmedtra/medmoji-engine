#!/usr/bin/env python3
"""DETOURER UNE IMAGE — par l'outil dedie de Higgsfield, jamais par du code maison.

Med, 30 aout 2026 : « plein de problemes, c'est quoi le truc radical qu'on peut
faire ? » — apres une soiree de detourage maison (flood-fill qui mangeait le
t-shirt blanc, fond vert qui laissait un lisere, seuillage qui trouait les
chaussures). Le truc radical : `remove_background`, l'outil dedie.

⚠️ L'ORDRE COMPTE : detourer AVANT de normaliser. Une image non detouree porte
son fond avec elle ; apres normalisation, son alpha est un RECTANGLE OPAQUE et
toute mesure de silhouette devient fausse. Mesure du 30 aout : le profil de
largeur rendait 868 px identiques du crane aux pieds — une valeur constante sur
toutes les lignes, donc l'instrument, pas le sujet.
"""
import importlib.util, json, os, re, sys, time, urllib.request

spec = importlib.util.spec_from_file_location('m', '/root/medtra-avatar/createur/mcp-hf.py')
mcp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp)


def texte(r):
    for b in r.get('content', []):
        t = b.get('text', '')
        try:
            return json.loads(t)
        except Exception:
            return t
    return r.get('structuredContent') or r


def creuser(o, cles):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in cles and isinstance(v, str):
                return v
            r = creuser(v, cles)
            if r:
                return r
    elif isinstance(o, list):
        for v in o:
            r = creuser(v, cles)
            if r:
                return r
    return None


def uuid_dans(t):
    m = re.search(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', str(t))
    return m.group(0) if m else None


def deposer(s, chemin):
    up = texte(s.outil('media_upload', {'method': 'upload_url',
                                        'filename': os.path.basename(chemin), 'type': 'image'}))
    url = creuser(up, {'upload_url', 'url', 'presigned_url', 'put_url'})
    mid = creuser(up, {'media_id', 'id', 'uuid'})
    # ⚠️ `media_upload` repond parfois en TEXTE (une commande curl toute faite)
    # plutot qu'en JSON structure. Chercher des cles ne suffit donc pas : on lit
    # aussi la forme texte. Mesure du 30 aout — le meme appel a rendu les deux
    # formes a dix minutes d'intervalle.
    if not (url and mid):
        brut = up if isinstance(up, str) else json.dumps(up, ensure_ascii=False)
        m = re.search(r"'(https://[^']+)'", brut) or re.search(r'"(https://[^"]+)"', brut)
        url = url or (m.group(1) if m else None)
        mid = mid or uuid_dans(brut)
    if not (url and mid):
        sys.exit(f'media_upload inattendu : {str(up)[:500]}')
    urllib.request.urlopen(urllib.request.Request(
        url, data=open(chemin, 'rb').read(), method='PUT',
        headers={'Content-Type': 'image/png'}), timeout=180)
    s.outil('media_confirm', {'type': 'image', 'media_id': mid})
    return mid


def detourer(chemin, dest):
    s = mcp.Session()
    mid = deposer(s, chemin)
    print(f'  depose {os.path.basename(chemin)} -> {mid}')
    r = texte(s.outil('remove_background', {'params': {'media_id': mid, 'media_type': 'image'}}))
    brut = json.dumps(r, ensure_ascii=False) if not isinstance(r, str) else r
    jid = uuid_dans(brut)
    if not jid:
        sys.exit(f'pas de job : {brut[:500]}')
    print(f'  job {jid}')
    for i in range(60):
        time.sleep(4)
        st = texte(s.outil('job_status', {'jobId': jid, 'sync': True}))
        b = json.dumps(st, ensure_ascii=False) if not isinstance(st, str) else st
        urls = re.findall(r'https://[^"\\ )]+?\.(?:png|webp)', b)
        if urls:
            with urllib.request.urlopen(urls[0], timeout=300) as x:
                open(dest, 'wb').write(x.read())
            print(f'  ECRIT {dest} ({os.path.getsize(dest)/1024:.0f} Ko)')
            return dest
        if '"failed"' in b or 'canceled' in b:
            sys.exit(f'echec : {b[:400]}')
    sys.exit('delai depasse')


if __name__ == '__main__':
    detourer(sys.argv[1], sys.argv[2])
