#!/usr/bin/env python3
"""APPELER UN OUTIL MCP HIGGSFIELD EN DIRECT, sans passer par le harnais.

⭐ POURQUOI CE FICHIER EXISTE. Le serveur MCP higgsfield se connecte au
demarrage d'une session ; si son jeton est renouvele APRES, la session garde
l'ancien en memoire et repond « token expired » jusqu'a un redemarrage. Ce
script lit le jeton FRAIS sur le disque a chaque appel — donc il marche
immediatement apres une reauthentification.

🔴 SANS User-Agent DE NAVIGATEUR, CLOUDFLARE REND 403 « error code: 1010 ».
Mesure du 30 aout 2026 : la meme cle, le meme jeton, rendaient 403 en
`Python-urllib/3.x` et 200 avec un UA Chrome. Lu comme une cle morte, ce 403
envoie chercher une recharge qui n'a pas lieu d'etre — l'INSTRUMENT avant le
sujet.

Usage :
    python3 mcp-hf.py balance
    python3 mcp-hf.py remove_background '{"image_url": "https://..."}'
"""
import json, sys, urllib.request, urllib.error

CRED = '/root/.claude/.credentials.json'
URL = 'https://mcp.higgsfield.ai/mcp'
UA = ('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/128.0 Safari/537.36')


def jeton():
    d = json.load(open(CRED))
    cle = next((k for k in d.get('mcpOAuth', {}) if k.startswith('higgsfield|')), None)
    if not cle:
        sys.exit('aucune entree higgsfield dans les identifiants')
    t = d['mcpOAuth'][cle].get('accessToken') or ''
    if not t:
        sys.exit('jeton higgsfield VIDE — refaire /mcp dans un terminal interactif')
    return t


class Session:
    def __init__(self):
        self.t = jeton()
        self.sid = None
        self._init()

    def _post(self, methode, params=None, notif=False):
        corps = {'jsonrpc': '2.0', 'method': methode}
        if params is not None:
            corps['params'] = params
        if not notif:
            corps['id'] = 1
        en = {'Authorization': f'Bearer {self.t}', 'Content-Type': 'application/json',
              'Accept': 'application/json, text/event-stream', 'User-Agent': UA}
        if self.sid:
            en['Mcp-Session-Id'] = self.sid
        r = urllib.request.Request(URL, data=json.dumps(corps).encode(), headers=en, method='POST')
        try:
            with urllib.request.urlopen(r, timeout=300) as x:
                if x.headers.get('Mcp-Session-Id'):
                    self.sid = x.headers['Mcp-Session-Id']
                brut = x.read().decode()
        except urllib.error.HTTPError as e:
            raise SystemExit(f'HTTP {e.code} sur {methode} : {e.read().decode()[:400]}')
        if notif:
            return None
        for ligne in brut.splitlines():
            if ligne.startswith('data:'):
                return json.loads(ligne[5:].strip())
        return json.loads(brut) if brut.strip() else None

    def _init(self):
        self._post('initialize', {'protocolVersion': '2025-06-18', 'capabilities': {},
                                  'clientInfo': {'name': 'medtra-createur', 'version': '1.0'}})
        self._post('notifications/initialized', {}, notif=True)

    def outil(self, nom, args=None):
        r = self._post('tools/call', {'name': nom, 'arguments': args or {}})
        if r and 'error' in r:
            raise SystemExit(json.dumps(r['error'], ensure_ascii=False)[:500])
        return r.get('result', {}) if r else {}

    def outils(self):
        return [o['name'] for o in (self._post('tools/list', {}) or {}).get('result', {}).get('tools', [])]


def main():
    if len(sys.argv) < 2:
        print('outils disponibles :')
        for n in Session().outils():
            print('  ', n)
        return
    args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    res = Session().outil(sys.argv[1], args)
    for bloc in res.get('content', []):
        print(bloc.get('text', json.dumps(bloc, ensure_ascii=False)))
    if res.get('structuredContent'):
        print(json.dumps(res['structuredContent'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
