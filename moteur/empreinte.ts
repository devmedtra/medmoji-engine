/**
 * MEDMOJI — EMPREINTE DÉTERMINISTE
 *
 * Med, 30 août 2026 : « Le rendu doit être déterministe. Même configuration =
 * même avatar. »
 *
 * 🔴 `JSON.stringify` NE SUFFIT PAS. L'ordre des clés d'un objet JavaScript
 * suit l'ordre d'insertion : deux codes qui construisent le même avatar dans un
 * ordre différent produisent deux chaînes différentes, donc deux empreintes
 * différentes, donc deux entrées de cache pour une seule image. Le cache
 * gonfle, le CDN sert deux fichiers identiques, et personne ne s'en aperçoit
 * jamais — c'est un défaut qui ne casse rien, il coûte seulement de l'argent
 * en silence.
 *
 * On canonise donc : clés triées à toute profondeur, nombres normalisés,
 * `undefined` et `null` traités pareil.
 *
 * ⚠️ LES NOMBRES SONT LE PIÈGE SUIVANT. `0.1 + 0.2` vaut 0.30000000000000004.
 * Deux plateformes qui calculent une échelle par des chemins différents peuvent
 * diverger au 16ᵉ chiffre et produire deux empreintes. On arrondit donc TOUT
 * nombre à 6 décimales avant l'empreinte — bien au-delà de la précision utile
 * (un millionième de 1536 px vaut 0,0015 px) et bien en deçà du bruit flottant.
 */
import { createHash } from 'node:crypto';

/** Arrondi stable : -0 devient 0, sinon deux zéros donnent deux empreintes. */
function nombre(n: number): number {
  if (!Number.isFinite(n)) throw new Error(`nombre non fini dans l'empreinte : ${n}`);
  const r = Math.round(n * 1e6) / 1e6;
  return Object.is(r, -0) ? 0 : r;
}

/** Rend une valeur canonique : ordre stable, nombres normalisés. */
export function canoniser(v: unknown): unknown {
  if (v === null || v === undefined) return null;
  if (typeof v === 'number') return nombre(v);
  // ⚠️ UNICODE. « é » s'écrit d'une façon sur iOS (U+00E9) et d'une autre sur
  // certains claviers Android (U+0065 U+0301) : deux chaînes visuellement
  // identiques, deux octets différents, deux empreintes. Le défaut ne se voit
  // jamais — il double simplement les entrées de cache. Signalé par le conseil
  // d'IA le 30 août 2026 (norme RFC 8785).
  if (typeof v === 'string') return v.normalize('NFC');
  if (Array.isArray(v)) return v.map(canoniser);
  if (typeof v === 'object') {
    const o = v as Record<string, unknown>;
    const trie: Record<string, unknown> = {};
    for (const k of Object.keys(o).sort()) {
      // ⚠️ On garde les clés absentes HORS de l'empreinte : `{a:1}` et
      // `{a:1, b:undefined}` doivent donner la même chose, sinon ajouter un
      // champ optionnel non renseigné invaliderait tout le cache existant.
      if (o[k] === undefined) continue;
      trie[k] = canoniser(o[k]);
    }
    return trie;
  }
  return v;
}

export function empreinte(...parts: unknown[]): string {
  const texte = JSON.stringify(parts.map(canoniser));
  return createHash('sha256').update(texte, 'utf8').digest('hex');
}

/**
 * La clé de cache d'un rendu.
 *
 * ⭐ La taille en fait partie : une même configuration rendue en 256 et en 1024
 * donne deux fichiers, donc deux clés. L'oublier ferait servir la vignette là
 * où l'on attend le plein écran.
 */
export function cleDeCache(avatar: unknown, moteur: string, taille: number): string {
  return empreinte(avatar, moteur, taille);
}
