/**
 * MEDMOJI — RULE ENGINE
 *
 * Med, 30 août 2026, dans la spec : « Si quelque chose est invalide :
 * NE PAS RENDER. Retourner une erreur structurée. »
 *
 * 🔴 IL NE CORRIGE RIEN, IL REFUSE. Un moteur qui rattrape silencieusement une
 * combinaison invalide livre une image que personne n'a validée — exactement ce
 * que toute cette architecture existe pour empêcher.
 *
 * ⭐ POURQUOI IL EST LA GARDE CONTRE L'HALLUCINATION. L'orchestrateur IA ne
 * produit que du JSON, et il peut inventer un identifiant qui n'existe pas.
 * Ici, un identifiant inventé est REFUSÉ — ce qui rend l'hallucination
 * inoffensive au lieu d'invisible. Sans ce contrôle, elle produirait un avatar
 * incomplet que rien ne signalerait.
 */
import type {
  Avatar, Catalogue, Piece, Validation, Erreur, Morphologie,
} from './modeles';
import { facteursDe, HORS_CORPULENCE } from './modeles';

const MOTEURS_CONNUS = ['medmoji_v1.0.0'];

function echec(erreur: Erreur, details: string, suggestion?: string): Validation {
  return { valide: false, erreur, details, suggestion };
}

/** Toutes les pièces référencées par un avatar, dans l'ordre de déclaration. */
export function piecesDe(a: Avatar): string[] {
  const ids: (string | null | undefined)[] = [
    a.cheveux,
    a.tenue.haut, a.tenue.manteau, a.tenue.bas, a.tenue.chaussures,
    ...(a.accessoires ?? []),
    ...Object.values(a.visage.pieces ?? {}),
  ];
  return ids.filter((x): x is string => typeof x === 'string' && x.length > 0);
}

export function valider(a: Avatar, cat: Catalogue): Validation {
  // ── 1. VERSION DU MOTEUR ─────────────────────────────────────────────
  if (!MOTEURS_CONNUS.includes(a.moteur)) {
    return echec('MOTEUR_INCONNU',
      `${a.moteur} — connus : ${MOTEURS_CONNUS.join(', ')}`);
  }

  const ids = piecesDe(a);
  const pieces: Piece[] = [];

  // ── 2. LES PIÈCES EXISTENT ───────────────────────────────────────────
  for (const id of ids) {
    const p = cat.pieces[id];
    if (!p) {
      // ⚠️ On propose un remplaçant de la même catégorie plutôt que d'échouer
      // sèchement : l'orchestrateur peut se rattraper sans repasser par Med.
      const prefixe = id.split('_')[0];
      const proche = Object.values(cat.pieces).find((q) => q.id.startsWith(prefixe));
      return echec('PIECE_INTROUVABLE', `${id} n'est pas au catalogue`, proche?.id);
    }
    pieces.push(p);
  }

  // ── 3. UN SEUL OCCUPANT PAR PLAN ─────────────────────────────────────
  // L'ordre de chargement ne doit jamais départager deux pièces.
  const parPlan = new Map<number, string>();
  for (const p of pieces) {
    const deja = parPlan.get(p.plan);
    if (deja) {
      return echec('PLAN_EN_DOUBLE', `${p.id} et ${deja} occupent tous deux le plan ${p.plan}`);
    }
    parPlan.set(p.plan, p.id);
  }

  // ── 4. STYLE ─────────────────────────────────────────────────────────
  for (const p of pieces) {
    if (p.style !== a.style) {
      return echec('STYLE_INCOMPATIBLE',
        `${p.id} est en ${p.style}, l'avatar en ${a.style}`);
    }
  }

  // ── 5. MORPHOLOGIE ───────────────────────────────────────────────────
  const m: Morphologie = a.corps.morphologie;
  for (const p of pieces) {
    const ok = p.morphologies.includes('toutes' as never) || p.morphologies.includes(m as never);
    if (!ok) {
      return echec('MORPHOLOGIE_INCOMPATIBLE',
        `${p.id} n'est pas déclarée pour la morphologie ${m}`);
    }
  }

  // ── 6. DÉFORMATION SUPPORTABLE ───────────────────────────────────────
  // ⭐ Étirer un raster de 25 % le déforme visiblement à un moment donné. La
  // pièce déclare son point de rupture, mesuré à la fabrication de 0,8× à 1,4×.
  // Au-delà, on refuse au lieu de livrer un vêtement distordu.
  for (const p of pieces) {
    // ⚠️ On passe par `facteursDe` — la MÊME fonction que le plan. Lire
    // `p.corpulence` directement ici laissait passer toute pièce sans
    // déclaration, que le plan étirait pourtant. Défaut trouvé par le témoin.
    if (p.deformationMax !== undefined && !HORS_CORPULENCE.includes(p.categorie)) {
      const f = facteursDe(p, m);
      const etirement = Math.max(f.x, f.y);
      if (etirement > p.deformationMax) {
        return echec('DEFORMATION_EXCESSIVE',
          `${p.id} devrait s'étirer à ${etirement.toFixed(2)}× en ${m}, ` +
          `au-delà de son maximum mesuré (${p.deformationMax})`);
      }
    }
  }

  // ── 7. INCOMPATIBILITÉS DÉCLARÉES ────────────────────────────────────
  const presents = new Set(ids);
  for (const p of pieces) {
    for (const x of p.incompatible ?? []) {
      if (presents.has(x)) {
        return echec('COMBINAISON_INVALIDE', `${p.id} est incompatible avec ${x}`);
      }
    }
  }

  // ── 8. OCCLUSION SANS CYCLE ──────────────────────────────────────────
  // A devant B, B devant A : le renderer n'a aucun moyen de trancher.
  const devant = new Map<string, Set<string>>();
  for (const p of pieces) {
    for (const [autre, sens] of Object.entries(p.occlusion ?? {})) {
      if (!presents.has(autre)) continue;
      const de = sens === 'devant' ? p.id : autre;
      const vers = sens === 'devant' ? autre : p.id;
      if (!devant.has(de)) devant.set(de, new Set());
      devant.get(de)!.add(vers);
    }
  }
  if (aUnCycle(devant)) {
    return echec('OCCLUSION_CYCLIQUE', 'les relations devant/derrière forment un cycle');
  }

  // ── 9. ANCRES CONNUES ────────────────────────────────────────────────
  const ANCRES = new Set(['col', 'epaule_g', 'epaule_d', 'taille', 'poignet_g',
                          'poignet_d', 'cheville_g', 'cheville_d', 'crane']);
  for (const p of pieces) {
    for (const nom of Object.keys(p.ancres ?? {})) {
      if (!ANCRES.has(nom)) {
        return echec('ANCRE_INCONNUE', `${p.id} déclare l'ancre inconnue « ${nom} »`);
      }
    }
  }

  // ── 10. BORNES DES MORPHS DE VISAGE ──────────────────────────────────
  for (const [nom, v] of Object.entries(a.visage.morphs ?? {})) {
    if (!Number.isFinite(v) || v < -1 || v > 1) {
      return echec('MORPH_HORS_BORNES', `${nom} = ${v}, attendu entre -1 et 1`);
    }
  }

  // ── 11. COUVERTURE ───────────────────────────────────────────────────
  // ⚠️ Un manteau seul suffit : il couvre le torse. C'est l'absence des DEUX
  // qui laisse le corps découvert.
  if (!a.tenue.haut && !a.tenue.manteau) {
    return echec('TORSE_DECOUVERT', 'ni haut ni manteau');
  }

  return { valide: true };
}

/** Détection de cycle par parcours en profondeur, avec pile d'exploration. */
function aUnCycle(g: Map<string, Set<string>>): boolean {
  const vu = new Set<string>();
  const pile = new Set<string>();
  const visiter = (n: string): boolean => {
    if (pile.has(n)) return true;
    if (vu.has(n)) return false;
    vu.add(n); pile.add(n);
    for (const s of g.get(n) ?? []) if (visiter(s)) return true;
    pile.delete(n);
    return false;
  };
  for (const n of g.keys()) if (visiter(n)) return true;
  return false;
}
