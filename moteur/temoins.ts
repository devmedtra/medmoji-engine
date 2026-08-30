/**
 * MEDMOJI — LES TÉMOINS DU MOTEUR
 *
 * ⚠️ CHAQUE CONTRÔLE EST VALIDÉ PAR UN TÉMOIN : on réintroduit le défaut, le
 * test doit hurler ; on le retire, il doit se taire. Un contrôle qui ne trouve
 * jamais rien est indiscernable d'un contrôle cassé — deux des garde-fous de ce
 * dépôt étaient FAUX à leur première écriture et ne l'auraient jamais avoué.
 *
 * Lancer :  npx tsx moteur/temoins.ts
 */
import { construirePlan } from './plan';
import { valider } from './validateur';
import { empreinte, canoniser } from './empreinte';
import type { Avatar, Catalogue, Piece } from './modeles';
import { PLANS } from './modeles';

let reussis = 0, echoues = 0;
function verifier(nom: string, condition: boolean, detail = '') {
  if (condition) { reussis++; console.log(`  ✓ ${nom}`); }
  else { echoues++; console.log(`  ✗ ${nom}${detail ? ` — ${detail}` : ''}`); }
}

// ── UN CATALOGUE MINIMAL, ÉCRIT À LA MAIN ────────────────────────────────
function piece(o: Partial<Piece> & { id: string; plan: number }): Piece {
  return {
    categorie: 'haut', fichier: `${o.id}.png`, morphologies: ['toutes'],
    style: 'medmoji_v1', version: '1.0', ...o,
  } as Piece;
}

const cat: Catalogue = {
  styles: { medmoji_v1: { angleLumiere: 45, contraste: 1.0 } },
  pieces: Object.fromEntries([
    // Un parka ample encaisse l'étirement ; le manteau ajusté ci-dessous non.
    piece({ id: 'coat_parka_green_001', categorie: 'manteau', plan: PLANS.manteau,
            deformationMax: 1.30, occlusion: { 'hand_l': 'devant' } }),
    piece({ id: 'pants_charcoal_002', categorie: 'bas', plan: PLANS.bas }),
    piece({ id: 'boots_black_004', categorie: 'chaussures', plan: PLANS.chaussures }),
    piece({ id: 'hair_afro_11', categorie: 'cheveux', plan: PLANS.cheveux,
            incompatible: ['hat_cap_003'] }),
    piece({ id: 'hat_cap_003', categorie: 'accessoire', plan: PLANS.accessoires }),
    piece({ id: 'hat_beanie_007', categorie: 'accessoire', plan: PLANS.accessoires }),
    piece({ id: 'tee_base_001', categorie: 'haut', plan: PLANS.haut }),
    piece({ id: 'coat_slim_009', categorie: 'manteau', plan: PLANS.manteau,
            deformationMax: 1.05 }),   // se déforme vite : sert au témoin 6
    piece({ id: 'coat_v2_001', categorie: 'manteau', plan: PLANS.manteau,
            style: 'medmoji_v2' }),
    piece({ id: 'shoes_dressy_012', categorie: 'chaussures', plan: PLANS.chaussures,
            morphologies: ['mince', 'moyenne'] }),
  ].map((p) => [p.id, p])),
};

const base: Avatar = {
  moteur: 'medmoji_v1.0.0',
  corps: { morphologie: 'moyenne', taille: 1.0, teint: 't3' },
  visage: { origine: 'scan', scanId: 'sc_test', morphs: { ecart_yeux: 0.02 } },
  cheveux: 'hair_afro_11',
  tenue: { haut: null, manteau: 'coat_parka_green_001',
           bas: 'pants_charcoal_002', chaussures: 'boots_black_004' },
  accessoires: [],
  pose: 'debout', expression: 'neutre', style: 'medmoji_v1',
};

console.log('\n═══ 1. L\'AVATAR DE RÉFÉRENCE PASSE ═══');
verifier('avatar valide', valider(base, cat).valide,
         JSON.stringify(valider(base, cat)));

console.log('\n═══ 2. CHAQUE CONTRÔLE ATTRAPE SON DÉFAUT ═══');
// ⚠️ Ces témoins sont l'inverse du premier : ils vérifient que le validateur
// REFUSE. Un validateur qui accepte tout passerait le test 1 sans problème.
const cas: [string, Avatar, string][] = [
  ['moteur inconnu',
   { ...base, moteur: 'medmoji_v9' }, 'MOTEUR_INCONNU'],
  ['pièce introuvable',
   { ...base, tenue: { ...base.tenue, manteau: 'coat_fantome_999' } }, 'PIECE_INTROUVABLE'],
  ['deux pièces au même plan',
   { ...base, accessoires: ['hat_cap_003', 'hat_beanie_007'] }, 'PLAN_EN_DOUBLE'],
  ['style incompatible',
   { ...base, tenue: { ...base.tenue, manteau: 'coat_v2_001' } }, 'STYLE_INCOMPATIBLE'],
  ['morphologie non déclarée',
   { ...base, corps: { ...base.corps, morphologie: 'forte' },
     tenue: { ...base.tenue, chaussures: 'shoes_dressy_012' } }, 'MORPHOLOGIE_INCOMPATIBLE'],
  ['déformation excessive',
   { ...base, corps: { ...base.corps, morphologie: 'forte' },
     tenue: { ...base.tenue, manteau: 'coat_slim_009' } }, 'DEFORMATION_EXCESSIVE'],
  ['combinaison interdite',
   { ...base, accessoires: ['hat_cap_003'] }, 'COMBINAISON_INVALIDE'],
  ['morph hors bornes',
   { ...base, visage: { ...base.visage, morphs: { ecart_yeux: 4.2 } } }, 'MORPH_HORS_BORNES'],
  ['torse découvert',
   { ...base, tenue: { ...base.tenue, manteau: null, haut: null } }, 'TORSE_DECOUVERT'],
];
for (const [nom, a, attendu] of cas) {
  const r = valider(a, cat);
  verifier(`refuse : ${nom}`, !r.valide && r.erreur === attendu,
           `obtenu ${r.erreur ?? 'VALIDE'}, attendu ${attendu}`);
}

console.log('\n═══ 3. LE VALIDATEUR PROPOSE UN REMPLAÇANT ═══');
const manquant = valider(
  { ...base, tenue: { ...base.tenue, manteau: 'coat_inexistant_42' } }, cat);
verifier('suggestion fournie', manquant.suggestion === 'coat_parka_green_001',
         `obtenu ${manquant.suggestion}`);

console.log('\n═══ 4. DÉTERMINISME DE L\'EMPREINTE ═══');
const p1 = construirePlan(base, cat);
const p2 = construirePlan(JSON.parse(JSON.stringify(base)), cat);
verifier('deux constructions identiques → même empreinte', p1.empreinte === p2.empreinte);

// 🔴 LE TÉMOIN QUI COMPTE : le même avatar écrit dans un ORDRE DE CLÉS
// différent. `JSON.stringify` seul donnerait deux empreintes ici, et le cache
// stockerait deux fois la même image sans que personne ne le voie jamais.
const desordre: Avatar = {
  style: 'medmoji_v1', expression: 'neutre', pose: 'debout',
  accessoires: [],
  tenue: { chaussures: 'boots_black_004', bas: 'pants_charcoal_002',
           manteau: 'coat_parka_green_001', haut: null },
  cheveux: 'hair_afro_11',
  visage: { morphs: { ecart_yeux: 0.02 }, scanId: 'sc_test', origine: 'scan' },
  corps: { teint: 't3', taille: 1.0, morphologie: 'moyenne' },
  moteur: 'medmoji_v1.0.0',
};
verifier('ordre des clés indifférent',
         construirePlan(desordre, cat).empreinte === p1.empreinte);

// Le bruit flottant ne doit pas changer l'empreinte.
verifier('0,1 + 0,2 ≡ 0,3', empreinte(0.1 + 0.2) === empreinte(0.3));
verifier('-0 ≡ 0', empreinte(-0) === empreinte(0));
verifier('clé absente ≡ clé indéfinie',
         empreinte({ a: 1 }) === empreinte({ a: 1, b: undefined }));
// 🔴 UNICODE — trouvé par le conseil d'IA. « é » précomposé (U+00E9) contre
// « e » + accent combinant (U+0065 U+0301) : même texte à l'écran, deux suites
// d'octets. Sans normalisation, deux entrées de cache pour une seule image, et
// personne ne le voit jamais.
verifier('unicode composé ≡ décomposé',
         empreinte('caf\u00e9') === empreinte('cafe\u0301'));

console.log('\n═══ 5. UN CHANGEMENT RÉEL CHANGE L\'EMPREINTE ═══');
// ⚠️ L'inverse du témoin 4, et il est indispensable : une fonction qui rend
// toujours la même empreinte passerait TOUS les tests ci-dessus.
const autre = construirePlan(
  { ...base, tenue: { ...base.tenue, chaussures: null, haut: 'tee_base_001' } }, cat);
verifier('tenue différente → empreinte différente', autre.empreinte !== p1.empreinte);
const grosse = construirePlan({ ...base, corps: { ...base.corps, morphologie: 'forte' } }, cat);
verifier('morphologie différente → empreinte différente', grosse.empreinte !== p1.empreinte);

console.log('\n═══ 6. L\'ORDRE DES PLANS EST RESPECTÉ ═══');
const plans = p1.couches.map((c) => c.plan);
verifier('couches triées par plan croissant',
         plans.every((v, i) => i === 0 || plans[i - 1] <= v), plans.join(', '));
verifier('le corps est présent', p1.couches.some((c) => c.pieceId.startsWith('corps_')));
verifier('les cheveux passent après le corps',
         p1.couches.findIndex((c) => c.pieceId === 'hair_afro_11') >
         p1.couches.findIndex((c) => c.pieceId.startsWith('corps_')));

console.log('\n═══ 7. LA CORPULENCE N\'ÉTIRE PAS LE VISAGE ═══');
const cheveuxForte = grosse.couches.find((c) => c.pieceId === 'hair_afro_11')!;
const manteauForte = grosse.couches.find((c) => c.pieceId === 'coat_parka_green_001')!;
verifier('cheveux à l\'échelle 1', cheveuxForte.echelle.x === 1,
         `obtenu ${cheveuxForte.echelle.x}`);
verifier('manteau élargi', manteauForte.echelle.x > 1.2,
         `obtenu ${manteauForte.echelle.x}`);

console.log('\n═══ 8. UN AVATAR INVALIDE NE SE REND PAS ═══');
let leve = false;
try { construirePlan({ ...base, moteur: 'medmoji_v9' }, cat); }
catch { leve = true; }
verifier('construirePlan refuse un avatar invalide', leve);

console.log(`\n${'─'.repeat(52)}`);
console.log(`  ${reussis} réussis · ${echoues} échoués`);
process.exit(echoues === 0 ? 0 : 1);
