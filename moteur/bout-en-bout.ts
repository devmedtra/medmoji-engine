/**
 * MEDMOJI — LE BOUT EN BOUT
 *
 * Un Avatar JSON entre, une image sort. Avec les VRAIES pièces fabriquées dans
 * la nuit du 29 au 30 août 2026 : le personnage neutre détouré et mis au
 * gabarit, et le parka extrait par segmentation.
 *
 * ⭐ CE QUE CE FICHIER PROUVE, et qu'aucun test unitaire ne prouve : que la
 * chaîne complète tient. Fabrique → catalogue → validation → plan → pixels.
 *
 * Lancer :  npx tsx moteur/bout-en-bout.ts
 */
import { writeFileSync } from 'node:fs';
import { construirePlan } from './plan';
import { rendre } from './rendu-node';
import { cleDeCache } from './empreinte';
import { PLANS, type Avatar, type Catalogue } from './modeles';

const RACINE = '/root/medtra-avatar/moteur/catalogue-test';

const catalogue: Catalogue = {
  styles: { medmoji_v1: { angleLumiere: 45, contraste: 1.0 } },
  pieces: {
    coat_parka_green_001: {
      id: 'coat_parka_green_001',
      categorie: 'manteau',
      plan: PLANS.manteau,
      fichier: 'manteau/coat_parka_green_001.png',
      // Mesurée sur l'image : le vêtement commence à 27,1 % de la hauteur.
      ligneRaccord: 0.271,
      deformationMax: 1.30,
      morphologies: ['toutes'],
      style: 'medmoji_v1',
      version: '1.0',
    },
  },
};

const avatar: Avatar = {
  moteur: 'medmoji_v1.0.0',
  corps: { morphologie: 'moyenne', taille: 1.0, teint: 't3' },
  visage: { origine: 'scan', scanId: 'sc_demo' },
  cheveux: null,
  tenue: { haut: null, manteau: 'coat_parka_green_001', bas: null, chaussures: null },
  accessoires: [],
  pose: 'debout', expression: 'neutre', style: 'medmoji_v1',
};

async function main() {
  const plan = construirePlan(avatar, catalogue);
  console.log(`plan       : ${plan.couches.length} couches`);
  for (const c of plan.couches) {
    console.log(`   plan ${String(c.plan).padStart(2)}  ${c.pieceId.padEnd(24)} ` +
                `échelle ${c.echelle.x.toFixed(2)}×${c.echelle.y.toFixed(2)}`);
  }
  console.log(`empreinte  : ${plan.empreinte.slice(0, 32)}…`);
  console.log(`clé cache  : ${cleDeCache(avatar, plan.moteur, 512).slice(0, 32)}…`);

  for (const taille of [1024, 256]) {
    const png = await rendre(plan, { racine: RACINE, taille });
    const dest = `/root/medtra-avatar/moteur/sortie-${taille}.png`;
    writeFileSync(dest, png);
    console.log(`rendu      : ${dest}  ${(png.length / 1024).toFixed(0)} Ko`);
  }

  // ── LE TÉMOIN QUI COMPTE : deux rendus successifs du même plan doivent
  //    produire les MÊMES OCTETS. Si sharp introduisait le moindre aléa —
  //    horodatage PNG, tramage, ordre de threads — le cache serait inutile et
  //    on ne le saurait jamais. ──
  const a = await rendre(plan, { racine: RACINE, taille: 512 });
  const b = await rendre(plan, { racine: RACINE, taille: 512 });
  console.log(`\ndéterminisme des pixels : ${a.equals(b) ? 'IDENTIQUE ✓' : 'DIVERGENT ✗'} ` +
              `(${a.length} vs ${b.length} octets)`);
  process.exit(a.equals(b) ? 0 : 1);
}

main().catch((e) => { console.error(String(e)); process.exit(1); });
