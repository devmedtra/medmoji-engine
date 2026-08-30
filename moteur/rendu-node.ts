/**
 * MEDMOJI — ADAPTATEUR DE RENDU (Node / sharp)
 *
 * ⭐ LE MOTEUR NE DESSINE PAS, IL PLANIFIE. Ce fichier est le seul qui touche
 * des pixels et le disque — et c'est pour ça qu'il est SÉPARÉ. L'app aura son
 * jumeau écrit avec Skia ; les deux exécutent le même plan et doivent produire
 * la même image. Toute décision prise ici et pas dans le plan est une occasion
 * de diverger entre les deux plateformes.
 *
 * 🔴 PIÈGE SHARP, MESURÉ LE 30 AOÛT : dans un même pipeline, `resize`
 * s'applique AVANT `composite`, quel que soit l'ordre d'écriture. Redimensionner
 * une couche puis la composer dans la foulée redimensionne donc le RÉSULTAT, pas
 * la couche. Chaque couche est matérialisée en tampon avant d'être composée.
 */
import sharp from 'sharp';
import { join } from 'node:path';
import type { PlanDeRendu } from './modeles';

export interface OptionsRendu {
  /** Racine du catalogue : les `fichier` des couches y sont relatifs. */
  racine: string;
  /** Côté le plus long de l'image produite. */
  taille?: number;
  /** Fond : transparent par défaut. */
  fond?: { r: number; g: number; b: number; alpha: number };
}

export async function rendre(plan: PlanDeRendu, o: OptionsRendu): Promise<Buffer> {
  const { largeur, hauteur } = plan.gabarit;

  // ── 1. CHAQUE COUCHE, MATÉRIALISÉE ───────────────────────────────────
  const composites: sharp.OverlayOptions[] = [];
  for (const c of plan.couches) {
    const chemin = join(o.racine, c.fichier);

    const nl = Math.round(largeur * c.echelle.x);
    const nh = Math.round(hauteur * c.echelle.y);

    // ⚠️ `.toBuffer()` ICI, et pas plus tard : c'est ce qui force le
    // redimensionnement à s'appliquer à la COUCHE.
    let buf: Buffer;
    try {
      buf = await sharp(chemin)
        .resize(nl, nh, { fit: 'fill', kernel: 'lanczos3' })
        .toBuffer();
    } catch (e) {
      // Une pièce manquante est une ERREUR, pas une couche qu'on saute en
      // silence : un avatar rendu sans son manteau ressemble à un avatar
      // valide, et rien ne signalerait le défaut.
      throw new Error(`couche introuvable ou illisible : ${c.fichier} (${c.pieceId})`);
    }

    composites.push({
      input: buf,
      left: Math.round(c.decalage.x),
      top: Math.round(c.decalage.y),
    });
  }

  // ── 2. COMPOSITION, dans l'ordre du plan ─────────────────────────────
  // `composite` respecte l'ordre du tableau ; le plan l'a déjà trié par
  // profondeur, donc on ne retrie surtout pas ici.
  let img = sharp({
    create: {
      width: largeur, height: hauteur, channels: 4,
      background: o.fond ?? { r: 0, g: 0, b: 0, alpha: 0 },
    },
  }).composite(composites);

  // ── 3. TAILLE FINALE ─────────────────────────────────────────────────
  if (o.taille) {
    // ⚠️ Nouveau pipeline : redimensionner la composition terminée, jamais
    // dans le même enchaînement que le composite.
    const compose = await img.png().toBuffer();
    return sharp(compose)
      .resize(o.taille, o.taille, { fit: 'inside', kernel: 'lanczos3' })
      .png({ compressionLevel: 9 })
      .toBuffer();
  }
  return img.png({ compressionLevel: 9 }).toBuffer();
}
