/**
 * MEDMOJI — ASSEMBLAGE : Avatar JSON → plan de rendu
 *
 * ⭐ LE MOTEUR NE DESSINE PAS. Il produit un PLAN : la liste ordonnée des
 * couches, leur échelle, leur décalage. Un adaptateur l'exécute ensuite —
 * sharp côté serveur, Skia côté app.
 *
 * C'est ce découpage qui rend le déterminisme TESTABLE. Comparer deux images
 * pixel à pixel dépend de la version de la bibliothèque graphique, du
 * rééchantillonnage, de l'anticrénelage. Comparer deux plans est exact, et
 * c'est là que vivent toutes les décisions du moteur.
 */
import {
  GABARIT, facteursDe, HORS_CORPULENCE,
  type Avatar, type Catalogue, type Couche, type PlanDeRendu,
} from './modeles';
import { valider, piecesDe } from './validateur';
import { empreinte } from './empreinte';


/**
 * Construit le plan. L'avatar DOIT avoir été validé — on revalide quand même,
 * parce qu'un plan construit sur un avatar invalide est un rendu que personne
 * n'a approuvé, et c'est précisément ce qu'on s'interdit.
 */
export function construirePlan(a: Avatar, cat: Catalogue): PlanDeRendu {
  const v = valider(a, cat);
  if (!v.valide) {
    throw new Error(`avatar invalide : ${v.erreur} — ${v.details}`);
  }

  const m = a.corps.morphologie;
  const couches: Couche[] = [];

  for (const id of piecesDe(a)) {
    const p = cat.pieces[id];
    const f = facteursDe(p, m);

    // ⚠️ LA TÊTE ET LE VISAGE NE SUIVENT PAS LA CORPULENCE. Élargir le crâne
    // de 25 % avec le torse déformerait le visage scanné de la personne — elle
    // ne se reconnaîtrait plus. Le visage garde son échelle propre ; seul le
    // bas du visage pourra plus tard suivre légèrement, par morph dédié.
    const suitLaCorpulence = !HORS_CORPULENCE.includes(p.categorie);
    const echelle = suitLaCorpulence ? f : { x: 1, y: 1 };

    // Le décalage recentre la pièce après mise à l'échelle : une pièce élargie
    // de 25 % déborderait à droite si on ne la recentrait pas.
    const dx = (GABARIT.largeur * (1 - echelle.x)) / 2;
    const dy = 0;

    couches.push({
      pieceId: p.id,
      fichier: p.fichier,
      plan: p.plan,
      echelle,
      decalage: { x: dx, y: dy },
      effet: p.effet,
    });
  }

  // ── LE CORPS. Il n'est pas dans `piecesDe` : il ne vient pas d'un choix de
  //    l'usager mais de sa morphologie et de son teint. ──
  const corpsId = `corps_${m}_${a.corps.teint}`;
  couches.push({
    pieceId: corpsId,
    fichier: `corps/${corpsId}.png`,
    plan: 10,
    echelle: { x: 1, y: 1 },
    decalage: { x: 0, y: 0 },
  });

  // 🔴 TRI STABLE ET TOTAL. Trier par plan seul laisse l'ordre des ex æquo à
  // l'implémentation de `sort` — et il diffère entre moteurs JavaScript. On
  // départage donc par identifiant, ce qui rend l'ordre reproductible partout.
  // Le validateur interdit déjà deux pièces au même plan ; ce second critère
  // est la ceinture en plus des bretelles.
  couches.sort((x, y) => x.plan - y.plan || x.pieceId.localeCompare(y.pieceId, 'en'));

  const gabarit = { largeur: GABARIT.largeur, hauteur: GABARIT.hauteur };
  return {
    moteur: a.moteur,
    gabarit,
    couches,
    empreinte: empreinte({ moteur: a.moteur, gabarit, couches }),
  };
}
