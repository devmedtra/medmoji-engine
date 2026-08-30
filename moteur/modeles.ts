/**
 * MEDMOJI — MODÈLES DE DONNÉES
 *
 * Med, 30 août 2026 : « je veux que MEDMOJI soit une véritable plateforme
 * Avatar Engine, pas simplement un générateur d'images. »
 *
 * 🔴 CE FICHIER NE CONNAÎT NI LE RÉSEAU, NI LE DISQUE, NI HIGGSFIELD.
 * Tout le moteur est pur : il prend un Avatar JSON et un catalogue en mémoire,
 * il rend un plan. C'est la seule façon d'obtenir les mêmes pixels dans l'app
 * et sur le serveur — un moteur qui lit un fichier ou appelle une API ne peut
 * pas être déterministe.
 */

/** Le gabarit, mesuré sur le personnage neutre validé le 30 août 2026. */
export const GABARIT = {
  largeur: 1536,
  hauteur: 2752,
  /** Le sommet du crâne, en fraction de la hauteur. */
  sommet: 0.091,
  /** Les pieds. */
  pieds: 0.952,
} as const;

/**
 * LES PLANS DE PROFONDEUR.
 *
 * ⚠️ Numérotés, jamais implicites : l'ordre de chargement des pièces ne doit
 * avoir AUCUN effet sur le rendu. Deux pièces au même plan sont une erreur,
 * pas un cas à départager au hasard.
 *
 * ⭐ Une coiffure occupe TROIS plans, pas un — corrigé le 30 août. Sans le
 * plan arrière, des cheveux longs passent devant le manteau ; sans les mèches
 * libres, la coiffure a l'air d'un casque moulé.
 */
export const PLANS = {
  cheveuxArriere: 5,
  corps: 10,
  chaussures: 20,
  bas: 30,
  haut: 40,
  bras: 50,
  manteau: 60,
  tete: 70,
  cheveux: 80,
  cheveuxMeches: 85,
  accessoires: 90,
} as const;

export type Categorie =
  | 'corps' | 'tete' | 'cheveux_arriere' | 'cheveux' | 'cheveux_meches'
  | 'haut' | 'manteau' | 'bas' | 'chaussures' | 'accessoire';

export type Morphologie = 'mince' | 'moyenne' | 'athletique' | 'musclee' | 'forte';

/** Position d'ancrage, en FRACTION du gabarit — jamais en pixels. */
export interface Ancre { x: number; y: number }

/** Facteurs d'échelle par morphologie. */
export interface FacteursCorpulence { x: number; y: number }

export interface Piece {
  id: string;
  categorie: Categorie;
  plan: number;
  /** Chemin de l'image, relatif au catalogue. Le moteur ne le lit jamais. */
  fichier: string;
  /** Ligne sous laquelle la pièce commence, en fraction de hauteur. Mesurée
   *  à la fabrication : le minimum de largeur trouve le cou sur un t-shirt et
   *  le col de fourrure sur un parka — 129 px d'écart, mesuré le 30 août. */
  ligneRaccord?: number;
  ancres?: Record<string, Ancre>;
  corpulence?: Partial<Record<Morphologie, FacteursCorpulence>>;
  /** Étirement au-delà duquel la pièce se déforme visiblement. Mesuré à la
   *  fabrication en la passant de 0,8× à 1,4×. */
  deformationMax?: number;
  /** Occlusion explicite : la seule information que la fabrique ne déduit pas.
   *  Une main est devant une manche courte, derrière une manche longue. */
  occlusion?: Record<string, 'devant' | 'derriere'>;
  morphologies: Morphologie[] | ['toutes'];
  style: string;
  /** Pièces avec lesquelles celle-ci ne peut pas coexister. */
  incompatible?: string[];
  rarete?: 'commun' | 'rare' | 'legendaire';
  /** Effet de rendu déclaré par la pièce (halo, dorure, chrome). */
  effet?: string;
  version: string;
}

export interface Catalogue {
  /** Indexé par id. Le moteur ne fait aucune entrée-sortie pour l'obtenir. */
  pieces: Record<string, Piece>;
  styles: Record<string, { angleLumiere: number; contraste: number }>;
}

export interface Avatar {
  /** Fige le comportement du renderer. Un avatar d'aujourd'hui doit encore se
   *  rendre dans deux ans, même quand v2 existe. */
  moteur: string;
  corps: { morphologie: Morphologie; taille: number; teint: string };
  visage: {
    origine: 'scan' | 'catalogue';
    scanId?: string;
    pieces?: Record<string, string>;
    /** Bornés à [-1, 1]. Mesurés sur les 478 repères, pas inventés. */
    morphs?: Record<string, number>;
  };
  cheveux?: string | null;
  tenue: {
    haut?: string | null;
    manteau?: string | null;
    bas?: string | null;
    chaussures?: string | null;
  };
  accessoires?: string[];
  pose: string;
  expression: string;
  style: string;
}

/** Une couche du plan de rendu : quoi dessiner, où, dans quel ordre. */
export interface Couche {
  pieceId: string;
  fichier: string;
  plan: number;
  /** Échelle appliquée, issue de la morphologie. */
  echelle: { x: number; y: number };
  /** Décalage en pixels du gabarit, après mise à l'échelle. */
  decalage: { x: number; y: number };
  effet?: string;
}

/**
 * LE PLAN DE RENDU — la sortie du moteur pur.
 *
 * ⭐ C'est LUI qu'on teste, pas les pixels. Deux plateformes qui produisent le
 * même plan produiront la même image ; comparer des plans est exact, rapide, et
 * ne dépend d'aucune bibliothèque graphique.
 */
export interface PlanDeRendu {
  moteur: string;
  gabarit: { largeur: number; hauteur: number };
  couches: Couche[];
  /** sha256 du plan canonique. La clé de cache. */
  empreinte: string;
}

/**
 * LES FACTEURS DE CORPULENCE PAR DÉFAUT, et l'unique fonction qui les résout.
 *
 * 🔴 UNE SEULE SOURCE DE VÉRITÉ. Cette fonction vivait en double : le
 * validateur ne regardait que les facteurs DÉCLARÉS par la pièce, le plan
 * appliquait ceux par DÉFAUT quand elle n'en déclarait pas. Résultat : une
 * pièce sans déclaration passait le contrôle de déformation, puis se faisait
 * étirer de 25 % au rendu. Le témoin l'a attrapée le 30 août 2026.
 *
 * Une règle appliquée à deux endroits n'est pas une règle, c'est deux
 * occasions de diverger.
 */
export const CORPULENCE_DEFAUT: Record<Morphologie, FacteursCorpulence> = {
  mince: { x: 0.90, y: 1.00 },
  moyenne: { x: 1.00, y: 1.00 },
  athletique: { x: 1.04, y: 1.02 },
  musclee: { x: 1.14, y: 1.02 },
  forte: { x: 1.25, y: 1.03 },
};

export function facteursDe(p: Piece, m: Morphologie): FacteursCorpulence {
  return p.corpulence?.[m] ?? CORPULENCE_DEFAUT[m];
}

/** Les catégories qui NE suivent PAS la corpulence — le visage scanné d'abord. */
export const HORS_CORPULENCE: readonly Categorie[] =
  ['tete', 'cheveux', 'cheveux_arriere', 'cheveux_meches'];

export type Erreur =
  | 'MOTEUR_INCONNU' | 'PIECE_INTROUVABLE' | 'PLAN_EN_DOUBLE'
  | 'COMBINAISON_INVALIDE' | 'MORPHOLOGIE_INCOMPATIBLE' | 'STYLE_INCOMPATIBLE'
  | 'OCCLUSION_CYCLIQUE' | 'TORSE_DECOUVERT' | 'MORPH_HORS_BORNES'
  | 'DEFORMATION_EXCESSIVE' | 'ANCRE_INCONNUE';

export interface Validation {
  valide: boolean;
  erreur?: Erreur;
  details?: string;
  suggestion?: string;
}
