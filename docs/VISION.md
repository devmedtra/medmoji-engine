# La vision

> « Je veux que MEDMOJI soit une véritable plateforme Avatar Engine, pas
> simplement un générateur d'images. »
> — Med, 30 août 2026

---

## La règle qui commande tout

**Ne pas construire :**

```
prompt de l'usager  →  génération d'image  →  personnage aléatoire
```

**Construire :**

```
prompt de l'usager
      ↓  interprétation par l'IA
Avatar JSON structuré
      ↓  validation
squelette → pièces → plans → occlusion → pose
      ↓
rendu déterministe
```

Ce n'est pas une préférence de style. Les chiffres l'imposent : générer à
l'achat coûte **375 fois** le prix d'un catalogue précalculé à un million
d'usagers, et surtout **personne ne regarde jamais l'image livrée**. Un
catalogue se valide une fois, pièce par pièce, par un humain.

---

## Le personnage

**Un seul**, androgyne. « On ne sait pas si c'est un homme ou une femme. »

Il tient cinq rôles à lui seul :

- le **mannequin de la vitrine** — c'est sur lui que toutes les tenues sont
  présentées avec leur prix ;
- la **référence** dont descendent morphologies, vêtements et animations ;
- le **medmoji par défaut** des non-abonnés — l'identité est portée par le nom
  affiché dessous, comme sur Snapchat ;
- la **morphologie neutre**, point zéro dont les cinq autres s'écartent ;
- les **traits neutres**, point de départ du créateur de visage.

Un seul personnage **divise la production par deux** : la garde-robe n'est
produite qu'une fois, pas une fois par genre. Le genre s'exprime par le visage,
les cheveux et la morphologie — jamais par deux silhouettes séparées.

> 🔴 Il ne doit **pas** pouvoir être lu comme un homme ou comme une femme :
> épaules ni larges ni tombantes, taille peu marquée, mâchoire douce, aucun
> signe genré.

Il porte la tenue de base Medmoji, identique pour tous. C'est elle qui donne de
la valeur à la première tenue achetée : au lancement, tout le monde porte la
même chose.

---

## L'économie du salon

> « Quand quelqu'un va dans un salon de coiffure, il paye pour une nouvelle
> coiffure. Le principe va être pareil. »

- **Payer une tenue débloque son animation.** L'achat n'est pas cosmétique, il
  ouvre du contenu.
- **On montre les maquettes avec le prix affiché.** Pas de surprise.
- **Pas d'essayage gratuit.** « Tu essayes, tu payes, c'est le prix à payer » —
  contrairement à Snapchat, où l'essai est gratuit. Un essai coûte une
  génération ; autant être clair dès le départ.
- **L'abonnement donne droit à un medmoji personnalisé.** Sans lui, un seul
  personnage par défaut.

### Ce que la fabrication maison a changé

Le catalogue coûtait 5 crédits par tenue chez un fournisseur. Depuis que la
Fabrique tourne en local sur GPU, **une tenue coûte du temps de calcul, pas de
l'argent**. La contrainte s'est déplacée : la ressource rare n'est plus le
budget, c'est le **nombre de pièces qu'un humain peut regarder et approuver**.

Toute l'architecture est bâtie autour de ça — la Fabrique produit des
candidats, un humain promeut, le Moteur assemble sans jamais improviser.

---

## Les moteurs

```
                    MEDMOJI
                       |
                AI ORCHESTRATOR
                       |
        +--------------+--------------+
        |              |              |
   FACE AGENT     CLOTHING AGENT   POSE AGENT
        |              |              |
        +--------------+--------------+
                       |
                 RULE ENGINE
                       |
                SKELETON / RIG
                       |
               COLLISION ENGINE
                       |
              COMPOSITING ENGINE
                       |
                 2D RENDERER
                       |
                FINAL AVATAR
```

L'IA sert à **comprendre la demande et à sélectionner les éléments**. Le moteur
graphique reste déterministe. Même configuration, même avatar.

### Ce que les agents n'ont pas le droit de faire

- **Un agent ne produit que du JSON.** Jamais un pixel, jamais un appel à un
  générateur d'images.
- **Un agent ne cite que des identifiants qui existent.** Il reçoit le catalogue
  filtré en entrée ; un identifiant inventé est refusé par le Rule Engine — ce
  qui rend l'hallucination inoffensive au lieu d'invisible.
- **Aucun agent ne parle au renderer.** Le seul chemin est l'Avatar JSON validé.

---

## Le visage : scanner plutôt que composer

> « Pour le visage, on va juste faire les gens scanner leur visage tout
> simplement, pour extraire — au lieu de générer chaque trait. Plus simple
> comme ça. »

Le plan d'origine générait 90 traits de visage puis laissait la personne les
combiner. Deux problèmes : 180 crédits de catalogue, et ça ne ressemblait à
personne en particulier.

**MediaPipe Face Landmarker** renverse le problème : 478 points de repère 3D et
52 coefficients d'expression, gratuitement, sur l'appareil.

Vérifié sur nos personnages de dessin animé le 30 août — le détecteur est
entraîné sur des visages réels, et nos yeux sont énormes :

```
478 points détectés sur les deux images testées
rapport yeux/largeur : 0,746 et 0,742
  → 0,5 % d'écart entre deux générations indépendantes
modèle : 3,6 Mo, il s'embarque dans l'app
```

Les paramètres cessent d'être des curseurs inventés : `ecart_yeux` est une
distance **mesurée sur le visage de la personne**. Et les 52 blendshapes sont
l'Expression Engine tout entier, dans le vocabulaire standard.

---

## L'échelle

> « Il faut que notre système soit award. On parle d'un million de personnes qui
> vont venir. »

Ce qui casse en premier n'est ni le serveur ni le stockage, mais **la mémoire du
téléphone** :

```
un calque en 1536×2752, décodé en RGBA :  16,12 Mio
  8 calques →  129 Mio     (258 Mio avec la copie GPU)
budget réel d'une app RN sur Android d'entrée de gamme : 200 à 400 Mo
```

Le gabarit est une taille de **source**, jamais de runtime. Un pion sur une
carte fait 48 px et coûte **800 fois moins**. D'où les trois niveaux de détail :
composition complète dans l'éditeur, image aplatie dans les listes, miniature
sur la carte.

Et un million d'usagers ne produit pas un million d'images à conserver, mais un
million de **JSON de 400 octets**, avec un cache d'images qui se reconstruit
tout seul.

---

## Les règles de production figées

1. **Assets à alpha propre.** Aucune ombre au sol dans les exports d'avatars ou
   de chaussures. L'ombre appartient au contexte, pas à la pièce — un medmoji
   posé sur une carte flotte sans sol, le même dans un vestiaire en reçoit une.
2. **Teinture en deux passes.** Un master en niveaux de gris, une matrice de
   couleur, et une correction de matité pour que le tissu sonne coton et non
   synthétique. Un asset, une infinité de coloris.
3. **Le corps de base est nu.** Un corps habillé laisse dépasser ce qu'il porte
   sous chaque vêtement qu'on lui pose.
4. **L'ordre des passes ne se négocie pas** : masque → teinture → ombre. Toute
   passe qui modifie les pixels détruit une segmentation calculée après elle.

---

## Ce qui reste ouvert

- **Poses fortes et angles de caméra.** Un personnage raster n'a pas de dos. Une
  pose de boxe ou une contre-plongée ne s'obtient pas en déformant une pose
  debout — ce sont des jeux de pièces à générer, et ils multiplient le catalogue.
- **Le clavier d'autocollants.** Le canal de distribution qui a fait Bitmoji.
  C'est du code natif, donc un build, pas une mise à jour à distance.
- **Le modèle maison.** Un LoRA entraîné sur les pièces validées : le catalogue
  qu'on construit *est* le jeu d'entraînement. Le style cesse alors de dépendre
  d'un fournisseur qui peut fermer un modèle du jour au lendemain.
