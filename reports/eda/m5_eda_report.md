# RetailMind AI — Rapport EDA M5

## 1. Perimetre et methode

Analyse descriptive de `sales_train_validation.csv`, `calendar.csv` et `sell_prices.csv`.
Les controles de coherence sont recalcules a partir des rapports locaux.

## 2. Indicateurs principaux

| Indicateur | Valeur |
|---|---:|
| Series produit-magasin | 30 490 |
| Produits distincts | 3 049 |
| Magasins | 10 |
| Jours de ventes | 1 913 |
| Periode | 2011-01-29 au 2016-04-24 |
| Unites vendues | 65 695 409 |
| Cellules sans vente | 68.2 % |
| Lignes de prix | 6 841 121 |
| Revenu brut estime (unites monetaires M5) | 187 676 570,02 |

## 3. Resultats commerciaux

### Categories (revenu decroissant)

| Categorie | Unites | Revenu estime |
|---|---:|---:|
| FOODS | 45 089 939 | 108 904 622,04 |
| HOUSEHOLD | 14 480 670 | 55 953 542,68 |
| HOBBIES | 6 124 800 | 22 818 405,30 |

### Magasins (revenu decroissant)

| Magasin | Unites | Revenu estime |
|---|---:|---:|
| CA_3 | 11 188 180 | 32 108 719,70 |
| CA_1 | 7 698 216 | 22 520 252,38 |
| TX_2 | 7 214 384 | 20 518 237,12 |
| TX_3 | 6 089 330 | 17 796 727,76 |
| WI_2 | 6 544 012 | 17 690 738,30 |
| CA_2 | 5 685 475 | 17 409 444,68 |
| WI_3 | 6 427 782 | 16 926 906,11 |
| TX_1 | 5 595 292 | 15 729 162,06 |
| WI_1 | 5 149 062 | 14 769 235,02 |
| CA_4 | 4 103 676 | 12 207 146,89 |

### Premier produit par revenu estime

- FOODS_3_586 : 1 462 145,56.

## 4. Saisonnalite et qualite

- Jour de semaine avec la moyenne la plus elevee : Samedi.
- Semaines de ventes : 274.
- Semaines calendaires couvertes par les prix : 282.
- Couples produit-magasin-semaine sans prix : 1 757 059.
- Unites vendues sans prix correspondant : 0.
- Doublons de cles produit-magasin-semaine : 0.
- Reconciliation des unites et revenus par semaine, categorie et magasin : validee.

## 5. Limites et interpretation

- Le revenu est estime a partir des prix hebdomadaires listes.
- Il ne constitue pas un chiffre d'affaires comptable.
- Taxes, retours et remises transactionnelles indisponibles.
- Une combinaison sans prix n'est pas une anomalie si elle ne comporte pas de vente positive sur la semaine.
- Les tendances observees ne prouvent pas de relation causale.
- Les donnees M5 sont historiques et ne decrivent pas les ventes actuelles.

## 6. Suite du projet

Construire le modele analytique PostgreSQL, les pipelines ETL, les tableaux de bord BI et les modeles de prevision.
