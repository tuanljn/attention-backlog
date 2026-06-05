# Attention Backlog Implementation

[![MPL-2.0 License](https://img.shields.io/badge/license-MPL_2.0-blue.svg)](https://www.mozilla.org/en-US/MPL/2.0/)

Livraison RTE des cinq items du backlog attention pour EnerGNN. Le code est ajouté en surcouche du framework EnerGNN (H2MG, Flax NNX, JAX) sans modifier l'API existante : les modules `LocalSumMessagePassingFunction` et `RecurrentCoupler` d'origine restent disponibles et servent de référence dans toutes les comparaisons.

## Items livrés

| # | Item | Type | Emplacement |
|---|---|---|---|
| 1 | GATv2 | message function | `src/energnn/model/coupler/message_passing/message_passing_function.py` |
| 2 | GlobalAggregation | aggregation operator | `src/energnn/model/coupler/message_passing/message_passing_function.py` |
| 3 | MultiHeadQKV | message function | `src/energnn/model/coupler/message_passing/message_passing_function.py` |
| 4 | Performer | message function (kernel-trick, single-head v1 ; FAVOR+ random-features différé) | `src/energnn/model/coupler/message_passing/message_passing_function.py` |
| 5 | VirtualAddressRecurrentCoupler | coupler | `src/energnn/model/coupler/message_passing/recurrent_coupler.py` |

## Gates de validation

- Approches 1 à 4 : Gates 1 à 7 passées (Gate 1 unitaire, Gate 2 statique, Gate 3 jouet LinearSystem, Gate 4 intégration H2MG, Gate 5 supervisé AC LF ieee9 + ieee14, Gate 6 perf ieee118/300, Gate 7 reproductibilité (empreinte numérique)).
- Approche 5 (`VirtualAddressRecurrentCoupler`) : Gates 1 à 5 passées. Gates 6 (perf) et 7 (consistency, empreinte numérique) non livrées à ce commit ; voir la branche `virtual-address-coupler` pour les mesures à venir.
- Gate 8 (snapshot RTE point figé) : hors scope tant que le snapshot RTE n'est pas reçu ; s'exécute rétroactivement sur les checkpoints sans ré-entraînement.

## Documentation

- `Rapport d'implémentation des mécanismes d'attention dans EnerGNN.pdf` (livré séparément) — rapport officiel item par item : méthode, hyperparamètres, résultats Gate 5/6/7, lecture des courbes. La source HTML (`baseline_walkthrough.html`) est versionnée dans le repo pour ré-édition.
- `BASELINES.md` -- chiffres de référence LocalSum sur LinearSystem, conversion IEEE et AC load-flow IEEE supervisé ; tout chiffre attention y est comparé directement.
- `docs/tutorials/` — sept notebooks reproductibles (00 baseline, 01-04 Items 1-4, 05 combinaisons, 06 Item 5 VAR) et leurs JSON sous `benchmarks/results/`.

## Licence

Mozilla Public License 2.0, conformément au framework EnerGNN amont (voir `LICENSE`).
