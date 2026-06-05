# `benchmarks/05_virtual_address/` — Approche 5 (VirtualAddressRecurrentCoupler)

Benchmarks de l'Item 5 (`VirtualAddressRecurrentCoupler`). Gates 1 à 5 livrées. Gates 6 (perf) et 7 (consistency, empreinte numérique) non livrées à ce commit ; voir la branche `virtual-address-coupler` pour les mesures à venir. Gate 8 (snapshot RTE) hors scope.

## Scripts

| Script | Gate | Substrate | Sortie |
|---|---|---|---|
| `baseline_var_linearsystem.py` | Gate 3 | LinearSystem Tiny + Small × 3 seeds | `../results/05_virtual_address/baseline_var_linearsystem.json` |
| `baseline_var_ac_load_flow.py` | Gate 5 | ieee9 + ieee14 × Tiny + Small × 3 seeds, AC LF `perturbation_scale=0.1`, `dataset_size=32` | `../results/05_virtual_address/baseline_var_ac_load_flow.json` |

Les Gates 6 (perf vs LocalSum) et 7 (consistency, empreinte numérique) sont mutualisées avec l'infrastructure générale du repo (`tests/perf/`, `tests/consistency/`) ; leur exécution pour l'Approche 5 est planifiée en commit de suivi.

## Configurations `message_functions`

Configuration de référence livrée pour la Gate 5 :

- **LocalSum seul** — mesure la contribution marginale de l'état virtuel vs `RecurrentCoupler + LocalSum`.

L'ablation **LocalSum + Performer** (combinaison via une message function topology-agnostic) est prévue en commit de suivi ; elle testera si le mécanisme d'adresse virtuelle récupère le signal topologique manquant.

## Résultats

Voir `../results/05_virtual_address/` pour les JSON bruts et le rapport `Rapport d'implémentation des mécanismes d'attention dans EnerGNN.pdf`, chapitre 15 pour l'analyse.
