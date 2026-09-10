# Chunking Comparison — Recall@k / MRR

Ground truth: `passages.is_selected` from MSMARCO-XI validation (real human relevance labels, not synthetic).

| Strategy | Mode | Chunks | Mean chunk len (words) | R@1 | R@3 | R@5 | R@10 | MRR@10 |
|---|---|---|---|---|---|---|---|---|
| fixed | hi_to_hi | 10130 | 60.5 | 0.358 | 0.611 | 0.740 | 0.827 | 0.510 |
| fixed | hi_to_en | 9952 | 50.3 | 0.277 | 0.534 | 0.650 | 0.749 | 0.430 |
| fixed | en_to_en | 9952 | 50.3 | 0.460 | 0.813 | 0.914 | 0.967 | 0.648 |
| semantic | hi_to_hi | 13196 | 45.9 | 0.356 | 0.606 | 0.723 | 0.809 | 0.504 |
| semantic | hi_to_en | 14378 | 34.8 | 0.230 | 0.418 | 0.507 | 0.591 | 0.341 |
| semantic | en_to_en | 14378 | 34.8 | 0.454 | 0.779 | 0.881 | 0.938 | 0.629 |
| metadata_aware | hi_to_hi | 12359 | 49.0 | 0.351 | 0.603 | 0.728 | 0.822 | 0.504 |
| metadata_aware | hi_to_en | 11666 | 42.9 | 0.235 | 0.454 | 0.563 | 0.657 | 0.367 |
| metadata_aware | en_to_en | 11666 | 42.9 | 0.447 | 0.801 | 0.905 | 0.960 | 0.637 |

## Stratified by query_type (recall@5)

| Strategy | Mode | query_type | n | R@5 | MRR@10 |
|---|---|---|---|---|---|
| fixed | hi_to_hi | DESCRIPTION | 728 | 0.712 | 0.479 |
| fixed | hi_to_hi | NUMERIC | 200 | 0.825 | 0.595 |
| fixed | hi_to_hi | ENTITY | 42 | 0.786 | 0.647 |
| fixed | hi_to_hi | PERSON | 25 | 0.800 | 0.502 |
| fixed | hi_to_hi | LOCATION | 5 | 0.800 | 0.500 |
| fixed | hi_to_en | DESCRIPTION | 728 | 0.610 | 0.391 |
| fixed | hi_to_en | NUMERIC | 200 | 0.755 | 0.554 |
| fixed | hi_to_en | ENTITY | 42 | 0.810 | 0.537 |
| fixed | hi_to_en | PERSON | 25 | 0.680 | 0.390 |
| fixed | hi_to_en | LOCATION | 5 | 0.800 | 0.420 |
| fixed | en_to_en | DESCRIPTION | 728 | 0.913 | 0.650 |
| fixed | en_to_en | NUMERIC | 200 | 0.905 | 0.647 |
| fixed | en_to_en | ENTITY | 42 | 0.952 | 0.686 |
| fixed | en_to_en | PERSON | 25 | 0.920 | 0.508 |
| fixed | en_to_en | LOCATION | 5 | 1.000 | 0.640 |
| semantic | hi_to_hi | DESCRIPTION | 728 | 0.695 | 0.468 |
| semantic | hi_to_hi | NUMERIC | 200 | 0.805 | 0.603 |
| semantic | hi_to_hi | ENTITY | 42 | 0.786 | 0.642 |
| semantic | hi_to_hi | PERSON | 25 | 0.760 | 0.477 |
| semantic | hi_to_hi | LOCATION | 5 | 0.800 | 0.700 |
| semantic | hi_to_en | DESCRIPTION | 728 | 0.453 | 0.296 |
| semantic | hi_to_en | NUMERIC | 200 | 0.665 | 0.503 |
| semantic | hi_to_en | ENTITY | 42 | 0.643 | 0.379 |
| semantic | hi_to_en | PERSON | 25 | 0.640 | 0.350 |
| semantic | hi_to_en | LOCATION | 5 | 0.200 | 0.067 |
| semantic | en_to_en | DESCRIPTION | 728 | 0.882 | 0.628 |
| semantic | en_to_en | NUMERIC | 200 | 0.870 | 0.634 |
| semantic | en_to_en | ENTITY | 42 | 0.905 | 0.639 |
| semantic | en_to_en | PERSON | 25 | 0.880 | 0.549 |
| semantic | en_to_en | LOCATION | 5 | 1.000 | 0.840 |
| metadata_aware | hi_to_hi | DESCRIPTION | 728 | 0.703 | 0.475 |
| metadata_aware | hi_to_hi | NUMERIC | 200 | 0.810 | 0.584 |
| metadata_aware | hi_to_hi | ENTITY | 42 | 0.738 | 0.600 |
| metadata_aware | hi_to_hi | PERSON | 25 | 0.760 | 0.525 |
| metadata_aware | hi_to_hi | LOCATION | 5 | 0.800 | 0.600 |
| metadata_aware | hi_to_en | DESCRIPTION | 728 | 0.514 | 0.335 |
| metadata_aware | hi_to_en | NUMERIC | 200 | 0.700 | 0.480 |
| metadata_aware | hi_to_en | ENTITY | 42 | 0.714 | 0.404 |
| metadata_aware | hi_to_en | PERSON | 25 | 0.640 | 0.380 |
| metadata_aware | hi_to_en | LOCATION | 5 | 0.600 | 0.217 |
| metadata_aware | en_to_en | DESCRIPTION | 728 | 0.904 | 0.636 |
| metadata_aware | en_to_en | NUMERIC | 200 | 0.905 | 0.642 |
| metadata_aware | en_to_en | ENTITY | 42 | 0.929 | 0.695 |
| metadata_aware | en_to_en | PERSON | 25 | 0.880 | 0.526 |
| metadata_aware | en_to_en | LOCATION | 5 | 1.000 | 0.640 |

**Fairness note:** strategies produce different chunk counts (see table above). A strategy with more chunks gets more independent shots at recall; chunk counts are reported alongside recall for exactly this reason -- read recall differences together with chunk count, not in isolation.