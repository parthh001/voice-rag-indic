# e5 prefix ablation (section 4.1)

`intfloat/multilingual-e5-small` is trained with mandatory `"query: "` / `"passage: "` prefixes. Omitting them does not error -- it silently degrades retrieval quality. Measured on 200 eval queries, fixed chunker, Hindi query -> Hindi corpus:

| Variant | Recall@5 | MRR@10 | Chunks indexed |
|---|---|---|---|
| With `query:`/`passage:` prefixes | 0.755 | 0.495 | 10130 |
| Without prefixes | 0.750 | 0.464 | 10130 |

**Honest note:** the measured Recall@5 delta (+0.005) is within noise for 200 queries -- not a dramatic effect on this particular slice, though the prefixes are still used everywhere in this project per the model card's documented training convention.