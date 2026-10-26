#!/usr/bin/env python3
"""
Check whether a KG dataset's entity/relation ID order matches a trained TransE checkpoint.

KGDataset assigns IDs by first appearance over train+valid+test, so a dataset copy with
a different line order silently breaks every checkpoint. This script scores true triples
vs. randomly corrupted ones with the checkpoint: a matching dataset gives a value close
to 1.0; a mismatched one gives about 0.5.

Usage (from the project root):
  python scripts/check_data_alignment.py --ckpt outputs/embeddings/fb15k237_transE.pt \
      --train data/FB15k/fb15k237-train.txt \
      --valid data/FB15k/fb15k237-valid.txt \
      --test  data/FB15k/fb15k237-test.txt
"""
import argparse
import random

import torch


def read_triples(path):
    triples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                parts = line.split()
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")
    E, R = state["entity_emb"], state["relation_emb"]

    train = read_triples(args.train)
    all_triples = train + read_triples(args.valid) + read_triples(args.test)
    ent, rel = {}, {}
    for h, r, t in all_triples:  # same order as KGDataset._build_vocab
        ent.setdefault(h, len(ent))
        ent.setdefault(t, len(ent))
        rel.setdefault(r, len(rel))

    print(f"checkpoint: entities={state['num_entities']}  relations={state['num_relations']}")
    print(f"dataset:    entities={len(ent)}  relations={len(rel)}")
    if len(ent) != state["num_entities"] or len(rel) != state["num_relations"]:
        print("RESULT: MISMATCH (entity/relation counts differ)")
        return

    random.seed(0)
    sample = random.sample(train, min(args.n, len(train)))
    h = torch.tensor([ent[x[0]] for x in sample])
    r = torch.tensor([rel[x[1]] for x in sample])
    t = torch.tensor([ent[x[2]] for x in sample])
    t_neg = torch.randint(0, len(ent), (len(sample),))
    pos = (E[h] + R[r] - E[t]).norm(dim=1)
    neg = (E[h] + R[r] - E[t_neg]).norm(dim=1)
    p = (pos < neg).float().mean().item()

    print(f"P(true triple scores better than corrupted) = {p:.3f}")
    print("RESULT:", "MATCH" if p > 0.9 else "MISMATCH (IDs do not line up with this checkpoint)")


if __name__ == "__main__":
    main()
