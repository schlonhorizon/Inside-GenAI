#!/usr/bin/env python3
"""Erzeugt vorberechnete Assets nach public/data/ (aufgerufen von .github/workflows/assets.yml).

Stand P0: nur das Gerüst. Die eigentlichen Schritte kommen mit den Phasen:
  golden      – Referenz-Logits/-Attention für die Golden-Tests (P1a/P2)
  embeddings  – Teilmenge deutscher fastText-Vektoren + 2D-Projektion (P3)
  toy         – Training des Spielzeugmodells (P1b)
"""
import argparse

KNOWN = {"golden": "P1a/P2", "embeddings": "P3", "toy": "P1b"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default=",".join(KNOWN))
    args = parser.parse_args()
    for t in [x.strip() for x in args.targets.split(",") if x.strip()]:
        if t not in KNOWN:
            raise SystemExit(f"Unbekanntes Ziel: {t} (bekannt: {', '.join(KNOWN)})")
        print(f"[assets] {t}: noch nicht implementiert (folgt in {KNOWN[t]})")


if __name__ == "__main__":
    main()
