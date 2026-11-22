#!/usr/bin/env python3
"""ChessBench move-choice training pool for es-bar (in-weight distillation of engine action-values).

Source: DeepMind searchless_chess TRAIN action-value shards (public GCS). Rows reproduce the DI 0.3 ChessBench-legal-move
format exactly (state, instructions, criteria {uci: {uci, san}} in the DI's hash order). Positions whose FEN, or whose
placement + side to move, occur in the official TEST action-value file (the DI's source) are excluded.
Target: soft, p(move) ~ exp((v - v_best) / TAU) over the engine win probabilities (near-equal moves share credit);
`expected` keeps one best move for the DI-format check.

  python3 build_chess_pool.py --kit KIT --test TEST.bag --train SHARD.bag [...] --out OUT.jsonl [--max N] [--tau 0.02]
"""
import argparse
import collections
import hashlib
import json
import math
import pathlib
import random
import sys

INSTR = ("Choose the strongest legal move for the side to move. Board rows run from rank 8 to rank 1; columns a to h. "
         "Uppercase pieces are white. All legal moves are supplied.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=30000)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--heldout-out", default="")
    ap.add_argument("--heldout-n", type=int, default=1000)
    a = ap.parse_args()
    sys.path.insert(0, a.kit)
    import chess
    from decision_index.suite.build.adapters_scored import bag_records, decode_bag

    test_fen, test_place = set(), set()
    for rec in bag_records(pathlib.Path(a.test)):
        fen, _, _ = decode_bag(rec)
        test_fen.add(fen)
        test_place.add(" ".join(fen.split()[:2]))
    print(f"test positions {len(test_fen):,}", flush=True)

    by_fen = collections.defaultdict(dict)
    for path in a.train:
        for rec in bag_records(pathlib.Path(path)):
            fen, move, value = decode_bag(rec)
            by_fen[fen][move] = value
    print(f"train positions {len(by_fen):,}", flush=True)

    rng = random.Random(a.seed)
    fens = sorted(by_fen)
    rng.shuffle(fens)
    counts = collections.Counter()
    ho = open(a.heldout_out, "w") if a.heldout_out else None
    with open(a.out, "w") as out:
        for fen in fens:
            if counts["kept"] >= a.max and (ho is None or counts["heldout"] >= a.heldout_n):
                break
            to_heldout = counts["kept"] >= a.max
            if fen in test_fen or " ".join(fen.split()[:2]) in test_place:
                counts["excluded_test"] += 1
                continue
            values = by_fen[fen]
            board = chess.Board(fen)
            moves = {m.uci(): m for m in board.legal_moves}
            if set(moves) != set(values) or len(moves) < 2:
                counts["incomplete"] += 1
                continue
            vmax = max(values.values())
            w = {m: math.exp((v - vmax) / a.tau) for m, v in values.items()}
            z = sum(w.values())
            soft = {m: x / z for m, x in w.items() if x / z >= 1e-4}
            zs = sum(soft.values())
            soft = {m: x / zs for m, x in soft.items()}
            order = sorted(moves, key=lambda m: hashlib.sha256((fen + ":" + m).encode()).digest())
            criteria = {m: {"uci": m, "san": board.san(moves[m])} for m in order}
            sid = hashlib.sha256(fen.encode()).hexdigest()
            if to_heldout:                               # unused train positions, DI scoring fields (ties, values) for evaluation
                best = max(values.values())
                ho.write(json.dumps({"id": f"HO-CHESS:{sid}", "family": "HELDOUT-ChessBench", "split": "heldout",
                                     "state": {"fen": fen, "board": str(board), "side_to_move": "white" if board.turn else "black"},
                                     "questions": {"move": {"type": "choice", "instructions": INSTR, "criteria": criteria}},
                                     "expected": {"move": max(values, key=values.get)},
                                     "scoring": {"values": values, "accepted": sorted(k for k, v in values.items() if v == best), "tie_tolerance": 0.0},
                                     "metadata": {"track": "chess_heldout_train_shard", "legal_moves": len(moves)}}) + "\n")
                counts["heldout"] += 1
                continue
            row = {"id": f"CHESSD:{sid}", "family": "LOOMD-chess_train", "split": "train", "group": sid,
                   "state": {"fen": fen, "board": str(board), "side_to_move": "white" if board.turn else "black"},
                   "questions": {"move": {"type": "choice", "instructions": INSTR, "criteria": criteria}},
                   "expected": {"move": max(values, key=values.get)}, "soft_gold": {"move": soft},
                   "meta": {"legal_moves": len(moves), "best_value": vmax, "tau": a.tau, "source": "searchless_chess train action-values"}}
            out.write(json.dumps(row) + "\n")
            counts["kept"] += 1
            counts[f"k>52"] += len(moves) > 52
    if ho:
        ho.close()
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
