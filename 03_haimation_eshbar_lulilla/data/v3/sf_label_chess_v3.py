#!/usr/bin/env python3
"""ChessBench-format move-choice pool labelled by Stockfish 16 (es-bar in-weight chess distillation).

The searchless_chess train shards are shuffled per (position, move) record, so one shard rarely holds a position's full
move table; the values are recomputed here the way searchless_chess made them: Stockfish 16, 50 ms per action, the
position after the move analysed and its score turned into the mover's win probability
(win = 0.5 + 0.5 * (2 / (1 + exp(-0.00368208 * cp)) - 1); mate -> 1 / 0). Positions come from the train shard's FENs,
excluding every position (FEN, or placement + side to move) of the official TEST action-value file.
Rows: the DI 0.3 ChessBench-legal-move format; soft target p ~ exp((v - v_best) / TAU); a held-out slice keeps the DI
scoring fields (all values, the tied-best set) for evaluation.

  python3 sf_label_chess.py --kit KIT --test TEST.bag --train SHARD.bag --stockfish BIN --out POOL.jsonl
         --heldout-out HO.jsonl [--n 30000] [--heldout-n 1000] [--workers 180] [--ms 50] [--tau 0.02]
"""
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import pathlib
import random
import sys

INSTR = ("Choose the strongest legal move for the side to move. Board rows run from rank 8 to rank 1; columns a to h. "
         "Uppercase pieces are white. All legal moves are supplied.")
_ENGINE = None


def win_prob(cp):
    return 0.5 + 0.5 * (2.0 / (1.0 + math.exp(-0.00368208 * cp)) - 1.0)


def _init(binary):
    global _ENGINE
    import chess.engine
    _ENGINE = chess.engine.SimpleEngine.popen_uci(binary)
    _ENGINE.configure({"Threads": 1, "Hash": 16})


def _values(args):
    fen, ms = args
    import chess
    import chess.engine
    board = chess.Board(fen)
    out = {}
    for mv in board.legal_moves:
        b = board.copy()
        b.push(mv)
        if b.is_checkmate():
            out[mv.uci()] = 1.0
            continue
        if b.is_game_over():
            out[mv.uci()] = 0.5
            continue
        info = _ENGINE.analyse(b, chess.engine.Limit(time=ms / 1000.0))
        sc = info["score"].pov(not b.turn)            # the mover's point of view
        if sc.is_mate():
            out[mv.uci()] = 1.0 if sc.mate() > 0 else 0.0
        else:
            out[mv.uci()] = win_prob(sc.score())
    return fen, out


def main():
    ap = argparse.ArgumentParser()
    for k in ("kit", "test", "train", "stockfish", "out", "heldout_out"):
        ap.add_argument(f"--{k.replace('_', '-')}", required=True)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--heldout-n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=180)
    ap.add_argument("--ms", type=int, default=50)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--exclude", default=None, help="file of FENs already labelled (pool / held-out): their placement + side are skipped")
    a = ap.parse_args()
    sys.path.insert(0, a.kit)
    import chess
    from decision_index.suite.build.adapters_scored import bag_records, decode_bag

    test_fen, test_place = set(), set()
    for rec in bag_records(pathlib.Path(a.test)):
        fen, _, _ = decode_bag(rec)
        test_fen.add(fen)
        test_place.add(" ".join(fen.split()[:2]))
    want = a.n + a.heldout_n
    seen, fens = set(), []
    if a.exclude:
        for l in open(a.exclude):
            if l.strip():
                seen.add(" ".join(l.split()[:2]))
    n_excl = len(seen)
    for rec in bag_records(pathlib.Path(a.train)):
        fen, _, _ = decode_bag(rec)
        place = " ".join(fen.split()[:2])
        if fen in test_fen or place in test_place or place in seen:
            continue
        b = chess.Board(fen)
        if b.is_game_over() or b.legal_moves.count() < 2:
            continue
        seen.add(place)
        fens.append(fen)
        if len(fens) >= want * 3:
            break
    random.Random(a.seed).shuffle(fens)
    fens = fens[:want]
    print(f"test positions {len(test_fen):,}; excluded placements {n_excl:,}; sampled {len(fens):,} train positions", flush=True)

    with mp.Pool(a.workers, initializer=_init, initargs=(a.stockfish,)) as pool:
        results = pool.map(_values, [(f, a.ms) for f in fens], chunksize=4)
    out, ho = open(a.out, "w"), open(a.heldout_out, "w")
    nk = nh = 0
    for i, (fen, values) in enumerate(results):
        board = chess.Board(fen)
        moves = {m.uci(): m for m in board.legal_moves}
        order = sorted(moves, key=lambda m: hashlib.sha256((fen + ":" + m).encode()).digest())
        criteria = {m: {"uci": m, "san": board.san(moves[m])} for m in order}
        sid = hashlib.sha256(fen.encode()).hexdigest()
        state = {"fen": fen, "board": str(board), "side_to_move": "white" if board.turn else "black"}
        best = max(values.values())
        q = {"move": {"type": "choice", "instructions": INSTR, "criteria": criteria}}
        if i < a.heldout_n:
            ho.write(json.dumps({"id": f"HO-CHESS:{sid}", "family": "HELDOUT-ChessBench", "split": "heldout_v3", "state": state, "questions": q,
                                 "expected": {"move": max(values, key=values.get)},
                                 "scoring": {"values": values, "accepted": sorted(k for k, v in values.items() if v == best), "tie_tolerance": 0.0},
                                 "metadata": {"track": "chess_heldout_sf16_v3", "legal_moves": len(moves)}}) + "\n")
            nh += 1
            continue
        w = {m: math.exp((v - best) / a.tau) for m, v in values.items()}
        z = sum(w.values())
        soft = {m: x / z for m, x in w.items() if x / z >= 1e-4}
        zs = sum(soft.values())
        out.write(json.dumps({"id": f"CHESSD:{sid}", "family": "LOOMD-chess_train", "split": "train", "group": sid, "state": state,
                              "questions": q, "expected": {"move": max(values, key=values.get)},
                              "soft_gold": {"move": {m: x / zs for m, x in soft.items()}},
                              "meta": {"legal_moves": len(moves), "best_value": best, "tau": a.tau,
                                       "values": "Stockfish 16, %d ms per action, mover win probability" % a.ms}}) + "\n")
        nk += 1
    out.close()
    ho.close()
    print(json.dumps({"pool": nk, "heldout": nh}), flush=True)


if __name__ == "__main__":
    main()
