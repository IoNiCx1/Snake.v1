"""benchmark.py - a FIXED exam for Snake that it never trains on.

    python benchmark.py                 # first run builds heldout_positions.json (300 positions), then scores the model
    python benchmark.py --positions 500 --rebuild      # build a new/bigger exam (old scores no longer comparable!)
    python benchmark.py --fast          # skip the centipawn-loss measurement (agreement only)

Why: agreement measured on games Snake has just studied is memorisation. This exam uses fresh positions
(random openings, then Stockfish-vs-Stockfish) that are never added to snake.db, so the numbers are comparable
across days. Every run appends a line to benchmark_log.csv so you can watch the trend.
Run it before and after a batch of games / a training session.
"""
import argparse, csv, json, math, os, random, shutil, sys, time
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

import chess, chess.engine
from snake_core import SnakeAgent, move_to_index

EXAM = "heldout_positions.json"
ap = argparse.ArgumentParser()
ap.add_argument("--positions", type=int, default=300)
ap.add_argument("--rebuild", action="store_true")
ap.add_argument("--fast", action="store_true")
ap.add_argument("--model", default="snake_model.pt")
args = ap.parse_args()

sf = os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish") or "/usr/games/stockfish"
if not os.path.exists(sf):
    sys.exit("Stockfish not found - set STOCKFISH_PATH")


def build_exam(n):
    print(f"Building a fresh exam of {n} positions with Stockfish (one-time, ~{int(n*0.12/60)+1} min)...")
    eng = chess.engine.SimpleEngine.popen_uci(sf)
    eng.configure({"Threads": 1, "Skill Level": 12})
    rows, seen = [], set()
    try:
        while len(rows) < n:
            b = chess.Board()
            for _ in range(random.randint(4, 12)):                       # random opening = variety
                if b.is_game_over(): break
                b.push(random.choice(list(b.legal_moves)))
            while not b.is_game_over() and b.ply() < 90 and len(rows) < n:
                if b.ply() % 3 == 0:
                    key = b.board_fen() + str(b.turn)
                    if key not in seen:
                        seen.add(key)
                        info = eng.analyse(b, chess.engine.Limit(time=0.1))
                        rows.append({"fen": b.fen(), "best": info["pv"][0].uci(),
                                     "cp": info["score"].pov(b.turn).score(mate_score=10000)})
                b.push(eng.play(b, chess.engine.Limit(time=0.02)).move)
    finally:
        eng.quit()
    json.dump(rows, open(EXAM, "w"))
    return rows


if args.rebuild or not os.path.exists(EXAM):
    rows = build_exam(args.positions)
else:
    rows = json.load(open(EXAM))

agent = SnakeAgent.load(args.model, device="cpu")
agree, losses, verr = 0, [], []
eng = None if args.fast else chess.engine.SimpleEngine.popen_uci(sf)
try:
    if eng: eng.configure({"Threads": 1, "Skill Level": 20})
    for r in rows:
        b = chess.Board(r["fen"])
        mine = agent.select_move(b)                                      # plain best move, no lookahead
        best = chess.Move.from_uci(r["best"])
        same = move_to_index(mine, b.turn) == move_to_index(best, b.turn)
        agree += same
        _, v = agent.evaluate(b)
        verr.append(abs(v - math.tanh(max(-1000, min(1000, r["cp"])) / 400)))
        if eng:
            if same:
                losses.append(0)
            else:
                a = b.copy(); a.push(mine)
                if a.is_game_over():
                    cp_a = 10000 if a.is_checkmate() else 0
                else:
                    cp_a = eng.analyse(a, chess.engine.Limit(time=0.05))["score"].pov(b.turn).score(mate_score=10000)
                clip = lambda x: max(-1000, min(1000, x))
                losses.append(max(0, clip(r["cp"]) - clip(cp_a)))
finally:
    if eng: eng.quit()

n = len(rows); p = agree / n; ci = 1.96 * math.sqrt(p * (1 - p) / n)
med = sorted(losses)[len(losses) // 2] if losses else None
print(f"\nSnake exam on {n} fresh positions   (model: {agent.meta.get('arch','resnet')}, "
      f"{agent.meta['train_steps']} training steps, {agent.meta.get('web_games',0)} website games studied)")
print(f"  agreement with Stockfish : {p*100:5.1f}%   (+/- {ci*100:.1f} at 95% confidence)")
if losses: print(f"  avg centipawn loss/move  : {sum(losses)/n:6.0f}   (median {med})   <- lower is better")
print(f"  value-head error (MAE)   : {sum(verr)/n:.3f}   <- lower is better")
new = not os.path.exists("benchmark_log.csv")
with open("benchmark_log.csv", "a", newline="") as f:
    w = csv.writer(f)
    if new: w.writerow(["time", "positions", "train_steps", "web_games", "agreement_pct", "avg_cp_loss", "value_mae"])
    w.writerow([time.strftime("%Y-%m-%d %H:%M"), n, agent.meta["train_steps"], agent.meta.get("web_games", 0),
                round(p * 100, 1), round(sum(losses) / n) if losses else "", round(sum(verr) / n, 3)])
print("  (logged to benchmark_log.csv)")
