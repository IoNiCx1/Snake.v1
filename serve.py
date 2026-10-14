"""serve.py - run the Snake website WITHOUT Jupyter.

    python serve.py                 # play with the model in snake_model.pt
    python serve.py 5050            # use another port
    python serve.py --new           # no model yet? create a fresh untrained Snake first
    python serve.py --new --arch transformer

Needs snake_core.py, snake_learn.py, snake_web.py, snake_ui.html next to this file.
With Stockfish installed you also get the Stockfish modes, eval bar, commentary and the Learning lab."""
import argparse, glob, os, shutil, sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)          # model + database are always looked up next to this file

from snake_core import SnakeAgent
from snake_web import create_app

ap = argparse.ArgumentParser()
ap.add_argument("port", nargs="?", type=int, default=5000)
ap.add_argument("--new", action="store_true", help="create a fresh untrained model if none exists")
ap.add_argument("--arch", default="resnet", choices=["resnet", "transformer"])
args = ap.parse_args()

MODEL = "snake_model.pt"
sf = os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")

if not os.path.exists(MODEL):
    others = glob.glob("*.pt") + glob.glob("deploy/*.pt")
    if args.new:
        ch, bl = (96, 4) if args.arch == "transformer" else (64, 4)
        SnakeAgent(channels=ch, blocks=bl, device="cpu", arch=args.arch).save(MODEL)
        print(f"Created a fresh untrained {args.arch} Snake -> {MODEL}")
    else:
        sys.exit(f"\n{MODEL} not found in {os.getcwd()}\n"
                 + (f"Model files I can see: {others}\n  (copy one here:  cp {others[0]} {MODEL})\n" if others else "")
                 + "Train one in the notebook first, or run:  python serve.py --new\n")

agent = SnakeAgent.load(MODEL, device="cpu")
print("Loaded Snake:", agent.meta)
print("Stockfish:", sf or "NOT FOUND (Stockfish modes + learning will be hidden)")

hub = None
if sf:
    from snake_learn import SnakeDB, LearningHub
    db = SnakeDB("snake.db")
    hub = LearningHub(agent, db, sf, model_path=MODEL)
    print(f"Memory: snake.db has {db.count()} positions - finished games will be added and studied.")

app = create_app(agent, stockfish_path=sf, hub=hub)
print(f"\n>>> Open http://127.0.0.1:{args.port} in your browser <<<\n")
app.run(host="127.0.0.1", port=args.port)
