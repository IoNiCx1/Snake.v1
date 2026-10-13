"""serve.py - run the Snake website WITHOUT Jupyter:  python serve.py [port]"""
import os, shutil, sys
from snake_core import SnakeAgent
from snake_web import create_app

port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
sf = os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")
agent = SnakeAgent.load("snake_model.pt", device="cpu")
print("Loaded Snake:", agent.meta)
print("Stockfish:", sf or "NOT FOUND (Stockfish modes will be hidden)")
app = create_app(agent, stockfish_path=sf)
print(f"\n>>> Open http://127.0.0.1:{port} in your browser <<<\n")
app.run(host="127.0.0.1", port=port)