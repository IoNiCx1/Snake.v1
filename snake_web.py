"""snake_web.py - Flask website + JSON API for Snake (used locally AND when deployed)."""
import atexit
import os
import threading
from pathlib import Path
import chess
import chess.engine
from flask import Flask, jsonify, request, Response
from snake_core import identify_game


class StockfishService:
    """Thread-safe wrapper around ONE Stockfish process (talked to over UCI). Opened lazily."""
    def __init__(self, path):
        self.path, self.lock, self.engine, self.name = path, threading.Lock(), None, "Stockfish"

    def _ensure(self):
        if self.engine is None:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.path)
            self.engine.configure({"Threads": 1, "Hash": 64})
            self.name = self.engine.id.get("name", "Stockfish")
            # must run BEFORE python joins the engine's helper thread, or exit hangs
            getattr(threading, "_register_atexit", atexit.register)(self.close)

    def close(self):
        with self.lock:
            if self.engine is not None:
                try:
                    self.engine.quit()
                except Exception:
                    pass
                self.engine = None

    def play(self, board, level, seconds):
        with self.lock:
            self._ensure()
            self.engine.configure({"Skill Level": max(0, min(20, int(level)))})
            return self.engine.play(board, chess.engine.Limit(time=float(seconds))).move

    def analyse(self, board, seconds=0.2):
        with self.lock:
            self._ensure()
            self.engine.configure({"Skill Level": 20})
            info = self.engine.analyse(board, chess.engine.Limit(time=seconds))
        sc = info["score"].white()
        pv = info.get("pv") or []
        return {"cp": sc.score(mate_score=10000), "mate": sc.mate(),
                "best": pv[0].uci() if pv else None,
                "best_san": board.san(pv[0]) if pv else None}


UI_FILE = Path(__file__).with_name("snake_ui.html")


def _build_board(moves, start_fen=None):
    board = chess.Board(start_fen) if start_fen else chess.Board()
    for u in moves:
        board.push_uci(u)          # raises ValueError if illegal -> handled by caller
    return board


def _state(board):
    legal = {}
    for m in board.legal_moves:
        legal.setdefault(chess.square_name(m.from_square), []).append(chess.square_name(m.to_square))
    replay, san = board.root().copy(), []
    for m in board.move_stack:
        san.append(replay.san(m)); replay.push(m)
    info = identify_game(board)
    return {"fen": board.fen(), "moves": [m.uci() for m in board.move_stack], "san": san,
            "start_fen": board.root().fen() if board.root().fen() != chess.STARTING_FEN else None,
            "turn": "white" if board.turn else "black", "legal": legal, "in_check": board.is_check(),
            "over": info["over"], "result": info["result"], "termination": info["termination"],
            "winner": info["winner"], "info": info}


def create_app(agent, stats_fn=None, stockfish_path=None, hub=None):
    app = Flask(__name__)
    sf = StockfishService(stockfish_path) if stockfish_path and os.path.exists(stockfish_path) else None

    def bad(msg, code=400):
        return jsonify({"error": msg}), code

    def read_board():
        data = request.get_json(force=True, silent=True) or {}
        try:
            return _build_board(data.get("moves", []), data.get("start_fen")), data
        except (ValueError, TypeError):
            return None, data

    @app.get("/")
    def index():
        return Response(UI_FILE.read_text(encoding="utf-8"), mimetype="text/html")

    @app.get("/api/capabilities")
    def capabilities():
        return jsonify(stockfish=sf is not None, learning=bool(hub and hub.enabled),
                       arch=agent.meta.get("arch", "resnet"))

    @app.get("/health")
    def health():
        return jsonify(ok=True)

    @app.get("/api/stats")
    def stats():
        s = stats_fn() if stats_fn else dict(agent.meta)
        return jsonify({"train_steps": s.get("train_steps", agent.meta.get("train_steps", 0)),
                        "games": s.get("games", agent.meta.get("games", 0)),
                        "web_games": agent.meta.get("web_games", 0),
                        "sf_level": s.get("sf_level", agent.meta.get("sf_level", 0)),
                        "arch": agent.meta.get("arch", "resnet")})

    @app.post("/api/state")
    def state():
        board, _ = read_board()
        return bad("invalid move list or start position") if board is None else jsonify(_state(board))

    @app.post("/api/validate_fen")
    def validate_fen():
        """Check a position built in the board editor; normalises impossible castling rights."""
        fen = str((request.get_json(force=True, silent=True) or {}).get("fen", ""))
        try:
            b = chess.Board(fen)
        except ValueError as e:
            return jsonify(ok=False, problems=[str(e)])
        b.castling_rights = b.clean_castling_rights()
        st = b.status()
        if st != chess.STATUS_VALID:
            names = [f.name.lower().replace("_", " ") for f in chess.Status if f in st and f != chess.STATUS_VALID]
            return jsonify(ok=False, problems=names)
        if b.is_game_over():
            return jsonify(ok=False, problems=["the game is already over in this position"])
        return jsonify(ok=True, fen=b.fen())

    @app.post("/api/move")
    def move():
        board, data = read_board()
        if board is None:
            return bad("invalid move list")
        if board.is_game_over(claim_draw=True):
            return bad("game is over")
        uci = str(data.get("move", ""))
        try:
            m = chess.Move.from_uci(uci)
        except ValueError:
            return bad("malformed move")
        piece = board.piece_at(m.from_square)            # auto-queen
        if piece and piece.piece_type == chess.PAWN and chess.square_rank(m.to_square) in (0, 7) and not m.promotion:
            m.promotion = chess.QUEEN
        if m not in board.legal_moves:                   # LEGAL MOVE IDENTIFIER on the server side
            return bad(f"illegal move: {uci}")
        board.push(m)
        if not board.is_game_over(claim_draw=True):
            if data.get("opponent") == "stockfish" and sf:
                board.push(sf.play(board, data.get("level", 3), 0.3))
            else:
                board.push(agent.select_move(board, lookahead=bool(data.get("lookahead", True))))
        return jsonify(_state(board))

    @app.post("/api/snake")
    def snake():
        board, data = read_board()
        if board is None:
            return bad("invalid move list")
        if board.is_game_over(claim_draw=True):
            return bad("game is over")
        board.push(agent.select_move(board, lookahead=bool(data.get("lookahead", True))))
        return jsonify(_state(board))

    @app.post("/api/sf_move")
    def sf_move():
        if not sf:
            return bad("Stockfish is not available on this server", 404)
        board, data = read_board()
        if board is None:
            return bad("invalid move list")
        if board.is_game_over(claim_draw=True):
            return bad("game is over")
        board.push(sf.play(board, data.get("level", 3), 0.3))
        return jsonify(_state(board))

    @app.post("/api/eval")
    def evaluate():
        if not sf:
            return bad("Stockfish is not available on this server", 404)
        board, _ = read_board()
        if board is None or board.is_game_over(claim_draw=True):
            return bad("invalid or finished game")
        return jsonify(sf.analyse(board))

    @app.post("/api/predict")
    def predict():
        """Snake's own thoughts about the position: % per legal move + how well it thinks it stands."""
        board, _ = read_board()
        if board is None or board.is_game_over(claim_draw=True):
            return bad("invalid or finished game")
        return jsonify(agent.predict(board, top=8))

    # ---------------- learning lab ----------------
    @app.get("/api/learn_config")
    def learn_config_get():
        if not hub:
            return bad("learning is not enabled on this server", 404)
        from snake_learn import cfg_ui
        return jsonify(cfg_ui(hub.cfg))

    @app.post("/api/learn_config")
    def learn_config_set():
        if not hub:
            return bad("learning is not enabled on this server", 404)
        from snake_learn import cfg_set, cfg_get, UI_META
        data = request.get_json(force=True, silent=True) or {}
        name = data.get("name")
        if name not in UI_META:
            return bad("unknown setting")
        try:
            return jsonify(name=name, value=cfg_set(hub.cfg, name, data.get("value")))
        except (ValueError, TypeError) as e:
            return bad(str(e))

    @app.post("/api/learn")
    def learn():
        if not hub:
            return bad("learning is not enabled on this server", 404)
        data = request.get_json(force=True, silent=True) or {}
        try:
            board = _build_board(data.get("moves", [])[:600], data.get("start_fen"))
        except (ValueError, TypeError):
            return bad("invalid move list")
        return jsonify(hub.submit([m.uci() for m in board.move_stack], data.get("start_fen"),
                                  str(data.get("snake_color", "white")), "web"))

    @app.get("/api/learn_status")
    def learn_status():
        if not hub:
            return bad("learning is not enabled on this server", 404)
        return jsonify(hub.get_status())

    return app
