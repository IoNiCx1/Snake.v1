"""
snake_learn.py - everything about LEARNING
  * LossCfg / LearnCfg : every knob, with UI metadata (the website builds its "Learning lab" from it)
  * SnakeDB            : SQLite memory (auto-migrates older snake.db files)
  * Trainer            : imitation + value + eval-DROP punishment + good-move reward
  * LearningHub        : "study session" after a game: label with Stockfish, then consolidate into the weights
"""
import math, os, queue, random, sqlite3, threading, time, traceback
from dataclasses import dataclass, field, fields, asdict
import numpy as np
import chess, chess.engine
import torch
import torch.nn.functional as F
from snake_core import encode_board, legal_mask, move_to_index, identify_game


# ==========================================================================
# 1. Configuration  (all of it is editable live from the website / notebook)
# ==========================================================================
@dataclass
class LossCfg:
    w_imitate: float = 1.0          # learn Stockfish's best move
    w_value: float = 1.0            # learn Stockfish's evaluation
    w_punish: float = 1.0           # punish moves that DROP the evaluation
    w_reward: float = 0.3           # reinforce alternative moves that keep the evaluation
    punish_mode: str = "unlikelihood"   # "unlikelihood" (stable) | "pg" (policy-gradient) | "off"
    squash: bool = False            # True: measure drops with tanh(cp/scale) -> hopeless positions punished less
    squash_scale: float = 400.0
    threshold_cp: float = 40.0      # ignore eval drops smaller than this (analysis noise)
    norm_cp: float = 300.0          # a drop of this many centipawns = full-strength punishment
    cp_clip: float = 1000.0         # clip evals to +-this before taking the difference (mate = 10000)
    reward_threshold_cp: float = 15.0   # a non-best move losing <= this is "as good as best"
    value_scale: float = 400.0      # value target = tanh(cp / value_scale)
    log_floor: float = -8.0         # PG mode: stop pushing log-prob down below this


@dataclass
class LearnCfg:
    auto_study: bool = True         # study every finished game on the website automatically
    learn_all_positions: bool = False   # also learn from the OTHER side's positions (imitate Stockfish)
    label_time: float = 0.10        # seconds Stockfish spends labelling each position
    epochs: int = 4                 # passes over the game
    batch_size: int = 16            # positions from the game per gradient step
    replay_ratio: float = 2.0       # extra old positions per game position (prevents forgetting)
    lr_scale: float = 1.0           # learning-rate multiplier during the study session
    save_after: bool = True         # write snake_model.pt after studying
    loss: LossCfg = field(default_factory=LossCfg)


# (label, help, min, max, step, options) for the UI. Anything missing falls back to a sensible default.
UI_META = {
    "auto_study": ("Study every finished game", "Snake learns from each game as soon as it ends"),
    "learn_all_positions": ("Learn from the other side too", "Also imitate Stockfish on positions where the opponent moved"),
    "label_time": ("Stockfish labelling time (s)", "More time = better labels, slower study", 0.02, 1.0, 0.01),
    "epochs": ("Study epochs", "How many times Snake re-reads the game", 1, 30, 1),
    "batch_size": ("Batch from the game", "Game positions per gradient step", 2, 128, 1),
    "replay_ratio": ("Replay ratio", "Old positions mixed in per game position", 0.0, 8.0, 0.5),
    "lr_scale": ("Learning-rate multiplier", "Study session speed (x base lr)", 0.1, 5.0, 0.1),
    "save_after": ("Save model after studying", "Write snake_model.pt"),
    "loss.w_imitate": ("Weight: imitate Stockfish", "", 0.0, 3.0, 0.05),
    "loss.w_value": ("Weight: value/eval", "", 0.0, 3.0, 0.05),
    "loss.w_punish": ("Weight: punishment", "Strength of the 'dumb move' penalty", 0.0, 5.0, 0.05),
    "loss.w_reward": ("Weight: reward", "Reinforce near-best alternative moves", 0.0, 3.0, 0.05),
    "loss.punish_mode": ("Punishment mode", "unlikelihood = -log(1-p) (stable); pg = policy gradient; off", None, None, None,
                         ["unlikelihood", "pg", "off"]),
    "loss.squash": ("Squash evals with tanh", "Hopeless positions: a further drop counts for less"),
    "loss.squash_scale": ("Squash scale (cp)", "", 100.0, 1000.0, 10.0),
    "loss.threshold_cp": ("Ignore drops below (cp)", "Dead-zone for analysis noise", 0.0, 300.0, 5.0),
    "loss.norm_cp": ("Full punishment at drop (cp)", "Punishment grows linearly up to this drop", 50.0, 1500.0, 10.0),
    "loss.cp_clip": ("Clip evals at (cp)", "Mate scores are clipped to this", 200.0, 3000.0, 50.0),
    "loss.reward_threshold_cp": ("Reward if loss under (cp)", "", 0.0, 100.0, 1.0),
}


def cfg_get(cfg, name):
    obj = cfg
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def cfg_set(cfg, name, value):
    """Validated, clamped assignment. Returns the stored value."""
    *path, leaf = name.split(".")
    obj = cfg
    for part in path:
        obj = getattr(obj, part)
    current = getattr(obj, leaf)
    meta = UI_META.get(name, ())
    if isinstance(current, bool):
        value = bool(value)
    elif isinstance(current, int):
        value = int(round(float(value)))
    elif isinstance(current, float):
        value = float(value)
    else:
        value = str(value)
        opts = meta[5] if len(meta) > 5 else None
        if opts and value not in opts:
            raise ValueError(f"{name} must be one of {opts}")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and len(meta) > 3 and meta[2] is not None:
        value = type(current)(min(max(value, meta[2]), meta[3]))
    setattr(obj, leaf, value)
    return value


def cfg_ui(cfg):
    """List of control descriptions for the website."""
    out = []
    for name, meta in UI_META.items():
        cur = cfg_get(cfg, name)
        item = {"name": name, "label": meta[0], "help": meta[1] if len(meta) > 1 else "", "value": cur}
        if isinstance(cur, bool):
            item["type"] = "bool"
        elif isinstance(cur, str):
            item.update(type="select", options=meta[5])
        else:
            item.update(type="int" if isinstance(cur, int) else "float", min=meta[2], max=meta[3], step=meta[4])
        out.append(item)
    return out


# ==========================================================================
# 2. Database
# ==========================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS games(
  id INTEGER PRIMARY KEY AUTOINCREMENT, started_at REAL, snake_color TEXT, sf_level INTEGER,
  result TEXT, snake_score REAL, termination TEXT, opening TEXT, n_plies INTEGER, pgn TEXT);
CREATE TABLE IF NOT EXISTS positions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, game_id INTEGER, ply INTEGER, fen TEXT,
  teacher_move TEXT, played_move TEXT, teacher_cp INTEGER, value_target REAL,
  phase TEXT, agreed INTEGER);
CREATE TABLE IF NOT EXISTS train_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, step INTEGER, policy_loss REAL,
  value_loss REAL, games INTEGER, sf_level INTEGER);
CREATE INDEX IF NOT EXISTS idx_pos_game ON positions(game_id);
"""
MIGRATIONS = {   # columns added later; older databases get them automatically
    "positions": [("cp_after", "INTEGER"), ("delta_cp", "INTEGER"), ("source", "TEXT")],
    "games": [("source", "TEXT")],
    "train_log": [("punish_loss", "REAL")],
}
ROW_COLS = "fen, teacher_move, value_target, played_move, teacher_cp, cp_after"


class SnakeDB:
    def __init__(self, path):
        self.path = path
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        for table, cols in MIGRATIONS.items():
            have = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            for name, typ in cols:
                if name not in have:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {typ}")
        self.conn.commit()

    def new_game(self, snake_color, sf_level, source="train"):
        with self.lock:
            cur = self.conn.execute("INSERT INTO games(started_at, snake_color, sf_level, source) VALUES(?,?,?,?)",
                                    (time.time(), snake_color, sf_level, source))
            self.conn.commit()
            return cur.lastrowid

    def add_position(self, game_id, ply, fen, teacher_move, played_move, cp, value_target, phase, agreed,
                     cp_after=None, delta_cp=None, source="train"):
        with self.lock:
            self.conn.execute(
                "INSERT INTO positions(game_id,ply,fen,teacher_move,played_move,teacher_cp,value_target,phase,"
                "agreed,cp_after,delta_cp,source) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (game_id, ply, fen, teacher_move, played_move, cp, value_target, phase, int(agreed),
                 cp_after, delta_cp, source))
            self.conn.commit()

    def finish_game(self, game_id, result, snake_score, termination, opening, n_plies, pgn):
        with self.lock:
            self.conn.execute("UPDATE games SET result=?, snake_score=?, termination=?, opening=?, n_plies=?, pgn=? WHERE id=?",
                              (result, snake_score, termination, opening, n_plies, pgn, game_id))
            self.conn.commit()

    def log_train(self, step, p_loss, v_loss, games, sf_level, punish_loss=None):
        with self.lock:
            self.conn.execute("INSERT INTO train_log(ts,step,policy_loss,value_loss,games,sf_level,punish_loss) "
                              "VALUES(?,?,?,?,?,?,?)", (time.time(), step, p_loss, v_loss, games, sf_level, punish_loss))
            self.conn.commit()

    def count(self, table="positions"):
        with self.lock:
            return self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def sample(self, n, recent_pool=500):
        """Half the batch from the newest positions (fresh mistakes), half uniformly from all history."""
        with self.lock:
            mx = self.conn.execute("SELECT MAX(id) FROM positions").fetchone()[0] or 0
            if mx == 0 or n <= 0:
                return []
            half = n // 2
            ids = [random.randint(max(1, mx - recent_pool + 1), mx) for _ in range(half)]
            ids += [random.randint(1, mx) for _ in range(n - half)]
            ids = list(set(ids))
            q = ",".join("?" * len(ids))
            return self.conn.execute(f"SELECT {ROW_COLS} FROM positions WHERE id IN ({q})", ids).fetchall()

    def query(self, sql, params=()):
        with self.lock:
            return self.conn.execute(sql, params).fetchall()


# ==========================================================================
# 3. Trainer  (imitation + value + punish eval-drops + reward)
# ==========================================================================
class Trainer:
    def __init__(self, agent, loss_cfg=None, lr=1e-3, weight_decay=1e-4):
        from collections import deque
        self.agent, self.loss_cfg, self.base_lr = agent, loss_cfg or LossCfg(), lr
        self.opt = torch.optim.AdamW(agent.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.history = deque(maxlen=20000)     # (step, imitate_loss, value_loss, punish_loss)

    # ---- the key idea: use the DIFFERENCE between the eval before and after Snake's move ----
    def drop_strength(self, cp_before, cp_after):
        """0 .. 1 punishment strength for a move that took the eval from cp_before to cp_after
        (both from the mover's point of view). A move that merely stays in a lost position
        (before == after) has drop 0 -> not punished, however negative the eval is."""
        c = self.loss_cfg
        if cp_before is None or cp_after is None:
            return 0.0, 0.0
        b = max(-c.cp_clip, min(c.cp_clip, float(cp_before)))
        a = max(-c.cp_clip, min(c.cp_clip, float(cp_after)))
        if c.squash:
            s = c.squash_scale
            drop, thr, norm = math.tanh(b / s) - math.tanh(a / s), math.tanh(c.threshold_cp / s), math.tanh(c.norm_cp / s)
        else:
            drop, thr, norm = b - a, c.threshold_cp, c.norm_cp
        punish = min(max(drop - thr, 0.0) / max(norm, 1e-6), 1.0)
        reward = 1.0 if (b - a) <= c.reward_threshold_cp else 0.0      # (always judged in cp)
        return punish, reward

    def _prepare(self, rows):
        dev = self.agent.device
        boards = [chess.Board(r[0]) for r in rows]
        X = torch.from_numpy(np.stack([encode_board(b) for b in boards])).to(dev)
        M = torch.from_numpy(np.stack([legal_mask(b) for b in boards])).to(dev)
        y_best, y_mine, pun, rew, y_val = [], [], [], [], []
        for b, r in zip(boards, rows):
            _, tmove, _, pmove, cp_b, cp_a = r
            best = move_to_index(chess.Move.from_uci(tmove), b.turn)
            y_best.append(best)
            y_val.append(math.tanh(max(-10000, min(10000, cp_b if cp_b is not None else 0)) / self.loss_cfg.value_scale)
                         if cp_b is not None else float(r[2]))
            mine = move_to_index(chess.Move.from_uci(pmove), b.turn) if pmove else best
            y_mine.append(mine)
            if pmove and mine != best:
                p, rw = self.drop_strength(cp_b, cp_a)
                pun.append(p); rew.append(rw if cp_a is not None else 0.0)
            else:
                pun.append(0.0); rew.append(0.0)
        t = lambda v, dt=torch.float32: torch.tensor(v, dtype=dt, device=dev)
        return X, M, t(y_best, torch.long), t(y_mine, torch.long), t(pun), t(rew), t(y_val)

    def _loss(self, rows):
        c = self.loss_cfg
        X, M, y_best, y_mine, pun, rew, y_val = self._prepare(rows)
        logits, value = self.agent.net(X)
        logp = F.log_softmax(logits.masked_fill(~M, -1e9), dim=1)           # only LEGAL moves compete
        imitate = F.nll_loss(logp, y_best)
        v_loss = F.mse_loss(value, y_val)
        lp_mine = logp.gather(1, y_mine.unsqueeze(1)).squeeze(1)
        if c.punish_mode == "unlikelihood":
            pen = -torch.log1p(-lp_mine.exp().clamp(max=1 - 1e-4))          # -log(1 - p(move))
        elif c.punish_mode == "pg":
            pen = lp_mine.clamp(min=c.log_floor)                            # push log p(move) DOWN
        else:
            pen = torch.zeros_like(lp_mine)
        punish = (pun * pen).mean()
        reward = (rew * (-lp_mine)).mean()
        total = c.w_imitate * imitate + c.w_value * v_loss + c.w_punish * punish + c.w_reward * reward
        return total, imitate, v_loss, punish, reward, logp, y_best, int((pun > 0).sum())

    def step(self, rows, lr_scale=1.0):
        with self.agent.lock:
            for g in self.opt.param_groups:
                g["lr"] = self.base_lr * lr_scale
            self.agent.net.train()
            total, imitate, v_loss, punish, reward, _, _, n_pun = self._loss(rows)
            self.opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.net.parameters(), 1.0)
            self.opt.step()
            self.agent.meta["train_steps"] += 1
            self.agent.net.eval()
            for g in self.opt.param_groups:
                g["lr"] = self.base_lr
        out = dict(imitate=imitate.item(), value=v_loss.item(), punish=punish.item(),
                   reward=reward.item(), n_punished=n_pun)
        self.history.append((self.agent.meta["train_steps"], out["imitate"], out["value"], out["punish"]))
        return out

    @torch.no_grad()
    def evaluate(self, rows):
        """No learning: how well does Snake currently match Stockfish on these positions?"""
        with self.agent.lock:
            self.agent.net.eval()
            _, imitate, v_loss, punish, _, logp, y_best, _ = self._loss(rows)
            agree = (logp.argmax(1) == y_best).float().mean().item()
        return dict(imitate=imitate.item(), value=v_loss.item(), punish=punish.item(), agree=agree)


# ==========================================================================
# 4. LearningHub: "study session" after a game
# ==========================================================================
class LearningHub:
    """
    After a game finishes (on the website), Snake 'sleeps on it':
      1) Stockfish labels every position Snake moved in (best move, eval before, eval after Snake's move)
      2) positions are stored in the database
      3) Snake re-reads the game for several epochs (mixed with old memories) -> weights change
    Jobs run one at a time in a background thread so the website stays responsive.
    """
    def __init__(self, agent, db, stockfish_path, cfg=None, trainer=None, model_path=None):
        self.agent, self.db, self.sf_path, self.model_path = agent, db, stockfish_path, model_path
        self.cfg = cfg or LearnCfg()
        self.trainer = trainer or Trainer(agent, self.cfg.loss)
        self.trainer.loss_cfg = self.cfg.loss                      # share live-editable loss settings
        self.q = queue.Queue()
        self.lock = threading.Lock()
        self.worker = None
        self.stop_event = threading.Event()
        self.jobs_done = 0
        self.status = {"state": "idle", "progress": 0, "total": 0, "message": "Snake is resting.",
                       "queue": 0, "last": None, "error": None}
        getattr(threading, "_register_atexit", lambda f: None)(self.stop_event.set)

    @property
    def enabled(self):
        return bool(self.sf_path and os.path.exists(self.sf_path))

    def get_status(self):
        with self.lock:
            s = dict(self.status); s["queue"] = self.q.qsize(); s["jobs_done"] = self.jobs_done
            return s

    def _set(self, **kw):
        with self.lock:
            self.status.update(kw)

    def submit(self, moves, start_fen=None, snake_color="white", source="web"):
        if not self.enabled:
            return {"queued": False, "reason": "Stockfish is not available, so Snake cannot label games."}
        if len(moves) < 2:
            return {"queued": False, "reason": "Game too short to study."}
        if snake_color not in ("white", "black", "none") or (snake_color == "none" and not self.cfg.learn_all_positions):
            return {"queued": False, "reason": "Snake did not play in this game (enable 'Learn from the other side too')."}
        self.q.put(dict(moves=list(moves), start_fen=start_fen, snake_color=snake_color, source=source))
        if self.worker is None or not self.worker.is_alive():
            self.worker = threading.Thread(target=self._loop, daemon=True)
            self.worker.start()
        return {"queued": True, "queue": self.q.qsize()}

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                job = self.q.get(timeout=2.0)
            except queue.Empty:
                return                                  # idle: thread ends, restarts on next submit
            try:
                self._process(job)
            except Exception:
                self._set(state="error", error=traceback.format_exc(), message="Study session failed.")

    # ---------------------------------------------------------------
    def _terminal_cp(self, board, mover_color):
        if board.is_checkmate():
            return 10000 if board.turn != mover_color else -10000     # side to move is mated
        return 0

    def _label(self, job):
        cfg = self.cfg
        board = chess.Board(job["start_fen"]) if job["start_fen"] else chess.Board()
        snake = {"white": chess.WHITE, "black": chess.BLACK}.get(job["snake_color"])
        game_id = self.db.new_game(job["snake_color"], -1, source=job["source"])
        rows, limit = [], chess.engine.Limit(time=cfg.label_time)
        eng = chess.engine.SimpleEngine.popen_uci(self.sf_path)
        try:
            eng.configure({"Threads": 1, "Hash": 64, "Skill Level": 20})
            total = len(job["moves"])
            for i, uci in enumerate(job["moves"]):
                if self.stop_event.is_set():
                    break
                move = chess.Move.from_uci(uci)
                mine = snake is not None and board.turn == snake
                if mine or cfg.learn_all_positions:
                    info = eng.analyse(board, limit)
                    best = info["pv"][0]
                    cp_b = info["score"].pov(board.turn).score(mate_score=10000)
                    same = move_to_index(move, board.turn) == move_to_index(best, board.turn)
                    cp_a = cp_b
                    if not same:
                        after = board.copy(); after.push(move)
                        if after.is_game_over():
                            cp_a = self._terminal_cp(after, board.turn)
                        else:
                            cp_a = eng.analyse(after, limit)["score"].pov(board.turn).score(mate_score=10000)
                    delta = cp_a - cp_b if mine else None
                    self.db.add_position(game_id, board.ply(), board.fen(), best.uci(), uci if mine else None,
                                         cp_b, math.tanh(cp_b / self.cfg.loss.value_scale),
                                         identify_game(board)["phase"], same, cp_a if mine else None, delta,
                                         source=job["source"])
                    self.db.conn.commit()
                    rows.append((board.fen(), best.uci(), math.tanh(cp_b / self.cfg.loss.value_scale),
                                 uci if mine else None, cp_b, cp_a if mine else None))
                board.push(move)
                self._set(state="labelling", progress=i + 1, total=total,
                          message=f"Stockfish is labelling the game... move {i + 1}/{total}")
        finally:
            try:
                eng.quit()
            except Exception:
                pass
        return game_id, rows, board

    def _process(self, job):
        cfg, tr = self.cfg, self.trainer
        self._set(state="labelling", progress=0, total=len(job["moves"]), error=None,
                  message="Snake is going to sleep on this game...")
        game_id, rows, final = self._label(job)
        if not rows:
            self._set(state="idle", message="Nothing to learn from in that game."); return
        gi = identify_game(final)
        self.db.finish_game(game_id, gi["result"], None, gi["termination"] or "UNFINISHED", gi["opening"],
                            final.ply(), "")
        before = tr.evaluate(rows)
        snake_rows = [r for r in rows if r[3]]
        clip = lambda x: max(-cfg.loss.cp_clip, min(cfg.loss.cp_clip, x))     # one allowed mate must not dominate
        deltas = [clip(r[5]) - clip(r[4]) for r in snake_rows]
        n_pun = sum(1 for r in snake_rows if tr.drop_strength(r[4], r[5])[0] > 0)
        n_rew = sum(1 for r in snake_rows if r[3] and tr.drop_strength(r[4], r[5])[1] > 0
                    and chess.Move.from_uci(r[3]) != chess.Move.from_uci(r[1]))
        steps, total_steps = 0, cfg.epochs * max(1, math.ceil(len(rows) / cfg.batch_size))
        for ep in range(cfg.epochs):
            random.shuffle(rows)
            for k in range(0, len(rows), cfg.batch_size):
                if self.stop_event.is_set():
                    break
                chunk = rows[k:k + cfg.batch_size]
                replay = self.db.sample(int(len(chunk) * cfg.replay_ratio))
                tr.step(chunk + replay, lr_scale=cfg.lr_scale)
                steps += 1
                self._set(state="consolidating", progress=steps, total=total_steps,
                          message=f"Snake is embedding the game into memory... epoch {ep + 1}/{cfg.epochs}")
        after = tr.evaluate(rows)
        self.agent.meta["web_games"] = self.agent.meta.get("web_games", 0) + 1
        if cfg.save_after and self.model_path:
            self.agent.save(self.model_path)
        self.jobs_done += 1
        self._set(state="done", progress=total_steps, total=total_steps, message="Snake has learned from the game.",
                  last={"game_id": game_id, "positions": len(rows), "snake_moves": len(snake_rows),
                        "avg_delta_cp": round(float(np.mean(deltas)), 1) if deltas else None,
                        "punished": n_pun, "rewarded": n_rew, "steps": steps,
                        "imitate_before": round(before["imitate"], 3), "imitate_after": round(after["imitate"], 3),
                        "agree_before": round(before["agree"] * 100, 1), "agree_after": round(after["agree"] * 100, 1),
                        "train_steps_total": self.agent.meta["train_steps"]})
