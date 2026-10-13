"""
snake_core.py  -  the "brain" of Snake
  * board encoding + move encoding (side-to-move perspective)
  * legal-move identifier (masking)
  * SnakeNet (policy + value network, PyTorch)
  * SnakeAgent (move selection, save / load)
  * identify_game (opening / phase / termination detector)
"""
import threading
import numpy as np
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

POLICY_SIZE = 64 * 64      # from-square * 64 + to-square (promotions -> queen)
N_PLANES = 18


# --------------------------------------------------------------------------
# 1. Move encoding + LEGAL MOVE IDENTIFIER
# --------------------------------------------------------------------------
# The board is always shown to the network from the point of view of the side
# to move (black positions are mirrored), so the net learns one "colour-free"
# way of playing. Moves are mirrored the same way.

def _orient(sq, turn):
    return sq if turn == chess.WHITE else chess.square_mirror(sq)


def move_to_index(move, turn):
    return _orient(move.from_square, turn) * 64 + _orient(move.to_square, turn)


def index_to_move(idx, board):
    f, t = divmod(int(idx), 64)
    f, t = _orient(f, board.turn), _orient(t, board.turn)   # mirror is its own inverse
    promo = None
    if board.piece_type_at(f) == chess.PAWN and chess.square_rank(t) in (0, 7):
        promo = chess.QUEEN                                  # auto-queen
    return chess.Move(f, t, promotion=promo)


def legal_mask(board):
    """Boolean vector of length 4096: True where the move is LEGAL right now."""
    mask = np.zeros(POLICY_SIZE, dtype=bool)
    for m in board.legal_moves:                              # python-chess knows every rule
        mask[move_to_index(m, board.turn)] = True            # (pins, checks, castling, e.p. ...)
    return mask


def is_legal_uci(board, uci):
    try:
        return chess.Move.from_uci(uci) in board.legal_moves
    except ValueError:
        return False


def encode_board(board):
    """18 x 8 x 8 float tensor: 12 piece planes, 4 castling, 1 en-passant, 1 halfmove clock."""
    b = board if board.turn == chess.WHITE else board.mirror()
    x = np.zeros((N_PLANES, 8, 8), dtype=np.float32)
    for sq, piece in b.piece_map().items():
        plane = piece.piece_type - 1 + (0 if piece.color == chess.WHITE else 6)
        x[plane, sq // 8, sq % 8] = 1.0
    x[12] = b.has_kingside_castling_rights(chess.WHITE)
    x[13] = b.has_queenside_castling_rights(chess.WHITE)
    x[14] = b.has_kingside_castling_rights(chess.BLACK)
    x[15] = b.has_queenside_castling_rights(chess.BLACK)
    if b.ep_square is not None:
        x[16, b.ep_square // 8, b.ep_square % 8] = 1.0
    x[17] = min(b.halfmove_clock, 100) / 100.0
    return x


# --------------------------------------------------------------------------
# 2. Neural network
# --------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(x + h)


class SnakeNet(nn.Module):
    def __init__(self, channels=64, blocks=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(N_PLANES, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.policy = nn.Sequential(nn.Conv2d(channels, 32, 1, bias=False), nn.BatchNorm2d(32),
                                    nn.ReLU(), nn.Flatten(), nn.Linear(32 * 64, POLICY_SIZE))
        self.value = nn.Sequential(nn.Conv2d(channels, 8, 1, bias=False), nn.BatchNorm2d(8),
                                   nn.ReLU(), nn.Flatten(), nn.Linear(8 * 64, 64), nn.ReLU(),
                                   nn.Linear(64, 1), nn.Tanh())

    def forward(self, x):
        h = self.tower(self.stem(x))
        return self.policy(h), self.value(h).squeeze(-1)


# --------------------------------------------------------------------------
# 3. Agent
# --------------------------------------------------------------------------
class SnakeAgent:
    def __init__(self, channels=64, blocks=4, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = SnakeNet(channels, blocks).to(self.device)
        self.lock = threading.RLock()        # training thread + web server share the net
        self.meta = {"channels": channels, "blocks": blocks, "train_steps": 0,
                     "games": 0, "sf_level": 0}

    @torch.no_grad()
    def evaluate(self, board):
        """-> (policy logits [4096] as numpy, value in [-1,1] for the side to move)"""
        x = torch.from_numpy(encode_board(board)).unsqueeze(0).to(self.device)
        with self.lock:
            self.net.eval()
            logits, value = self.net(x)
        return logits[0].cpu().numpy(), float(value[0])

    def select_move(self, board, temperature=0.0, lookahead=False, top_k=4, value_weight=1.0):
        """
        Pick a LEGAL move.
          temperature = 0   -> best move;  > 0 -> sample (exploration while training)
          lookahead   = True-> mate-in-1 check + re-rank the top_k moves with the value head
        """
        legal = list(board.legal_moves)
        if not legal:
            raise ValueError("no legal moves - game is over")
        logits, _ = self.evaluate(board)
        masked = np.where(legal_mask(board), logits, -1e9)      # illegal moves can never win

        if lookahead:
            for m in legal:                                      # always take a mate in 1
                board.push(m); mate = board.is_checkmate(); board.pop()
                if mate:
                    return m
            p = np.exp(masked - masked.max()); p /= p.sum()
            cand = np.argsort(-masked)[:min(top_k, len(legal))]
            best, best_score = None, -1e9
            for idx in cand:
                m = index_to_move(idx, board)
                board.push(m)
                if board.is_game_over():
                    child = 0.0                                  # draw-ish
                else:
                    child = -self.evaluate(board)[1]             # opponent's value, flipped
                board.pop()
                score = p[idx] + value_weight * child
                if score > best_score:
                    best, best_score = m, score
            return best

        if temperature <= 1e-6:
            idx = int(np.argmax(masked))
        else:
            z = (masked - masked.max()) / temperature
            p = np.exp(z); p /= p.sum()
            idx = int(np.random.choice(POLICY_SIZE, p=p))
        return index_to_move(idx, board)

    # ---- persistence ----
    def save(self, path):
        with self.lock:
            torch.save({"state_dict": self.net.state_dict(), "meta": self.meta}, path)

    @classmethod
    def load(cls, path, device=None):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        meta = ck["meta"]
        agent = cls(meta["channels"], meta["blocks"], device)
        agent.net.load_state_dict(ck["state_dict"])
        agent.meta = meta
        agent.net.eval()
        return agent


# --------------------------------------------------------------------------
# 4. (Optional) Chess GAME IDENTIFIER: opening, phase, how the game ended
# --------------------------------------------------------------------------
OPENINGS = {
    "e4 e5 Nf3 Nc6 Bb5": "Ruy Lopez", "e4 e5 Nf3 Nc6 Bc4": "Italian Game",
    "e4 e5 Nf3 Nc6 d4": "Scotch Game", "e4 e5 Nf3 Nc6 Nc3 Nf6": "Four Knights Game",
    "e4 e5 Nf3 Nf6": "Petrov's Defence", "e4 e5 f4": "King's Gambit",
    "e4 e5 Nc3": "Vienna Game", "e4 e5 Bc4": "Bishop's Opening", "e4 e5": "Open Game",
    "e4 c5": "Sicilian Defence", "e4 c5 Nf3 d6": "Sicilian: Najdorf/Dragon family",
    "e4 c5 Nf3 Nc6": "Sicilian: Classical", "e4 c5 c3": "Sicilian: Alapin",
    "e4 e6": "French Defence", "e4 c6": "Caro-Kann Defence", "e4 d5": "Scandinavian Defence",
    "e4 d6": "Pirc Defence", "e4 g6": "Modern Defence", "e4 Nf6": "Alekhine's Defence",
    "d4 d5 c4 e6": "Queen's Gambit Declined", "d4 d5 c4 dxc4": "Queen's Gambit Accepted",
    "d4 d5 c4": "Queen's Gambit", "d4 d5 Bf4": "London System", "d4 d5": "Closed Game",
    "d4 Nf6 c4 g6": "King's Indian / Grunfeld setup", "d4 Nf6 c4 e6 Nc3 Bb4": "Nimzo-Indian Defence",
    "d4 Nf6 c4 e6": "Queen's Indian / Nimzo setup", "d4 Nf6 Bf4": "London System",
    "d4 Nf6": "Indian Defence", "d4 f5": "Dutch Defence",
    "c4": "English Opening", "Nf3": "Reti Opening", "f4": "Bird's Opening",
    "b3": "Nimzo-Larsen Attack", "g3": "King's Fianchetto Opening",
}
_OPENING_KEYS = sorted(((k.split(), v) for k, v in OPENINGS.items()), key=lambda kv: -len(kv[0]))


def identify_game(board):
    """Describe a game from its move stack: opening, phase, material, termination."""
    replay, sans = chess.Board(), []
    for m in board.move_stack:
        sans.append(replay.san(m)); replay.push(m)
    sans_clean = [s.rstrip("+#") for s in sans]
    opening = "Unknown / out of book"
    for tokens, name in _OPENING_KEYS:
        if sans_clean[:len(tokens)] == tokens:
            opening = name
            break
    if not sans:
        opening = "Starting position"

    values = {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    def material(color):
        return sum(v * len(board.pieces(pt, color)) for pt, v in values.items())
    total = material(chess.WHITE) + material(chess.BLACK)
    if board.fullmove_number <= 10 and total >= 56:
        phase = "opening"
    elif total <= 24:
        phase = "endgame"
    else:
        phase = "middlegame"

    info = {"opening": opening, "phase": phase, "ply": board.ply(),
            "material_white": material(chess.WHITE), "material_black": material(chess.BLACK),
            "in_check": board.is_check(), "over": False, "result": "*", "termination": None,
            "winner": None}
    outcome = board.outcome(claim_draw=True)
    if outcome:
        info.update(over=True, result=outcome.result(), termination=outcome.termination.name,
                    winner=None if outcome.winner is None else ("white" if outcome.winner else "black"))
    return info
