"""snake_web.py - Flask website + JSON API for Snake (used locally AND when deployed)."""
import atexit
import os
import threading
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

INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Snake - AI Chess Bot</title>
<style>
  :root{--light:#f0d9b5;--dark:#b58863;--bg:#1e1e24;--card:#2a2a33;--text:#ececf1;--accent:#6ee7a8}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,Segoe UI,Roboto,sans-serif;
  display:flex;flex-direction:column;align-items:center;padding:16px}
  h1{margin:8px 0 2px;font-size:1.6rem} h1 span{color:var(--accent)} .sub{opacity:.7;margin-bottom:14px;font-size:.9rem}
  #wrap{display:flex;flex-wrap:wrap;gap:20px;justify-content:center;width:100%;max-width:960px}
  #board{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));grid-template-rows:repeat(8,minmax(0,1fr));
         width:min(92vw,520px);height:min(92vw,520px);border:4px solid #444;border-radius:6px;overflow:hidden}
  .sq{display:flex;align-items:center;justify-content:center;position:relative;cursor:pointer;user-select:none;
      min-width:0;min-height:0;overflow:hidden;font-size:min(8vw,50px);line-height:1}
  .sq.l{background:var(--light)} .sq.d{background:var(--dark)}
  .sq.sel{outline:4px solid #4da3ff;outline-offset:-4px} .sq.last{box-shadow:inset 0 0 0 100px rgba(255,235,59,.38)}
  .sq.chk{box-shadow:inset 0 0 0 100px rgba(255,60,60,.5)} .sq.hint{box-shadow:inset 0 0 0 100px rgba(80,200,120,.45)}
  .sq.tgt::after{content:"";width:28%;height:28%;border-radius:50%;background:rgba(0,0,0,.28);position:absolute}
  .sq.tgt.cap::after{width:88%;height:88%;background:none;border:5px solid rgba(0,0,0,.3)}
  .w{color:#fff;text-shadow:0 0 2px #000,0 0 2px #000,0 0 3px #000} .b{color:#111;text-shadow:0 0 1px #fff}
  .coord{position:absolute;font-size:10px;opacity:.65;color:#000} .coord.r{top:2px;left:3px} .coord.f{bottom:1px;right:3px}
  #panel{background:var(--card);border-radius:10px;padding:16px;width:min(92vw,320px);display:flex;flex-direction:column;gap:10px}
  #status{font-weight:600;min-height:2.6em} .row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
  button{background:var(--accent);color:#102;border:0;border-radius:8px;padding:9px 12px;font-weight:700;cursor:pointer}
  button.alt{background:#44445a;color:var(--text)} button:disabled{opacity:.5;cursor:wait}
  select,input[type=range]{accent-color:var(--accent)} select{background:#44445a;color:var(--text);border:0;border-radius:6px;padding:6px}
  #moves{font-family:ui-monospace,Menlo,monospace;font-size:.85rem;max-height:150px;overflow:auto;background:#222229;border-radius:6px;padding:8px;min-height:44px}
  .tag{font-size:.8rem;opacity:.8} label{font-size:.85rem;display:flex;gap:6px;align-items:center}
  #sfbox{display:none;flex-direction:column;gap:8px;border-top:1px solid #444;padding-top:10px}
  #bar{height:16px;background:#111;border-radius:8px;overflow:hidden;border:1px solid #555}
  #barfill{height:100%;width:50%;background:#f2f2f2;transition:width .3s}
</style></head><body>
<h1>&#128013; <span>Snake</span> - AI Chess Bot</h1>
<div class="sub">A neural network trained live against Stockfish</div>
<div id="wrap">
  <div id="board"></div>
  <div id="panel">
    <div id="status">Loading...</div>
    <div class="tag" id="info"></div>
    <div id="sfbox">
      <label>Mode:&nbsp;<select id="mode" onchange="startNew()">
        <option value="human">You vs Snake</option>
        <option value="humansf">You vs Stockfish</option>
        <option value="spectate">Watch: Snake vs Stockfish</option></select></label>
      <label>Stockfish level: <b id="lvlv">3</b>
        <input type="range" id="lvl" min="0" max="20" value="3" oninput="document.getElementById('lvlv').textContent=this.value"></label>
      <label id="snakewhiteRow"><input type="checkbox" id="snakewhite" checked> Snake plays White (watch mode)</label>
      <label><input type="checkbox" id="evalon" onchange="refreshEval()"> Stockfish evaluation bar + hint</label>
      <div id="bar"><div id="barfill"></div></div><div class="tag" id="evaltxt"></div>
    </div>
    <div class="row" id="playbtns">
      <button onclick="newGame('white')">Play White</button>
      <button onclick="newGame('black')" class="alt">Play Black</button>
    </div>
    <div class="row" id="watchbtns" style="display:none">
      <button onclick="startWatch()">Start match</button>
      <button onclick="watching=false" class="alt">Pause</button>
    </div>
    <label><input type="checkbox" id="look" checked> Smart mode (1-ply lookahead)</label>
    <div id="moves"></div>
    <div class="tag" id="stats"></div>
  </div>
</div>
<script>
const GLYPH={k:"\u265A\uFE0E",q:"\u265B\uFE0E",r:"\u265C\uFE0E",b:"\u265D\uFE0E",n:"\u265E\uFE0E",p:"\u265F\uFE0E"};
let moves=[], st=null, me="white", sel=null, busy=false, hasSF=false, watching=false, hintSq=null, token=0;
const $=id=>document.getElementById(id), mode=()=>hasSF?$("mode").value:"human", lvl=()=>+$("lvl").value;

async function api(path, body){
  const r=await fetch(path,{method:body?"POST":"GET",headers:{"Content-Type":"application/json"},body:body?JSON.stringify(body):undefined});
  const j=await r.json(); if(!r.ok) throw new Error(j.error||"error"); return j;
}
function parseFen(fen){
  const rows=fen.split(" ")[0].split("/"), g={};
  rows.forEach((row,i)=>{let f=0;for(const ch of row){if(/\d/.test(ch)){f+=+ch}else{g["abcdefgh"[f]+(8-i)]=ch;f++}}});
  return g;
}
function render(){
  const g=parseFen(st.fen), last=moves.length?moves[moves.length-1]:null, el=$("board");
  el.innerHTML="";
  const files=me==="white"?"abcdefgh":"hgfedcba", ranks=me==="white"?[8,7,6,5,4,3,2,1]:[1,2,3,4,5,6,7,8];
  const targets=sel&&st.legal[sel]?st.legal[sel]:[];
  ranks.forEach((r,ri)=>files.split("").forEach((f,fi)=>{
    const s=f+r, d=document.createElement("div"), light=("abcdefgh".indexOf(f)+r)%2===0;  // a1 is DARK
    d.className="sq "+(light?"l":"d");
    if(s===sel)d.classList.add("sel");
    if(last&&(s===last.slice(0,2)||s===last.slice(2,4)))d.classList.add("last");
    if(hintSq&&(s===hintSq[0]||s===hintSq[1]))d.classList.add("hint");
    if(targets.includes(s)){d.classList.add("tgt");if(g[s])d.classList.add("cap")}
    const p=g[s];
    if(p){ if(p.toLowerCase()==="k"&&st.in_check&&((p===p.toUpperCase())===(st.turn==="white")))d.classList.add("chk");
      d.innerHTML=`<span class="${p===p.toUpperCase()?"w":"b"}">${GLYPH[p.toLowerCase()]}</span>`;}
    if(fi===0){const c=document.createElement("span");c.className="coord r";c.textContent=r;d.appendChild(c)}
    if(ri===7){const c=document.createElement("span");c.className="coord f";c.textContent=f;d.appendChild(c)}
    d.onclick=()=>clickSq(s);
    el.appendChild(d);
  }));
  const m=mode(), opp=m==="humansf"?"Stockfish":"Snake", why=t=>t.toLowerCase().replace(/_/g," ");
  let txt;
  if(m==="spectate"){
    const sw=$("snakewhite").checked, nm=c=>(c==="white")===sw?"Snake":`Stockfish L${lvl()}`;
    if(st.over) txt=st.result==="1/2-1/2"?`Draw (${why(st.termination)})`:`${nm(st.winner)} (${st.winner}) wins by ${why(st.termination)}`;
    else txt=`${nm(st.turn)} (${st.turn}) is thinking...`;
  } else if(st.over){
    txt=st.result==="1/2-1/2"?`Draw (${why(st.termination)})`:
      `${st.winner==="white"?"White":"Black"} wins by ${why(st.termination)}`+((st.winner===me)?" - you win!":` - ${opp} wins`);
  } else txt=st.turn===me?"Your move"+(st.in_check?" (you are in check!)":""):`${opp} is thinking...`;
  $("status").textContent=txt;
  $("info").textContent=`Opening: ${st.info.opening} | Phase: ${st.info.phase} | Move ${Math.floor(st.info.ply/2)+1}`;
  const san=st.san||[]; let h="";
  for(let i=0;i<san.length;i+=2)h+=`${i/2+1}. ${san[i]} ${san[i+1]||""}<br>`;
  const mv=$("moves"); mv.innerHTML=h||"<i>No moves yet</i>"; mv.scrollTop=mv.scrollHeight;
  $("playbtns").style.display=m==="spectate"?"none":"flex"; $("watchbtns").style.display=m==="spectate"?"flex":"none";
  $("snakewhiteRow").style.display=m==="spectate"?"flex":"none";
}
async function refreshEval(){
  hintSq=null; const my=++token;
  if(!hasSF||!$("evalon").checked||!st){ $("evaltxt").textContent=""; $("barfill").style.width="50%"; if(st)render(); return; }
  if(st.over){ return; }
  try{
    const e=await api("/api/eval",{moves}); if(my!==token)return;
    const pct=50+50*Math.tanh(e.cp/400), sc=e.mate!==null?`mate in ${Math.abs(e.mate)} for ${e.mate>0?"White":"Black"}`:(e.cp>=0?"+":"")+(e.cp/100).toFixed(2);
    $("barfill").style.width=pct+"%";
    const myTurn=mode()!=="spectate"&&st.turn===me;
    $("evaltxt").textContent=`Eval (White's view): ${sc}`+(myTurn&&e.best_san?` | Stockfish suggests: ${e.best_san}`:"");
    hintSq=myTurn&&e.best?[e.best.slice(0,2),e.best.slice(2,4)]:null; render();
  }catch(err){}
}
async function clickSq(s){
  if(busy||st.over||st.turn!==me||mode()==="spectate")return;
  const g=parseFen(st.fen);
  if(sel&&st.legal[sel]&&st.legal[sel].includes(s)){
    const uci=sel+s; sel=null; hintSq=null; busy=true;
    try{
      st=await api("/api/move",{moves,move:uci,lookahead:$("look").checked,opponent:mode()==="humansf"?"stockfish":"snake",level:lvl()});
      moves=st.moves;
    }catch(e){alert(e.message)}
    busy=false; render(); refreshEval(); loadStats(); return;
  }
  sel=(g[s]&&st.legal[s])?s:null; render();
}
async function opponentMove(){
  const body={moves,lookahead:$("look").checked,level:lvl()};
  return mode()==="humansf"?api("/api/sf_move",body):api("/api/snake",body);
}
async function newGame(color){
  watching=false; me=color; sel=null; hintSq=null; moves=[]; busy=true;
  st=await api("/api/state",{moves:[]}); render();
  if(me==="black"){ st=await opponentMove(); moves=st.moves; }
  busy=false; render(); refreshEval();
}
function startNew(){ $("mode").value==="spectate"?startWatch():newGame("white"); }
async function startWatch(){
  watching=false; await new Promise(r=>setTimeout(r,50));
  const sw=$("snakewhite").checked; me=sw?"white":"black"; sel=null; hintSq=null; moves=[]; busy=true;
  st=await api("/api/state",{moves:[]}); render(); watching=true;
  while(watching&&!st.over){
    const snakeTurn=(st.turn==="white")===sw, body={moves,lookahead:$("look").checked,level:lvl()};
    try{ st=await (snakeTurn?api("/api/snake",body):api("/api/sf_move",body)); moves=st.moves; }
    catch(e){ watching=false; alert(e.message); break; }
    render(); refreshEval(); await new Promise(r=>setTimeout(r,450));
  }
  busy=false; watching=false; render();
}
async function loadStats(){
  try{const s=await api("/api/stats"); $("stats").textContent=
    `Snake has trained for ${s.train_steps} steps over ${s.games} games (Stockfish level ${s.sf_level}).`}catch(e){}
}
(async()=>{
  try{ const c=await api("/api/capabilities"); hasSF=!!c.stockfish; if(hasSF){$("sfbox").style.display="flex"} }catch(e){}
  await newGame("white"); loadStats();
})();
</script></body></html>"""


def _build_board(moves):
    board = chess.Board()
    for u in moves:
        board.push_uci(u)          # raises ValueError if illegal -> handled by caller
    return board


def _state(board):
    legal = {}
    for m in board.legal_moves:
        legal.setdefault(chess.square_name(m.from_square), []).append(chess.square_name(m.to_square))
    replay, san = chess.Board(), []
    for m in board.move_stack:
        san.append(replay.san(m)); replay.push(m)
    info = identify_game(board)
    return {"fen": board.fen(), "moves": [m.uci() for m in board.move_stack], "san": san,
            "turn": "white" if board.turn else "black", "legal": legal, "in_check": board.is_check(),
            "over": info["over"], "result": info["result"], "termination": info["termination"],
            "winner": info["winner"], "info": info}


def create_app(agent, stats_fn=None, stockfish_path=None):
    app = Flask(__name__)
    sf = StockfishService(stockfish_path) if stockfish_path and os.path.exists(stockfish_path) else None

    def bad(msg, code=400):
        return jsonify({"error": msg}), code

    def read_board():
        data = request.get_json(force=True, silent=True) or {}
        try:
            return _build_board(data.get("moves", [])), data
        except (ValueError, TypeError):
            return None, data

    @app.get("/")
    def index():
        return Response(INDEX_HTML, mimetype="text/html")

    @app.get("/api/capabilities")
    def capabilities():
        return jsonify(stockfish=sf is not None)

    @app.get("/health")
    def health():
        return jsonify(ok=True)

    @app.get("/api/stats")
    def stats():
        s = stats_fn() if stats_fn else dict(agent.meta)
        return jsonify({"train_steps": s.get("train_steps", agent.meta.get("train_steps", 0)),
                        "games": s.get("games", agent.meta.get("games", 0)),
                        "sf_level": s.get("sf_level", agent.meta.get("sf_level", 0))})

    @app.post("/api/state")
    def state():
        board, _ = read_board()
        return bad("invalid move list") if board is None else jsonify(_state(board))

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

    return app