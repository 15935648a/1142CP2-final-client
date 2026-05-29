#!/usr/bin/env python3
"""
Rokumoku Arena bot client.

Usage:
    ARENA_URL=http://... BOT_API_KEY=ra_bot_... ROOM_ID=... python bot.py

Optional env vars:
    BOT_SIMS         MCTS simulations per move  (default 150)
    BOT_BID_VALUE    Armageddon bid seconds      (default 30)
    BOT_BID_COLOR    Preferred color if bid wins (default black)
"""

import json
import logging
import os
import sys
import threading
import time
import uuid

import numpy as np
import requests
import torch

# Sandbox has 2 CPUs — cap threads to avoid context-switch overhead
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from connect6s.config import Config
from connect6s.config_large import ConfigLarge
from connect6s.game import GameState, BOARD_SIZE
from connect6s.heuristic import find_forced_move
from connect6s.heuristic_agent import HeuristicAgent
from connect6s.mcts import MCTS
from connect6s.network import build_net

# ── Environment ───────────────────────────────────────────────────────────────
ARENA_URL   = os.environ.get("ARENA_URL", "http://arena-api-server:8080").rstrip("/")
BOT_API_KEY = os.environ.get("BOT_API_KEY", "")
ROOM_ID     = os.environ.get("ROOM_ID", "")
BOT_HEURISTIC = bool(os.environ.get("BOT_HEURISTIC", ""))  # use alpha-beta instead of MCTS
BOT_SIMS      = 0 if BOT_HEURISTIC else int(os.environ.get("BOT_SIMS", "0"))  # heuristic always timed
BOT_COLOR     = os.environ.get("BOT_COLOR", "").lower()    # "black" or "white", empty = any
BID_VALUE   = float(os.environ.get("BOT_BID_VALUE", "30.0"))
BID_COLOR   = os.environ.get("BOT_BID_COLOR", "black")

HEARTBEAT_INTERVAL = 5.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bot")


# ── Board reconstruction ──────────────────────────────────────────────────────
def rebuild_from_moves(moves: list) -> GameState:
    """Replay server move_history to produce exact GameState."""
    state = GameState()
    for mv in moves:
        coords    = mv["coords"]   # [row, col]
        r, c      = coords[0], coords[1]
        is_strong = bool(mv.get("strong", False))
        state = state.make_move(r, c, is_strong)
    return state


# Board-char → signed int8 (black>0, white<0; |2|=strong). Matches GameState encoding.
_CHAR2CELL = {".": 0, "b": 1, "B": 2, "w": -1, "W": -2}
_DIRS4 = [(0, 1), (1, 0), (1, 1), (1, -1)]


def _decode_compact(board_compact: str, board_size) -> list:
    """Decode 'board_compact' string into 2D char board (template fallback)."""
    try:
        size = int(board_size)
    except (TypeError, ValueError):
        size = BOARD_SIZE
    if size <= 0:
        size = BOARD_SIZE
    text = str(board_compact or "")
    return [[text[r * size + c] if r * size + c < len(text) else "."
             for c in range(size)] for r in range(size)]


class SnapshotState:
    """Duck-typed GameState built from a live board snapshot.

    Fallback for when 'move_history' is absent (non-documented field).
    Implements only the interface used by find_forced_move, HeuristicAgent,
    and ArenaBot._pick_and_send — NOT a full game engine.
    """

    def __init__(self, board2d, current_player, move_count, strong_pieces):
        n = BOARD_SIZE
        self.board = np.zeros((n, n), dtype=np.int8)
        for r in range(min(n, len(board2d))):
            row = board2d[r]
            for c in range(min(n, len(row))):
                self.board[r, c] = _CHAR2CELL.get(row[c], 0)
        self.current_player = current_player          # +1 black, -1 white
        self.move_count     = move_count
        self.strong_pieces  = strong_pieces           # {1: black_cnt, -1: white_cnt}
        self.game_over      = self._detect_game_over()

    def _detect_game_over(self) -> bool:
        n = BOARD_SIZE
        for r in range(n):
            for c in range(n):
                v = int(self.board[r, c])
                if v == 0:
                    continue
                sign = 1 if v > 0 else -1
                for dr, dc in _DIRS4:
                    cnt = 1
                    rr, cc = r + dr, c + dc
                    while 0 <= rr < n and 0 <= cc < n and (int(self.board[rr, cc]) > 0) == (sign > 0) and self.board[rr, cc] != 0:
                        cnt += 1
                        if cnt >= 6:
                            return True
                        rr += dr
                        cc += dc
        return bool(not np.any(self.board == 0))

    def get_observation(self) -> np.ndarray:
        p = self.current_player
        obs = np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        obs[0] = (self.board ==  p).astype(np.float32)
        obs[1] = (self.board == 2 * p).astype(np.float32)
        obs[2] = (self.board == -p).astype(np.float32)
        obs[3] = (self.board == -2 * p).astype(np.float32)
        obs[4] = float(self.strong_pieces[p])
        obs[5] = float(self.strong_pieces[-p])
        return obs

    def get_legal_moves(self):
        p = self.current_player
        has_strong = self.strong_pieces[p] > 0
        moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cell = int(self.board[r, c])
                if cell == 0:
                    moves.append((r, c, False))
                    if has_strong:
                        moves.append((r, c, True))
                elif has_strong and abs(cell) == 1:
                    moves.append((r, c, True))
        return moves

    def index_to_action(self, idx):
        base = BOARD_SIZE * BOARD_SIZE
        if idx >= base:
            return (idx - base) // BOARD_SIZE, (idx - base) % BOARD_SIZE, True
        return idx // BOARD_SIZE, idx % BOARD_SIZE, False


def rebuild_from_board(room: dict, my_player_id) -> SnapshotState | None:
    """Build SnapshotState from the documented live board fields."""
    board2d = room.get("board")
    if not isinstance(board2d, list):
        board2d = _decode_compact(room.get("board_compact"), room.get("board_size"))
    if not board2d:
        return None

    # Player to move (GameState sign): color int 1=black→+1, 2=white→-1
    pc   = room.get("player_colors") or {}
    turn = room.get("turn_info") or {}
    turn_pid = turn.get("player_id", my_player_id)
    turn_color = pc.get(str(turn_pid))
    cur = 1 if turn_color == 1 else (-1 if turn_color == 2 else 1)

    # strong_pieces_available keyed by color int: "1"=black, "2"=white
    avail = room.get("strong_pieces_available") or {}
    strong = {1:  int(avail.get("1", 0) or 0),
              -1: int(avail.get("2", 0) or 0)}

    move_count = int(room.get("move_count", 0) or 0)
    return SnapshotState(board2d, cur, move_count, strong)


# ── SSE stream parser ─────────────────────────────────────────────────────────
def iter_sse(response):
    """Yield (event_name, data_dict) from an SSE response."""
    name  = "message"
    lines: list[str] = []
    for raw in response.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        if raw.startswith("event:"):
            name = raw[6:].strip()
        elif raw.startswith("data:"):
            lines.append(raw[5:].strip())
        elif raw == "":
            if lines:
                try:
                    yield name, json.loads("\n".join(lines))
                except json.JSONDecodeError as e:
                    log.warning(f"SSE JSON error: {e}")
            name  = "message"
            lines = []


# ── Bot ───────────────────────────────────────────────────────────────────────
class ArenaBot:
    def __init__(self, mcts: MCTS):
        self.mcts        = mcts
        self.session     = requests.Session()
        self.presence_id = uuid.uuid4().hex
        self.username: str | None = None
        self.my_player_id: int | None = None  # 1 or 2
        self.game_state: GameState | None = None
        # Dedup keys to avoid double-submitting
        self._last_move_key: str | None = None
        self._last_ready_key: str | None = None
        self._last_bid_key: str | None = None
        self._logged_game_over: set = set()
        self._game_color: dict = {}   # game_id -> my color int (1=black,2=white)
        self._warned_board_fallback = False
        self._stop = threading.Event()

    # ── Low-level HTTP ────────────────────────────────────────────────────────
    def _post(self, path: str, body=None, params=None):
        r = self.session.post(f"{ARENA_URL}{path}", json=body, params=params)
        if not r.ok:
            log.warning(f"POST {path} → {r.status_code}: {r.text[:200]}")
        return r

    # ── Arena actions ─────────────────────────────────────────────────────────
    def login(self):
        r = self._post("/api/auth/login",
                       {"provider": "api_key", "api_key": BOT_API_KEY})
        r.raise_for_status()
        self.username = r.json()["data"]["user"]["username"]
        log.info(f"Logged in as {self.username}")

    def _heartbeat_loop(self):
        while not self._stop.wait(HEARTBEAT_INTERVAL):
            try:
                self._post(f"/api/rooms/{ROOM_ID}/heartbeat",
                           params={"presence_id": self.presence_id})
            except Exception as e:
                log.warning(f"Heartbeat error: {e}")

    def leave(self):
        self._stop.set()
        try:
            self._post(f"/api/rooms/{ROOM_ID}/seat", {"seat": "spectator"})
        except Exception:
            pass
        try:
            r = self.session.delete(
                f"{ARENA_URL}/api/rooms/{ROOM_ID}/presence/{self.presence_id}")
            if not r.ok:
                log.warning(f"DELETE presence → {r.status_code}: {r.text[:200]}")
        except Exception:
            pass
        log.info("Left room")

    def _seat(self, color: str):
        log.info(f"Requesting seat {color}")
        self._post(f"/api/rooms/{ROOM_ID}/seat", {"seat": color})

    def _ready(self):
        self._post(f"/api/rooms/{ROOM_ID}/ready")
        log.info("Sent ready")

    def _move(self, row: int, col: int, strong: bool):
        self._post(f"/api/rooms/{ROOM_ID}/move",
                   {"row": row, "col": col, "strong": strong})
        log.info(f"Moved ({row},{col}) strong={strong}")

    def _bid(self, value: float, color: str):
        self._post(f"/api/rooms/{ROOM_ID}/bid",
                   {"bid": value, "color": color})
        log.info(f"Bid value={value} color={color}")

    # ── MCTS decision ─────────────────────────────────────────────────────────
    def _time_budget(self, time_left: float, move_no: int = 0) -> float:
        """Spend more time in midgame/endgame, less in opening."""
        fraction = min(0.15 + move_no * 0.005, 0.30)  # 15% → 30% over 30 moves
        usable   = max(time_left - 10.0, 0.0)
        budget   = min(usable * fraction, 20.0)
        return max(budget, 1.0)

    def _pick_and_send(self, move_key: str, time_left: float = 0.0, move_no: int = 0):
        if move_key == self._last_move_key:
            return
        if self.game_state is None or self.game_state.game_over:
            log.error("Cannot move: no valid game state")
            return
        self._last_move_key = move_key
        self.mcts.clear_cache()

        try:
            forced = find_forced_move(self.game_state)
            if forced:
                row, col, s = forced
                log.info(f"Forced move: ({row},{col}) strong={s}")
                self._move(row, col, s)
                return

            t0 = time.time()
            if BOT_SIMS > 0:
                # fixed-sims mode (override from env)
                probs = self.mcts.get_action_probs(
                    self.game_state, temperature=0.0, add_noise=False
                )
            elif time_left > 0:
                budget = self._time_budget(time_left, move_no)
                if BOT_HEURISTIC:
                    budget = min(budget, 10.0)  # alpha-beta: deeper search worth the time
                log.info(f"Thinking for {budget:.1f}s (bank={time_left:.1f}s move={move_no})")
                probs = self.mcts.get_action_probs_timed(
                    self.game_state, seconds=budget, add_noise=False
                )
            elif BOT_HEURISTIC:
                # no time info but heuristic mode — use default 3s timed search
                budget = 3.0
                log.info(f"Thinking for {budget:.1f}s (no bank, heuristic default)")
                probs = self.mcts.get_action_probs_timed(
                    self.game_state, seconds=budget, add_noise=False
                )
            else:
                # no time info (sandbox inject) — fallback fixed sims
                probs = self.mcts.get_action_probs(
                    self.game_state, temperature=0.0, add_noise=False
                )
            log.info(f"MCTS done in {time.time() - t0:.2f}s")
            idx         = int(np.argmax(probs))
            row, col, s = self.game_state.index_to_action(idx)
            self._move(row, col, s)

        except Exception as e:
            log.error(f"MCTS failed: {e!r} — falling back to greedy move")
            try:
                valid = self.game_state.get_valid_moves()
                if valid:
                    idx = valid[len(valid) // 2]
                    row, col, s = self.game_state.index_to_action(idx)
                    log.info(f"Greedy fallback: ({row},{col}) strong={s}")
                    self._move(row, col, s)
            except Exception as e2:
                log.error(f"Fallback also failed: {e2!r}")

    # ── Template-aligned field helpers ─────────────────────────────────────────
    @staticmethod
    def _seated_username(room: dict, seat: str):
        seat_key = "1" if seat == "black" else "2"
        return (room.get("seated_usernames") or {}).get(seat_key)

    def _find_my_player_id(self, room: dict):
        for pid, uname in (room.get("player_usernames") or {}).items():
            if uname == self.username:
                return int(pid)
        if self._seated_username(room, "black") == self.username:
            return 1
        if self._seated_username(room, "white") == self.username:
            return 2
        return None

    def _current_seat(self, room: dict):
        if self._seated_username(room, "black") == self.username:
            return "black"
        if self._seated_username(room, "white") == self.username:
            return "white"
        return None

    # ── Snapshot handler ──────────────────────────────────────────────────────
    def _handle_snapshot(self, room: dict):
        self.my_player_id = self._find_my_player_id(room)

        # Take an empty seat if not seated yet
        if self._current_seat(room) not in ("black", "white"):
            order = ["white", "black"] if BOT_COLOR == "white" else ["black", "white"]
            for color in order:
                if self._seated_username(room, color) is None:
                    self._seat(color)
                    break
            return

        if self.my_player_id is None:
            return

        pid = str(self.my_player_id)

        # Track my color for this game while room is internally consistent
        game_id = room.get("current_game_id", "")
        my_color = (room.get("player_colors") or {}).get(pid)
        if my_color is not None and game_id:
            self._game_color[game_id] = my_color

        # Ready phase
        if room.get("awaiting_player_confirmation"):
            game_id   = room.get("current_game_id", "")
            ready_key = f"{game_id}:{self.my_player_id}"
            if ready_key != self._last_ready_key:
                ready_info = room.get("ready_info") or {}
                confirmed  = (ready_info.get("confirmed") or {})
                if not confirmed.get(pid):
                    self._last_ready_key = ready_key
                    self._ready()

        # Rebuild game state. Primary: move_history (exact). Fallback: live board.
        moves = room.get("move_history") or []
        try:
            if moves:
                self.game_state = rebuild_from_moves(moves)
            else:
                self.game_state = rebuild_from_board(room, self.my_player_id)
                if self.game_state is not None and not self._warned_board_fallback:
                    self._warned_board_fallback = True
                    log.warning("move_history absent — using live-board fallback")
        except Exception as e:
            log.error(f"State rebuild failed: {e}")
            self.game_state = None

        # Log game result when game ends (once per game)
        if self.game_state and self.game_state.game_over:
            if game_id not in self._logged_game_over:
                self._logged_game_over.add(game_id)
                winner = room.get("winner") or room.get("result") or "?"
                # color captured during play (stale-proof). int: 1=black, 2=white
                color_int = self._game_color.get(game_id)
                color_name = {1: "black", 2: "white"}.get(color_int, "?")
                i_am_black = (color_int == 1)
                log.info(f"[GAME OVER] winner={winner} "
                         f"I_am={color_name}(p{self.my_player_id}) moves={len(moves)}")
                for i, m in enumerate(moves):
                    # black moves at even indices (black moves first)
                    who = "ME  " if (i % 2 == 0) == i_am_black else "OPP "
                    coords = m.get("coords") or []
                    r_str = coords[0] if len(coords) > 0 else "?"
                    c_str = coords[1] if len(coords) > 1 else "?"
                    log.info(f"  [HIST] move={i} {who} ({r_str},{c_str}) strong={m.get('strong',False)}")

        # Move phase
        if room.get("awaiting_move") and self.game_state:
            turn = room.get("turn_info") or {}
            if turn.get("player_id") == self.my_player_id:
                game_id   = room.get("current_game_id", "")
                move_no   = room.get("move_count", 0)
                move_key  = f"{game_id}:{move_no}"
                time_left = float((room.get("player_time_left") or {}).get(pid, 0.0) or 0.0)
                self._pick_and_send(move_key, time_left, move_no)

        # Bid phase
        if room.get("awaiting_bid"):
            bid_req = room.get("bid_request") or {}
            if bid_req.get("viewer_player_id") == self.my_player_id:
                if not bid_req.get("viewer_submitted"):
                    game_id  = room.get("current_game_id", "")
                    deadline = bid_req.get("bid_deadline", "")
                    bid_key  = f"{game_id}:{deadline}"
                    if bid_key != self._last_bid_key:
                        self._last_bid_key = bid_key
                        my_info  = (bid_req.get("players") or {}).get(str(self.my_player_id), {})
                        max_bid  = float(my_info.get("max_bid") or 120.0)
                        safe_bid = min(BID_VALUE, max_bid)
                        self._bid(safe_bid, BID_COLOR)

    # ── SSE stream loop ───────────────────────────────────────────────────────
    def _stream_loop(self):
        url    = f"{ARENA_URL}/api/rooms/{ROOM_ID}/stream"
        params = {"presence_id": self.presence_id}
        with self.session.get(url, params=params, stream=True,
                              timeout=(30, None)) as resp:
            resp.raise_for_status()
            log.info("SSE stream opened")
            for event, data in iter_sse(resp):
                if event == "sync":
                    continue
                self._handle_snapshot(data)

    # ── Main entry ────────────────────────────────────────────────────────────
    def run(self):
        self.login()
        log.info(f"Joining room {ROOM_ID!r}  presence={self.presence_id}")
        threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="heartbeat"
        ).start()

        try:
            while True:
                try:
                    self._stream_loop()
                except Exception as e:
                    log.error(f"Stream error: {e!r}, retry in 2s...")
                    time.sleep(2)
        finally:
            self.leave()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    missing = [v for v in ("BOT_API_KEY", "ROOM_ID") if not os.environ.get(v)]
    if missing:
        sys.exit(f"Missing env vars: {', '.join(missing)}")

    cfg = ConfigLarge() if os.getenv("BOT_LARGE") else Config()

    if BOT_HEURISTIC:
        log.info("Mode: heuristic alpha-beta (BOT_HEURISTIC=1)")
        agent = HeuristicAgent(depth=4)
    else:
        device = torch.device("cpu")
        net    = build_net(cfg.NUM_RES_BLOCKS, cfg.NUM_CHANNELS, device=device)
        if os.path.exists(cfg.BEST_MODEL_PATH):
            ckpt = torch.load(cfg.BEST_MODEL_PATH, map_location=device)
            sd   = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            net.load_state_dict(sd)
            log.info(f"Loaded model: {cfg.BEST_MODEL_PATH}")
        else:
            log.warning("No trained model — using random weights!")
        net.eval()
        agent = MCTS(
            net,
            num_simulations  = BOT_SIMS if BOT_SIMS > 0 else 800,
            c_puct           = cfg.C_PUCT,
            dirichlet_alpha  = cfg.DIRICHLET_ALPHA,
            dirichlet_eps    = 0.0,
            leaf_batch_size  = cfg.LEAF_BATCH_SIZE,
            heuristic_weight = 1.5,
        )
        log.info("Mode: AlphaZero MCTS")

    ArenaBot(agent).run()


if __name__ == "__main__":
    main()
