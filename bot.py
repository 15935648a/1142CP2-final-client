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
from connect6s.game import GameState
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
                    budget = min(budget, 5.0)  # alpha-beta: >5s gives diminishing returns
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

    # ── Snapshot handler ──────────────────────────────────────────────────────
    def _handle_snapshot(self, room: dict):
        # Determine my seat
        su = room.get("seated_usernames", {})
        if su.get("1") == self.username:
            self.my_player_id = 1
        elif su.get("2") == self.username:
            self.my_player_id = 2
        else:
            self.my_player_id = None

        # Take an empty seat if available (black=P1=key"1", white=P2=key"2")
        if self.my_player_id is None:
            order = ["black", "white"]
            if BOT_COLOR == "white":
                order = ["white", "black"]
            seat_key = {"black": "1", "white": "2"}
            for color in order:
                if su.get(seat_key[color]) is None:
                    self._seat(color)
                    break
            return

        pid = str(self.my_player_id)

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

        # Rebuild game state from authoritative move history
        moves = room.get("move_history") or []
        try:
            self.game_state = rebuild_from_moves(moves)
        except Exception as e:
            log.error(f"State rebuild failed: {e}")
            self.game_state = None

        # Log game result when game ends
        if self.game_state and self.game_state.game_over:
            winner = room.get("winner") or room.get("result") or "?"
            my_color = "black" if self.my_player_id == 1 else "white"
            log.info(f"[GAME OVER] winner={winner} I_am={my_color}(p{self.my_player_id}) moves={len(moves)}")

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
