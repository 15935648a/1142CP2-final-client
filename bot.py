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

import numpy as np
import requests
import torch

# Sandbox has 2 CPUs — cap threads to avoid context-switch overhead
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from connect6s.config import Config
from connect6s.game import GameState
from connect6s.mcts import MCTS
from connect6s.network import build_net

# ── Environment ───────────────────────────────────────────────────────────────
ARENA_URL   = os.environ.get("ARENA_URL", "http://arena-api-server:8080").rstrip("/")
BOT_API_KEY = os.environ.get("BOT_API_KEY", "")
ROOM_ID     = os.environ.get("ROOM_ID", "")
BOT_SIMS    = int(os.environ.get("BOT_SIMS", "150"))
BID_VALUE   = float(os.environ.get("BOT_BID_VALUE", "30.0"))
BID_COLOR   = os.environ.get("BOT_BID_COLOR", "black")

HEARTBEAT_INTERVAL = 10.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bot")


# ── Board reconstruction ──────────────────────────────────────────────────────
def rebuild_from_moves(moves: list) -> GameState:
    """Replay server move list to produce exact GameState."""
    state = GameState()
    for mv in moves:
        r         = mv["position"]["row"]
        c         = mv["position"]["col"]
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


# ── Sentinel exceptions ───────────────────────────────────────────────────────
class _Reconnect(Exception):
    pass

class _SessionExpired(Exception):
    pass


# ── Bot ───────────────────────────────────────────────────────────────────────
class ArenaBot:
    def __init__(self, mcts: MCTS):
        self.mcts        = mcts
        self.session     = requests.Session()
        self.presence_id: str | None = None
        self.my_seat: str | None = None   # "P1" or "P2"
        self.game_state: GameState | None = None
        self.permissions: dict = {}
        self._stop = threading.Event()

    # ── Low-level HTTP ────────────────────────────────────────────────────────
    def _post(self, path: str, body=None):
        r = self.session.post(f"{ARENA_URL}{path}", json=body)
        if not r.ok:
            log.warning(f"POST {path} → {r.status_code}: {r.text[:200]}")
        return r

    def _delete(self, path: str):
        try:
            self.session.delete(f"{ARENA_URL}{path}")
        except Exception:
            pass

    # ── Arena actions ─────────────────────────────────────────────────────────
    def login(self):
        self._post("/api/auth/login", {"api_key": BOT_API_KEY}).raise_for_status()
        log.info("Logged in")

    def join_room(self):
        r = self._post(f"/api/rooms/{ROOM_ID}/join")
        r.raise_for_status()
        self.presence_id = r.json()["data"]["presence_id"]
        log.info(f"Joined room {ROOM_ID!r}  presence={self.presence_id}")

    def _heartbeat_loop(self):
        while not self._stop.wait(HEARTBEAT_INTERVAL):
            try:
                self._post(
                    f"/api/rooms/{ROOM_ID}/presence/{self.presence_id}/heartbeat"
                )
            except Exception as e:
                log.warning(f"Heartbeat error: {e}")

    def leave(self):
        self._stop.set()
        if self.presence_id:
            self._delete(f"/api/rooms/{ROOM_ID}/presence/{self.presence_id}")
        log.info("Left room")

    def _seat(self, target: str):
        log.info(f"Requesting seat {target}")
        self._post(f"/api/rooms/{ROOM_ID}/seat", {"seat_target": target})

    def _ready(self, state: bool = True):
        self._post(f"/api/rooms/{ROOM_ID}/ready", {"ready_state": state})
        log.info(f"Ready → {state}")

    def _move(self, row: int, col: int, strong: bool):
        self._post(f"/api/rooms/{ROOM_ID}/move",
                   {"position": {"row": row, "col": col}, "strong": strong})
        log.info(f"Moved ({row},{col}) strong={strong}")

    def _bid(self, value: float, color: str):
        self._post(f"/api/rooms/{ROOM_ID}/bid",
                   {"bid": {"value": value, "color": color}})
        log.info(f"Bid value={value} color={color}")

    # ── MCTS decision ─────────────────────────────────────────────────────────
    def _pick_and_send(self):
        if self.game_state is None or self.game_state.game_over:
            log.error("Cannot move: no valid game state")
            return
        self.mcts.clear_cache()
        t0    = time.time()
        probs = self.mcts.get_action_probs(
            self.game_state, temperature=0.0, add_noise=False
        )
        log.info(f"MCTS done in {time.time() - t0:.2f}s")
        idx         = int(np.argmax(probs))
        row, col, s = self.game_state.index_to_action(idx)
        self._move(row, col, s)

    # ── Turn check ────────────────────────────────────────────────────────────
    def _my_turn(self, turn: dict) -> bool:
        return bool(self.my_seat and turn.get("player_id") == self.my_seat)

    # ── Event handlers ────────────────────────────────────────────────────────
    def _on_snapshot(self, data: dict):
        room    = data.get("room", {})
        viewer  = room.get("viewer", {})
        self.my_seat     = viewer.get("seat")   # "P1" / "P2" / "spectator" / null
        self.permissions = room.get("permissions", {})
        log.info(f"Snapshot: seat={self.my_seat}")

        # Rebuild board
        display = data.get("display_board")
        if display:
            moves = display.get("moves", [])
            self.game_state = rebuild_from_moves(moves)
            log.info(f"State rebuilt from {len(moves)} moves")

        p = self.permissions
        # Sit if unseated
        if self.my_seat not in ("P1", "P2"):
            if p.get("can_sit_P1"):
                self._seat("P1")
            elif p.get("can_sit_P2"):
                self._seat("P2")

        # Act on current game state from snapshot (handles reconnect mid-game)
        match = data.get("match")
        if match:
            gstate = match.get("current_game_state") or {}
            phase  = gstate.get("phase", "")
            turn   = gstate.get("turn")
            if phase == "ready" and p.get("can_ready"):
                self._ready(True)
            elif phase == "armageddon_bid" and p.get("can_bid"):
                self._bid(BID_VALUE, BID_COLOR)
            elif phase == "playing" and turn and self._my_turn(turn):
                self._pick_and_send()

    def _on_room_state(self, data: dict):
        self.permissions = data.get("permissions", self.permissions)
        p = self.permissions
        if self.my_seat not in ("P1", "P2"):
            if p.get("can_sit_P1"):
                self._seat("P1")
            elif p.get("can_sit_P2"):
                self._seat("P2")

    def _on_game_phase(self, data: dict):
        phase = data.get("phase", "")
        turn  = data.get("turn")
        log.info(f"Game phase → {phase}")

        if phase == "starting":
            # Server rebound display_board; reconnect to get fresh snapshot
            raise _Reconnect("game starting")

        if phase == "ready" and self.permissions.get("can_ready"):
            self._ready(True)

        elif phase == "armageddon_bid" and self.permissions.get("can_bid"):
            self._bid(BID_VALUE, BID_COLOR)

        elif phase == "playing" and turn and self._my_turn(turn):
            self._pick_and_send()

    def _on_move(self, data: dict):
        mv = data.get("move", {})
        if self.game_state and mv:
            try:
                r         = mv["position"]["row"]
                c         = mv["position"]["col"]
                is_strong = bool(mv.get("strong", False))
                self.game_state = self.game_state.make_move(r, c, is_strong)
            except Exception as e:
                log.error(f"Move delta failed: {e} — state desynced, will resync on reconnect")
                self.game_state = None

        next_turn = data.get("next_turn")
        if next_turn and self._my_turn(next_turn):
            self._pick_and_send()

    def _on_server_goodbye(self, data: dict):
        reason = data.get("reason", "unknown")
        log.warning(f"server_goodbye: {reason}")
        if reason in ("heartbeat_timeout", "session_expired"):
            raise _SessionExpired(reason)
        raise _Reconnect(reason)

    # ── Dispatch ──────────────────────────────────────────────────────────────
    _DISPATCH = {
        "snapshot":       _on_snapshot,
        "room_state":     _on_room_state,
        "game_phase":     _on_game_phase,
        "move":           _on_move,
        "server_goodbye": _on_server_goodbye,
    }

    def _dispatch(self, event: str, data: dict):
        handler = self._DISPATCH.get(event)
        if handler:
            handler(self, data)
        # Ignored: subscription_ack, sync, clock, ready_update,
        #          bid_update, match_score, match_finished, match_error,
        #          seat_update, chat

    # ── SSE stream loop ───────────────────────────────────────────────────────
    def _stream_loop(self):
        url    = f"{ARENA_URL}/api/rooms/{ROOM_ID}/events"
        params = {"presence_id": self.presence_id}
        with self.session.get(url, params=params, stream=True,
                              timeout=(30, None)) as resp:
            resp.raise_for_status()
            log.info("SSE stream opened")
            for event, data in iter_sse(resp):
                self._dispatch(event, data)

    # ── Main entry ────────────────────────────────────────────────────────────
    def run(self):
        self.login()
        self.join_room()
        threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="heartbeat"
        ).start()

        try:
            while True:
                try:
                    self._stream_loop()
                except _SessionExpired as e:
                    log.info(f"Re-logging in ({e})...")
                    self.login()
                    self.join_room()
                except _Reconnect as e:
                    log.info(f"Reconnecting ({e})...")
                    time.sleep(0.5)
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

    cfg    = Config()
    device = torch.device("cpu")   # sandbox: no GPU
    net    = build_net(cfg.NUM_RES_BLOCKS, cfg.NUM_CHANNELS, device=device)

    if os.path.exists(cfg.BEST_MODEL_PATH):
        ckpt = torch.load(cfg.BEST_MODEL_PATH, map_location=device)
        sd   = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        net.load_state_dict(sd)
        log.info(f"Loaded model: {cfg.BEST_MODEL_PATH}")
    else:
        log.warning("No trained model — using random weights!")

    net.eval()
    mcts = MCTS(
        net,
        num_simulations  = BOT_SIMS,
        c_puct           = cfg.C_PUCT,
        dirichlet_alpha  = cfg.DIRICHLET_ALPHA,
        dirichlet_eps    = 0.0,        # no exploration noise in tournament
        leaf_batch_size  = cfg.LEAF_BATCH_SIZE,
    )

    ArenaBot(mcts).run()


if __name__ == "__main__":
    main()
