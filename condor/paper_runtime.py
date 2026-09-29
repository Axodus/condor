"""Operator CLI for PAPER_MAINNET_DATA."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from condor.live_execution import make_paper_supervisor
from condor.orderflow_orchestrator import SYMBOLS, STRATEGIES, STRATEGY_REVISION
from condor.paper_execution import PAPER_FILL_MODEL_VERSION, PaperExecutionAdapter


async def run_preflight(supervisor: Any) -> dict[str, Any]:
    orchestrator = supervisor.orchestrator
    runtime = orchestrator.runtime
    root = orchestrator.config.artifact_root
    writable = False
    free_bytes = 0
    storage_error = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / f".durability_probe_{int(time.time() * 1000)}"
        probe.write_text("DURABILITY_PROBE_OK\\n", encoding="utf-8")
        probe.flush if False else None
        writable = probe.read_text(encoding="utf-8") == "DURABILITY_PROBE_OK\\n"
        probe.unlink()
        import shutil
        free_bytes = shutil.disk_usage(root).free
    except Exception as exc:
        storage_error = type(exc).__name__

    jev_health: dict[str, Any] = {}
    jev_probe: dict[str, Any] = {}
    try:
        jev = orchestrator.jev_client
        if jev is not None:
            jev_health = await asyncio.wait_for(jev.health(), timeout=5.0)
            probe_fn = getattr(jev, "decision_probe", None)
            jev_probe = await asyncio.wait_for(probe_fn(), timeout=5.0) if probe_fn else {"status": "PASS", "execution": "NONE"}
    except Exception as exc:
        jev_health = {"status": "ERROR", "error": type(exc).__name__}

    feed = await runtime.preflight(SYMBOLS)
    terminal = await runtime.terminal_state()
    feed_status = feed.get("feed", {}).get("status") if "feed" in feed else feed.get("status")
    ready = (
        writable
        and free_bytes > 1_000_000_000
        and jev_health.get("status") == "HEALTHY"
        and jev_probe.get("status") == "PASS"
        and feed_status == "HEALTHY"
        and not terminal.get("open_orders")
        and not terminal.get("positions")
    )
    return {
        "status": "READY" if ready else "BLOCKED",
        "execution_environment": "PAPER_MAINNET_DATA",
        "market_data_environment": "MAINNET",
        "remote_execution_disabled": True,
        "paper_fill_model_version": PAPER_FILL_MODEL_VERSION,
        "strategy_revision": STRATEGY_REVISION,
        "symbols": list(SYMBOLS),
        "strategies": list(STRATEGIES),
        "cells": len(orchestrator.cells),
        "storage": {"root": str(root), "writable": writable, "free_bytes": free_bytes, "error": storage_error},
        "jev": {"health": jev_health, "decision_probe": jev_probe},
        "market_feed": feed,
        "baseline": {"open_orders": len(terminal.get("open_orders", [])), "positions": len(terminal.get("positions", []))},
        "mainnet_mutations": 0,
        "real_capital": 0,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def reconcile(root: Path, run_id: str | None) -> dict[str, Any]:
    target = root / run_id if run_id else root
    orders = read_jsonl(target / "orders.jsonl")
    fills = read_jsonl(target / "fills.jsonl")
    positions = read_jsonl(target / "positions.jsonl")
    active_orders = [o for o in orders if o.get("status") in {"SUBMITTED", "PARTIALLY_FILLED"}]
    return {
        "run_id": run_id or target.name,
        "execution_environment": "PAPER_MAINNET_DATA",
        "orders": len(orders),
        "fills": len(fills),
        "positions": positions,
        "open_orders": active_orders,
        "terminal_clean": not active_orders and all(p.get("amount") in {None, "0", "0.0"} for p in positions[-12:]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trinity PAPER_MAINNET_DATA operator CLI")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--reconcile", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--duration", type=float, default=86400.0)
    parser.add_argument("--artifact-root")
    args = parser.parse_args()
    root = Path(args.artifact_root) if args.artifact_root else None
    supervisor = make_paper_supervisor(artifact_root=root, duration_seconds=args.duration)

    if args.status:
        print(json.dumps(supervisor.status(), indent=2))
        return
    if args.reconcile:
        print(json.dumps(reconcile(root or supervisor.orchestrator.config.artifact_root, args.run_id), indent=2))
        return
    if args.stop:
        state = supervisor.status()
        pid = state.get("pid")
        if pid and pid != os.getpid():
            os.kill(pid, signal.SIGTERM)
            print(json.dumps({"status": "STOPPING", "pid": pid}))
        else:
            print(json.dumps({"status": "NO_RUNNING_PID"}))
        return

    async def execute() -> None:
        try:
            if args.preflight:
                print(json.dumps(await run_preflight(supervisor), indent=2))
            elif args.probe:
                adapter = PaperExecutionAdapter(supervisor.orchestrator.runtime, supervisor.orchestrator.sink)
                print(json.dumps(await adapter.bounded_dispatch_probe(), indent=2))
            elif args.run:
                await supervisor.run_forever()
            else:
                parser.print_help()
        finally:
            await supervisor.orchestrator.runtime.feed.close()
            if supervisor.orchestrator.jev_client:
                await supervisor.orchestrator.jev_client.close()

    asyncio.run(execute())


if __name__ == "__main__":
    main()
