"""Strict ExecutionIntent dispatch and persistent supervisor primitives.

The module is deliberately transport agnostic: Hummingbot remains the only
exchange transport.  It wraps the existing Hummingbot runtime interface and
adds the validation, durable state, and startup gates required by the live
Testnet runner.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Awaitable, Callable

from condor.orderflow_orchestrator import (
    CONNECTOR,
    SYMBOLS,
    STRATEGY_REVISION,
    ExecutionIntent,
    JsonlSink,
    OrchestratorBlocked,
    OrderFlowOrchestrator,
    RunnerState,
)


class HummingbotExecutionAdapter:
    """Dispatch validated intents through the already configured Hummingbot runtime."""

    def __init__(self, runtime: Any, sink: JsonlSink | None = None) -> None:
        self.runtime = runtime
        self.sink = sink

    @staticmethod
    def validate_intent(intent: ExecutionIntent, valid_cells: set[str]) -> None:
        intent.validate()
        if intent.cell_id not in valid_cells:
            raise OrchestratorBlocked("UNKNOWN_CELL_ID")
        if intent.target_environment != "TESTNET":
            raise OrchestratorBlocked("NON_TESTNET_DISPATCH_REJECTED")
        if intent.strategy_revision != STRATEGY_REVISION:
            raise OrchestratorBlocked("STRATEGY_REVISION_MISMATCH")

    async def dispatch(
        self, intent: ExecutionIntent, valid_cells: set[str]
    ) -> dict[str, Any]:
        self.validate_intent(intent, valid_cells)
        result = await self.runtime.dispatch_intent(intent)
        record = {
            **asdict(intent),
            "event_type": "ORDER_DISPATCHED",
            "provenance": "HUMMINGBOT_EXECUTION_STATE",
            "observed_at": time.time(),
            "response": result,
        }
        if self.sink is not None:
            await self.sink.write("execution_events", record)
        return result

    async def bounded_dispatch_probe(
        self,
        *,
        symbol: str = "BTCUSDT",
        quantity: str = "0.001",
    ) -> dict[str, Any]:
        """Execute one bounded Testnet dispatch probe and ensure clean terminal state."""
        if symbol not in SYMBOLS:
            raise OrchestratorBlocked(f"UNAPPROVED_SYMBOL:{symbol}")
        cell_id = f"{symbol}:orderflow.momentum.aggression:{STRATEGY_REVISION}"
        correlation_id = f"probe-{int(time.time()*1000)}"
        intent = ExecutionIntent(
            run_id="probe-run",
            cell_id=cell_id,
            correlation_id=correlation_id,
            jev_decision_id="probe-jev-dec",
            symbol=symbol,
            action="ENTRY",
            side="BUY",
            strategy_id="orderflow.momentum.aggression",
            strategy_revision=STRATEGY_REVISION,
            quantity=quantity,
            order_type="MARKET",
            price=None,
            execution_constraints={"probe": True},
            target_environment="TESTNET",
        )
        dispatch_result = await self.dispatch(intent, {cell_id})
        # Allow brief settlement, then cancel all and flatten any residual probe position
        await asyncio.sleep(0.5)
        await self.runtime.cancel_all("probe-run")
        await self.runtime.flatten("probe-run")
        # Poll for terminal settlement (up to 5 attempts)
        open_orders: list[Any] = []
        positions: list[Any] = []
        for _ in range(5):
            await asyncio.sleep(0.5)
            terminal = await self.runtime.terminal_state()
            open_orders = [
                o for o in terminal.get("open_orders", [])
                if str(o.get("status", "")).upper() in ("NEW", "PARTIALLY_FILLED", "SUBMITTED")
            ]
            positions = [
                p for p in terminal.get("positions", [])
                if abs(float(p.get("amount") or p.get("position_amount") or 0.0)) > 1e-6
            ]
            if not open_orders and not positions:
                break
        if open_orders or positions:
            raise OrchestratorBlocked("PROBE_TERMINAL_RECONCILIATION_FAILED")
        return {
            "status": "PASS",
            "symbol": symbol,
            "cell_id": cell_id,
            "correlation_id": correlation_id,
            "dispatch_response": dispatch_result,
            "terminal_open_orders": len(open_orders),
            "terminal_positions": len(positions),
            "target_environment": "TESTNET",
        }


class PersistentExecutionSupervisor:
    """Durable supervisor for arming and running one immutable experiment."""

    def __init__(
        self,
        orchestrator: OrderFlowOrchestrator,
        *,
        state_path: Path | None = None,
        service_identity: str = "condor-orderflow-testnet",
    ) -> None:
        self.orchestrator = orchestrator
        self.state_path = state_path or (
            orchestrator.artifact_root / "supervisor_state.json"
        )
        self.service_identity = service_identity

    def _write_state(self, **fields: Any) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        current: dict[str, Any] = {}
        if self.state_path.exists():
            try:
                current = json.loads(self.state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                current = {}
        current.update(fields)
        current.update(
            {
                "run_id": self.orchestrator.run_id,
                "service_identity": self.service_identity,
                "pid": os.getpid(),
                "updated_at": time.time(),
                "state": self.orchestrator.state.value,
            }
        )
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.state_path)

    async def arm(self) -> dict[str, Any]:
        if self.orchestrator.state not in (RunnerState.PRE_RUN, RunnerState.BLOCKED):
            raise OrchestratorBlocked("RUN_ALREADY_ARMED_OR_STARTED")
        preflight = await self.orchestrator.startup_gate(start_timer=False)
        self.orchestrator.state = RunnerState.ARMED
        self._write_state(
            state="ARMED",
            cells_armed=len(self.orchestrator.cells),
            baseline_open_orders=len(preflight.get("open_orders", [])),
            baseline_positions=len(preflight.get("positions", [])),
            timer_started=False,
        )
        return preflight

    async def start(self) -> None:
        if self.orchestrator.state != RunnerState.ARMED:
            raise OrchestratorBlocked("TWELVE_CELLS_NOT_ARMED")
        await self.orchestrator.start_24h_timer()
        self._write_state(
            state="RUNNING",
            timer_started=True,
            run_started_at=self.orchestrator.started_at,
            run_deadline=self.orchestrator.deadline,
        )

    def status(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"state": "PRE_RUN", "run_id": self.orchestrator.run_id}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    async def run_forever(self) -> None:
        """Run the supervised 24-hour loop until deadline or termination signal."""
        await self.arm()
        await self.start()
        self.orchestrator._tasks = [
            asyncio.create_task(self.orchestrator._heartbeat_loop()),
            asyncio.create_task(self.orchestrator._market_loop()),
        ]
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.orchestrator._stop.set)
            except (NotImplementedError, RuntimeError):
                pass
        try:
            await self.orchestrator._stop.wait()
        finally:
            reason = (
                "HARD_STOP_24H"
                if self.orchestrator.deadline
                and self.orchestrator.clock() >= self.orchestrator.deadline
                else "STOP_REQUESTED"
            )
            await self.orchestrator.stop(reason=reason)

    async def trigger_kill_switch(self, reason: str) -> None:
        """Emergency kill-switch: block entries, cancel orders, flatten positions."""
        self.orchestrator.state = RunnerState.FAILED
        await self.orchestrator.sink.write(
            "incidents",
            {
                "run_id": self.orchestrator.run_id,
                "type": "KILL_SWITCH_TRIGGERED",
                "reason": reason,
                "triggered_at": time.time(),
            },
        )
        await self.orchestrator.runtime.cancel_all(self.orchestrator.run_id)
        await self.orchestrator.runtime.flatten(self.orchestrator.run_id)
        self._write_state(
            state="FAILED",
            kill_switch_triggered=True,
            kill_switch_reason=reason,
        )

    async def recover_or_resume(self) -> dict[str, Any]:
        """Reconcile persisted state and exchange state after a restart."""
        if not self.state_path.exists():
            return {"action": "FRESH_START", "run_id": self.orchestrator.run_id}
        saved = json.loads(self.state_path.read_text(encoding="utf-8"))
        terminal = await self.orchestrator.runtime.terminal_state()
        open_orders = terminal.get("open_orders", [])
        positions = terminal.get("positions", [])
        if open_orders or positions:
            previous_run_id = saved.get("run_id", self.orchestrator.run_id)
            await self.orchestrator.runtime.cancel_all(previous_run_id)
            await self.orchestrator.runtime.flatten(previous_run_id)
            self._write_state(state="RECOVERED_FLATTENED", recovered_at=time.time())
            return {
                "action": "RECOVERED_AND_FLATTENED",
                "previous_run_id": saved.get("run_id"),
                "terminal_open_orders": len(open_orders),
                "terminal_positions": len(positions),
            }
        return {"action": "CLEAN_BASELINE", "previous_run_id": saved.get("run_id")}


def make_live_supervisor(
    *,
    artifact_root: Path | None = None,
    account_name: str = "master_account",
    duration_seconds: float = 86_400.0,
    config_path: Path | str | None = None,
) -> PersistentExecutionSupervisor:
    """Factory for the Testnet execution supervisor with Hummingbot."""
    from hummingbot_api_client import HummingbotAPIClient
    from config_manager import ConfigManager
    from condor.orderflow_orchestrator import HummingbotApiRuntime, RunnerConfig
    from condor.jev_qualification import RealJevClient

    resolved_cfg = config_path or (
        Path("condor/config.yml") if Path("condor/config.yml").exists() else Path("config.yml")
    )
    config_mgr = ConfigManager(str(resolved_cfg))
    server_config = config_mgr.get_server("main") or {}
    base_url = os.getenv("HUMMINGBOT_API_URL") or f"http://{server_config.get('host', '127.0.0.1')}:{server_config.get('port', 8000)}"
    username = os.getenv("HUMMINGBOT_API_USER") or server_config.get("username", "admin")
    password = os.getenv("HUMMINGBOT_API_PASSWORD") or server_config.get("password", "")
    client = HummingbotAPIClient(base_url=base_url, username=username, password=password)
    runtime = HummingbotApiRuntime(client, account_name=account_name)
    root_env = os.getenv("ORDERFLOW_ARTIFACT_ROOT")
    resolved_root = Path(root_env) if root_env else (artifact_root or Path("/run/media/mzfshark/Storage/Axodus/Trading/market-data/runs/live_24h_testnet"))
    config = RunnerConfig(
        duration_seconds=duration_seconds,
        artifact_root=resolved_root,
        execution_environment="TESTNET",
        market_data_environment="TESTNET",
    )
    jev_client = RealJevClient.from_environment()
    orchestrator = OrderFlowOrchestrator(runtime, config, jev_client=jev_client)
    return PersistentExecutionSupervisor(orchestrator)


def make_paper_supervisor(
    *,
    artifact_root: Path | None = None,
    duration_seconds: float = 86_400.0,
    rest_base: str = "https://fapi.binance.com",
    ws_base: str = "wss://fstream.binance.com/stream",
) -> PersistentExecutionSupervisor:
    """Factory for the PAPER_MAINNET_DATA supervisor with Binance Mainnet public feed."""
    from condor.mainnet_market_data import BinanceMainnetPublicMarketFeed
    from condor.paper_execution import PaperExecutionRuntime, PaperExecutionAdapter
    from condor.jev_qualification import RealJevClient
    from condor.orderflow_orchestrator import RunnerConfig

    feed = BinanceMainnetPublicMarketFeed(rest_base=rest_base, ws_base=ws_base)
    root_env = os.getenv("ORDERFLOW_PAPER_ARTIFACT_ROOT") or os.getenv("ORDERFLOW_ARTIFACT_ROOT")
    resolved_root = (
        Path(root_env)
        if root_env
        else (
            artifact_root
            or Path("/run/media/mzfshark/Storage/Axodus/Trading/market-data/runs/live_24h_paper")
        )
    )
    config = RunnerConfig(
        duration_seconds=duration_seconds,
        artifact_root=resolved_root,
        execution_environment="PAPER_MAINNET_DATA",
        market_data_environment="MAINNET",
        testnet_only=False,
    )
    jev_client = RealJevClient.from_environment()
    runtime = PaperExecutionRuntime(feed=feed)
    orchestrator = OrderFlowOrchestrator(runtime, config, jev_client=jev_client)
    # Connect sink to runtime after orchestrator initializes sink
    runtime.sink = orchestrator.sink
    return PersistentExecutionSupervisor(
        orchestrator, service_identity="condor-orderflow-paper-mainnet"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Persistent Live Execution Supervisor")
    parser.add_argument("--status", action="store_true", help="Print current status")
    parser.add_argument("--probe", action="store_true", help="Run bounded Testnet dispatch probe")
    parser.add_argument("--run", action="store_true", help="Run the persistent supervised 24-hour loop")
    args = parser.parse_args()

    supervisor = make_live_supervisor()
    if args.run:
        async def _run():
            await supervisor.orchestrator.runtime.client.init()
            try:
                await supervisor.run_forever()
            finally:
                await supervisor.orchestrator.runtime.client.close()
        asyncio.run(_run())
    elif args.status:
        print(json.dumps(supervisor.status(), indent=2))
    elif args.probe:
        async def _probe():
            await supervisor.orchestrator.runtime.client.init()
            try:
                adapter = HummingbotExecutionAdapter(supervisor.orchestrator.runtime)
                res = await adapter.bounded_dispatch_probe()
                print("DISPATCH_PROBE:", json.dumps(res, indent=2))
            finally:
                await supervisor.orchestrator.runtime.client.close()

        asyncio.run(_probe())
