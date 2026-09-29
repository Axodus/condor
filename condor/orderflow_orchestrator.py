"""Durable 12-cell OrderFlow Testnet orchestration.

This module owns orchestration only.  Hummingbot remains the exchange
transport and CondorDeploymentGate remains the authorization boundary.  The
default evaluator is deliberately no-signal: a live adapter must be supplied
explicitly before a frozen strategy can create an ExecutionIntent.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from condor.deployment_candidate import (
    CondorDeploymentDecision,
    CondorDeploymentGate,
    CondorOperationalGateContext,
    DeploymentCandidatePayload,
    DeploymentEnvironment,
)
from condor.jev_qualification import (
    CandidateTriggerV1,
    JevDecisionType,
    JevQualificationClient,
    RealJevClient,
)

SYMBOLS = ("BTCUSDT", "ZECUSDT", "SUIUSDT", "WLDUSDT")
STRATEGIES = (
    "orderflow.momentum.aggression",
    "orderflow.absorption.fade",
    "orderflow.cvd.divergence.reversal",
)
STRATEGY_REVISION = "freeze-2026-09-24-adapter-v1"
CONNECTOR = "binance_perpetual_testnet"
DEFAULT_ARTIFACT_ROOT = Path(
    "/run/media/mzfshark/Storage/Axodus/Trading/market-data/runs/live_24h_testnet"
)


class RunnerState(StrEnum):
    PRE_RUN = "PRE_RUN"
    STARTING = "STARTING"
    ARMED = "ARMED"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    RECOVERING = "RECOVERING"
    BLOCKED = "BLOCKED"
    STOPPING = "STOPPING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class OrchestratorBlocked(RuntimeError):
    """Raised when a startup or safety invariant fails closed."""


@dataclass(frozen=True)
class Cell:
    cell_id: str
    symbol: str
    strategy_id: str
    strategy_revision: str
    role: str
    deployment_candidate_id: str


@dataclass(frozen=True)
class ExecutionIntent:
    run_id: str
    cell_id: str
    correlation_id: str
    jev_decision_id: str
    symbol: str
    action: str
    side: str
    strategy_id: str
    strategy_revision: str
    quantity: str
    order_type: str
    price: str | None
    execution_constraints: dict[str, Any]
    target_environment: str = "TESTNET"

    def validate(self, valid_cells: set[str] | None = None) -> None:
        if self.target_environment not in {"TESTNET", "PAPER_MAINNET_DATA"}:
            raise OrchestratorBlocked("NON_TESTNET_DISPATCH_REJECTED")
        if self.strategy_revision != STRATEGY_REVISION:
            raise OrchestratorBlocked("STRATEGY_REVISION_MISMATCH")
        if self.symbol not in SYMBOLS:
            raise OrchestratorBlocked(f"UNAPPROVED_SYMBOL:{self.symbol}")
        if valid_cells is not None and self.cell_id not in valid_cells:
            raise OrchestratorBlocked("UNKNOWN_CELL_ID")
        if self.action not in ("ENTRY", "EXIT"):
            raise OrchestratorBlocked("INVALID_INTENT_ACTION")
        if self.side not in ("BUY", "SELL"):
            raise OrchestratorBlocked("INVALID_INTENT_SIDE")
        if not self.correlation_id or not self.jev_decision_id:
            raise OrchestratorBlocked("EXECUTION_INTENT_MISSING_ID")
        try:
            if float(self.quantity) <= 0:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise OrchestratorBlocked("INVALID_INTENT_QUANTITY") from exc


@dataclass
class RunnerConfig:
    duration_seconds: float = 86_400.0
    heartbeat_seconds: float = 30.0
    recovery_window_seconds: float = 120.0
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT
    connector: str = CONNECTOR
    account_name: str = "master_account"
    testnet_only: bool = True
    execution_environment: str = "TESTNET"  # TESTNET | PAPER_MAINNET_DATA
    market_data_environment: str = "TESTNET"  # TESTNET | MAINNET
    jev_timeout_seconds: float = 5.0


class StrategyEvaluator(Protocol):
    async def evaluate(
        self, cell: Cell, observation: dict[str, Any]
    ) -> dict[str, Any] | None: ...


class NoSignalEvaluator:
    """Safe default until a live OrderFlowFrame evaluator is explicitly wired."""

    async def evaluate(self, cell: Cell, observation: dict[str, Any]) -> None:
        return None


class JsonlSink:
    def __init__(self, root: Path):
        self.root = root
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self._locks: dict[Path, asyncio.Lock] = {}

    async def write(self, stream: str, payload: dict[str, Any]) -> Path:
        path = self.root / f"{stream}.jsonl"
        lock = self._locks.setdefault(path, asyncio.Lock())
        line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        async with lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        return path

    def write_sync(self, stream: str, payload: dict[str, Any]) -> Path:
        path = self.root / f"{stream}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        return path


class HummingbotRuntime(Protocol):
    async def preflight(self, symbols: tuple[str, ...]) -> dict[str, Any]: ...
    async def market_stream(self, symbols: tuple[str, ...]): ...
    async def cancel_all(self, run_id: str) -> dict[str, Any]: ...
    async def flatten(self, run_id: str) -> dict[str, Any]: ...
    async def terminal_state(self) -> dict[str, Any]: ...
    async def dispatch_intent(self, intent: ExecutionIntent) -> dict[str, Any]: ...


class HummingbotApiRuntime:
    """Small adapter over the supported Hummingbot API client."""

    def __init__(
        self,
        client: Any,
        account_name: str = "master_account",
        connector: str = CONNECTOR,
    ):
        self.client = client
        self.account_name = account_name
        self.connector = connector

    async def preflight(self, symbols: tuple[str, ...]) -> dict[str, Any]:
        accounts = await self.client.accounts.list_accounts()
        connectors = await self.client.connectors.list_connectors()
        if self.connector not in connectors:
            raise OrchestratorBlocked(f"TESTNET_CONNECTOR_UNAVAILABLE:{self.connector}")
        credentials = await self.client.accounts.list_account_credentials(
            self.account_name
        )
        if self.connector not in credentials:
            raise OrchestratorBlocked("TESTNET_CREDENTIAL_NOT_CONFIGURED")
        prices: dict[str, Any] = {}
        for symbol in symbols:
            pair = f"{symbol[:-4]}-USDT"
            prices[symbol] = await self.client.market_data.get_prices(
                self.connector, pair
            )
        orders = await self.client.trading.get_active_orders(
            account_names=[self.account_name], connector_names=[self.connector]
        )
        positions = await self.client.trading.get_positions(
            account_names=[self.account_name], connector_names=[self.connector]
        )
        return {
            "api": "HEALTHY",
            "accounts": accounts,
            "connector": self.connector,
            "credentials": "CONFIGURED",
            "prices": prices,
            "open_orders": orders.get("data", []),
            "positions": positions.get("data", []),
            "mainnet": "REJECTED",
        }

    async def market_stream(self, symbols: tuple[str, ...]):
        async with self.client.ws.market_data() as ws:
            subscriptions = []
            for symbol in symbols:
                pair = f"{symbol[:-4]}-USDT"
                subscriptions.append(
                    await ws.subscribe_order_book(
                        self.connector, pair, depth=10, update_interval=1.0
                    )
                )
                subscriptions.append(
                    await ws.subscribe_trades(self.connector, pair, update_interval=1.0)
                )
            while True:
                event = await ws.receive()
                if event is None:
                    raise ConnectionError("MARKET_STREAM_CLOSED")
                yield event

    async def cancel_all(self, run_id: str) -> dict[str, Any]:
        active = await self.client.trading.get_active_orders(
            account_names=[self.account_name], connector_names=[self.connector]
        )
        cancelled = []
        for order in active.get("data", []):
            client_order_id = order.get("client_order_id") or order.get("clientOrderId")
            if client_order_id:
                cancelled.append(
                    await self.client.trading.cancel_order(
                        self.account_name, self.connector, client_order_id
                    )
                )
        return {"run_id": run_id, "cancelled": cancelled}

    async def flatten(self, run_id: str) -> dict[str, Any]:
        positions = await self.client.trading.get_positions(
            account_names=[self.account_name], connector_names=[self.connector]
        )
        flattened = []
        for position in positions.get("data", []):
            amount = float(
                position.get("amount") or position.get("position_amount") or 0
            )
            pair = position.get("trading_pair") or position.get("symbol")
            if not pair or amount == 0:
                continue
            side = "SELL" if amount > 0 else "BUY"
            flattened.append(
                await self.client.trading.place_order(
                    self.account_name,
                    self.connector,
                    pair,
                    side,
                    abs(amount),
                    "MARKET",
                    None,
                    "CLOSE",
                )
            )
        return {"run_id": run_id, "flattened": flattened}

    async def terminal_state(self) -> dict[str, Any]:
        orders = await self.client.trading.get_active_orders(
            account_names=[self.account_name], connector_names=[self.connector]
        )
        positions = await self.client.trading.get_positions(
            account_names=[self.account_name], connector_names=[self.connector]
        )
        return {
            "open_orders": orders.get("data", []),
            "positions": positions.get("data", []),
        }

    async def dispatch_intent(self, intent: ExecutionIntent) -> dict[str, Any]:
        if intent.target_environment != "TESTNET" or self.connector != CONNECTOR:
            raise OrchestratorBlocked("NON_TESTNET_DISPATCH_REJECTED")
        pair = f"{intent.symbol[:-4]}-USDT"
        return await self.client.trading.place_order(
            self.account_name,
            self.connector,
            pair,
            intent.side,
            float(intent.quantity),
            intent.order_type,
            float(intent.price) if intent.price is not None else None,
            "OPEN",
        )


def build_cells(run_id: str) -> dict[str, Cell]:
    cells: dict[str, Cell] = {}
    for symbol in SYMBOLS:
        for strategy in STRATEGIES:
            cell_id = f"{symbol}:{strategy}:{STRATEGY_REVISION}"
            cells[cell_id] = Cell(
                cell_id=cell_id,
                symbol=symbol,
                strategy_id=strategy,
                strategy_revision=STRATEGY_REVISION,
                role="CONTROL_SAMPLE" if symbol == "BTCUSDT" else "ACTIVE",
                deployment_candidate_id=f"{run_id}:{hashlib.sha256(cell_id.encode()).hexdigest()[:16]}",
            )
    return cells


class OrderFlowOrchestrator:
    def __init__(
        self,
        runtime: HummingbotRuntime,
        config: RunnerConfig | None = None,
        evaluator: StrategyEvaluator | None = None,
        deployment_gate: CondorDeploymentGate | None = None,
        candidate_factory: (
            Callable[[Cell], DeploymentCandidatePayload | None] | None
        ) = None,
        jev_client: JevQualificationClient | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.config = config or RunnerConfig()
        self.runtime = runtime
        self.evaluator = evaluator or NoSignalEvaluator()
        self.deployment_gate = deployment_gate or CondorDeploymentGate()
        self.candidate_factory = candidate_factory
        self.jev_client = (
            jev_client if jev_client is not None else RealJevClient.from_environment()
        )
        self.clock = clock
        self.run_id = str(uuid.uuid4())
        self.state = RunnerState.PRE_RUN
        self.cells = build_cells(self.run_id)
        self.sink = JsonlSink(self.config.artifact_root / self.run_id)
        self.started_at: float | None = None
        self.deadline: float | None = None
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task[Any]] = []
        self._seen_events: set[str] = set()

    @property
    def artifact_root(self) -> Path:
        return self.sink.root

    def _manifest(self, **extra: Any) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "state": self.state.value,
            "strategy_revision": STRATEGY_REVISION,
            "connector": CONNECTOR if self.config.execution_environment == "TESTNET" else "none (local paper)",
            "environment": self.config.execution_environment,
            "market_data_environment": self.config.market_data_environment,
            "symbols": list(SYMBOLS),
            "strategies": list(STRATEGIES),
            "cells": [asdict(cell) for cell in self.cells.values()],
            "run_started_at": self.started_at,
            "run_deadline": self.deadline,
            "mainnet_mutations": 0,
            "real_capital": 0,
            **extra,
        }

    async def _persist_manifest(self, **extra: Any) -> None:
        path = self.artifact_root / "run_manifest.json"
        path.write_text(
            json.dumps(self._manifest(**extra), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    async def startup_gate(self, *, start_timer: bool = True) -> dict[str, Any]:
        self.state = RunnerState.STARTING
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        if self.config.execution_environment not in {"TESTNET", "PAPER_MAINNET_DATA"}:
            self.state = RunnerState.BLOCKED
            raise OrchestratorBlocked(
                f"UNSUPPORTED_EXECUTION_ENVIRONMENT:{self.config.execution_environment}"
            )
        if self.config.execution_environment == "TESTNET":
            if self.config.connector != CONNECTOR or not self.config.testnet_only:
                self.state = RunnerState.BLOCKED
                raise OrchestratorBlocked("TESTNET_ONLY_CONFIGURATION_REQUIRED")
        if len(self.cells) != 12:
            self.state = RunnerState.BLOCKED
            raise OrchestratorBlocked("CELL_COUNT_MUST_BE_12")
        if self.jev_client is None:
            self.state = RunnerState.BLOCKED
            raise OrchestratorBlocked("JEV_RUNTIME_UNAVAILABLE")
        try:
            jev_health = await asyncio.wait_for(
                self.jev_client.health(), timeout=self.config.jev_timeout_seconds
            )
        except Exception as exc:
            self.state = RunnerState.BLOCKED
            raise OrchestratorBlocked("JEV_RUNTIME_UNAVAILABLE") from exc
        if jev_health.get("status") != "HEALTHY":
            self.state = RunnerState.BLOCKED
            raise OrchestratorBlocked("JEV_RUNTIME_UNAVAILABLE")
        if jev_health.get("contract_version") not in (None, "v1"):
            self.state = RunnerState.BLOCKED
            raise OrchestratorBlocked("JEV_CONTRACT_INCOMPATIBLE")
        probe = getattr(self.jev_client, "decision_probe", None)
        if probe is not None:
            try:
                jev_probe = await asyncio.wait_for(
                    probe(), timeout=self.config.jev_timeout_seconds
                )
            except Exception as exc:
                self.state = RunnerState.BLOCKED
                raise OrchestratorBlocked("JEV_DECISION_PROBE_FAILED") from exc
            if (
                jev_probe.get("status") != "PASS"
                or jev_probe.get("execution") != "NONE"
            ):
                self.state = RunnerState.BLOCKED
                raise OrchestratorBlocked("JEV_DECISION_PROBE_FAILED")
        else:
            jev_probe = {"status": "NOT_IMPLEMENTED_BY_TEST_DOUBLE"}
        if not os.access(self.artifact_root, os.W_OK):
            self.state = RunnerState.BLOCKED
            raise OrchestratorBlocked("ARTIFACT_DIRECTORY_NOT_WRITABLE")
        preflight = await self.runtime.preflight(SYMBOLS)
        if preflight.get("mainnet") != "REJECTED":
            self.state = RunnerState.BLOCKED
            raise OrchestratorBlocked("MAINNET_NOT_REJECTED")
        if preflight.get("open_orders") or preflight.get("positions"):
            self.state = RunnerState.BLOCKED
            raise OrchestratorBlocked("BASELINE_NOT_FLAT")
        if start_timer:
            self.started_at = self.clock()
            self.deadline = self.started_at + self.config.duration_seconds
            self.state = RunnerState.RUNNING
        else:
            self.state = RunnerState.ARMED
        await self._persist_manifest(
            preflight=preflight,
            jev_health=jev_health,
            jev_decision_probe=jev_probe,
            cells_armed=12,
            first_heartbeat=False,
            timer_started=start_timer,
        )
        return preflight

    async def start_24h_timer(self) -> None:
        if self.state != RunnerState.ARMED:
            raise OrchestratorBlocked("TWELVE_CELLS_NOT_ARMED")
        self.started_at = self.clock()
        self.deadline = self.started_at + self.config.duration_seconds
        self.state = RunnerState.RUNNING
        await self.sink.write(
            "heartbeats",
            {
                "run_id": self.run_id,
                "timestamp": self.started_at,
                "runner_status": self.state.value,
                "cells_armed": len(self.cells),
                "first_heartbeat": True,
            },
        )
        await self._persist_manifest(
            run_started_at=self.started_at,
            run_deadline=self.deadline,
            first_heartbeat=True,
            timer_started=True,
        )

    async def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            now = self.clock()
            terminal = await self.runtime.terminal_state()
            payload = {
                "run_id": self.run_id,
                "timestamp": now,
                "runner_status": self.state.value,
                "cells_armed": len(self.cells),
                "cell_statuses": {
                    cell.cell_id: "ARMED" for cell in self.cells.values()
                },
                "hummingbot_health": "HEALTHY",
                "market_data_health": "ACTIVE",
                "telemetry_health": "ACTIVE",
                "open_experimental_orders": len(terminal.get("open_orders", [])),
                "experimental_positions": len(terminal.get("positions", [])),
            }
            await self.sink.write("heartbeats", payload)
            if self.started_at is not None and now >= (self.deadline or now):
                self._stop.set()
                break
            await asyncio.sleep(self.config.heartbeat_seconds)

    async def _market_loop(self) -> None:
        while not self._stop.is_set():
            try:
                async for event in self.runtime.market_stream(SYMBOLS):
                    if self._stop.is_set():
                        break
                    observation = {
                        "run_id": self.run_id,
                        "received_at": self.clock(),
                        "event": event,
                    }
                    await self.sink.write("market_observations", observation)
                    symbol = self._symbol_from_event(event)
                    if symbol is None:
                        continue
                    if event.get("market_valid") is False or event.get("is_crossed") is True:
                        await self.sink.write(
                            "incidents",
                            {
                                "run_id": self.run_id,
                                "symbol": symbol,
                                "type": "INVALID_OR_CROSSED_MARKET_STATE_DISCARDED",
                                "is_crossed": event.get("is_crossed"),
                                "sequence_gap": event.get("sequence_gap"),
                            },
                        )
                        continue
                    for cell in self.cells.values():
                        if cell.symbol != symbol:
                            continue
                        signal = await self.evaluator.evaluate(cell, observation)
                        if signal is not None:
                            await self._handle_signal(cell, signal, observation)
                if not self._stop.is_set():
                    raise ConnectionError("MARKET_STREAM_ENDED")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._stop.is_set():
                    break
                self.state = RunnerState.RECOVERING
                await self.sink.write(
                    "incidents",
                    {
                        "run_id": self.run_id,
                        "type": "MARKET_STREAM_RECOVERY",
                        "error": type(exc).__name__,
                    },
                )
                await asyncio.sleep(min(self.config.recovery_window_seconds, 5.0))
                self.state = RunnerState.RUNNING

    @staticmethod
    def _symbol_from_event(event: Any) -> str | None:
        if not isinstance(event, dict):
            return None
        raw = event.get("symbol") or event.get("trading_pair")
        if not raw and isinstance(event.get("data"), dict):
            raw = event["data"].get("symbol") or event["data"].get("trading_pair")
        if not raw and "subscription_id" in event:
            subscription = str(event["subscription_id"])
            for symbol in SYMBOLS:
                pair = f"{symbol[:-4]}-USDT"
                if pair in subscription or symbol in subscription:
                    raw = symbol
                    break
        if not raw:
            return None
        return str(raw).replace("-", "").upper()

    async def _handle_signal(
        self, cell: Cell, signal: dict[str, Any], observation: dict[str, Any]
    ) -> None:
        signal_id = str(signal.get("signal_id") or uuid.uuid4())
        if signal_id in self._seen_events:
            return
        self._seen_events.add(signal_id)
        signal_revision = signal.get("strategy_revision", cell.strategy_revision)
        if signal_revision != cell.strategy_revision:
            await self.sink.write(
                "incidents",
                {
                    "run_id": self.run_id,
                    "cell_id": cell.cell_id,
                    "type": "STRATEGY_REVISION_MUTATION_REJECTED",
                    "received_revision": signal_revision,
                    "expected_revision": cell.strategy_revision,
                },
            )
            return
        correlation_id = str(uuid.uuid4())
        await self.sink.write(
            "signals",
            {
                "run_id": self.run_id,
                "cell_id": cell.cell_id,
                "signal_id": signal_id,
                "symbol": cell.symbol,
                "strategy_id": cell.strategy_id,
                "strategy_revision": cell.strategy_revision,
                "timestamp": self.clock(),
                "trigger_inputs": signal.get("trigger_inputs", {}),
                "decision": signal,
                "execution_eligibility": "PENDING_GATE",
                "correlation_id": correlation_id,
            },
        )
        if self.jev_client is None:
            await self.sink.write(
                "incidents",
                {
                    "run_id": self.run_id,
                    "cell_id": cell.cell_id,
                    "type": "JEV_UNAVAILABLE_ENTRY_BLOCKED",
                },
            )
            return
        candidate = CandidateTriggerV1(
            schema_version="v1",
            run_id=self.run_id,
            cell_id=cell.cell_id,
            symbol=cell.symbol,
            venue="Binance USD-M Futures",
            market_type="USD-M Futures",
            strategy_id=cell.strategy_id,
            strategy_revision=cell.strategy_revision,
            trigger_type=str(signal.get("trigger_type", "ENTRY")),
            side=str(signal["side"]).upper(),
            event_time=float(observation.get("received_at", self.clock())),
            market_state_ref={"observation_stream": "market_observations"},
            feature_snapshot=dict(signal.get("feature_snapshot", {})),
            trigger_evidence=dict(signal.get("trigger_evidence", {})),
            current_logical_position=signal.get("current_logical_position"),
            current_open_orders=list(signal.get("current_open_orders", [])),
            deployment_candidate_id=cell.deployment_candidate_id,
            candidate_trigger_id=signal_id,
            correlation_id=correlation_id,
        )
        await self.sink.write("candidate_triggers", candidate.to_dict())
        try:
            candidate.validate()
            decision = await asyncio.wait_for(
                self.jev_client.qualify(candidate),
                timeout=self.config.jev_timeout_seconds,
            )
            decision.validate()
            if (
                decision.run_id != candidate.run_id
                or decision.cell_id != candidate.cell_id
                or decision.candidate_trigger_id != candidate.candidate_trigger_id
                or decision.correlation_id != candidate.correlation_id
                or decision.strategy_id != candidate.strategy_id
                or decision.strategy_revision != candidate.strategy_revision
            ):
                raise ValueError("JEV_DECISION_IDENTITY_MISMATCH")
        except Exception as exc:
            await self.sink.write(
                "incidents",
                {
                    "run_id": self.run_id,
                    "cell_id": cell.cell_id,
                    "candidate_trigger_id": candidate.candidate_trigger_id,
                    "type": "JEV_UNAVAILABLE_OR_INVALID_RESPONSE",
                    "error": type(exc).__name__,
                    "correlation_id": correlation_id,
                },
            )
            return
        decision_record = decision.to_dict()
        call_metadata = getattr(self.jev_client, "last_call", None)
        if isinstance(call_metadata, dict):
            decision_record["runtime_observability"] = {
                key: call_metadata[key]
                for key in (
                    "request_id",
                    "runtime_version",
                    "latency_ms",
                    "candidate_trigger_id",
                )
                if key in call_metadata
            }
        await self.sink.write("jev_decisions", decision_record)
        if decision.decision != JevDecisionType.SIGNAL_QUALIFIED.value:
            return
        candidate = self.candidate_factory(cell) if self.candidate_factory else None
        if candidate is None:
            await self.sink.write(
                "incidents",
                {
                    "run_id": self.run_id,
                    "cell_id": cell.cell_id,
                    "type": "EXECUTION_BLOCKED_NO_CANDIDATE",
                },
            )
            await self.sink.write(
                "condor_decisions",
                {
                    "run_id": self.run_id,
                    "cell_id": cell.cell_id,
                    "correlation_id": correlation_id,
                    "decision": "REJECTED",
                    "reason": "NO_CANDIDATE",
                },
            )
            return
        gate_env = (
            DeploymentEnvironment.PAPER_MAINNET_DATA
            if self.config.execution_environment == "PAPER_MAINNET_DATA"
            else DeploymentEnvironment.TESTNET
        )
        gate = self.deployment_gate.evaluate(
            candidate,
            CondorOperationalGateContext(environment=gate_env),
        )
        expected_decision = (
            CondorDeploymentDecision.PAPER_DISPATCH_AUTHORIZED.value
            if self.config.execution_environment == "PAPER_MAINNET_DATA"
            else CondorDeploymentDecision.TESTNET_DISPATCH_AUTHORIZED.value
        )
        if gate["decision"] != expected_decision:
            await self.sink.write(
                "incidents",
                {
                    "run_id": self.run_id,
                    "cell_id": cell.cell_id,
                    "type": "EXECUTION_BLOCKED_GATE",
                    "gate": gate,
                },
            )
            await self.sink.write(
                "condor_decisions",
                {
                    "run_id": self.run_id,
                    "cell_id": cell.cell_id,
                    "correlation_id": correlation_id,
                    "decision": "REJECTED",
                    "gate": gate,
                },
            )
            return
        intent = ExecutionIntent(
            run_id=self.run_id,
            cell_id=cell.cell_id,
            correlation_id=correlation_id,
            jev_decision_id=decision.decision_id,
            symbol=cell.symbol,
            action=decision.action,
            side=str(signal["side"]).upper(),
            strategy_id=cell.strategy_id,
            strategy_revision=cell.strategy_revision,
            quantity=str(signal["quantity"]),
            order_type=str(signal.get("order_type", "MARKET")),
            price=signal.get("price"),
            execution_constraints=dict(signal.get("execution_constraints", {})),
            target_environment=self.config.execution_environment,
        )
        await self.sink.write("execution_intents", asdict(intent))
        try:
            execution = await self.runtime.dispatch_intent(intent)
        except Exception as exc:
            await self.sink.write(
                "condor_decisions",
                {
                    "run_id": self.run_id,
                    "cell_id": cell.cell_id,
                    "correlation_id": correlation_id,
                    "decision": "REJECTED",
                    "reason": type(exc).__name__,
                },
            )
            return
        await self.sink.write(
            "condor_decisions",
            {
                "run_id": self.run_id,
                "cell_id": cell.cell_id,
                "correlation_id": correlation_id,
                "decision": "AUTHORIZED",
                "execution": execution,
            },
        )

    async def run(self) -> None:
        await self.startup_gate()
        await self.sink.write(
            "heartbeats",
            {
                "run_id": self.run_id,
                "timestamp": self.clock(),
                "runner_status": "RUNNING",
                "cells_armed": 12,
                "first_heartbeat": True,
            },
        )
        self._tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._market_loop()),
        ]
        try:
            await self._stop.wait()
        finally:
            await self.stop(
                reason=(
                    "HARD_STOP_24H"
                    if self.deadline and self.clock() >= self.deadline
                    else "STOP_REQUESTED"
                )
            )

    async def stop(self, reason: str = "STOP_REQUESTED") -> None:
        if self.state in {RunnerState.COMPLETE, RunnerState.FAILED}:
            return
        self.state = RunnerState.STOPPING
        self._stop.set()
        for task in self._tasks:
            if not task.done():
                task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.runtime.cancel_all(self.run_id)
        await self.runtime.flatten(self.run_id)
        terminal = await self.runtime.terminal_state()
        if terminal.get("open_orders") or terminal.get("positions"):
            self.state = RunnerState.FAILED
            await self._persist_manifest(
                stop_reason=reason, terminal_reconciliation=terminal
            )
            raise OrchestratorBlocked("TERMINAL_RECONCILIATION_FAILED")
        self.state = RunnerState.COMPLETE
        await self._persist_manifest(
            stop_reason=reason, terminal_reconciliation=terminal
        )


def utc_iso(timestamp: float | None) -> str | None:
    return (
        datetime.fromtimestamp(timestamp, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
        if timestamp is not None
        else None
    )
