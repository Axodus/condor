from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from condor.jev_qualification import DeterministicMockJevClient
from condor.live_execution import (
    HummingbotExecutionAdapter,
    PersistentExecutionSupervisor,
)
from condor.orderflow_orchestrator import (
    CONNECTOR,
    STRATEGIES,
    STRATEGY_REVISION,
    SYMBOLS,
    ExecutionIntent,
    JsonlSink,
    OrchestratorBlocked,
    OrderFlowOrchestrator,
    RunnerConfig,
    RunnerState,
    build_cells,
)


class FakeRuntime:
    def __init__(self, flat: bool = True) -> None:
        self.flat = flat
        self.dispatched: list[ExecutionIntent] = []
        self.cancelled = 0
        self.flattened = 0

    async def preflight(self, symbols: tuple[str, ...]) -> dict[str, Any]:
        return {
            "api": "HEALTHY",
            "connector": CONNECTOR,
            "credentials": "CONFIGURED",
            "mainnet": "REJECTED",
            "open_orders": [] if self.flat else [{"id": "x"}],
            "positions": [],
            "prices": {s: {"prices": {s: 1.0}} for s in symbols},
        }

    async def market_stream(self, symbols: tuple[str, ...]):
        while True:
            await asyncio.sleep(3600)
            yield {"symbol": symbols[0]}

    async def cancel_all(self, run_id: str) -> dict[str, Any]:
        self.cancelled += 1
        return {"run_id": run_id}

    async def flatten(self, run_id: str) -> dict[str, Any]:
        self.flattened += 1
        return {"run_id": run_id}

    async def terminal_state(self) -> dict[str, Any]:
        return {"open_orders": [], "positions": []}

    async def dispatch_intent(self, intent: ExecutionIntent) -> dict[str, Any]:
        self.dispatched.append(intent)
        return {
            "status": "SUBMITTED",
            "client_order_id": intent.correlation_id,
            "order_id": "testnet-ord-12345",
        }


def sample_intent(
    cell_id: str = "BTCUSDT:orderflow.momentum.aggression:freeze-2026-09-24-adapter-v1",
    symbol: str = "BTCUSDT",
    revision: str = STRATEGY_REVISION,
    env: str = "TESTNET",
    action: str = "ENTRY",
    side: str = "BUY",
    qty: str = "0.001",
) -> ExecutionIntent:
    return ExecutionIntent(
        run_id="run-test-1",
        cell_id=cell_id,
        correlation_id="corr-test-1",
        jev_decision_id="jev-dec-1",
        symbol=symbol,
        action=action,
        side=side,
        strategy_id="orderflow.momentum.aggression",
        strategy_revision=revision,
        quantity=qty,
        order_type="MARKET",
        price=None,
        execution_constraints={},
        target_environment=env,
    )


@pytest.mark.asyncio
async def test_execution_adapter_valid_dispatch(tmp_path: Path):
    runtime = FakeRuntime()
    sink = JsonlSink(tmp_path)
    adapter = HummingbotExecutionAdapter(runtime, sink=sink)
    cells = build_cells("run-test-1")
    valid_cells = set(cells.keys())

    intent = sample_intent()
    result = await adapter.dispatch(intent, valid_cells)

    assert result["status"] == "SUBMITTED"
    assert len(runtime.dispatched) == 1
    assert (tmp_path / "execution_events.jsonl").exists()


@pytest.mark.asyncio
async def test_execution_adapter_rejects_unknown_cell():
    runtime = FakeRuntime()
    adapter = HummingbotExecutionAdapter(runtime)
    intent = sample_intent(cell_id="UNKNOWN_CELL:123")
    with pytest.raises(OrchestratorBlocked, match="UNKNOWN_CELL_ID"):
        await adapter.dispatch(intent, {"BTCUSDT:cell:v1"})


@pytest.mark.asyncio
async def test_execution_adapter_rejects_mainnet_intent():
    runtime = FakeRuntime()
    adapter = HummingbotExecutionAdapter(runtime)
    intent = sample_intent(env="MAINNET")
    with pytest.raises(OrchestratorBlocked, match="NON_TESTNET_DISPATCH_REJECTED"):
        await adapter.dispatch(intent, {intent.cell_id})


@pytest.mark.asyncio
async def test_execution_adapter_rejects_strategy_revision_mutation():
    runtime = FakeRuntime()
    adapter = HummingbotExecutionAdapter(runtime)
    intent = sample_intent(revision="mutated-revision-v2")
    with pytest.raises(OrchestratorBlocked, match="STRATEGY_REVISION_MISMATCH"):
        await adapter.dispatch(intent, {intent.cell_id})


@pytest.mark.asyncio
async def test_execution_adapter_rejects_unapproved_symbol():
    runtime = FakeRuntime()
    adapter = HummingbotExecutionAdapter(runtime)
    intent = sample_intent(symbol="DOGEUSDT")
    with pytest.raises(OrchestratorBlocked, match="UNAPPROVED_SYMBOL"):
        await adapter.dispatch(intent, {intent.cell_id})


@pytest.mark.asyncio
async def test_cell_isolation_same_symbol_multiple_strategies(tmp_path: Path):
    runtime = FakeRuntime()
    sink = JsonlSink(tmp_path)
    adapter = HummingbotExecutionAdapter(runtime, sink=sink)

    cell_a = "BTCUSDT:orderflow.momentum.aggression:freeze-2026-09-24-adapter-v1"
    cell_b = "BTCUSDT:orderflow.absorption.fade:freeze-2026-09-24-adapter-v1"
    valid_cells = {cell_a, cell_b}

    intent_a = sample_intent(cell_id=cell_a)
    intent_b = sample_intent(cell_id=cell_b)

    res_a = await adapter.dispatch(intent_a, valid_cells)
    res_b = await adapter.dispatch(intent_b, valid_cells)

    assert res_a["status"] == "SUBMITTED"
    assert res_b["status"] == "SUBMITTED"
    assert len(runtime.dispatched) == 2
    assert runtime.dispatched[0].cell_id == cell_a
    assert runtime.dispatched[1].cell_id == cell_b


@pytest.mark.asyncio
async def test_persistent_supervisor_arms_without_starting_clock(tmp_path: Path):
    runtime = FakeRuntime()
    orchestrator = OrderFlowOrchestrator(
        runtime,
        RunnerConfig(duration_seconds=3600, artifact_root=tmp_path),
        jev_client=DeterministicMockJevClient(),
    )
    supervisor = PersistentExecutionSupervisor(orchestrator)

    preflight = await supervisor.arm()
    assert preflight["api"] == "HEALTHY"
    assert orchestrator.state == RunnerState.ARMED

    status = supervisor.status()
    assert status["state"] == "ARMED"
    assert status["cells_armed"] == 12
    assert status["timer_started"] is False


@pytest.mark.asyncio
async def test_persistent_supervisor_starts_clock_when_armed(tmp_path: Path):
    runtime = FakeRuntime()
    orchestrator = OrderFlowOrchestrator(
        runtime,
        RunnerConfig(duration_seconds=3600, artifact_root=tmp_path),
        jev_client=DeterministicMockJevClient(),
    )
    supervisor = PersistentExecutionSupervisor(orchestrator)

    await supervisor.arm()
    await supervisor.start()

    assert orchestrator.state == RunnerState.RUNNING
    assert orchestrator.started_at is not None
    assert orchestrator.deadline == orchestrator.started_at + 3600

    status = supervisor.status()
    assert status["state"] == "RUNNING"
    assert status["timer_started"] is True
    assert status["run_started_at"] == orchestrator.started_at
    assert status["run_deadline"] == orchestrator.deadline


@pytest.mark.asyncio
async def test_bounded_probe_dispatches_and_reconciles(tmp_path: Path):
    runtime = FakeRuntime()
    adapter = HummingbotExecutionAdapter(runtime, sink=JsonlSink(tmp_path))
    result = await adapter.bounded_dispatch_probe()
    assert result["status"] == "PASS"
    assert result["terminal_open_orders"] == 0
    assert result["terminal_positions"] == 0
    assert len(runtime.dispatched) == 1


@pytest.mark.asyncio
async def test_kill_switch_persists_incident_and_flattens(tmp_path: Path):
    runtime = FakeRuntime()
    orchestrator = OrderFlowOrchestrator(
        runtime,
        RunnerConfig(artifact_root=tmp_path),
        jev_client=DeterministicMockJevClient(),
    )
    supervisor = PersistentExecutionSupervisor(orchestrator)
    await supervisor.arm()
    await supervisor.trigger_kill_switch("TEST_FAILURE")
    assert orchestrator.state == RunnerState.FAILED
    assert runtime.cancelled == 1
    assert runtime.flattened == 1
    assert supervisor.status()["kill_switch_triggered"] is True
    assert "KILL_SWITCH_TRIGGERED" in (tmp_path / orchestrator.run_id / "incidents.jsonl").read_text()


@pytest.mark.asyncio
async def test_recovery_reports_clean_baseline(tmp_path: Path):
    runtime = FakeRuntime()
    orchestrator = OrderFlowOrchestrator(
        runtime,
        RunnerConfig(artifact_root=tmp_path),
        jev_client=DeterministicMockJevClient(),
    )
    supervisor = PersistentExecutionSupervisor(orchestrator)
    await supervisor.arm()
    result = await supervisor.recover_or_resume()
    assert result["action"] == "CLEAN_BASELINE"
