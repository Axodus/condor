from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from condor.jev_qualification import DeterministicMockJevClient
from condor.orderflow_orchestrator import (
    CONNECTOR,
    SYMBOLS,
    STRATEGIES,
    HummingbotRuntime,
    OrchestratorBlocked,
    OrderFlowOrchestrator,
    RunnerConfig,
    RunnerState,
    build_cells,
)


class FakeRuntime:
    def __init__(self, tmp_path: Path, flat: bool = True):
        self.flat = flat
        self.cancelled = 0
        self.flattened = 0
        self.stream_started = False

    async def preflight(self, symbols):
        assert symbols == SYMBOLS
        return {
            "api": "HEALTHY", "connector": CONNECTOR, "credentials": "CONFIGURED",
            "mainnet": "REJECTED", "open_orders": [] if self.flat else [{"id": "x"}],
            "positions": [], "prices": {symbol: {"prices": {symbol: 1}} for symbol in symbols},
        }

    async def market_stream(self, symbols):
        self.stream_started = True
        while True:
            await asyncio.sleep(3600)
            yield {"symbol": symbols[0]}

    async def cancel_all(self, run_id):
        self.cancelled += 1
        return {"run_id": run_id}

    async def flatten(self, run_id):
        self.flattened += 1
        return {"run_id": run_id}

    async def terminal_state(self):
        return {"open_orders": [], "positions": []}


def make_runner(runtime, config):
    return OrderFlowOrchestrator(runtime, config, jev_client=DeterministicMockJevClient())


def test_builds_exact_matrix_and_stable_roles():
    cells = build_cells("run")
    assert len(cells) == 12
    assert {cell.symbol for cell in cells.values()} == set(SYMBOLS)
    assert {cell.strategy_id for cell in cells.values()} == set(STRATEGIES)
    assert sum(cell.role == "CONTROL_SAMPLE" for cell in cells.values()) == 3


@pytest.mark.asyncio
async def test_startup_gate_sets_run_identity_and_persists_first_heartbeat(tmp_path):
    runner = make_runner(FakeRuntime(tmp_path), RunnerConfig(duration_seconds=60, artifact_root=tmp_path))
    await runner.startup_gate()
    assert runner.state == RunnerState.RUNNING
    assert runner.started_at is not None
    assert runner.deadline == runner.started_at + 60
    await runner.sink.write("heartbeats", {"run_id": runner.run_id, "first_heartbeat": True})
    assert (runner.artifact_root / "run_manifest.json").exists()
    assert (runner.artifact_root / "heartbeats.jsonl").exists()


@pytest.mark.asyncio
async def test_startup_fails_closed_if_baseline_not_flat(tmp_path):
    runner = make_runner(FakeRuntime(tmp_path, flat=False), RunnerConfig(artifact_root=tmp_path))
    with pytest.raises(OrchestratorBlocked, match="BASELINE_NOT_FLAT"):
        await runner.startup_gate()
    assert runner.state == RunnerState.BLOCKED


@pytest.mark.asyncio
async def test_stop_cancels_flattens_and_reconciles(tmp_path):
    runtime = FakeRuntime(tmp_path)
    runner = make_runner(runtime, RunnerConfig(artifact_root=tmp_path))
    await runner.startup_gate()
    await runner.stop("TEST")
    assert runtime.cancelled == 1
    assert runtime.flattened == 1
    assert runner.state == RunnerState.COMPLETE


@pytest.mark.asyncio
async def test_no_signal_default_never_creates_execution_intent(tmp_path):
    runtime = FakeRuntime(tmp_path)
    runner = make_runner(runtime, RunnerConfig(artifact_root=tmp_path))
    await runner.startup_gate()
    assert not (runner.artifact_root / "execution_intents.jsonl").exists()
    await runner.stop("TEST")
