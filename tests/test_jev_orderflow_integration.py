from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from condor.deployment_candidate import DeploymentCandidatePayload
from condor.jev_qualification import (
    CandidateTriggerV1,
    DeterministicMockJevClient,
    JevDecisionType,
    JevValidationError,
)
from condor.orderflow_orchestrator import (
    CONNECTOR,
    SYMBOLS,
    Cell,
    ExecutionIntent,
    OrderFlowOrchestrator,
    RunnerConfig,
    build_cells,
)


class Runtime:
    def __init__(self) -> None:
        self.intents: list[ExecutionIntent] = []

    async def preflight(self, symbols: tuple[str, ...]):
        return {
            "api": "HEALTHY",
            "connector": CONNECTOR,
            "credentials": "CONFIGURED",
            "mainnet": "REJECTED",
            "open_orders": [],
            "positions": [],
            "prices": {symbol: {"prices": {symbol: 1}} for symbol in symbols},
        }

    async def market_stream(self, symbols):
        while True:
            await asyncio.sleep(3600)
            yield {"symbol": symbols[0]}

    async def dispatch_intent(self, intent: ExecutionIntent):
        self.intents.append(intent)
        return {"status": "SUBMITTED", "client_order_id": intent.correlation_id}

    async def cancel_all(self, run_id: str):
        return {"run_id": run_id}

    async def flatten(self, run_id: str):
        return {"run_id": run_id}

    async def terminal_state(self):
        return {"open_orders": [], "positions": []}


def candidate_payload(cell: Cell, status: str = "VALIDATED_POSITIVE", enabled: bool = True):
    return DeploymentCandidatePayload(
        symbol=cell.symbol,
        venue="Binance USD-M Futures",
        market_type="USD-M Futures",
        strategy_id=cell.strategy_id,
        strategy_revision=cell.strategy_revision,
        strategy_parameter_hash="parameter-hash",
        research_status=status,
        is_evidence_ref={"status": "PASS"},
        oos_evidence_ref={"status": "POSITIVE"},
        economic_disposition="VIABLE",
        instrument_spec_ref="instrument-spec",
        instrument_spec_hash="instrument-hash",
        fee_model_ref="fee-model",
        deployment_candidate=enabled,
        execution_requirements={"order_type": "LIMIT"},
        research_artifact_hashes={"evidence": "hash"},
    )


def signal(signal_id: str = "signal-1", revision: str | None = None):
    result = {
        "signal_id": signal_id,
        "side": "BUY",
        "trigger_type": "ENTRY",
        "quantity": "0.001",
        "order_type": "LIMIT",
        "price": "65000",
        "feature_snapshot": {"obi": 0.8},
        "trigger_evidence": {"reason": "frozen-adapter"},
    }
    if revision is not None:
        result["strategy_revision"] = revision
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decision",
    [JevDecisionType.NO_ACTION.value, JevDecisionType.SIGNAL_REJECTED.value],
)
async def test_jev_non_qualified_decisions_never_create_intent(tmp_path: Path, decision: str):
    runtime = Runtime()
    jev = DeterministicMockJevClient(default_decision=decision)
    runner = OrderFlowOrchestrator(runtime, RunnerConfig(artifact_root=tmp_path), jev_client=jev)
    await runner.startup_gate()
    await runner._handle_signal(list(runner.cells.values())[0], signal(), {"received_at": time.time()})

    assert not (runner.artifact_root / "execution_intents.jsonl").exists()
    assert runtime.intents == []
    assert (runner.artifact_root / "candidate_triggers.jsonl").exists()
    assert (runner.artifact_root / "jev_decisions.jsonl").exists()


@pytest.mark.asyncio
async def test_qualified_jev_decision_creates_typed_intent_and_dispatches(tmp_path: Path):
    runtime = Runtime()
    jev = DeterministicMockJevClient(default_decision=JevDecisionType.SIGNAL_QUALIFIED.value)
    runner = OrderFlowOrchestrator(
        runtime,
        RunnerConfig(artifact_root=tmp_path),
        jev_client=jev,
        candidate_factory=candidate_payload,
    )
    await runner.startup_gate()
    await runner._handle_signal(list(runner.cells.values())[0], signal(), {"received_at": time.time()})

    assert len(runtime.intents) == 1
    intent = runtime.intents[0]
    assert intent.action == "ENTRY"
    assert intent.target_environment == "TESTNET"
    assert intent.jev_decision_id
    assert json.loads((runner.artifact_root / "execution_intents.jsonl").read_text())["action"] == "ENTRY"


@pytest.mark.asyncio
async def test_condor_rejection_prevents_dispatch_after_jev_qualification(tmp_path: Path):
    runtime = Runtime()
    jev = DeterministicMockJevClient(default_decision=JevDecisionType.SIGNAL_QUALIFIED.value)
    runner = OrderFlowOrchestrator(
        runtime,
        RunnerConfig(artifact_root=tmp_path),
        jev_client=jev,
        candidate_factory=lambda cell: candidate_payload(cell, status="VALIDATED_NEGATIVE", enabled=False),
    )
    await runner.startup_gate()
    await runner._handle_signal(list(runner.cells.values())[0], signal(), {"received_at": time.time()})

    assert runtime.intents == []
    assert not (runner.artifact_root / "execution_intents.jsonl").exists()
    decisions = (runner.artifact_root / "condor_decisions.jsonl").read_text().splitlines()
    assert json.loads(decisions[0])["decision"] == "REJECTED"


@pytest.mark.asyncio
async def test_invalid_jev_response_fails_closed(tmp_path: Path):
    class InvalidJev:
        async def health(self):
            return {"status": "HEALTHY"}

        async def qualify(self, candidate: CandidateTriggerV1):
            return {"decision": "SIGNAL_QUALIFIED"}

    runtime = Runtime()
    runner = OrderFlowOrchestrator(runtime, RunnerConfig(artifact_root=tmp_path), jev_client=InvalidJev())
    await runner.startup_gate()
    await runner._handle_signal(list(runner.cells.values())[0], signal(), {"received_at": time.time()})

    assert runtime.intents == []
    assert (runner.artifact_root / "incidents.jsonl").exists()
    assert not (runner.artifact_root / "execution_intents.jsonl").exists()


@pytest.mark.asyncio
async def test_strategy_revision_mutation_is_rejected(tmp_path: Path):
    runtime = Runtime()
    runner = OrderFlowOrchestrator(
        runtime,
        RunnerConfig(artifact_root=tmp_path),
        jev_client=DeterministicMockJevClient(),
    )
    await runner.startup_gate()
    await runner._handle_signal(
        list(runner.cells.values())[0], signal(revision="new-revision"), {"received_at": time.time()}
    )

    assert runtime.intents == []
    incident = (runner.artifact_root / "incidents.jsonl").read_text()
    assert "STRATEGY_REVISION_MUTATION_REJECTED" in incident
