from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
import pytest_asyncio
from aiohttp import web

from condor.deployment_candidate import DeploymentCandidatePayload
from condor.jev_qualification import (
    CandidateTriggerV1,
    JevActionType,
    JevDecisionType,
    JevValidationError,
    RealJevClient,
)
from condor.orderflow_orchestrator import (
    Cell,
    ExecutionIntent,
    OrderFlowOrchestrator,
    RunnerConfig,
)


class MockJevHttpServer:
    def __init__(self) -> None:
        self.app = web.Application()
        self.app.router.add_get("/health", self.handle_health)
        self.app.router.add_post("/v1/systemone", self.handle_systemone)
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None
        self.port = 0
        self.health_status = 200
        self.health_body = {"status": "ok", "version": "jev-1.13.0"}
        self.decision_mode = "qualified"
        self.custom_wire: dict | None = None
        self.delay_seconds = 0.0
        self.requests: list[dict] = []

    async def handle_health(self, request: web.Request) -> web.Response:
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        return web.json_response(self.health_body, status=self.health_status)

    async def handle_systemone(self, request: web.Request) -> web.Response:
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        self.requests.append(await request.json())
        if self.custom_wire is not None:
            return web.json_response(self.custom_wire)
        reason = {
            "qualified": "causal_evidence",
            "rejected": "regime_mismatch",
            "no_action": "no_actionable_trigger",
        }[self.decision_mode]
        return web.json_response(
            {
                "model": "jev-1.13.0",
                "answers": {
                    "qualification_decision": {
                        "type": "choice",
                        "choice": self.decision_mode,
                    },
                    "reason_code": {"type": "choice", "choice": reason},
                },
            }
        )

    async def start(self) -> str:
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        self.port = self.site._server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{self.port}"

    async def stop(self) -> None:
        if self.runner is not None:
            await self.runner.cleanup()


@pytest_asyncio.fixture
async def jev_server():
    server = MockJevHttpServer()
    base_url = await server.start()
    try:
        yield server, base_url
    finally:
        await server.stop()


def sample_candidate(
    cell_id: str = "BTCUSDT:momentum:v1", symbol: str = "BTCUSDT"
) -> CandidateTriggerV1:
    return CandidateTriggerV1(
        schema_version="v1",
        run_id="run-test-123",
        cell_id=cell_id,
        symbol=symbol,
        venue="Binance USD-M Futures",
        market_type="USD-M Futures",
        strategy_id="orderflow.momentum.aggression",
        strategy_revision="freeze-2026-09-24-adapter-v1",
        trigger_type="ENTRY",
        side="BUY",
        event_time=time.time(),
        market_state_ref={"snapshot": 1},
        feature_snapshot={"delta": 15.2, "obi": 0.75},
        trigger_evidence={"rule": "aggression_spike"},
        current_logical_position=None,
        current_open_orders=[],
        deployment_candidate_id="dc-123",
        candidate_trigger_id="trigger-abc-1",
        correlation_id="corr-xyz-1",
    )


@pytest.mark.asyncio
async def test_real_jev_health_success(jev_server):
    _, base_url = jev_server
    client = RealJevClient(
        api_key="test-key",
        endpoint=f"{base_url}/v1/systemone",
        health_endpoint=f"{base_url}/health",
    )
    try:
        health = await client.health()
        assert health["status"] == "HEALTHY"
        assert health["contract_version"] == "v1"
        assert health["runtime_version"] == "jev-1.13.0"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_real_jev_health_requires_credentials():
    client = RealJevClient(api_key=None, endpoint="http://127.0.0.1:9/v1/systemone")
    try:
        health = await client.health()
        assert health == {
            "status": "UNAVAILABLE",
            "provider": "typesafe-jev",
            "contract_version": "v1",
            "reason": "JEV_CREDENTIAL_NOT_CONFIGURED",
        }
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_real_jev_health_falls_back_to_decision_probe(jev_server):
    server, base_url = jev_server
    server.health_status = 404
    client = RealJevClient(
        api_key="test-key",
        endpoint=f"{base_url}/v1/systemone",
        health_endpoint=f"{base_url}/missing",
    )
    try:
        health = await client.health()
        assert health["status"] == "HEALTHY"
        assert health["decision_probe"] == "PASS"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_real_jev_explicit_decision_probe(jev_server):
    _, base_url = jev_server
    client = RealJevClient(api_key="test-key", endpoint=f"{base_url}/v1/systemone")
    try:
        probe = await client.decision_probe()
        assert probe["status"] == "PASS"
        assert probe["schema_version"] == "v1"
        assert probe["execution"] == "NONE"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_real_jev_qualifies_all_decision_paths(jev_server):
    server, base_url = jev_server
    client = RealJevClient(api_key="test-key", endpoint=f"{base_url}/v1/systemone")
    candidate = sample_candidate()
    try:
        for mode, expected in (
            ("qualified", JevDecisionType.SIGNAL_QUALIFIED.value),
            ("rejected", JevDecisionType.SIGNAL_REJECTED.value),
            ("no_action", JevDecisionType.NO_ACTION.value),
        ):
            server.decision_mode = mode
            decision = await client.qualify(candidate)
            assert decision.decision == expected
            assert decision.run_id == candidate.run_id
            assert decision.cell_id == candidate.cell_id
            assert decision.candidate_trigger_id == candidate.candidate_trigger_id
            assert decision.correlation_id == candidate.correlation_id
            assert decision.strategy_id == candidate.strategy_id
            assert decision.strategy_revision == candidate.strategy_revision
        assert decision.action == JevActionType.NONE.value
        assert decision.side == "NONE"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_real_jev_invalid_schema_and_timeout_fail_closed(jev_server):
    server, base_url = jev_server
    server.custom_wire = {
        "model": "jev-1.13.0",
        "answers": {"qualification_decision": "invalid"},
    }
    client = RealJevClient(api_key="test-key", endpoint=f"{base_url}/v1/systemone")
    try:
        with pytest.raises(JevValidationError):
            await client.qualify(sample_candidate())
    finally:
        await client.close()

    server.custom_wire = None
    server.delay_seconds = 0.2
    client = RealJevClient(
        api_key="test-key", endpoint=f"{base_url}/v1/systemone", timeout_seconds=0.05
    )
    try:
        with pytest.raises(RuntimeError, match="JEV_RUNTIME_UNAVAILABLE"):
            await client.qualify(sample_candidate())
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_real_jev_calls_preserve_cell_identity(jev_server):
    _, base_url = jev_server
    client = RealJevClient(api_key="test-key", endpoint=f"{base_url}/v1/systemone")
    try:
        candidates = [
            sample_candidate(f"{symbol}:momentum:v1", symbol)
            for symbol in ("BTCUSDT", "ZECUSDT", "SUIUSDT", "WLDUSDT")
        ]
        decisions = await asyncio.gather(
            *(client.qualify(candidate) for candidate in candidates)
        )
        assert [decision.cell_id for decision in decisions] == [
            candidate.cell_id for candidate in candidates
        ]
    finally:
        await client.close()


class MockRuntime:
    def __init__(self) -> None:
        self.intents: list[ExecutionIntent] = []

    async def preflight(self, symbols: tuple[str, ...]):
        return {
            "api": "HEALTHY",
            "connector": "binance_perpetual_testnet",
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


def candidate_payload(
    cell: Cell, status: str = "VALIDATED_POSITIVE", enabled: bool = True
):
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


@pytest.mark.asyncio
async def test_orchestrator_uses_real_jev_client_for_startup_and_dispatch(
    jev_server, tmp_path: Path
):
    _, base_url = jev_server
    runtime = MockRuntime()
    client = RealJevClient(
        api_key="test-key",
        endpoint=f"{base_url}/v1/systemone",
        health_endpoint=f"{base_url}/health",
    )
    runner = OrderFlowOrchestrator(
        runtime,
        RunnerConfig(artifact_root=tmp_path),
        jev_client=client,
        candidate_factory=candidate_payload,
    )
    try:
        await runner.startup_gate()
        await runner._handle_signal(
            list(runner.cells.values())[0],
            {
                "signal_id": "sig-100",
                "side": "BUY",
                "trigger_type": "ENTRY",
                "quantity": "0.001",
                "order_type": "LIMIT",
                "price": "65000",
                "feature_snapshot": {"obi": 0.82},
                "trigger_evidence": {"reason": "momentum_breakout"},
            },
            {"received_at": time.time()},
        )
        assert len(runtime.intents) == 1
        assert runtime.intents[0].action == "ENTRY"
    finally:
        await client.close()
