"""Tests for Condor Ingestion of Quants-Lab Deployment Candidates."""
from __future__ import annotations

import pytest

from condor.deployment_candidate import (
    CondorDeploymentDecision,
    CondorDeploymentGate,
    CondorOperationalGateContext,
    DeploymentCandidatePayload,
    DeploymentEnvironment,
)

# Override autouse conftest fixture that imports httpx for unrelated DEX pool throttle tests
@pytest.fixture(autouse=True)
def _reset_gecko_throttle():
    yield


def _make_candidate(status: str = "VALIDATED_POSITIVE", is_candidate: bool = True) -> DeploymentCandidatePayload:
    return DeploymentCandidatePayload(
        symbol="WLDUSDT",
        venue="Binance USD-M Futures",
        market_type="USD-M Futures",
        strategy_id="orderflow.absorption.fade",
        strategy_revision="freeze-2026-09-24-adapter-v1",
        strategy_parameter_hash="hash_abc",
        research_status=status,
        is_evidence_ref={"trade_count": 12},
        oos_evidence_ref={"trade_count": 16, "net_pnl": "1.27"},
        economic_disposition="ECONOMICALLY_VIABLE",
        instrument_spec_ref="WLDUSDT",
        instrument_spec_hash="spec_hash_123",
        fee_model_ref="maker=0.0002 taker=0.0005",
        deployment_candidate=is_candidate,
        execution_requirements={"maker_taker": "MAKER"},
        research_artifact_hashes={"signals": "sig_hash"},
    )


def test_condor_rejects_negative_research_candidate():
    gate = CondorDeploymentGate()
    candidate = _make_candidate(status="VALIDATED_NEGATIVE", is_candidate=False)
    result = gate.evaluate(candidate)
    assert result["decision"] == CondorDeploymentDecision.CANDIDATE_REJECTED.value
    assert "RESEARCH_STATUS_NOT_POSITIVE_VALIDATED_NEGATIVE" in result["blockers"]


def test_condor_authorizes_testnet_when_operational_context_passed():
    gate = CondorDeploymentGate()
    candidate = _make_candidate(status="VALIDATED_POSITIVE", is_candidate=True)
    result = gate.evaluate(candidate, CondorOperationalGateContext(environment=DeploymentEnvironment.TESTNET))
    assert result["decision"] == CondorDeploymentDecision.TESTNET_DISPATCH_AUTHORIZED.value
    assert result["executionTarget"] == "HUMMINGBOT_TESTNET_CONNECTOR"
    assert result["blockers"] == []


def test_condor_blocks_mainnet_by_default_due_to_governance_blocker():
    gate = CondorDeploymentGate()
    candidate = _make_candidate(status="VALIDATED_POSITIVE", is_candidate=True)
    result = gate.evaluate(candidate, CondorOperationalGateContext(environment=DeploymentEnvironment.MAINNET))
    assert result["decision"] == CondorDeploymentDecision.MAINNET_BLOCKED.value
    assert "DEBT-AUD-A-03" in result["blockers"]
    assert "CAPITAL_AUTHORITY_NOT_APPROVED" in result["blockers"]
