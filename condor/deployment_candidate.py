"""Condor ingestion and deployment gating for Quants-Lab candidates."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class DeploymentEnvironment(StrEnum):
    TESTNET = "TESTNET"
    MAINNET = "MAINNET"


class CondorDeploymentDecision(StrEnum):
    TESTNET_DISPATCH_AUTHORIZED = "TESTNET_DISPATCH_AUTHORIZED"
    TESTNET_BLOCKED = "TESTNET_BLOCKED"
    MAINNET_DISPATCH_AUTHORIZED = "MAINNET_DISPATCH_AUTHORIZED"
    MAINNET_BLOCKED = "MAINNET_BLOCKED"
    CANDIDATE_REJECTED = "CANDIDATE_REJECTED"


@dataclass(frozen=True)
class DeploymentCandidatePayload:
    """Credential-free typed payload received from Quants-Lab."""

    symbol: str
    venue: str
    market_type: str
    strategy_id: str
    strategy_revision: str
    strategy_parameter_hash: str
    research_status: str
    is_evidence_ref: dict[str, Any]
    oos_evidence_ref: dict[str, Any] | None
    economic_disposition: str
    instrument_spec_ref: str
    instrument_spec_hash: str
    fee_model_ref: str
    deployment_candidate: bool
    execution_requirements: dict[str, Any]
    research_artifact_hashes: dict[str, str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeploymentCandidatePayload":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CondorOperationalGateContext:
    environment: DeploymentEnvironment = DeploymentEnvironment.TESTNET
    hummingbot_adapter_available: bool = True
    credentials_configured: bool = True
    safety_supervisor_active: bool = True
    risk_authority_approved: bool = True
    capital_authority_approved: bool = False
    real_capital_authorized: bool = False
    execution_readiness_passed: bool = True
    protection_readiness_passed: bool = True
    active_blockers: list[str] = field(default_factory=lambda: ["DEBT-AUD-A-03"])


class CondorDeploymentGate:
    """Evaluates research candidates; Hummingbot remains the execution owner."""

    def evaluate(
        self,
        candidate: DeploymentCandidatePayload,
        context: CondorOperationalGateContext | None = None,
    ) -> dict[str, Any]:
        ctx = context or CondorOperationalGateContext()
        blockers: list[str] = []

        if not candidate.deployment_candidate or candidate.research_status != "VALIDATED_POSITIVE":
            return {
                "decision": CondorDeploymentDecision.CANDIDATE_REJECTED.value,
                "environment": ctx.environment.value,
                "blockers": [f"RESEARCH_STATUS_NOT_POSITIVE_{candidate.research_status}"],
                "candidate": candidate.to_dict(),
            }

        if not ctx.hummingbot_adapter_available:
            blockers.append("HUMMINGBOT_ADAPTER_UNAVAILABLE")
        if not ctx.credentials_configured:
            blockers.append("CREDENTIALS_NOT_CONFIGURED")
        if not ctx.execution_readiness_passed:
            blockers.append("EXECUTION_READINESS_NOT_PASSED")
        if not ctx.protection_readiness_passed:
            blockers.append("PROTECTION_READINESS_NOT_PASSED")
        if not ctx.safety_supervisor_active:
            blockers.append("SAFETY_SUPERVISOR_INACTIVE")
        if not ctx.risk_authority_approved:
            blockers.append("RISK_AUTHORITY_NOT_APPROVED")

        if ctx.environment == DeploymentEnvironment.TESTNET:
            decision = CondorDeploymentDecision.TESTNET_DISPATCH_AUTHORIZED if not blockers else CondorDeploymentDecision.TESTNET_BLOCKED
            return {
                "decision": decision.value,
                "environment": DeploymentEnvironment.TESTNET.value,
                "blockers": blockers,
                "executionTarget": "HUMMINGBOT_TESTNET_CONNECTOR" if not blockers else None,
                "candidate": candidate.to_dict(),
            }

        if not ctx.capital_authority_approved:
            blockers.append("CAPITAL_AUTHORITY_NOT_APPROVED")
        if not ctx.real_capital_authorized:
            blockers.append("REAL_CAPITAL_NOT_AUTHORIZED")
        blockers.extend(ctx.active_blockers)
        decision = CondorDeploymentDecision.MAINNET_DISPATCH_AUTHORIZED if not blockers else CondorDeploymentDecision.MAINNET_BLOCKED
        return {
            "decision": decision.value,
            "environment": DeploymentEnvironment.MAINNET.value,
            "blockers": sorted(set(blockers)),
            "executionTarget": "HUMMINGBOT_MAINNET_CONNECTOR" if not blockers else None,
            "candidate": candidate.to_dict(),
        }
