from __future__ import annotations

from dotenv import load_dotenv
"""Jev qualification contracts and client interface.

Jev evaluates and qualifies/rejects candidate strategy triggers. Jev does not
own exchange credentials, order execution, position ledger, or deployment
authority. Condor and Hummingbot remain the authorization and execution owners.
"""

import asyncio
import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import aiohttp
from pathlib import Path

# Load local environment if available
for _env_path in (Path('.env.local'), Path('../.env.local'), Path('../../.env.local')):
    if _env_path.exists():
        load_dotenv(_env_path)
        break


class JevDecisionType(StrEnum):
    NO_ACTION = "NO_ACTION"
    SIGNAL_REJECTED = "SIGNAL_REJECTED"
    SIGNAL_QUALIFIED = "SIGNAL_QUALIFIED"


class JevActionType(StrEnum):
    NONE = "NONE"
    ENTRY = "ENTRY"
    EXIT = "EXIT"


class MicrotrendJevReasonCode(StrEnum):
    """Backward-compatible reason-code vocabulary for Microtrend v3.

    The outer JevDecisionV1 enum remains the established contract. These codes
    only qualify the outcome for the Microtrend pullback path and do not alter
    orderflow-strategy semantics or grant execution authority.
    """

    CONTINUATION = "MICROTREND_CONTINUATION"
    NEUTRAL = "MICROTREND_NEUTRAL"
    REVERSAL_RISK = "MICROTREND_REVERSAL_RISK"
    INVALID = "MICROTREND_INVALID"


class JevValidationError(ValueError):
    """Raised when Jev candidate trigger or decision contract fails validation."""


@dataclass(frozen=True)
class CandidateTriggerV1:
    """Deterministic strategy trigger candidate submitted to Jev for qualification."""

    schema_version: str
    run_id: str
    cell_id: str
    symbol: str
    venue: str
    market_type: str
    strategy_id: str
    strategy_revision: str
    trigger_type: str  # ENTRY | EXIT
    side: str  # BUY | SELL
    event_time: float
    market_state_ref: dict[str, Any]
    feature_snapshot: dict[str, Any]
    trigger_evidence: dict[str, Any]
    current_logical_position: dict[str, Any] | None
    current_open_orders: list[dict[str, Any]]
    deployment_candidate_id: str
    candidate_trigger_id: str
    correlation_id: str

    def validate(self) -> None:
        if self.schema_version != "v1":
            raise JevValidationError(
                f"unsupported schema_version: {self.schema_version}"
            )
        if not self.run_id or not self.cell_id or not self.candidate_trigger_id:
            raise JevValidationError(
                "run_id, cell_id, and candidate_trigger_id are required"
            )
        if self.trigger_type not in ("ENTRY", "EXIT"):
            raise JevValidationError(f"invalid trigger_type: {self.trigger_type}")
        if self.side not in ("BUY", "SELL"):
            raise JevValidationError(f"invalid side: {self.side}")
        if not self.strategy_id or not self.strategy_revision:
            raise JevValidationError("strategy_id and strategy_revision are required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateTriggerV1:
        candidate = cls(
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        )
        candidate.validate()
        return candidate


@dataclass(frozen=True)
class JevDecisionV1:
    """Typed deterministic qualification returned by Jev."""

    schema_version: str
    decision_id: str
    run_id: str
    cell_id: str
    candidate_trigger_id: str
    correlation_id: str
    strategy_id: str
    strategy_revision: str
    decision: str  # NO_ACTION | SIGNAL_REJECTED | SIGNAL_QUALIFIED
    action: str  # NONE | ENTRY | EXIT
    side: str  # BUY | SELL | NONE
    reason_codes: list[str]
    input_evidence_refs: dict[str, Any]
    market_state_hash: str
    decided_at: float

    def validate(self) -> None:
        if self.schema_version != "v1":
            raise JevValidationError(
                f"unsupported schema_version: {self.schema_version}"
            )
        if self.decision not in (
            JevDecisionType.NO_ACTION.value,
            JevDecisionType.SIGNAL_REJECTED.value,
            JevDecisionType.SIGNAL_QUALIFIED.value,
        ):
            raise JevValidationError(f"invalid decision: {self.decision}")
        if self.action not in (
            JevActionType.NONE.value,
            JevActionType.ENTRY.value,
            JevActionType.EXIT.value,
        ):
            raise JevValidationError(f"invalid action: {self.action}")
        if (
            self.decision == JevDecisionType.SIGNAL_REJECTED.value
            and not self.reason_codes
        ):
            raise JevValidationError(
                "reason_codes must not be empty on SIGNAL_REJECTED"
            )
        if (
            self.decision == JevDecisionType.NO_ACTION.value
            and self.action != JevActionType.NONE.value
        ):
            raise JevValidationError("action must be NONE on NO_ACTION")
        if (
            self.decision == JevDecisionType.SIGNAL_QUALIFIED.value
            and self.action == JevActionType.NONE.value
        ):
            raise JevValidationError("action must not be NONE on SIGNAL_QUALIFIED")
        if (
            self.decision == JevDecisionType.SIGNAL_QUALIFIED.value
            and not self.reason_codes
        ):
            raise JevValidationError(
                "reason_codes must not be empty on SIGNAL_QUALIFIED"
            )
        if (
            self.decision != JevDecisionType.SIGNAL_QUALIFIED.value
            and self.side != "NONE"
        ):
            raise JevValidationError("side must be NONE unless the signal is qualified")
        if not self.decision_id or not self.candidate_trigger_id:
            raise JevValidationError(
                "decision_id and candidate_trigger_id are required"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JevDecisionV1:
        decision = cls(
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        )
        decision.validate()
        return decision


def microtrend_outcome(decision: JevDecisionV1) -> MicrotrendJevReasonCode:
    """Validate and classify a Microtrend-v3 Jev outcome from its reason code.

    A non-Microtrend decision, mismatched outer decision, action, or side is
    invalid. This keeps the generic Jev contract stable while providing a
    deterministic adapter boundary for Microtrend consumers.
    """

    decision.validate()
    reasons = set(decision.reason_codes)
    mappings = (
        (MicrotrendJevReasonCode.CONTINUATION, JevDecisionType.SIGNAL_QUALIFIED.value, JevActionType.ENTRY.value),
        (MicrotrendJevReasonCode.NEUTRAL, JevDecisionType.NO_ACTION.value, JevActionType.NONE.value),
        (MicrotrendJevReasonCode.REVERSAL_RISK, JevDecisionType.SIGNAL_REJECTED.value, JevActionType.NONE.value),
        (MicrotrendJevReasonCode.INVALID, JevDecisionType.SIGNAL_REJECTED.value, JevActionType.NONE.value),
    )
    for outcome, expected_decision, expected_action in mappings:
        if outcome.value not in reasons:
            continue
        if decision.decision != expected_decision or decision.action != expected_action:
            raise JevValidationError(f"MICROTREND_REASON_CONTRACT_INVALID:{outcome.value}")
        if outcome is MicrotrendJevReasonCode.CONTINUATION:
            if decision.side not in {"BUY", "SELL"}:
                raise JevValidationError("MICROTREND_CONTINUATION_SIDE_INVALID")
        elif decision.side != "NONE":
            raise JevValidationError(f"MICROTREND_REASON_SIDE_INVALID:{outcome.value}")
        return outcome
    raise JevValidationError("MICROTREND_REASON_CODE_MISSING")


class JevQualificationClient(Protocol):
    """Protocol for Jev qualification service."""

    async def qualify(self, candidate: CandidateTriggerV1) -> JevDecisionV1: ...
    async def health(self) -> dict[str, Any]: ...

    async def close(self) -> None: ...


_ENV_DEFAULT = object()


class RealJevClient:
    """HTTP client for the canonical TypeSafe Jev runtime.

    The wire protocol is the existing System One contract used by Trading
    Core.  This client only evaluates a typed candidate; it has no exchange
    credentials and no execution capabilities.
    """

    contract_version = "v1"

    def __init__(
        self,
        *,
        api_key: str | None | object = _ENV_DEFAULT,
        endpoint: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        health_endpoint: str | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.api_key = (
            (
                os.getenv("TRADING_TYPESAFE_API_KEY")
                or os.getenv("TYPESAFE_API_KEY")
                or os.getenv("TYPESAFE_AI_API_KEY")
            )
            if api_key is _ENV_DEFAULT
            else api_key
        )
        self.endpoint = (
            endpoint
            or os.getenv("TRADING_TYPESAFE_API_ENDPOINT")
            or os.getenv("TYPESAFE_API_ENDPOINT")
            or "https://api.typesafe.ai/v1/systemone"
        )
        self.model = (
            model
            or os.getenv("TRADING_TYPESAFE_MODEL")
            or os.getenv("TYPESAFE_MODEL")
            or "jev-1.13.0"
        )
        raw_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else float(os.getenv("TRADING_TYPESAFE_API_TIMEOUT_MS", "5000")) / 1000
        )
        self.timeout_seconds = max(0.1, raw_timeout)
        self.health_endpoint = (
            health_endpoint
            or os.getenv("TRADING_TYPESAFE_HEALTH_ENDPOINT")
            or f"{self.endpoint.rstrip('/')}/health"
        )
        self._session = session
        self._owns_session = session is None
        self.last_call: dict[str, Any] | None = None

    @classmethod
    def from_environment(cls) -> "RealJevClient":
        return cls()

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("JEV_CREDENTIAL_NOT_CONFIGURED")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if (
            self._owns_session
            and self._session is not None
            and not self._session.closed
        ):
            await self._session.close()

    async def health(self) -> dict[str, Any]:
        started = time.monotonic()
        if not self.api_key:
            return {
                "status": "UNAVAILABLE",
                "provider": "typesafe-jev",
                "contract_version": self.contract_version,
                "reason": "JEV_CREDENTIAL_NOT_CONFIGURED",
            }
        session = await self._get_session()
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            async with session.get(
                self.health_endpoint, headers=self._headers(), timeout=timeout
            ) as response:
                if response.status < 400:
                    payload = await _safe_json_async(response)
                    return {
                        "status": "HEALTHY",
                        "provider": "typesafe-jev",
                        "contract_version": self.contract_version,
                        "runtime_version": _runtime_version(payload, self.model),
                        "latency_ms": round((time.monotonic() - started) * 1000, 3),
                    }
                if response.status not in (404, 405):
                    return {
                        "status": "UNAVAILABLE",
                        "provider": "typesafe-jev",
                        "contract_version": self.contract_version,
                        "reason": f"JEV_HEALTH_HTTP_{response.status}",
                    }
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass

        # The System One endpoint is the canonical authority in Trading Core;
        # installations without /health are probed with a non-executing call.
        probe = _health_probe_candidate()
        try:
            decision = await self.qualify(probe)
            decision.validate()
            return {
                "status": "HEALTHY",
                "provider": "typesafe-jev",
                "contract_version": self.contract_version,
                "runtime_version": self.model,
                "latency_ms": round((time.monotonic() - started) * 1000, 3),
                "decision_probe": "PASS",
            }
        except Exception as exc:
            return {
                "status": "UNAVAILABLE",
                "provider": "typesafe-jev",
                "contract_version": self.contract_version,
                "reason": type(exc).__name__,
            }

    async def decision_probe(self) -> dict[str, Any]:
        candidate = _health_probe_candidate()
        decision = await self.qualify(candidate)
        decision.validate()
        return {
            "status": "PASS",
            "schema_version": decision.schema_version,
            "decision": decision.decision,
            "execution": "NONE",
            "runtime_version": (self.last_call or {}).get(
                "runtime_version", self.model
            ),
        }

    async def qualify(self, candidate: CandidateTriggerV1) -> JevDecisionV1:
        candidate.validate()
        request_id = f"jev-request:{uuid.uuid4().hex}"
        started = time.monotonic()
        payload = {
            "state": {
                "candidate_trigger": candidate.to_dict(),
                "contract": "CandidateTriggerV1->JevDecisionV1",
                "schema_version": self.contract_version,
            },
            "model": self.model,
            "questions": {
                "qualification_decision": {
                    "type": "choice",
                    "instructions": "Return the typed qualification decision for this frozen strategy candidate. Do not authorize execution.",
                    "criteria": {
                        "qualified": "SIGNAL_QUALIFIED",
                        "rejected": "SIGNAL_REJECTED",
                        "no_action": "NO_ACTION",
                    },
                },
                "reason_code": {
                    "type": "choice",
                    "instructions": "Return one stable reason code for the decision.",
                    "criteria": {
                        "causal_evidence": "Candidate evidence is causally sufficient",
                        "insufficient_evidence": "Evidence is insufficient",
                        "regime_mismatch": "Observed regime does not qualify",
                        "no_actionable_trigger": "No actionable trigger",
                    },
                },
            },
        }
        session = await self._get_session()
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            async with session.post(
                self.endpoint, headers=self._headers(), json=payload, timeout=timeout
            ) as response:
                if not response.ok:
                    raise RuntimeError(f"JEV_HTTP_{response.status}")
                wire = await response.json()
            decision = _decision_from_wire(wire, candidate, self.model)
            decision.validate()
            self.last_call = {
                "request_id": request_id,
                "runtime_version": _runtime_version(wire, self.model),
                "latency_ms": round((time.monotonic() - started) * 1000, 3),
                "decision": decision.decision,
                "candidate_trigger_id": candidate.candidate_trigger_id,
            }
            return decision
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise RuntimeError("JEV_RUNTIME_UNAVAILABLE") from exc


async def _safe_json_async(response: aiohttp.ClientResponse) -> dict[str, Any]:
    try:
        value = await response.json()
        return value if isinstance(value, dict) else {}
    except (ValueError, aiohttp.ContentTypeError):
        return {}


def _runtime_version(payload: Any, default: str) -> str:
    return (
        str(
            payload.get("runtime_version")
            or payload.get("version")
            or payload.get("model")
            or default
        )
        if isinstance(payload, dict)
        else default
    )


def _answer_value(answer: Any, key: str) -> str:
    if not isinstance(answer, dict):
        raise JevValidationError(f"JEV_ANSWER_INVALID:{key}")
    value = answer.get("choice") or answer.get("value") or answer.get("noul")
    if not isinstance(value, str):
        raise JevValidationError(f"JEV_ANSWER_INVALID:{key}")
    return value


def _decision_from_wire(
    payload: Any, candidate: CandidateTriggerV1, model: str
) -> JevDecisionV1:
    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
        raise JevValidationError("JEV_RESPONSE_SHAPE_INVALID")
    answers = payload["answers"]
    raw_decision = _answer_value(
        answers.get("qualification_decision"), "qualification_decision"
    )
    raw_reason = _answer_value(answers.get("reason_code"), "reason_code")
    mapping = {
        "qualified": JevDecisionType.SIGNAL_QUALIFIED.value,
        "rejected": JevDecisionType.SIGNAL_REJECTED.value,
        "no_action": JevDecisionType.NO_ACTION.value,
    }
    decision = mapping.get(raw_decision, raw_decision)
    if decision not in {item.value for item in JevDecisionType}:
        raise JevValidationError(f"JEV_DECISION_INVALID:{decision}")
    action = (
        candidate.trigger_type
        if decision == JevDecisionType.SIGNAL_QUALIFIED.value
        else JevActionType.NONE.value
    )
    side = (
        candidate.side if decision == JevDecisionType.SIGNAL_QUALIFIED.value else "NONE"
    )
    reasons = [raw_reason] if raw_reason else ["NO_ACTIONABLE_TRIGGER"]
    state_hash = hashlib.sha256(
        json.dumps(
            candidate.feature_snapshot, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return JevDecisionV1(
        schema_version="v1",
        decision_id=f"jev_dec_{uuid.uuid4().hex[:12]}",
        run_id=candidate.run_id,
        cell_id=candidate.cell_id,
        candidate_trigger_id=candidate.candidate_trigger_id,
        correlation_id=candidate.correlation_id,
        strategy_id=candidate.strategy_id,
        strategy_revision=candidate.strategy_revision,
        decision=decision,
        action=action,
        side=side,
        reason_codes=reasons,
        input_evidence_refs={
            "candidate_trigger_id": candidate.candidate_trigger_id,
            "model": model,
        },
        market_state_hash=state_hash,
        decided_at=time.time(),
    )


def _health_probe_candidate() -> CandidateTriggerV1:
    return CandidateTriggerV1(
        schema_version="v1",
        run_id="health-probe",
        cell_id="health-probe",
        symbol="BTCUSDT",
        venue="Binance USD-M Futures",
        market_type="USD-M Futures",
        strategy_id="health-probe",
        strategy_revision="freeze-2026-09-24-adapter-v1",
        trigger_type="ENTRY",
        side="BUY",
        event_time=time.time(),
        market_state_ref={"probe": True},
        feature_snapshot={"probe": True},
        trigger_evidence={"probe": True},
        current_logical_position=None,
        current_open_orders=[],
        deployment_candidate_id="health-probe",
        candidate_trigger_id=f"health-probe-{uuid.uuid4().hex}",
        correlation_id=f"health-probe-{uuid.uuid4().hex}",
    )


class DeterministicMockJevClient:
    """Deterministic qualification provider for testing and verification."""

    def __init__(
        self,
        default_decision: str = JevDecisionType.SIGNAL_QUALIFIED.value,
        rejection_reason: str = "REGIME_VOLATILITY_MISMATCH",
    ) -> None:
        self.default_decision = default_decision
        self.rejection_reason = rejection_reason
        self.is_healthy = True
        self.evaluations_count = 0

    async def health(self) -> dict[str, Any]:
        return {
            "status": "HEALTHY" if self.is_healthy else "UNAVAILABLE",
            "provider": "deterministic-mock-jev",
        }

    async def qualify(self, candidate: CandidateTriggerV1) -> JevDecisionV1:
        if not self.is_healthy:
            raise RuntimeError("JEV_SERVICE_UNAVAILABLE")
        candidate.validate()
        self.evaluations_count += 1
        state_hash = hashlib.sha256(
            json.dumps(candidate.feature_snapshot, sort_keys=True).encode()
        ).hexdigest()
        decision_id = f"jev_dec_{uuid.uuid4().hex[:12]}"

        if self.default_decision == JevDecisionType.SIGNAL_QUALIFIED.value:
            return JevDecisionV1(
                schema_version="v1",
                decision_id=decision_id,
                run_id=candidate.run_id,
                cell_id=candidate.cell_id,
                candidate_trigger_id=candidate.candidate_trigger_id,
                correlation_id=candidate.correlation_id,
                strategy_id=candidate.strategy_id,
                strategy_revision=candidate.strategy_revision,
                decision=JevDecisionType.SIGNAL_QUALIFIED.value,
                action=candidate.trigger_type,
                side=candidate.side,
                reason_codes=["QUALIFIED_MICROSTRUCTURE_PRESSURE"],
                input_evidence_refs={"trigger_evidence": candidate.trigger_evidence},
                market_state_hash=state_hash,
                decided_at=time.time(),
            )
        elif self.default_decision == JevDecisionType.SIGNAL_REJECTED.value:
            return JevDecisionV1(
                schema_version="v1",
                decision_id=decision_id,
                run_id=candidate.run_id,
                cell_id=candidate.cell_id,
                candidate_trigger_id=candidate.candidate_trigger_id,
                correlation_id=candidate.correlation_id,
                strategy_id=candidate.strategy_id,
                strategy_revision=candidate.strategy_revision,
                decision=JevDecisionType.SIGNAL_REJECTED.value,
                action=JevActionType.NONE.value,
                side="NONE",
                reason_codes=[self.rejection_reason],
                input_evidence_refs={"trigger_evidence": candidate.trigger_evidence},
                market_state_hash=state_hash,
                decided_at=time.time(),
            )
        else:
            return JevDecisionV1(
                schema_version="v1",
                decision_id=decision_id,
                run_id=candidate.run_id,
                cell_id=candidate.cell_id,
                candidate_trigger_id=candidate.candidate_trigger_id,
                correlation_id=candidate.correlation_id,
                strategy_id=candidate.strategy_id,
                strategy_revision=candidate.strategy_revision,
                decision=JevDecisionType.NO_ACTION.value,
                action=JevActionType.NONE.value,
                side="NONE",
                reason_codes=["NO_ACTIONABLE_TRIGGER"],
                input_evidence_refs={},
                market_state_hash=state_hash,
                decided_at=time.time(),
            )
