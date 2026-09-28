"""Controlled Testnet execution and telemetry roundtrip certification.

Validates AXODUS-TRADING-VAL-CONDOR-HUMMINGBOT-TESTNET-01:
  1. Quants-Lab emits a controlled DeploymentCandidate (BTCUSDT control).
  2. Condor ingests and evaluates operational gates (TESTNET authorized, MAINNET blocked).
  3. Hummingbot execution transport interface simulates a bounded Testnet lifecycle:
     order submission -> exchange ack -> lifecycle telemetry -> controlled cancel/fill -> flatten.
  4. Telemetry roundtrip is ingested back into Quants-Lab comparison harness.
  5. Terminal reconciliation asserts 0 residual open orders and 0 positions.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any

from condor.deployment_candidate import (
    CondorDeploymentDecision,
    CondorDeploymentGate,
    CondorOperationalGateContext,
    DeploymentCandidatePayload,
    DeploymentEnvironment,
)


class HummingbotTestnetExecutionSim:
    """Deterministic simulator for Hummingbot Binance Perpetual Testnet connector lifecycle."""

    def __init__(self, connector_name: str = "binance_perpetual_testnet") -> None:
        self.connector_name = connector_name
        self.active_orders: dict[str, dict[str, Any]] = {}
        self.positions: dict[str, dict[str, Any]] = {}
        self.telemetry_events: list[dict[str, Any]] = []

    def submit_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        post_only: bool = True,
    ) -> dict[str, Any]:
        order_id = f"testnet_hb_{int(time.time() * 1000)}_{len(self.active_orders) + 1}"
        ack = {
            "order_id": order_id,
            "symbol": symbol,
            "connector": self.connector_name,
            "side": side,
            "order_type": order_type,
            "amount": amount,
            "price": price,
            "post_only": post_only,
            "status": "SUBMITTED_AND_ACKNOWLEDGED",
            "timestamp_ms": int(time.time() * 1000),
        }
        self.active_orders[order_id] = ack
        self.telemetry_events.append({"event": "ORDER_ACK", "payload": ack})
        return ack

    def cancel_order(self, order_id: str, reason: str = "TEST_CANCEL_BEFORE_FILL") -> dict[str, Any]:
        if order_id not in self.active_orders:
            raise ValueError(f"order not found: {order_id}")
        event = {
            "order_id": order_id,
            "status": "CANCELLED",
            "reason": reason,
            "timestamp_ms": int(time.time() * 1000),
        }
        del self.active_orders[order_id]
        self.telemetry_events.append({"event": "ORDER_CANCEL", "payload": event})
        return event

    def record_controlled_fill(
        self,
        order_id: str,
        fill_price: float,
        fill_amount: float,
        role: str = "MAKER",
        fee: float = 0.0002,
    ) -> dict[str, Any]:
        if order_id not in self.active_orders:
            raise ValueError(f"order not found: {order_id}")
        order = self.active_orders.pop(order_id)
        fill_event = {
            "fill_id": f"fill_{order_id}",
            "order_id": order_id,
            "symbol": order["symbol"],
            "price": fill_price,
            "amount": fill_amount,
            "role": role,
            "fee": fee,
            "timestamp_ms": int(time.time() * 1000),
        }
        self.positions[order["symbol"]] = {
            "symbol": order["symbol"],
            "amount": fill_amount if order["side"] == "BUY" else -fill_amount,
            "entry_price": fill_price,
        }
        self.telemetry_events.append({"event": "FILL", "payload": fill_event})
        return fill_event

    def flatten_position(self, symbol: str) -> dict[str, Any]:
        if symbol not in self.positions:
            return {"status": "ALREADY_FLAT", "symbol": symbol}
        pos = self.positions.pop(symbol)
        event = {
            "event": "POSITION_FLATTENED",
            "symbol": symbol,
            "closed_amount": pos["amount"],
            "timestamp_ms": int(time.time() * 1000),
        }
        self.telemetry_events.append({"event": "FLATTEN", "payload": event})
        return event


def test_end_to_end_testnet_execution_and_telemetry_roundtrip(tmp_path):
    # 1. Candidate Emission (Quants-Lab contract shape)
    candidate = DeploymentCandidatePayload(
        symbol="BTCUSDT",
        venue="Binance USD-M Futures",
        market_type="USD-M Futures",
        strategy_id="orderflow.momentum.aggression",
        strategy_revision="freeze-2026-09-24-adapter-v1",
        strategy_parameter_hash="param_hash_btc_ctrl",
        research_status="VALIDATED_POSITIVE",
        is_evidence_ref={"trade_count": 139},
        oos_evidence_ref={"trade_count": 50, "net_pnl": "0.00"},
        economic_disposition="CONTROL_SAMPLE",
        instrument_spec_ref="BTCUSDT",
        instrument_spec_hash="spec_hash_btc",
        fee_model_ref="maker=0.0002 taker=0.0005",
        deployment_candidate=True,
        execution_requirements={"maker_taker": "TAKER", "post_only": False},
        research_artifact_hashes={"signals": "sig_hash_btc"},
    )

    # 2. Condor Ingestion & Operational Gate
    gate = CondorDeploymentGate()
    testnet_decision = gate.evaluate(
        candidate, CondorOperationalGateContext(environment=DeploymentEnvironment.TESTNET)
    )
    assert testnet_decision["decision"] == CondorDeploymentDecision.TESTNET_DISPATCH_AUTHORIZED.value
    assert testnet_decision["executionTarget"] == "HUMMINGBOT_TESTNET_CONNECTOR"

    # Mainnet blocked check
    mainnet_decision = gate.evaluate(
        candidate, CondorOperationalGateContext(environment=DeploymentEnvironment.MAINNET)
    )
    assert mainnet_decision["decision"] == CondorDeploymentDecision.MAINNET_BLOCKED.value
    assert "DEBT-AUD-A-03" in mainnet_decision["blockers"]

    # 3. Hummingbot Testnet Execution Lifecycle
    hb_exec = HummingbotTestnetExecutionSim()
    order = hb_exec.submit_order(symbol="BTCUSDT", side="BUY", order_type="LIMIT", amount=0.001, price=65000.0, post_only=True)
    assert len(hb_exec.active_orders) == 1

    # Cancel before fill
    cancel = hb_exec.cancel_order(order["order_id"])
    assert cancel["status"] == "CANCELLED"
    assert len(hb_exec.active_orders) == 0

    # Submit second order & simulate controlled fill
    order2 = hb_exec.submit_order(symbol="BTCUSDT", side="BUY", order_type="LIMIT", amount=0.001, price=65000.0, post_only=True)
    fill = hb_exec.record_controlled_fill(order2["order_id"], fill_price=65000.0, fill_amount=0.001, role="MAKER", fee=0.0002)
    assert fill["role"] == "MAKER"
    assert len(hb_exec.active_orders) == 0
    assert "BTCUSDT" in hb_exec.positions

    # Flatten
    flatten = hb_exec.flatten_position("BTCUSDT")
    assert flatten["event"] == "POSITION_FLATTENED"
    assert len(hb_exec.positions) == 0

    # 4. Telemetry Export & Ingestion by Quants-Lab Runner
    import sys
    sys.path.insert(0, "/opt/Axodus/Trading/quants-lab")
    from research_notebooks.orderflow_backtest.testnet_orderflow_live_runner import (
        Dedicated24hTestnetRunner,
    )

    quants_runner = Dedicated24hTestnetRunner(output_root=tmp_path)
    quants_runner.start()

    # Ingest lifecycle events into Quants-Lab cell
    for ev in hb_exec.telemetry_events:
        event_type = ev["event"]
        payload = ev["payload"]
        if event_type == "ORDER_ACK":
            quants_runner.ingest_event("BTCUSDT", "orderflow.momentum.aggression", "ORDER", payload, timestamp_ms=payload["timestamp_ms"])
        elif event_type == "ORDER_CANCEL":
            quants_runner.ingest_event("BTCUSDT", "orderflow.momentum.aggression", "CANCEL", payload, timestamp_ms=payload["timestamp_ms"])
        elif event_type == "FILL":
            quants_runner.ingest_event("BTCUSDT", "orderflow.momentum.aggression", "FILL", payload, timestamp_ms=payload["timestamp_ms"])
        elif event_type == "FLATTEN":
            quants_runner.ingest_event("BTCUSDT", "orderflow.momentum.aggression", "TRADE", {"tradeId": "flat_trade", "netPnl": "0.0"}, timestamp_ms=payload["timestamp_ms"])

    btc_cell = quants_runner.cells["BTCUSDT:orderflow.momentum.aggression:freeze-2026-09-24-adapter-v1"]
    summary = btc_cell.summary()
    assert summary["orders"] == 3  # 2 orders + 1 cancel record
    assert summary["fills"] == 1
    assert summary["trades"] == 1
    assert not summary["openOrder"]
    assert not summary["activePosition"]

    # 5. Terminal Invariants
    assert len(hb_exec.active_orders) == 0
    assert len(hb_exec.positions) == 0
