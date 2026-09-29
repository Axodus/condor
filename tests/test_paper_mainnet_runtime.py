from __future__ import annotations

from decimal import Decimal

import pytest

from condor.deployment_candidate import (
    CondorDeploymentDecision,
    CondorDeploymentGate,
    CondorOperationalGateContext,
    DeploymentCandidatePayload,
    DeploymentEnvironment,
)
from condor.mainnet_market_data import OrderBookState
from condor.orderflow_orchestrator import (
    ExecutionIntent,
    OrchestratorBlocked,
    STRATEGY_REVISION,
)
from condor.paper_execution import (
    MAKER_FEE_RATE,
    PAPER_FILL_MODEL_VERSION,
    PaperExecutionAdapter,
    PaperExecutionRuntime,
)


def candidate() -> DeploymentCandidatePayload:
    return DeploymentCandidatePayload(
        symbol="BTCUSDT",
        venue="Binance USD-M Futures",
        market_type="USD-M Futures",
        strategy_id="orderflow.absorption.fade",
        strategy_revision=STRATEGY_REVISION,
        strategy_parameter_hash="frozen",
        research_status="VALIDATED_POSITIVE",
        is_evidence_ref={"hash": "is"},
        oos_evidence_ref={"hash": "oos"},
        economic_disposition="ECONOMICALLY_VIABLE",
        instrument_spec_ref="BTCUSDT",
        instrument_spec_hash="spec",
        fee_model_ref="frozen",
        deployment_candidate=True,
        execution_requirements={"post_only": False},
        research_artifact_hashes={"trades": "trades"},
    )


def intent(
    *,
    cell_id: str = "BTCUSDT:orderflow.absorption.fade:freeze-2026-09-24-adapter-v1",
    side: str = "BUY",
    order_type: str = "MARKET",
    price: str | None = None,
    quantity: str = "2",
    constraints: dict | None = None,
) -> ExecutionIntent:
    return ExecutionIntent(
        run_id="paper-run",
        cell_id=cell_id,
        correlation_id=f"corr-{cell_id}",
        jev_decision_id="jev",
        symbol="BTCUSDT",
        action="ENTRY",
        side=side,
        strategy_id="orderflow.absorption.fade",
        strategy_revision=STRATEGY_REVISION,
        quantity=quantity,
        order_type=order_type,
        price=price,
        execution_constraints=constraints or {},
        target_environment="PAPER_MAINNET_DATA",
    )


def runtime_with_book() -> PaperExecutionRuntime:
    runtime = PaperExecutionRuntime()
    runtime.feed.books["BTCUSDT"].apply_snapshot(
        bids=[["99", "10"], ["98", "10"]],
        asks=[["101", "1"], ["102", "10"]],
        last_update_id=10,
    )
    return runtime


def test_paper_gate_routes_to_local_engine():
    decision = CondorDeploymentGate().evaluate(
        candidate(),
        CondorOperationalGateContext(
            environment=DeploymentEnvironment.PAPER_MAINNET_DATA,
            credentials_configured=False,
        ),
    )
    assert decision["decision"] == CondorDeploymentDecision.PAPER_DISPATCH_AUTHORIZED
    assert decision["executionTarget"] == "PAPER_EXECUTION_ENGINE"


def test_mainnet_gate_remains_blocked():
    decision = CondorDeploymentGate().evaluate(
        candidate(),
        CondorOperationalGateContext(environment=DeploymentEnvironment.MAINNET),
    )
    assert decision["decision"] == CondorDeploymentDecision.MAINNET_BLOCKED


@pytest.mark.asyncio
async def test_paper_adapter_rejects_testnet_and_never_calls_remote():
    runtime = runtime_with_book()
    adapter = PaperExecutionAdapter(runtime)
    with pytest.raises(OrchestratorBlocked, match="NON_PAPER"):
        await adapter.dispatch(
            intent(). __class__(**{**intent().__dict__, "target_environment": "TESTNET"}),
            {intent().cell_id},
        )


@pytest.mark.asyncio
async def test_taker_walks_depth_and_models_fee():
    runtime = runtime_with_book()
    result = await PaperExecutionAdapter(runtime).dispatch(
        intent(quantity="2"), {intent().cell_id}
    )
    assert result["status"] == "FILLED"
    assert result["average_price"] == "101.5"
    assert result["levels_consumed"] == 2
    assert result["liquidity_role"] == "TAKER"
    position = runtime.positions_by_cell[intent().cell_id]
    assert position.amount == Decimal("2")
    assert position.total_fees == Decimal("0.1015")


@pytest.mark.asyncio
async def test_taker_partial_fill_when_depth_is_insufficient():
    runtime = runtime_with_book()
    result = await PaperExecutionAdapter(runtime).dispatch(
        intent(quantity="20"), {intent().cell_id}
    )
    assert result["status"] == "PARTIALLY_FILLED"
    assert result["filled_quantity"] == "11"


@pytest.mark.asyncio
async def test_maker_does_not_fill_on_touch_and_fills_after_queue_volume():
    runtime = runtime_with_book()
    order_result = await PaperExecutionAdapter(runtime).dispatch(
        intent(order_type="LIMIT", price="99"), {intent().cell_id}
    )
    assert order_result["status"] == "SUBMITTED"
    assert order_result["liquidity_role"] == "MAKER"
    await runtime.on_trade_print("BTCUSDT", Decimal("99"), Decimal("10"))
    assert not runtime.positions_by_cell
    await runtime.on_trade_print("BTCUSDT", Decimal("99"), Decimal("1"))
    assert runtime.positions_by_cell[intent().cell_id].amount == Decimal("1")
    assert runtime.orders[order_result["order_id"]].fills[0]["commission_provenance"] == "MODELLED"


@pytest.mark.asyncio
async def test_maker_cancel_isolated_between_same_symbol_cells():
    runtime = runtime_with_book()
    adapter = PaperExecutionAdapter(runtime)
    cell_a = "BTCUSDT:orderflow.momentum.aggression:freeze-2026-09-24-adapter-v1"
    cell_b = "BTCUSDT:orderflow.absorption.fade:freeze-2026-09-24-adapter-v1"
    result_a = await adapter.dispatch(intent(cell_id=cell_a, order_type="LIMIT", price="99"), {cell_a, cell_b})
    result_b = await adapter.dispatch(intent(cell_id=cell_b, order_type="LIMIT", price="98"), {cell_a, cell_b})
    await runtime.cancel_all("paper-run")
    assert result_a["order_id"] in runtime.orders
    assert result_b["order_id"] in runtime.orders
    assert runtime.orders[result_a["order_id"]].status == "CANCELLED"
    assert runtime.orders[result_b["order_id"]].status == "CANCELLED"
    assert not runtime.positions_by_cell


def test_fill_model_and_fee_constants_are_frozen():
    assert PAPER_FILL_MODEL_VERSION == "conservative-v1"
    assert MAKER_FEE_RATE == Decimal("0.0002")


def test_order_book_detects_crossed_state_and_requires_rebootstrap():
    book = OrderBookState(symbol="BTCUSDT")
    book.apply_snapshot(
        bids=[["100", "1"]],
        asks=[["101", "1"]],
        last_update_id=100,
    )
    assert not book.is_crossed
    assert not book.rebootstrap_required

    # Apply an invalid crossed update (bid >= ask)
    valid = book.apply_diff_depth(bids=[["102", "1"]], asks=[], first_update_id=101, final_update_id=101)
    assert not valid
    assert book.is_crossed
    assert book.rebootstrap_required


def test_order_book_detects_sequence_gap():
    book = OrderBookState(symbol="BTCUSDT")
    book.apply_snapshot(
        bids=[["100", "1"]],
        asks=[["101", "1"]],
        last_update_id=100,
    )
    # Attempt to apply an update with first_update_id > last_update_id + 1 (gap)
    valid = book.apply_diff_depth(
        bids=[["99", "1"]], asks=[], first_update_id=105, final_update_id=110
    )
    assert not valid
    assert book.sequence_gap
    assert book.rebootstrap_required


@pytest.mark.asyncio
async def test_paper_runtime_flatten_closes_positions_to_zero():
    runtime = runtime_with_book()
    adapter = PaperExecutionAdapter(runtime)
    cell_id = "BTCUSDT:orderflow.absorption.fade:freeze-2026-09-24-adapter-v1"

    # Open long position of 2 BTC
    await adapter.dispatch(intent(quantity="2"), {cell_id})
    assert runtime.positions_by_cell[cell_id].amount == Decimal("2")

    # Flatten
    flat_res = await runtime.flatten("paper-run")
    assert flat_res["flattened_positions"] == 1
    assert runtime.positions_by_cell[cell_id].amount == Decimal("0")
    terminal = await runtime.terminal_state()
    assert len(terminal["positions"]) == 0
    assert len(terminal["open_orders"]) == 0


@pytest.mark.asyncio
async def test_paper_adapter_rejects_mainnet_mutation_intent():
    runtime = runtime_with_book()
    adapter = PaperExecutionAdapter(runtime)
    mainnet_intent = intent()
    # Force intent with forbidden MAINNET
    object.__setattr__(mainnet_intent, "target_environment", "MAINNET")
    with pytest.raises(OrchestratorBlocked, match="NON_PAPER"):
        await adapter.dispatch(mainnet_intent, {mainnet_intent.cell_id})


@pytest.mark.asyncio
async def test_bounded_dispatch_probe_returns_pass_and_flat_terminal_state():
    runtime = runtime_with_book()
    adapter = PaperExecutionAdapter(runtime)
    probe_res = await adapter.bounded_dispatch_probe(symbol="BTCUSDT", quantity="0.001")
    assert probe_res["status"] == "PASS"
    assert probe_res["target_environment"] == "PAPER_MAINNET_DATA"
    assert probe_res["terminal_open_orders"] == 0
    assert probe_res["terminal_positions"] == 0
