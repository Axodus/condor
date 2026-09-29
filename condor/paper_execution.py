"""Deterministic Paper Execution Engine & Adapter for Mainnet Market Data.

Provides local simulated execution against Binance Futures Mainnet public market
data. Remote order submission, remote cancellation, and exchange credentials are
impossible in this runtime.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import asdict, dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, AsyncIterator

from condor.mainnet_market_data import BinanceMainnetPublicMarketFeed, OrderBookState
from condor.orderflow_orchestrator import (
    STRATEGY_REVISION,
    SYMBOLS,
    ExecutionIntent,
    JsonlSink,
    OrchestratorBlocked,
)

PAPER_FILL_MODEL_VERSION = "conservative-v1"
MAKER_FEE_RATE = Decimal("0.0002")  # 0.0200%
TAKER_FEE_RATE = Decimal("0.0005")  # 0.0500%


def _round_decimal(val: Decimal, places: int = 8) -> Decimal:
    return val.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP)


@dataclass
class PaperOrder:
    order_id: str
    client_order_id: str
    run_id: str
    cell_id: str
    symbol: str
    side: str  # BUY | SELL
    order_type: str  # MARKET | LIMIT
    requested_quantity: Decimal
    remaining_quantity: Decimal
    filled_quantity: Decimal = Decimal("0")
    price: Decimal | None = None
    post_only: bool = False
    status: str = "SUBMITTED"  # SUBMITTED | PARTIALLY_FILLED | FILLED | CANCELLED | REJECTED
    liquidity_role: str = "UNKNOWN"  # TAKER | MAKER | UNKNOWN
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    queue_ahead_volume: Decimal = Decimal("0")
    cumulative_quote: Decimal = Decimal("0")
    cumulative_fee: Decimal = Decimal("0")
    fills: list[dict[str, Any]] = field(default_factory=list)

    @property
    def average_price(self) -> Decimal | None:
        if self.filled_quantity > 0:
            return self.cumulative_quote / self.filled_quantity
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "run_id": self.run_id,
            "cell_id": self.cell_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "requested_quantity": str(self.requested_quantity),
            "remaining_quantity": str(self.remaining_quantity),
            "filled_quantity": str(self.filled_quantity),
            "price": str(self.price) if self.price is not None else None,
            "post_only": self.post_only,
            "status": self.status,
            "liquidity_role": self.liquidity_role,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "queue_ahead_volume": str(self.queue_ahead_volume),
            "cumulative_quote": str(self.cumulative_quote),
            "cumulative_fee": str(self.cumulative_fee),
            "average_price": str(self.average_price) if self.average_price else None,
            "fills_count": len(self.fills),
            "execution_mode": "PAPER",
            "execution_environment": "PAPER_MAINNET_DATA",
            "provenance": "PAPER_EXECUTION_ENGINE",
        }


@dataclass
class PaperPosition:
    run_id: str
    cell_id: str
    symbol: str
    amount: Decimal = Decimal("0")  # >0 Long, <0 Short
    entry_price: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    total_fees: Decimal = Decimal("0")
    updated_at: float = field(default_factory=time.time)

    def update_with_fill(
        self,
        side: str,
        fill_qty: Decimal,
        fill_price: Decimal,
        fee: Decimal,
        now: float | None = None,
    ) -> Decimal:
        """Update position on fill and return realized PnL delta."""
        self.updated_at = now or time.time()
        self.total_fees += fee
        realized_delta = Decimal("0")

        if side == "BUY":
            if self.amount >= 0:
                # Increasing Long
                total_qty = self.amount + fill_qty
                if total_qty > 0:
                    self.entry_price = (
                        (self.amount * self.entry_price) + (fill_qty * fill_price)
                    ) / total_qty
                self.amount = total_qty
            else:
                # Closing / Reducing Short
                closed_qty = min(abs(self.amount), fill_qty)
                realized_delta = (self.entry_price - fill_price) * closed_qty
                self.realized_pnl += realized_delta
                new_amount = self.amount + fill_qty
                if new_amount > 0:
                    self.entry_price = fill_price
                elif new_amount == 0:
                    self.entry_price = Decimal("0")
                self.amount = new_amount
        else:  # SELL
            if self.amount <= 0:
                # Increasing Short
                total_qty = abs(self.amount) + fill_qty
                if total_qty > 0:
                    self.entry_price = (
                        (abs(self.amount) * self.entry_price) + (fill_qty * fill_price)
                    ) / total_qty
                self.amount = -total_qty
            else:
                # Closing / Reducing Long
                closed_qty = min(self.amount, fill_qty)
                realized_delta = (fill_price - self.entry_price) * closed_qty
                self.realized_pnl += realized_delta
                new_amount = self.amount - fill_qty
                if new_amount < 0:
                    self.entry_price = fill_price
                elif new_amount == 0:
                    self.entry_price = Decimal("0")
                self.amount = new_amount

        return realized_delta

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "cell_id": self.cell_id,
            "symbol": self.symbol,
            "amount": str(self.amount),
            "entry_price": str(self.entry_price),
            "realized_pnl": str(self.realized_pnl),
            "total_fees": str(self.total_fees),
            "net_pnl": str(self.realized_pnl - self.total_fees),
            "updated_at": self.updated_at,
            "execution_mode": "PAPER",
            "execution_environment": "PAPER_MAINNET_DATA",
            "provenance": "PAPER_EXECUTION_ENGINE",
        }


class PaperExecutionRuntime:
    """Local paper execution runtime with depth-walking taker & volume-queued maker models."""

    def __init__(
        self,
        feed: BinanceMainnetPublicMarketFeed | None = None,
        *,
        sink: JsonlSink | None = None,
    ) -> None:
        self.feed = feed or BinanceMainnetPublicMarketFeed()
        self.sink = sink
        self.orders: dict[str, PaperOrder] = {}
        self.open_orders_by_cell: dict[str, list[PaperOrder]] = {}
        self.positions_by_cell: dict[str, PaperPosition] = {}
        self.execution_events: list[dict[str, Any]] = []

    def get_order_book(self, symbol: str) -> OrderBookState:
        return self.feed.books.setdefault(symbol, OrderBookState(symbol=symbol))

    def get_position(self, run_id: str, cell_id: str, symbol: str) -> PaperPosition:
        if cell_id not in self.positions_by_cell:
            self.positions_by_cell[cell_id] = PaperPosition(
                run_id=run_id, cell_id=cell_id, symbol=symbol
            )
        return self.positions_by_cell[cell_id]

    async def preflight(self, symbols: tuple[str, ...] = SYMBOLS) -> dict[str, Any]:
        feed_preflight = await self.feed.preflight(symbols)
        return {
            "execution_mode": "PAPER",
            "execution_environment": "PAPER_MAINNET_DATA",
            "market_data_environment": "MAINNET",
            "paper_fill_model_version": PAPER_FILL_MODEL_VERSION,
            "remote_trading_enabled": False,
            "mainnet": "REJECTED",
            "open_orders": [],
            "positions": [],
            "feed": feed_preflight,
            "symbols": list(symbols),
        }

    async def dispatch_intent(self, intent: ExecutionIntent) -> dict[str, Any]:
        if intent.target_environment != "PAPER_MAINNET_DATA":
            raise OrchestratorBlocked(
                f"NON_PAPER_INTENT_REJECTED:{intent.target_environment}"
            )
        if intent.symbol not in SYMBOLS:
            raise OrchestratorBlocked(f"UNAPPROVED_SYMBOL:{intent.symbol}")

        book = self.get_order_book(intent.symbol)
        now = time.time()
        order_id = f"paper_ord_{uuid.uuid4().hex[:12]}"
        req_qty = Decimal(str(intent.quantity))
        limit_price = Decimal(str(intent.price)) if intent.price else None
        post_only = bool(intent.execution_constraints.get("post_only", False))

        order = PaperOrder(
            order_id=order_id,
            client_order_id=intent.correlation_id,
            run_id=intent.run_id,
            cell_id=intent.cell_id,
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            requested_quantity=req_qty,
            remaining_quantity=req_qty,
            price=limit_price,
            post_only=post_only,
            created_at=now,
            updated_at=now,
        )
        self.orders[order_id] = order

        # Determine whether order is Marketable / Taker
        is_marketable = False
        if intent.order_type == "MARKET":
            is_marketable = True
        elif intent.order_type == "LIMIT" and limit_price is not None:
            if intent.side == "BUY" and book.best_ask is not None and limit_price >= book.best_ask:
                is_marketable = True
            elif intent.side == "SELL" and book.best_bid is not None and limit_price <= book.best_bid:
                is_marketable = True

        if is_marketable and post_only:
            order.status = "REJECTED"
            order.updated_at = now
            record = {
                "event_type": "ORDER_REJECTED",
                "reason": "POST_ONLY_RESTING_VIOLATION",
                "order": order.to_dict(),
                "observed_at": now,
            }
            self.execution_events.append(record)
            if self.sink is not None:
                await self.sink.write("execution_events", record)
            return {"status": "REJECTED", "reason": "POST_ONLY_RESTING_VIOLATION", "order_id": order_id}

        if is_marketable:
            return await self._execute_taker_order(order, book, now)

        # Otherwise, resting Limit Order (Maker candidate)
        return await self._register_maker_order(order, book, now)

    async def _execute_taker_order(
        self, order: PaperOrder, book: OrderBookState, now: float
    ) -> dict[str, Any]:
        """Walk order book levels deterministically for taker fill."""
        order.liquidity_role = "TAKER"
        levels = book.top_asks(50) if order.side == "BUY" else book.top_bids(50)
        pos = self.get_position(order.run_id, order.cell_id, order.symbol)

        filled_notional = Decimal("0")
        levels_consumed = 0

        for p, q_avail in levels:
            if order.remaining_quantity <= 0:
                break
            if order.price is not None:
                if order.side == "BUY" and p > order.price:
                    break
                if order.side == "SELL" and p < order.price:
                    break

            take_qty = min(order.remaining_quantity, q_avail)
            if take_qty <= 0:
                continue

            fill_id = f"fill_{order.order_id}_{len(order.fills)+1}"
            notional = take_qty * p
            fee = _round_decimal(notional * TAKER_FEE_RATE, 8)

            order.filled_quantity += take_qty
            order.remaining_quantity -= take_qty
            order.cumulative_quote += notional
            order.cumulative_fee += fee
            filled_notional += notional
            levels_consumed += 1

            realized_pnl_delta = pos.update_with_fill(
                order.side, take_qty, p, fee, now=now
            )

            fill_record = {
                "fill_id": fill_id,
                "order_id": order.order_id,
                "client_order_id": order.client_order_id,
                "cell_id": order.cell_id,
                "symbol": order.symbol,
                "side": order.side,
                "price": str(p),
                "quantity": str(take_qty),
                "notional": str(notional),
                "liquidity_role": "TAKER",
                "fee": str(fee),
                "fee_rate": str(TAKER_FEE_RATE),
                "commission_provenance": "MODELLED",
                "realized_pnl_delta": str(realized_pnl_delta),
                "timestamp": now,
                "execution_mode": "PAPER",
                "execution_environment": "PAPER_MAINNET_DATA",
                "provenance": "PAPER_EXECUTION_ENGINE",
            }
            order.fills.append(fill_record)
            if self.sink is not None:
                await self.sink.write("fills", fill_record)

        if order.remaining_quantity == 0:
            order.status = "FILLED"
        elif order.filled_quantity > 0:
            order.status = "PARTIALLY_FILLED"
        else:
            order.status = "REJECTED"  # Insufficient book depth

        order.updated_at = now
        ack_event = {
            "event_type": "ORDER_FILLED" if order.status == "FILLED" else "ORDER_PARTIALLY_FILLED",
            "order": order.to_dict(),
            "position": pos.to_dict(),
            "levels_consumed": levels_consumed,
            "observed_at": now,
            "provenance": "PAPER_EXECUTION_ENGINE",
            "execution_environment": "PAPER_MAINNET_DATA",
        }
        self.execution_events.append(ack_event)
        if self.sink is not None:
            await self.sink.write("orders", order.to_dict())
            await self.sink.write("positions", pos.to_dict())
            await self.sink.write("execution_events", ack_event)

        return {
            "status": order.status,
            "order_id": order.order_id,
            "client_order_id": order.client_order_id,
            "filled_quantity": str(order.filled_quantity),
            "average_price": str(order.average_price) if order.average_price else None,
            "liquidity_role": "TAKER",
            "fee": str(order.cumulative_fee),
            "levels_consumed": levels_consumed,
        }

    async def _register_maker_order(
        self, order: PaperOrder, book: OrderBookState, now: float
    ) -> dict[str, Any]:
        """Register resting limit order and compute initial queue ahead volume."""
        order.liquidity_role = "MAKER"
        # Determine queue ahead volume at order.price in current book
        if order.price is not None:
            if order.side == "BUY":
                order.queue_ahead_volume = book.bids.get(order.price, Decimal("0"))
            else:
                order.queue_ahead_volume = book.asks.get(order.price, Decimal("0"))

        order.status = "SUBMITTED"
        order.updated_at = now
        self.open_orders_by_cell.setdefault(order.cell_id, []).append(order)

        ack_event = {
            "event_type": "ORDER_ACKNOWLEDGED",
            "order": order.to_dict(),
            "queue_ahead_volume": str(order.queue_ahead_volume),
            "observed_at": now,
            "provenance": "PAPER_EXECUTION_ENGINE",
            "execution_environment": "PAPER_MAINNET_DATA",
        }
        self.execution_events.append(ack_event)
        if self.sink is not None:
            await self.sink.write("orders", order.to_dict())
            await self.sink.write("execution_events", ack_event)

        return {
            "status": "SUBMITTED",
            "order_id": order.order_id,
            "client_order_id": order.client_order_id,
            "queue_ahead_volume": str(order.queue_ahead_volume),
            "liquidity_role": "MAKER",
        }

    async def on_trade_print(
        self,
        symbol: str,
        trade_price: Decimal,
        trade_qty: Decimal,
        timestamp: float | None = None,
    ) -> list[dict[str, Any]]:
        """Process market trade print to advance maker order queues deterministically."""
        now = timestamp or time.time()
        fills_generated: list[dict[str, Any]] = []

        for cell_id, cell_orders in list(self.open_orders_by_cell.items()):
            active_remaining: list[PaperOrder] = []
            for order in cell_orders:
                if order.symbol != symbol or order.status not in ("SUBMITTED", "PARTIALLY_FILLED"):
                    continue
                if order.price is None:
                    continue

                eligible_fill_qty = Decimal("0")

                if order.side == "BUY":
                    if trade_price < order.price:
                        # Traded through order price -> queue fully cleared
                        eligible_fill_qty = order.remaining_quantity
                    elif trade_price == order.price:
                        if order.queue_ahead_volume > 0:
                            consumed_queue = min(order.queue_ahead_volume, trade_qty)
                            order.queue_ahead_volume -= consumed_queue
                            excess_qty = trade_qty - consumed_queue
                            if excess_qty > 0 and order.queue_ahead_volume == 0:
                                eligible_fill_qty = min(order.remaining_quantity, excess_qty)
                        else:
                            eligible_fill_qty = min(order.remaining_quantity, trade_qty)
                else:  # SELL
                    if trade_price > order.price:
                        # Traded through order price -> queue fully cleared
                        eligible_fill_qty = order.remaining_quantity
                    elif trade_price == order.price:
                        if order.queue_ahead_volume > 0:
                            consumed_queue = min(order.queue_ahead_volume, trade_qty)
                            order.queue_ahead_volume -= consumed_queue
                            excess_qty = trade_qty - consumed_queue
                            if excess_qty > 0 and order.queue_ahead_volume == 0:
                                eligible_fill_qty = min(order.remaining_quantity, excess_qty)
                        else:
                            eligible_fill_qty = min(order.remaining_quantity, trade_qty)

                if eligible_fill_qty > 0:
                    fill_id = f"fill_{order.order_id}_{len(order.fills)+1}"
                    notional = eligible_fill_qty * order.price
                    fee = _round_decimal(notional * MAKER_FEE_RATE, 8)

                    order.filled_quantity += eligible_fill_qty
                    order.remaining_quantity -= eligible_fill_qty
                    order.cumulative_quote += notional
                    order.cumulative_fee += fee
                    order.updated_at = now

                    pos = self.get_position(order.run_id, order.cell_id, order.symbol)
                    realized_delta = pos.update_with_fill(
                        order.side, eligible_fill_qty, order.price, fee, now=now
                    )

                    fill_record = {
                        "fill_id": fill_id,
                        "order_id": order.order_id,
                        "client_order_id": order.client_order_id,
                        "cell_id": order.cell_id,
                        "symbol": order.symbol,
                        "side": order.side,
                        "price": str(order.price),
                        "quantity": str(eligible_fill_qty),
                        "notional": str(notional),
                        "liquidity_role": "MAKER",
                        "fee": str(fee),
                        "fee_rate": str(MAKER_FEE_RATE),
                        "commission_provenance": "MODELLED",
                        "realized_pnl_delta": str(realized_delta),
                        "timestamp": now,
                        "execution_mode": "PAPER",
                        "execution_environment": "PAPER_MAINNET_DATA",
                        "provenance": "PAPER_EXECUTION_ENGINE",
                    }
                    order.fills.append(fill_record)
                    fills_generated.append(fill_record)

                    if self.sink is not None:
                        await self.sink.write("fills", fill_record)
                        await self.sink.write("positions", pos.to_dict())

                    if order.remaining_quantity == 0:
                        order.status = "FILLED"
                    else:
                        order.status = "PARTIALLY_FILLED"
                        active_remaining.append(order)

                    if self.sink is not None:
                        await self.sink.write("orders", order.to_dict())
                else:
                    active_remaining.append(order)

            self.open_orders_by_cell[cell_id] = active_remaining

        return fills_generated

    async def cancel_all(self, run_id: str) -> dict[str, Any]:
        now = time.time()
        cancelled_count = 0
        for cell_id, orders in list(self.open_orders_by_cell.items()):
            for order in orders:
                if order.status in ("SUBMITTED", "PARTIALLY_FILLED"):
                    order.status = "CANCELLED"
                    order.updated_at = now
                    cancelled_count += 1
                    if self.sink is not None:
                        await self.sink.write("orders", order.to_dict())
            self.open_orders_by_cell[cell_id] = []
        return {"run_id": run_id, "cancelled_orders": cancelled_count}

    async def flatten(self, run_id: str) -> dict[str, Any]:
        """Close any non-zero paper position at current market prices."""
        now = time.time()
        flattened_count = 0
        for cell_id, pos in list(self.positions_by_cell.items()):
            if pos.amount == Decimal("0"):
                continue
            book = self.get_order_book(pos.symbol)
            close_side = "SELL" if pos.amount > 0 else "BUY"
            qty = abs(pos.amount)
            exit_price = book.best_bid if close_side == "SELL" else book.best_ask
            if exit_price is None:
                exit_price = pos.entry_price

            notional = qty * exit_price
            fee = _round_decimal(notional * TAKER_FEE_RATE, 8)
            realized_delta = pos.update_with_fill(
                close_side, qty, exit_price, fee, now=now
            )
            flattened_count += 1

            flat_event = {
                "event_type": "POSITION_FLATTENED",
                "cell_id": cell_id,
                "symbol": pos.symbol,
                "closed_amount": str(qty),
                "exit_price": str(exit_price),
                "realized_pnl_delta": str(realized_delta),
                "total_realized_pnl": str(pos.realized_pnl),
                "net_pnl": str(pos.realized_pnl - pos.total_fees),
                "timestamp": now,
                "execution_mode": "PAPER",
                "execution_environment": "PAPER_MAINNET_DATA",
                "provenance": "PAPER_EXECUTION_ENGINE",
            }
            self.execution_events.append(flat_event)
            if self.sink is not None:
                await self.sink.write("positions", pos.to_dict())
                await self.sink.write("execution_events", flat_event)

        return {"run_id": run_id, "flattened_positions": flattened_count}

    async def terminal_state(self) -> dict[str, Any]:
        open_orders: list[dict[str, Any]] = []
        for cell_id, orders in self.open_orders_by_cell.items():
            for order in orders:
                if order.status in ("SUBMITTED", "PARTIALLY_FILLED"):
                    open_orders.append(order.to_dict())
        positions: list[dict[str, Any]] = []
        for cell_id, pos in self.positions_by_cell.items():
            if pos.amount != Decimal("0"):
                positions.append(pos.to_dict())

        return {
            "open_orders": open_orders,
            "positions": positions,
            "open_orders_count": len(open_orders),
            "positions_count": len(positions),
            "cells_with_positions": len(positions),
            "execution_mode": "PAPER",
            "execution_environment": "PAPER_MAINNET_DATA",
        }

    async def market_stream(
        self, symbols: tuple[str, ...] = SYMBOLS
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream live events and update local maker order book queues on trade prints."""
        async for event in self.feed.market_stream(symbols):
            if event.get("event_type") == "TRADE":
                p = Decimal(str(event.get("price", "0")))
                q = Decimal(str(event.get("quantity", "0")))
                sym = event.get("symbol", "")
                t = float(event.get("event_time", time.time() * 1000)) / 1000
                if p > 0 and q > 0 and sym:
                    await self.on_trade_print(sym, p, q, timestamp=t)
            yield event


class PaperExecutionAdapter:
    """Adapter connecting Condor Orchestrator to the PaperExecutionRuntime."""

    def __init__(
        self, runtime: PaperExecutionRuntime, sink: JsonlSink | None = None
    ) -> None:
        self.runtime = runtime
        self.sink = sink

    @staticmethod
    def validate_intent(intent: ExecutionIntent, valid_cells: set[str]) -> None:
        if intent.target_environment != "PAPER_MAINNET_DATA":
            raise OrchestratorBlocked("NON_PAPER_MAINNET_DATA_INTENT_REJECTED")
        intent.validate()
        if intent.cell_id not in valid_cells:
            raise OrchestratorBlocked("UNKNOWN_CELL_ID")
        if intent.strategy_revision != STRATEGY_REVISION:
            raise OrchestratorBlocked("STRATEGY_REVISION_MISMATCH")

    async def dispatch(
        self, intent: ExecutionIntent, valid_cells: set[str]
    ) -> dict[str, Any]:
        self.validate_intent(intent, valid_cells)
        result = await self.runtime.dispatch_intent(intent)
        record = {
            **asdict(intent),
            "event_type": "ORDER_DISPATCHED",
            "execution_mode": "PAPER",
            "execution_environment": "PAPER_MAINNET_DATA",
            "provenance": "PAPER_EXECUTION_ENGINE",
            "observed_at": time.time(),
            "response": result,
        }
        if self.sink is not None:
            await self.sink.write("execution_events", record)
        return result

    async def bounded_dispatch_probe(
        self,
        *,
        symbol: str = "BTCUSDT",
        quantity: str = "0.001",
    ) -> dict[str, Any]:
        """Execute one non-trading paper probe against real/mock book and verify terminal state."""
        if symbol not in SYMBOLS:
            raise OrchestratorBlocked(f"UNAPPROVED_SYMBOL:{symbol}")
        cell_id = f"{symbol}:orderflow.momentum.aggression:{STRATEGY_REVISION}"
        correlation_id = f"paper-probe-{int(time.time()*1000)}"

        # Ensure book is bootstrapped with at least minimal depth for the probe
        book = self.runtime.get_order_book(symbol)
        if not book.is_bootstrapped or not book.asks:
            book.apply_snapshot(
                bids=[[Decimal("65000.0"), Decimal("10.0")]],
                asks=[[Decimal("65001.0"), Decimal("10.0")]],
                last_update_id=1000,
            )

        intent = ExecutionIntent(
            run_id="paper-probe-run",
            cell_id=cell_id,
            correlation_id=correlation_id,
            jev_decision_id="probe-jev-dec",
            symbol=symbol,
            action="ENTRY",
            side="BUY",
            strategy_id="orderflow.momentum.aggression",
            strategy_revision=STRATEGY_REVISION,
            quantity=quantity,
            order_type="MARKET",
            price=None,
            execution_constraints={"probe": True},
            target_environment="PAPER_MAINNET_DATA",
        )
        dispatch_result = await self.dispatch(intent, {cell_id})
        # Settle and flatten position
        await self.runtime.cancel_all("paper-probe-run")
        await self.runtime.flatten("paper-probe-run")
        terminal = await self.runtime.terminal_state()

        if terminal.get("open_orders") or terminal.get("positions"):
            raise OrchestratorBlocked("PAPER_PROBE_TERMINAL_RECONCILIATION_FAILED")

        return {
            "status": "PASS",
            "execution_mode": "PAPER",
            "symbol": symbol,
            "cell_id": cell_id,
            "correlation_id": correlation_id,
            "dispatch_response": dispatch_result,
            "terminal_open_orders": len(terminal.get("open_orders", [])),
            "terminal_positions": len(terminal.get("positions", [])),
            "target_environment": "PAPER_MAINNET_DATA",
            "provenance": "PAPER_EXECUTION_ENGINE",
        }
