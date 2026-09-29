"""Binance Futures Mainnet Public Market Data Feed.

Strictly read-only and public: no trading credentials, no private streams, no
order submission, and no account mutations. Maintains causal L2 order books and
streams real-time market events for BTCUSDT, ZECUSDT, SUIUSDT, and WLDUSDT.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, AsyncIterator

import aiohttp

DEFAULT_MAINNET_REST = "https://fapi.binance.com"
DEFAULT_MAINNET_WS = "wss://fstream.binance.com/stream"
APPROVED_SYMBOLS = ("BTCUSDT", "ZECUSDT", "SUIUSDT", "WLDUSDT")


@dataclass
class OrderBookState:
    """Causal local L2 order book representation for one symbol."""

    symbol: str
    bids: dict[Decimal, Decimal] = field(default_factory=dict)
    asks: dict[Decimal, Decimal] = field(default_factory=dict)
    last_update_id: int = 0
    last_event_time: float = 0.0
    is_crossed: bool = False
    is_bootstrapped: bool = False
    sequence_gap: bool = False
    rebootstrap_required: bool = False

    @property
    def best_bid(self) -> Decimal | None:
        if not self.bids:
            return None
        return max(self.bids.keys())

    @property
    def best_ask(self) -> Decimal | None:
        if not self.asks:
            return None
        return min(self.asks.keys())

    @property
    def spread(self) -> Decimal | None:
        bid = self.best_bid
        ask = self.best_ask
        if bid is not None and ask is not None:
            return ask - bid
        return None

    def top_bids(self, n: int = 20) -> list[tuple[Decimal, Decimal]]:
        sorted_prices = sorted(self.bids.keys(), reverse=True)[:n]
        return [(p, self.bids[p]) for p in sorted_prices]

    def top_asks(self, n: int = 20) -> list[tuple[Decimal, Decimal]]:
        sorted_prices = sorted(self.asks.keys())[:n]
        return [(p, self.asks[p]) for p in sorted_prices]

    def apply_snapshot(
        self,
        bids: list[list[Any]],
        asks: list[list[Any]],
        last_update_id: int,
        timestamp: float | None = None,
    ) -> None:
        self.bids.clear()
        self.asks.clear()
        for p_raw, q_raw in bids:
            p = Decimal(str(p_raw))
            q = Decimal(str(q_raw))
            if q > 0:
                self.bids[p] = q
        for p_raw, q_raw in asks:
            p = Decimal(str(p_raw))
            q = Decimal(str(q_raw))
            if q > 0:
                self.asks[p] = q
        self.last_update_id = last_update_id
        self.last_event_time = timestamp or time.time()
        self.is_bootstrapped = True
        self.sequence_gap = False
        self.rebootstrap_required = False
        self._check_crossed()

    def apply_diff_depth(
        self,
        bids: list[list[Any]],
        asks: list[list[Any]],
        first_update_id: int | None,
        final_update_id: int,
        prev_final_update_id: int | None = None,
        timestamp: float | None = None,
    ) -> bool:
        if not self.is_bootstrapped or self.rebootstrap_required:
            self.sequence_gap = True
            self.rebootstrap_required = True
            return False
        if first_update_id is not None and self.last_update_id:
            contiguous = first_update_id <= self.last_update_id + 1 <= final_update_id
            if prev_final_update_id is not None:
                contiguous = contiguous and prev_final_update_id == self.last_update_id
            if not contiguous:
                self.sequence_gap = True
                self.rebootstrap_required = True
                return False
        for p_raw, q_raw in bids:
            p = Decimal(str(p_raw))
            q = Decimal(str(q_raw))
            if q == 0:
                self.bids.pop(p, None)
            else:
                self.bids[p] = q
        for p_raw, q_raw in asks:
            p = Decimal(str(p_raw))
            q = Decimal(str(q_raw))
            if q == 0:
                self.asks.pop(p, None)
            else:
                self.asks[p] = q
        self.last_update_id = final_update_id
        self.last_event_time = timestamp or time.time()
        valid = self._check_crossed()
        if not valid:
            self.rebootstrap_required = True
        return valid

    def _check_crossed(self) -> bool:
        bid = self.best_bid
        ask = self.best_ask
        if bid is not None and ask is not None and bid >= ask:
            self.is_crossed = True
            return False
        self.is_crossed = False
        return True

    def to_snapshot_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "best_bid": str(self.best_bid) if self.best_bid else None,
            "best_ask": str(self.best_ask) if self.best_ask else None,
            "spread": str(self.spread) if self.spread is not None else None,
            "bids_count": len(self.bids),
            "asks_count": len(self.asks),
            "last_update_id": self.last_update_id,
            "is_crossed": self.is_crossed,
            "is_bootstrapped": self.is_bootstrapped,
            "sequence_gap": self.sequence_gap,
            "rebootstrap_required": self.rebootstrap_required,
        }


class BinanceMainnetPublicMarketFeed:
    """Public read-only market feed for Binance USD-M Futures Mainnet."""

    def __init__(
        self,
        *,
        rest_base: str = DEFAULT_MAINNET_REST,
        ws_base: str = DEFAULT_MAINNET_WS,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.rest_base = rest_base.rstrip("/")
        self.ws_base = ws_base
        self._session = session
        self._owns_session = session is None
        self.books: dict[str, OrderBookState] = {
            sym: OrderBookState(symbol=sym) for sym in APPROVED_SYMBOLS
        }
        self.last_trade_times: dict[str, float] = {}
        self.last_depth_times: dict[str, float] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10.0)
            )
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session is not None and not self._session.closed:
            await self._session.close()

    async def preflight(self, symbols: tuple[str, ...] = APPROVED_SYMBOLS) -> dict[str, Any]:
        """Verify public Mainnet REST and bootstrap depth without trading credentials."""
        session = await self._get_session()
        # 1. Ping / Time
        async with session.get(f"{self.rest_base}/fapi/v1/time") as resp:
            if resp.status != 200:
                raise ConnectionError(f"BINANCE_PUBLIC_REST_UNAVAILABLE: {resp.status}")
            time_data = await resp.json()
            server_time = time_data.get("serverTime")

        # 2. Fetch depth snapshot for each symbol
        prices: dict[str, dict[str, Any]] = {}
        for sym in symbols:
            url = f"{self.rest_base}/fapi/v1/depth?symbol={sym}&limit=50"
            async with session.get(url) as resp:
                if resp.status != 200:
                    raise ConnectionError(f"DEPTH_SNAPSHOT_FAILED:{sym}:{resp.status}")
                data = await resp.json()
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                last_update_id = int(data.get("lastUpdateId", 0))
                if not bids or not asks:
                    raise ValueError(f"EMPTY_DEPTH_SNAPSHOT:{sym}")
                book = self.books.setdefault(sym, OrderBookState(symbol=sym))
                book.apply_snapshot(bids, asks, last_update_id)
                if book.is_crossed:
                    raise ValueError(f"CROSSED_BOOK_ON_BOOTSTRAP:{sym}")
                prices[sym] = {
                    "best_bid": str(book.best_bid),
                    "best_ask": str(book.best_ask),
                    "spread": str(book.spread),
                    "bids_depth": len(bids),
                    "asks_depth": len(asks),
                    "last_update_id": last_update_id,
                }

        return {
            "market_data_environment": "MAINNET",
            "rest_base": self.rest_base,
            "server_time": server_time,
            "credentials_required": False,
            "remote_trading_enabled": False,
            "symbols": list(symbols),
            "prices": prices,
            "status": "HEALTHY",
        }

    async def fetch_snapshot(self, symbol: str, limit: int = 100) -> OrderBookState:
        session = await self._get_session()
        url = f"{self.rest_base}/fapi/v1/depth?symbol={symbol}&limit={limit}"
        async with session.get(url) as resp:
            if resp.status != 200:
                raise ConnectionError(f"DEPTH_SNAPSHOT_FAILED:{symbol}:{resp.status}")
            data = await resp.json()
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            last_update_id = int(data.get("lastUpdateId", 0))
            book = self.books.setdefault(symbol, OrderBookState(symbol=symbol))
            book.apply_snapshot(bids, asks, last_update_id)
            return book

    async def market_stream(
        self, symbols: tuple[str, ...] = APPROVED_SYMBOLS
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield live depth and trade updates from public WebSocket multiplex stream."""
        # Pre-bootstrap books if not already bootstrapped
        for sym in symbols:
            if not self.books[sym].is_bootstrapped:
                try:
                    await self.fetch_snapshot(sym, limit=50)
                except Exception:
                    pass

        stream_names = []
        for sym in symbols:
            s_lower = sym.lower()
            stream_names.append(f"{s_lower}@depth@100ms")
            stream_names.append(f"{s_lower}@aggTrade")

        ws_url = f"{self.ws_base}?streams={'/'.join(stream_names)}"
        session = await self._get_session()

        while True:
            try:
                async with session.ws_connect(
                    ws_url, heartbeat=15.0, timeout=10.0
                ) as ws:
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                payload = json.loads(msg.data)
                            except json.JSONDecodeError:
                                continue
                            event = self._process_stream_payload(payload)
                            if event is not None:
                                yield event
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            break
            except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError):
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break

    def _process_stream_payload(self, raw: dict[str, Any]) -> dict[str, Any] | None:
        data = raw.get("data") if "stream" in raw else raw
        if not isinstance(data, dict):
            return None

        event_type = data.get("e")
        sym_raw = data.get("s")
        if not sym_raw:
            return None
        symbol = str(sym_raw).upper()
        if symbol not in self.books:
            return None

        book = self.books[symbol]
        now = time.time()

        if event_type == "depthUpdate":
            bids = data.get("b", [])
            asks = data.get("a", [])
            final_u = int(data.get("u", 0))
            valid = book.apply_diff_depth(
                bids,
                asks,
                int(data.get("U", 0)) if data.get("U") is not None else None,
                final_u,
                int(data.get("pu")) if data.get("pu") is not None else None,
                timestamp=now,
            )
            self.last_depth_times[symbol] = now
            return {
                "symbol": symbol,
                "event_type": "DEPTH_UPDATE",
                "event_time": data.get("E", int(now * 1000)),
                "transaction_time": data.get("T", int(now * 1000)),
                "first_update_id": data.get("U"),
                "final_update_id": final_u,
                "prev_final_update_id": data.get("pu"),
                "best_bid": str(book.best_bid) if book.best_bid else None,
                "best_ask": str(book.best_ask) if book.best_ask else None,
                "spread": str(book.spread) if book.spread is not None else None,
                "is_crossed": book.is_crossed,
                "market_valid": valid and not book.rebootstrap_required,
                "sequence_gap": book.sequence_gap,
                "rebootstrap_required": book.rebootstrap_required,
                "bids": bids,
                "asks": asks,
                "provenance": "MAINNET_PUBLIC_MARKET_DATA",
                "execution_environment": "PAPER_MAINNET_DATA",
            }

        if event_type == "aggTrade":
            price = Decimal(str(data.get("p", "0")))
            qty = Decimal(str(data.get("q", "0")))
            is_buyer_maker = bool(data.get("m", False))
            aggressor_side = "SELL" if is_buyer_maker else "BUY"
            self.last_trade_times[symbol] = now
            return {
                "symbol": symbol,
                "event_type": "TRADE",
                "trade_id": str(data.get("a")),
                "price": str(price),
                "quantity": str(qty),
                "aggressor_side": aggressor_side,
                "is_buyer_maker": is_buyer_maker,
                "event_time": data.get("E", int(now * 1000)),
                "transaction_time": data.get("T", int(now * 1000)),
                "best_bid": str(book.best_bid) if book.best_bid else None,
                "best_ask": str(book.best_ask) if book.best_ask else None,
                "provenance": "MAINNET_PUBLIC_MARKET_DATA",
                "execution_environment": "PAPER_MAINNET_DATA",
            }

        return None
