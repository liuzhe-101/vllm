# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resident, interval-aggregated metrics for the frontend output path."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.outputs import PoolingRequestOutput, RequestOutput

logger = init_logger("vllm.lwd_meter")

_METER_INTERVAL_S = 300.0
_MAX_PENDING_SSE_OUTPUTS = 4096


class _LwdFrontendSpan:
    __slots__ = ("count", "_max_s", "_total_s")

    def __init__(self) -> None:
        self.count = 0
        self._total_s = 0.0
        self._max_s = 0.0

    def add(self, seconds: float) -> None:
        self.count += 1
        self._total_s += seconds
        self._max_s = max(self._max_s, seconds)

    def summary(self) -> str:
        return (
            f"avg_ms={self._total_s / self.count * 1000:.3f} "
            f"max_ms={self._max_s * 1000:.3f} count={self.count}"
        )


@dataclass(slots=True)
class LwdCollectorState:
    first_queued: float | None = None
    output_count: int = 0
    merge_count: int = 0

    def reset(self) -> None:
        self.first_queued = None
        self.output_count = 0
        self.merge_count = 0


class LwdFrontendMeter:
    """Aggregate frontend timing without per-output logging or synchronization."""

    def __init__(self) -> None:
        self._last_flush = time.monotonic()
        self._pending_sse: OrderedDict[int, tuple[RequestOutput, float]] = OrderedDict()
        self._reset_window()

    def _reset_window(self) -> None:
        self._engine_to_processor = _LwdFrontendSpan()
        self._process_outputs = _LwdFrontendSpan()
        self._collector_wait = _LwdFrontendSpan()
        self._processor_to_sse = _LwdFrontendSpan()
        self._merge_events = 0
        self._merged_outputs = 0
        self._merged_tokens = 0
        self._max_tokens_per_chunk = 0

    def note_process_outputs(
        self, started: float, engine_core_timestamp: float | None
    ) -> None:
        ended = time.monotonic()
        if engine_core_timestamp is not None:
            self._engine_to_processor.add(started - engine_core_timestamp)
        self._process_outputs.add(ended - started)
        self._maybe_flush(ended)

    def note_collector_put(self, state: LwdCollectorState, merged: bool) -> None:
        if state.first_queued is None or not merged:
            state.first_queued = time.monotonic()
            state.output_count = 1
            state.merge_count = 0
            return
        state.output_count += 1
        state.merge_count += 1

    def note_collector_get(
        self,
        output: RequestOutput | PoolingRequestOutput | Exception,
        state: LwdCollectorState,
    ) -> None:
        if state.first_queued is None:
            return
        dequeued = time.monotonic()
        self._collector_wait.add(dequeued - state.first_queued)
        if isinstance(output, RequestOutput):
            tokens = sum(len(completion.token_ids) for completion in output.outputs)
            self._max_tokens_per_chunk = max(self._max_tokens_per_chunk, tokens)
            if state.merge_count:
                self._merge_events += state.merge_count
                self._merged_outputs += state.output_count
                self._merged_tokens += tokens
            self._remember_dequeued(output, dequeued)
        state.reset()
        self._maybe_flush(dequeued)

    def _remember_dequeued(self, output: RequestOutput, dequeued: float) -> None:
        key = id(output)
        self._pending_sse.pop(key, None)
        self._pending_sse[key] = (output, dequeued)
        if len(self._pending_sse) > _MAX_PENDING_SSE_OUTPUTS:
            self._pending_sse.popitem(last=False)

    def note_processor_to_sse(self, output: RequestOutput) -> None:
        entry = self._pending_sse.pop(id(output), None)
        if entry is None or entry[0] is not output:
            return
        yielded = time.monotonic()
        self._processor_to_sse.add(yielded - entry[1])
        self._maybe_flush(yielded)

    def _maybe_flush(self, now: float) -> None:
        if now - self._last_flush >= _METER_INTERVAL_S:
            self._flush(now)

    def flush(self) -> None:
        """Emit and clear the current partial window; empty flushes are silent."""
        self._flush(time.monotonic())

    def _flush(self, now: float) -> None:
        parts = []
        for name, span in (
            ("engine_to_processor", self._engine_to_processor),
            ("process_outputs", self._process_outputs),
            ("collector_wait", self._collector_wait),
            ("processor_to_sse", self._processor_to_sse),
        ):
            if span.count:
                parts.append(f"{name}[{span.summary()}]")
        for name, value in (
            ("merge_events", self._merge_events),
            ("merged_outputs", self._merged_outputs),
            ("merged_tokens", self._merged_tokens),
            ("max_tokens_per_chunk", self._max_tokens_per_chunk),
        ):
            if value:
                parts.append(f"{name}={value}")
        if parts:
            logger.info("[lwd][frontend] %s", " ".join(parts))
        self._reset_window()
        self._last_flush = now


LWD_FRONTEND_METER = LwdFrontendMeter()
