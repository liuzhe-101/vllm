# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-overhead interval meters for the LWD token-output path."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from vllm.logger import init_logger

logger = init_logger("vllm.lwd_meter")

_METER_INTERVAL_S = 300.0
_SLOW_CLOUD_STEP_S = 1.0
_MAX_SLOW_EXAMPLES = 3


class LwdSpan:
    """Count, mean and maximum of one timing segment, in seconds."""

    __slots__ = ("count", "_max_s", "_total_s")

    def __init__(self) -> None:
        self.count = 0
        self._total_s = 0.0
        self._max_s = 0.0

    def add(self, seconds: float) -> None:
        self.count += 1
        self._total_s += seconds
        self._max_s = max(self._max_s, seconds)

    def __str__(self) -> str:
        if not self.count:
            return "-/-"
        return f"{self._total_s / self.count * 1000:.1f}/{self._max_s * 1000:.1f}"


class LwdIntervalMeter:
    """Accumulate per event and report at most once per interval."""

    def __init__(self) -> None:
        self._last_report = time.monotonic()
        self._reset()

    def _reset(self) -> None:
        raise NotImplementedError

    def _report(self) -> None:
        raise NotImplementedError

    def _roll(self) -> None:
        now = time.monotonic()
        if now - self._last_report < _METER_INTERVAL_S:
            return
        self._last_report = now
        self._report()
        self._reset()

    def flush(self) -> None:
        """Report the final partial window, normally once at shutdown."""
        self._report()
        self._reset()
        self._last_report = time.monotonic()


@dataclass(slots=True)
class LwdBatchShape:
    decode_reqs: int = 0
    decode_expected: int = 0
    decode_tokens: int = 0
    prefill_reqs: int = 0
    prefill_tokens: int = 0
    window_reqs: int = 0
    window_tokens: int = 0
    room: int = 0
    running: int = 0
    waiting: int = 0
    preempted: int = 0
    first_prefill_index: int = -1
    draft_slots: int = 0
    output_tokens: int = 0

    def __str__(self) -> str:
        return (
            f"decode={self.decode_tokens}/{self.decode_expected}/"
            f"{self.decode_reqs}(actual/expected/reqs) "
            f"prefill={self.prefill_tokens}/{self.prefill_reqs} "
            f"windows={self.window_tokens}/{self.window_reqs} room={self.room} "
            f"running={self.running} waiting={self.waiting} "
            f"preempted={self.preempted} first_prefill={self.first_prefill_index} "
            f"draft_slots={self.draft_slots} output_tokens={self.output_tokens}"
        )


@dataclass(slots=True)
class LwdCloudStepTiming:
    started: float = field(default_factory=time.monotonic)
    pre_gap: float = 0.0
    dummy: float = 0.0
    dp_sync: float = 0.0
    sched: float = 0.0
    model_wait: float = 0.0
    update: float = 0.0
    c2e_send: float = 0.0
    native_post: float = 0.0
    wall: float = 0.0
    c2e_count: int = 0
    down_seqno: int = -1
    shape: LwdBatchShape = field(default_factory=LwdBatchShape)

    @property
    def other(self) -> float:
        """Submit, grammar, abort and LWD input work not split out yet."""
        return max(
            0.0,
            self.wall
            - self.sched
            - self.model_wait
            - self.update
            - self.c2e_send
            - self.native_post,
        )

    @property
    def total(self) -> float:
        return self.wall + self.dummy + self.dp_sync


class LwdShapeWindow:
    """Aggregates for one non-empty scheduler batch shape."""

    def __init__(self) -> None:
        self.steps = 0
        self.total = LwdSpan()
        self.model_wait = LwdSpan()
        self.model_s = 0.0
        self.prefill_tokens = 0
        self.output_tokens = 0

    def note(self, timing: LwdCloudStepTiming) -> None:
        self.steps += 1
        self.total.add(timing.total)
        self.model_wait.add(timing.model_wait)
        self.model_s += timing.model_wait
        self.prefill_tokens += timing.shape.prefill_tokens
        self.output_tokens += timing.shape.output_tokens

    def summary(self, all_model_s: float) -> str:
        ms_per_token = (
            f"{self.model_s * 1000 / self.output_tokens:.2f}"
            if self.output_tokens
            else "-"
        )
        share = self.model_s / all_model_s * 100 if all_model_s else 0.0
        return (
            f"steps={self.steps} prefill={self.prefill_tokens} "
            f"output={self.output_tokens} total={self.total} "
            f"model={self.model_wait} model_ms/token={ms_per_token} "
            f"model_share={share:.1f}%"
        )


class LwdCloudStepMeter(LwdIntervalMeter):
    """Break down engine wall time; average/max values are milliseconds."""

    def __init__(self) -> None:
        self._last_step_end = 0.0
        super().__init__()

    def _reset(self) -> None:
        self._steps = 0
        self._empty = 0
        self._shapes = {
            kind: LwdShapeWindow() for kind in ("mixed", "decode_only", "prefill_only")
        }
        self._slow = 0
        self._slow_examples: list[LwdCloudStepTiming] = []
        self._ranges = 0
        self._range_ingest = LwdSpan()
        self._hint_post = LwdSpan()
        for name in (
            "total",
            "pre_gap",
            "dummy",
            "dp_sync",
            "sched",
            "model_wait",
            "update",
            "c2e_send",
            "native_post",
            "other",
        ):
            setattr(self, f"_{name}", LwdSpan())

    def note_step_end(self) -> None:
        self._last_step_end = time.monotonic()

    def note_range(self, ingest_wait: float, hint_post: float) -> None:
        self._ranges += 1
        self._range_ingest.add(ingest_wait)
        self._hint_post.add(hint_post)
        self._roll()

    def take_pre_gap(self) -> float:
        if not self._last_step_end:
            return 0.0
        return time.monotonic() - self._last_step_end

    @staticmethod
    def _kind(timing: LwdCloudStepTiming) -> str:
        shape = timing.shape
        has_prefill = bool(shape.prefill_tokens or shape.window_tokens)
        has_decode = bool(shape.decode_expected or shape.decode_tokens)
        if has_prefill:
            return "mixed" if has_decode else "prefill_only"
        if has_decode or shape.output_tokens or timing.c2e_count:
            return "decode_only"
        return "empty"

    def note_step(self, timing: LwdCloudStepTiming) -> None:
        kind = self._kind(timing)
        if kind == "empty":
            self._empty += 1
            self._roll()
            return
        self._shapes[kind].note(timing)
        self._steps += 1
        values = {
            "total": timing.total,
            "pre_gap": timing.pre_gap,
            "dummy": timing.dummy,
            "dp_sync": timing.dp_sync,
            "sched": timing.sched,
            "model_wait": timing.model_wait,
            "update": timing.update,
            "c2e_send": timing.c2e_send,
            "native_post": timing.native_post,
            "other": timing.other,
        }
        for name, value in values.items():
            getattr(self, f"_{name}").add(value)
        if timing.total >= _SLOW_CLOUD_STEP_S:
            self._slow += 1
            self._slow_examples.append(timing)
            self._slow_examples.sort(key=lambda item: item.total, reverse=True)
            del self._slow_examples[_MAX_SLOW_EXAMPLES:]
        self._roll()

    def _report_shapes(self) -> None:
        all_model_s = sum(window.model_s for window in self._shapes.values())
        shapes = "; ".join(
            f"{kind}[{window.summary(all_model_s)}]"
            for kind, window in self._shapes.items()
            if window.steps
        )
        if shapes:
            logger.info("[lwd][cloud-shape] %s", shapes)

    def _report(self) -> None:
        if self._steps:
            logger.info(
                "[lwd][cloud-step] %d steps (%d slow, %d empty excluded): "
                "total=%s pre_gap=%s dummy=%s dp_sync=%s sched=%s "
                "model_wait=%s update=%s c2e_send=%s native_post=%s "
                "other=%s (avg/max ms)",
                self._steps,
                self._slow,
                self._empty,
                self._total,
                self._pre_gap,
                self._dummy,
                self._dp_sync,
                self._sched,
                self._model_wait,
                self._update,
                self._c2e_send,
                self._native_post,
                self._other,
            )
            self._report_shapes()
        elif self._empty:
            logger.info("[lwd][cloud-step] %d empty steps excluded", self._empty)
        if self._ranges:
            logger.info(
                "[lwd][cloud-up] %d ranges ingest_wait=%s hint_post=%s (avg/max ms)",
                self._ranges,
                self._range_ingest,
                self._hint_post,
            )
        if self._slow_examples:
            examples = "; ".join(
                f"total={item.total * 1000:.1f} model={item.model_wait * 1000:.1f} "
                f"sched={item.sched * 1000:.1f} c2e={item.c2e_send * 1000:.1f} "
                f"down={item.down_seqno} [{item.shape}]"
                for item in self._slow_examples
            )
            logger.info("[lwd][cloud-slow-top] %s", examples)


class LwdControlPlaneMeter(LwdIntervalMeter):
    """Time from the inproc outbound queue to the peer socket."""

    def __init__(self, label: str) -> None:
        self._label = label
        super().__init__()

    def _reset(self) -> None:
        self._deliveries = 0
        self._full = 0
        self._backlogs = 0
        self._max_depth = 0
        self._queue_wait = LwdSpan()
        self._held_poll = LwdSpan()

    def note_delivery(self, queue_wait: float, remaining: int) -> None:
        self._deliveries += 1
        self._max_depth = max(self._max_depth, remaining + 1)
        self._queue_wait.add(queue_wait)
        self._roll()

    def note_backlog(self, depth: int) -> None:
        self._backlogs += 1
        self._max_depth = max(self._max_depth, depth)

    def note_poll(self, elapsed: float, held_messages: int) -> None:
        if held_messages:
            self._held_poll.add(elapsed)

    def note_full(self) -> None:
        self._full += 1

    def _report(self) -> None:
        if not self._deliveries and not self._full:
            return
        logger.info(
            "[lwd][control-io] %s delivered=%d full=%d backlogs=%d "
            "max_depth=%d fifo_wait=%s held_poll=%s (avg/max ms)",
            self._label,
            self._deliveries,
            self._full,
            self._backlogs,
            self._max_depth,
            self._queue_wait,
            self._held_poll,
        )


class LwdExecutorOutputMeter(LwdIntervalMeter):
    """Worker output-thread harvest, async output resolution and response enqueue."""

    def __init__(self, rank: int) -> None:
        self._rank = rank
        super().__init__()

    def _reset(self) -> None:
        self._outputs = 0
        self._carriers = 0
        self._meta_harvest = LwdSpan()
        self._output_get = LwdSpan()
        self._response_enqueue = LwdSpan()

    def note_harvest(self, seconds: float) -> None:
        self._carriers += 1
        self._meta_harvest.add(seconds)

    def note_output(self, output_get: float, response_enqueue: float) -> None:
        self._outputs += 1
        self._output_get.add(output_get)
        self._response_enqueue.add(response_enqueue)
        self._roll()

    def _report(self) -> None:
        if not self._outputs and not self._carriers:
            return
        logger.info(
            "[lwd][executor-output] rank%d outputs=%d carriers=%d "
            "meta_harvest=%s output_get=%s response_enqueue=%s (avg/max ms)",
            self._rank,
            self._outputs,
            self._carriers,
            self._meta_harvest,
            self._output_get,
            self._response_enqueue,
        )


@dataclass(slots=True)
class LwdEdgeUnembedTiming:
    queue: float
    worker_wait: float
    update: float
    other: float
    roundtrip: float
    tokens: int


class LwdEdgeOutputMeter(LwdIntervalMeter):
    """Split edge C2E queuing from UNEMBED wait and delivery update."""

    def _reset(self) -> None:
        self._outputs = 0
        self._tokens = 0
        self._unembed_queue = LwdSpan()
        self._worker_wait = LwdSpan()
        self._update = LwdSpan()
        self._other = LwdSpan()
        self._roundtrip = LwdSpan()

    def note_unembed(self, timing: LwdEdgeUnembedTiming) -> None:
        self._outputs += 1
        self._tokens += timing.tokens
        self._unembed_queue.add(timing.queue)
        self._worker_wait.add(timing.worker_wait)
        self._update.add(timing.update)
        self._other.add(timing.other)
        self._roundtrip.add(timing.roundtrip)
        self._roll()

    def _report(self) -> None:
        if not self._outputs:
            return
        logger.info(
            "[lwd][edge-output] %d unembeds tokens=%d queue=%s worker_wait=%s "
            "update=%s other=%s roundtrip=%s (avg/max ms)",
            self._outputs,
            self._tokens,
            self._unembed_queue,
            self._worker_wait,
            self._update,
            self._other,
            self._roundtrip,
        )
