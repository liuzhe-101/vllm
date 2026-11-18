# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-model-family request metadata the edge extracts at admission.

Only the edge holds the real prompt. Model families whose cloud-side forward
reads more than the prompt embeddings get that extra input derived here, once
per request, in a form that carries no token id: M-RoPE positions for the
Qwen-VL family. The edge scheduler and engine use the family selected for the
model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm.lwd_logger import lwd_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from vllm.config import VllmConfig
    from vllm.v1.request import Request

    LwdMropePositionsFn = Callable[..., tuple[torch.Tensor, Any]]


class LwdModelFamily:
    """Default model family, which derives no additional request metadata."""

    def admit(self, request: Request) -> None:
        """Hook for retaining per-request metadata derived at admission."""

    def has_mrope_positions(self, request_id: str) -> bool:
        """Whether the request carries M-RoPE positions. Read per range
        item, so it must not serialize them."""
        return False

    def prompt_mrope_positions(self, request_id: str) -> bytes | None:
        """Raw int64 `[3, num_prompt_tokens]` M-RoPE positions of the whole
        prompt; None when the request carries none."""
        return None

    def release(self, request_id: str) -> None:
        """Release per-request metadata retained by `admit()`."""

    def held_request_ids(self) -> list[str]:
        """Request IDs whose admission metadata is still retained."""
        return []


class LwdMropeModelFamily(LwdModelFamily):
    """M-RoPE model family: whole-prompt positions for multimodal prompts.

    Positions need the real prompt and the image grids, so the edge computes
    them for the whole prompt at admission and the request notify carries
    them whole.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        self._vllm_config = vllm_config
        # Resolved on the first multimodal request; model classes lacking it
        # fail fast there.
        self._positions_fn: LwdMropePositionsFn | None = None
        # Whole-prompt [3, N] CPU positions per multimodal request.
        self._positions: dict[str, torch.Tensor] = {}

    def admit(self, request: Request) -> None:
        if not _lwd_request_has_mm(request):
            return
        assert request.prompt_token_ids is not None
        mm_features = [f for f in request.mm_features if f.modality != "prompt_embeds"]
        positions, _delta = self._resolve_positions_fn(request)(
            input_tokens=list(request.prompt_token_ids),
            mm_features=mm_features,
            config=self._vllm_config.model_config.hf_config,
        )
        assert positions.shape[1] == request.num_prompt_tokens, (
            f"[LWD] mrope positions width {positions.shape[1]} != prompt "
            f"len {request.num_prompt_tokens} (request {request.request_id})"
        )
        self._positions[request.request_id] = positions

    def _resolve_positions_fn(self, request: Request) -> LwdMropePositionsFn:
        if self._positions_fn is None:
            from vllm.model_executor.models import ModelRegistry

            model_config = self._vllm_config.model_config
            model_cls, _ = ModelRegistry.resolve_model_cls(
                model_config.architectures, model_config
            )
            fn = getattr(model_cls, "_get_mrope_input_positions", None)
            if fn is None:
                raise ValueError(
                    f"[LWD] multimodal requests in prefill-only mode "
                    f"require a model class exposing "
                    f"_get_mrope_input_positions(input_tokens, mm_features, "
                    f"config); got {model_cls.__name__} "
                    f"(request {request.request_id})"
                )
            self._positions_fn = fn
        return self._positions_fn

    def has_mrope_positions(self, request_id: str) -> bool:
        return request_id in self._positions

    def prompt_mrope_positions(self, request_id: str) -> bytes | None:
        positions = self._positions.get(request_id)
        if positions is None:
            return None
        return positions.contiguous().numpy().tobytes()

    def release(self, request_id: str) -> None:
        self._positions.pop(request_id, None)

    def held_request_ids(self) -> list[str]:
        return list(self._positions)


def _lwd_request_has_mm(request: Request) -> bool:
    """Whether the request carries real multimodal data (prompt_embeds
    passthrough does not count -- it has no grid metadata, mrope is
    handled as text positions, consistent with the native definition)."""
    return any(f.modality != "prompt_embeds" for f in request.mm_features)


def lwd_select_model_family(vllm_config: VllmConfig) -> LwdModelFamily:
    """The request metadata family of the served model."""
    model_config = vllm_config.model_config
    family: LwdModelFamily
    if model_config.uses_mrope:
        family = LwdMropeModelFamily(vllm_config)
    else:
        family = LwdModelFamily()
    lwd_logger.info("[LWD] request metadata family: %s", type(family).__name__)
    return family
