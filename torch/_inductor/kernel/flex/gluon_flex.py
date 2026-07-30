# mypy: allow-untyped-defs
"""Gluon FlexAttention templates for Inductor.

Offers hand-tuned Gluon flash-attention forward kernels as flex-attention
autotuning candidates through ``InductorChoices.append_flex_attention_choices``
(the extension seam TLX uses), installed via ``config.inductor_choices_class``.

A template is a ``GluonTemplate`` -- a ``TritonTemplate`` that emits a
``@gluon.jit`` body -- so ``score_mod``/``mask_mod`` are rendered by Inductor's
existing ``modification()`` machinery and the kernel runs through the standard
Triton scheduling/compile path.

Targets are described by ``GluonFlexTarget``: matmul tile geometry, which
template bodies exist, and the DMA staging ladder. Kernel *bodies* are per target
because Gluon's primitives differ by family, but everything here -- config
filtering, subgraph rendering, autotuning -- is shared. Adding a target is a new
descriptor plus its template files.
"""

import dataclasses
import functools
import logging
import os
from typing import Any, NamedTuple
from typing_extensions import override

import torch
from torch._inductor import config
from torch._inductor.choices import InductorChoices
from torch._inductor.codegen.gluon.cdna4 import Cdna4GluonTemplate
from torch._inductor.codegen.gluon.gluon_template import GluonTemplate

from .common import load_flex_template
from .gluon_dma_layouts import as_template_options, CDNA4_DMA_LADDER, DmaLayouts


log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, eq=False)
class GluonFlexTarget:
    """Everything target-specific about offering a Gluon flex-attention body.

    ``eq=False`` keeps the default identity hash, so a target can be used as a
    cache key. Field-based equality would hash ``dma_ladder`` and fail, which is
    also why this is not a NamedTuple -- being a tuple implies hashability that a
    mapping field does not provide.
    """

    name: str
    template_cls: type[GluonTemplate]
    sync_template: str
    # None when the target has no async body yet; only the sync one is offered.
    async_template: str | None
    # Matmul instruction tile (MFMA on CDNA, WMMA on RDNA/gfx1250, MMA on NVIDIA).
    mma_m: int
    mma_n: int
    max_warps: int
    # (head_dim, block_n, num_warps) -> staging layouts for the async body.
    dma_ladder: dict[tuple[int, int, int], DmaLayouts]

    def warps_for(self, block_m: int) -> int:
        """The matmul layout splits BLOCK_M across waves."""
        return min(self.max_warps, max(1, block_m // self.mma_m))

    def tiles_evenly(self, block_m: int, block_n: int) -> bool:
        """Blocks smaller than one matmul tile give a degenerate layout."""
        return block_m % self.mma_m == 0 and block_n % self.mma_n == 0


CDNA4_TARGET = GluonFlexTarget(
    name="cdna4",
    template_cls=Cdna4GluonTemplate,
    sync_template="gluon_flex_attention",
    async_template="gluon_flex_attention_async",
    mma_m=32,
    mma_n=32,
    max_warps=8,
    dma_ladder=CDNA4_DMA_LADDER,
)


@functools.cache
def _active_target() -> GluonFlexTarget | None:
    """The descriptor for this device, or None if no Gluon body targets it."""
    if not torch.version.hip:
        return None
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        return None
    if "gfx95" in arch:
        return CDNA4_TARGET
    return None


@functools.cache
def _get_gluon_flex_template(target: GluonFlexTarget, name: str) -> GluonTemplate:
    from .flex_attention import flex_attention_grid

    return target.template_cls(
        name=name,
        grid=flex_attention_grid,
        source=load_flex_template(name) + load_flex_template("utilities"),
        always_freeze_layout=True,
    )


def _async_body(
    target: GluonFlexTarget, conf: Any, kernel_options: dict[str, Any]
) -> tuple[str, DmaLayouts] | None:
    """This config's async body and its staging layouts, or None if unsupported.

    This *is* the gate: the async body renders its layout declarations from these
    values, so a config is offered exactly when the ladder has an entry for it.
    """
    name = target.async_template
    if name is None:
        return None
    qk_rounded = kernel_options.get("QK_HEAD_DIM_ROUNDED")
    v_rounded = kernel_options.get("V_HEAD_DIM_ROUNDED")
    # The ladder is keyed on one head dim; K^T and V are staged with the same one.
    if qk_rounded is None or qk_rounded != v_rounded:
        return None
    key = (qk_rounded, conf.block_n, kernel_options["num_warps"])
    layouts = target.dma_ladder.get(key)
    return None if layouts is None else (name, layouts)


class _ExtraConfig(NamedTuple):
    """A block shape offered on top of the ones flex proposes."""

    block_m: int
    block_n: int


# flex's own config list stops at BLOCK_M=128, but a taller query tile is what the
# hand-tuned gfx950 kernel runs: one K/V block then serves twice the query rows, so
# the DMA traffic per unit of math halves. Only usable when the sparse Q block is a
# multiple of it, which the loop below already checks.
EXTRA_CONFIGS = (_ExtraConfig(256, 64), _ExtraConfig(256, 32))


def _score_mod_is_identity(subgraphs) -> bool:
    """True when score_mod passes the score through unchanged.

    Inductor lowers an identity score_mod to a buffer that just reads the
    ``score`` input, so the template can then fold sm_scale into the same FMA
    that does the exp2 change of base instead of scaling the tile separately.
    """
    from torch._inductor import ir

    if not subgraphs:
        return False
    return isinstance(getattr(subgraphs[0], "data", None), ir.InputBuffer)


class GluonInductorChoices(InductorChoices):
    """InductorChoices that offers the Gluon flash-attention templates."""

    def uuid(self) -> str:
        return "gluon-flex-attention-v3"

    @override
    def append_flex_attention_choices(
        self,
        choices: list[Any],
        configs: list[Any],
        input_nodes: list[Any],
        subgraphs: list[Any],
        layout: Any,
        kernel_options: dict[str, Any],
        sparse_q_block_size: int,
        sparse_kv_block_size: int,
    ) -> list[Any]:
        target = _active_target()
        if not config.gluon_flex_attention or target is None:
            return choices

        query, _key, _value, logsumexp, max_scores = input_nodes[:5]
        if query.get_dtype() not in (torch.float16, torch.bfloat16):
            return choices

        # None of this depends on the block shape, so build it once rather than
        # per config.
        base_opts = kernel_options.copy()
        for k in list(base_opts.keys()):
            if k.startswith("fwd_"):
                base_opts[k[4:]] = base_opts.pop(k)
            elif k.startswith("bwd_"):
                base_opts.pop(k)
        base_opts["USE_TMA"] = False
        # LDS ring depth: measured 2 > 3 > 4 on gfx950 (deeper rings cost more
        # occupancy at D=128 than the extra overlap buys: 608 -> 559 TFLOPS
        # non-causal).
        base_opts["GLUON_NUM_BUF"] = 2
        base_opts["SCORE_MOD_IS_IDENTITY"] = _score_mod_is_identity(subgraphs)
        # The Gluon body schedules its own loop; keep Triton's software pipeliner
        # out of it.
        base_opts["num_stages"] = 1
        base_opts.setdefault("SPARSE_Q_BLOCK_SIZE", sparse_q_block_size)
        base_opts.setdefault("SPARSE_KV_BLOCK_SIZE", sparse_kv_block_size)
        # Validate against the effective sizes rather than the arguments: the
        # setdefaults above leave a caller-pinned value in place, and that is what
        # the kernel is built with. Same reasoning as the Triton path.
        sparse_q = base_opts["SPARSE_Q_BLOCK_SIZE"]
        sparse_kv = base_opts["SPARSE_KV_BLOCK_SIZE"]

        for conf in tuple(configs) + EXTRA_CONFIGS:
            if not target.tiles_evenly(conf.block_m, conf.block_n):
                continue
            if sparse_kv % conf.block_n != 0 or sparse_q % conf.block_m != 0:
                continue

            opts = dict(base_opts)
            opts["BLOCK_M"] = conf.block_m
            opts["BLOCK_N"] = conf.block_n
            opts["num_warps"] = target.warps_for(conf.block_m)
            # num_warps is consumed by the kernel constructor and never reaches
            # the template environment, so surface it under its own name too.
            opts["GLUON_NUM_WARPS"] = opts["num_warps"]

            # Offer the synchronous body always, and the async one wherever the
            # ladder has staging layouts, so autotuning picks between them.
            bodies: list[tuple[str, DmaLayouts | None]] = [(target.sync_template, None)]
            async_body = _async_body(target, conf, opts)
            if async_body is not None:
                bodies.append(async_body)

            for template_name, dma_layouts in bodies:
                body_opts = dict(opts)
                if dma_layouts is not None:
                    body_opts.update(as_template_options(dma_layouts))
                # Unrolling the KV loop pays off on some shapes and costs on
                # others (it helps D=64 causal, hurts D=128), so offer both and
                # let autotuning decide per shape.
                unroll_factors = (
                    (1, 2) if template_name == target.async_template else (1,)
                )
                for unroll in unroll_factors:
                    body_opts["GLUON_UNROLL"] = unroll
                    error = _get_gluon_flex_template(
                        target, template_name
                    ).maybe_append_choice(
                        choices=choices,
                        input_nodes=input_nodes,
                        layout=layout,
                        subgraphs=subgraphs,
                        mutated_inputs=[logsumexp, max_scores],
                        call_sizes=query.get_size(),
                        **body_opts,
                    )
                    if error is not None:
                        log.debug(
                            "gluon flex choice skipped: target=%s body=%s "
                            "BLOCK_M=%s BLOCK_N=%s unroll=%s: %s: %s",
                            target.name,
                            template_name,
                            conf.block_m,
                            conf.block_n,
                            unroll,
                            type(error).__name__,
                            error,
                        )
        return choices


def enable_gluon_flex_attention() -> None:
    """Turn on the Gluon flex-attention choices (currently AMD gfx95x only).

    Also serializes Triton compilation, because compiling a Gluon kernel while
    anything else compiles is unsafe in Triton's AMD backend: it segfaults the
    process from the autotune thread pool, and fails a compile worker with
    StopIteration. Serializing avoids both -- 8/8 clean against a ~50% crash rate,
    and 480/480 clean against failing within 18 to 293 compiles -- at roughly 2x
    compile time. Remove this once the underlying issue is fixed.

    TORCHINDUCTOR_COMPILE_THREADS is the lever rather than config.compile_threads
    because the autotune thread pool reads only the environment
    (select_algorithm.get_num_workers), while the subprocess pool goes through
    config. An explicit caller setting is left alone.
    """
    config.gluon_flex_attention = True
    config.inductor_choices_class = GluonInductorChoices
    if "TORCHINDUCTOR_COMPILE_THREADS" not in os.environ:
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        config.compile_threads = 1
