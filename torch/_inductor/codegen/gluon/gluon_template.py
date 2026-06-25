# mypy: allow-untyped-defs
"""Minimal Gluon template + choice caller + template buffer.

Mirrors the CuteDSL template surface closely enough to participate in
autotuning and route through CUDACombinedScheduling, but trimmed to the
essentials for a prototype.
"""
import itertools
from collections.abc import Callable, Iterable
from typing import Any

from torch._inductor.codegen.common import KernelTemplate
from torch._inductor.ir import (
    Buffer,
    ChoiceCaller,
    IRNode,
    Layout,
    TemplateBuffer,
    TensorBox,
)
from torch._inductor.utils import Placeholder
from torch._inductor.virtualized import V

from .gluon_kernel import GluonTemplateKernel


class GluonTemplateBuffer(TemplateBuffer):
    """Template buffer for Gluon kernels; routed to GluonScheduling."""

    def __init__(
        self,
        layout: Layout,
        inputs,
        make_kernel_render: Callable[..., Any],
        template: Any,
        mutated_inputs: Iterable[IRNode] | None = None,
    ) -> None:
        super().__init__(layout, inputs, make_kernel_render)
        self.template = template
        self.mutated_inputs = mutated_inputs
        self.outputs: list[Buffer] = [self]

    def get_outputs(self) -> list[Buffer]:
        return self.outputs


class GluonTemplate(KernelTemplate):
    """Template for generating Gluon kernels."""

    index_counter = itertools.count()
    all_templates: dict[str, "GluonTemplate"] = {}

    def __init__(self, name: str, source: str) -> None:
        super().__init__(name)
        self.source = source
        self.template = KernelTemplate._template_from_string(source)
        GluonTemplate.all_templates[name] = self

    def maybe_append_choice(self, choices: list[Any], **kwargs: Any):
        try:
            choices.append(self.generate(**kwargs))
            return None
        except NotImplementedError as e:
            return e
        except Exception as e:  # noqa: BLE001
            return NotImplementedError(f"Gluon template failed: {e}")

    def generate(self, **kwargs: Any) -> ChoiceCaller:
        input_nodes = kwargs.pop("input_nodes")
        layout = kwargs.pop("layout")
        mutated_inputs = kwargs.pop("mutated_inputs", None)
        bench_fn = kwargs.pop("bench_fn", None)
        template_kwargs = dict(kwargs)

        kernel_name = f"gluon_{self.name}_{next(self.index_counter)}"
        self.output_node: Buffer = Buffer(name="buf_out", layout=layout)

        def make_kernel_render(out_node, hint_override: int | None = None):
            render_kernel = GluonTemplateKernel(
                kernel_name=str(Placeholder.KERNEL_NAME),
                input_nodes=input_nodes,
                output_node=out_node,
            )

            def render():
                return render_kernel.render(self.template, **template_kwargs)

            return render_kernel, render

        return GluonTemplateCaller(
            name=kernel_name,
            input_nodes=input_nodes,
            layout=layout,
            make_kernel_render=make_kernel_render,
            template=self,
            mutated_inputs=mutated_inputs,
            template_kwargs=template_kwargs,
            bench_fn=bench_fn,
        )


class GluonTemplateCaller(ChoiceCaller):
    """ChoiceCaller for Gluon templates."""

    def __init__(
        self,
        name: str,
        input_nodes,
        layout: Layout,
        make_kernel_render: Any,
        template: "GluonTemplate",
        mutated_inputs: Iterable[IRNode] | None = None,
        template_kwargs: dict[str, Any] | None = None,
        bench_fn: Callable[..., None] | None = None,
    ):
        desc = f"Gluon template {name}"
        if template_kwargs:
            desc += " (" + ", ".join(f"{k}={v}" for k, v in template_kwargs.items()) + ")"
        super().__init__(name=name, input_nodes=input_nodes, layout=layout, description=desc)
        self.make_kernel_render = make_kernel_render
        self.template = template
        self.mutated_inputs = mutated_inputs
        self._bench_fn = bench_fn

    def __str__(self) -> str:
        return f"GluonTemplateCaller({self.name})"

    def benchmark(self, *args, out) -> float:
        # Prototype: run the in-process kernel closure if provided, else treat
        # as free so autotuning can still select it.
        if self._bench_fn is not None:
            import torch

            *inputs, _ = args
            self._bench_fn(*inputs, out)
            torch.cuda.synchronize()
        return 0.0

    def output_node(self) -> TensorBox:
        buffer = GluonTemplateBuffer(
            layout=self.layout,
            inputs=self.input_nodes,
            make_kernel_render=self.make_kernel_render,
            template=self.template,
            mutated_inputs=self.mutated_inputs,
        )
        return TensorBox.create(buffer)

    def call_name(self) -> str:
        return self.name

    def to_callable(self):
        return self.make_kernel_render

    def hash_key(self) -> str:
        return "-".join([self.name.rsplit("_", 1)[0], self.template.name])

    def info_dict(self) -> dict[str, Any]:
        return {"name": self.name, "backend": "Gluon", "template": self.template.name}
