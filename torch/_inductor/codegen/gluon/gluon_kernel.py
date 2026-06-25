# mypy: allow-untyped-defs
"""Minimal template kernel for the experimental Gluon backend.

Stripped-down analogue of CuteDSLTemplateKernel: handles signature generation
(`def_kernel`/`get_output`), constexpr defines, and the kernel call emission.
It intentionally omits subgraph/modification rendering -- that path is
validated separately -- so this prototype focuses on the scheduling/codegen/
async-compile plumbing.
"""
from collections.abc import Callable
from typing import Any

from torch._inductor.codegen.common import IndentedBuffer, Kernel
from torch._inductor.ir import Buffer
from torch._inductor.utils import OrderedSet
from torch._inductor.virtualized import V


MAIN_SUFFIX = "main"


class GluonKernelWrapper:
    """Provides a `.run()` interface for a compiled Gluon launcher function."""

    def __init__(self, kernel_fn: Callable[..., Any], kernel_path: str | None = None):
        self.kernel_fn = kernel_fn
        self.kernel_path = kernel_path

    def run(self, *args, stream=None, **kwargs):
        return self.kernel_fn(*args, stream=stream, **kwargs)


class GluonTemplateKernel(Kernel):
    """Code generation + argument management for a Gluon template kernel."""

    def __init__(
        self,
        kernel_name: str,
        input_nodes: list[Buffer],
        output_node: Buffer,
    ) -> None:
        super().__init__()
        self.kernel_name = kernel_name
        self.input_nodes = input_nodes
        self.output_node = output_node
        self.render_hooks: dict[str, Any] = {}
        self.named_input_nodes: dict[str, Buffer] = {}
        for i, input_node in enumerate(input_nodes):
            node_name = getattr(input_node, "name", f"input_{i}")
            self.named_input_nodes[node_name] = input_node

    def gen_imports(self) -> str:
        imports = IndentedBuffer()
        imports.splice(
            """
            import torch
            import triton
            from triton.experimental import gluon
            from triton.experimental.gluon import language as gl
            """
        )
        return imports.getvalue()

    def gen_defines(self, **kwargs) -> str:
        params = IndentedBuffer()
        for name, val in kwargs.items():
            params.writeline(f"{name} = {val!r}")
        return params.getvalue()

    def def_kernel(self, *argnames):
        renames = IndentedBuffer(initial_indent=1)
        self._template_input_args: list[tuple[str, Buffer]] = []
        self._seen_input_args: OrderedSet[str] = OrderedSet()

        for i, input_node in enumerate(self.input_nodes):
            buf_name = input_node.get_name()
            self.args.input(buf_name)
            if i < len(argnames):
                template_name = argnames[i]
                arg_name = f"arg_{template_name}"
                self.args.input_buffers[buf_name] = arg_name
                renames.writeline(f"{template_name} = {arg_name}")
                self._template_input_args.append((arg_name, input_node))
                self._seen_input_args.add(arg_name)

        if self.output_node:
            self.args.output(self.output_node.get_name())

        def hook():
            code = IndentedBuffer()
            params = [arg_name for arg_name, _ in self._template_input_args]
            arg_defs, _, _, _ = self.args.python_argdefs()
            for arg_def in arg_defs:
                if arg_def.full_name() not in self._seen_input_args:
                    params.append(arg_def.full_name())
            params.append("stream")
            code.writeline(f"def {self.kernel_name}_{MAIN_SUFFIX}({', '.join(params)}):")
            with code.indent():
                code.splice(renames.getvalue())
            return code.getvalue()

        self.render_hooks["<DEF_KERNEL>"] = hook
        return "<DEF_KERNEL>"

    def get_output(self):
        buf_name = self.output_node.get_name()
        return self.args.output_buffers[buf_name]

    def render(self, template, **kwargs):
        from torch._inductor.select_algorithm import PartialRender

        template_env = {
            "def_kernel": self.def_kernel,
            "gen_defines": lambda: self.gen_defines(**kwargs),
            "get_output": self.get_output,
        }
        rendered = template.render(
            kernel_name=self.kernel_name,
            input_nodes=self.input_nodes,
            output_node=self.output_node,
            **template_env,
            **kwargs,
        )
        full_code = self.gen_imports() + rendered
        return PartialRender(full_code, self.render_hooks)

    def call_kernel(self, name: str, node=None):
        wrapper = V.graph.wrapper_code
        call_args = []
        arg_types = []
        for _, input_node in self._template_input_args:
            call_args.append(input_node.get_name())
            arg_types.append(V.graph.get_dtype(input_node.get_name()))

        arg_defs, py_call_args, _, py_arg_types = self.args.python_argdefs()
        for arg_def, call_arg, arg_type in zip(arg_defs, py_call_args, py_arg_types):
            if arg_def.full_name() in self._seen_input_args:
                continue
            call_args.append(call_arg)
            arg_types.append(arg_type)

        wrapper.generate_kernel_call(name, call_args, triton=True, arg_types=arg_types)
