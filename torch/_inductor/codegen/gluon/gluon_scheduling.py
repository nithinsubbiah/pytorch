# mypy: allow-untyped-defs
"""Minimal scheduling for the experimental Gluon template backend.

Delegated to by CUDACombinedScheduling for GluonTemplateBuffer nodes. No
fusion support in this prototype.
"""
import hashlib
import logging
from collections.abc import Sequence
from typing import cast

from torch._inductor.utils import Placeholder
from torch.utils._ordered_set import OrderedSet

from ... import config
from ...codecache import code_hash, get_path
from ...scheduler import BaseSchedulerNode, BaseScheduling, SchedulerNode
from ...select_algorithm import PartialRender
from ...utils import get_fused_kernel_name, get_kernel_metadata
from ...virtualized import V
from ..common import BackendFeature, IndentedBuffer
from .gluon_template import GluonTemplateBuffer


log = logging.getLogger(__name__)


class GluonScheduling(BaseScheduling):
    @classmethod
    def get_backend_features(cls, device) -> OrderedSet[BackendFeature]:
        return OrderedSet()

    @staticmethod
    def is_gluon_template(node: BaseSchedulerNode) -> bool:
        return isinstance(node, SchedulerNode) and isinstance(
            node.node, GluonTemplateBuffer
        )

    def can_fuse_vertical(self, node1, node2) -> bool:
        return False

    def can_fuse_horizontal(self, node1, node2) -> bool:
        return False

    def define_kernel(self, src_code_str: str, node_schedule) -> str:
        wrapper = V.graph.wrapper_code
        if src_code_str in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code_str]

        fused_name = (
            get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
            if config.triton.descriptive_names
            else ""
        )
        kernel_hash = hashlib.sha256(src_code_str.encode("utf-8")).hexdigest()[:8]
        kernel_name = f"gluon_{fused_name}_{kernel_hash}" if fused_name else f"gluon_{kernel_hash}"
        wrapper.src_to_kernel[src_code_str] = kernel_name
        src_code_str = src_code_str.replace(str(Placeholder.KERNEL_NAME), kernel_name)

        _, _, kernel_path = get_path(code_hash(src_code_str), "py")
        compile_wrapper = IndentedBuffer()
        compile_wrapper.writeline(f"async_compile.gluon({kernel_name!r}, r'''")
        compile_wrapper.splice(src_code_str, strip=True)
        compile_wrapper.writeline("''')")

        metadata_comment = f"# kernel path: {kernel_path}"
        origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
        metadata_comment += "\n" + origins + "\n" + detailed_origins
        wrapper.define_kernel(kernel_name, compile_wrapper.getvalue(), metadata_comment)
        return kernel_name

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ):
        if epilogue_nodes:
            raise AssertionError("Gluon template does not support epilogue nodes")
        if prologue_nodes:
            raise AssertionError("Gluon template does not support prologue nodes")

        template_node = cast(SchedulerNode, template_node)
        gtb: GluonTemplateBuffer = cast(GluonTemplateBuffer, template_node.node)

        kernel, render = gtb.make_kernel_render(gtb)
        template_node.mark_run()
        src_code = render()
        src_code_str = src_code.finalize_all() if isinstance(src_code, PartialRender) else src_code

        with V.set_kernel_handler(kernel):
            node_schedule = [template_node]
            kernel_name = self.define_kernel(src_code_str, node_schedule)
        self.codegen_comment(node_schedule, kernel_name)
        kernel.call_kernel(kernel_name, gtb)
        V.graph.removed_buffers |= kernel.removed_buffers
        self.free_buffers_in_scheduler()
