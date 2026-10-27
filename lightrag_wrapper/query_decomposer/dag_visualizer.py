"""Draw a query plan and save it as a timestamped PNG."""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path

from lightrag_wrapper.models import QueryPlanNode

_DEFAULT_DIRECTORY = Path("data") / "query_dags"
_RESULT_FRAME = "#9f1239"
_NODE_COLORS = {
    "retrieve": "#dbeafe",
    "filter": "#fef3c7",
    "transform": "#e0e7ff",
    "aggregate": "#d1fae5",
    "compare": "#fce7f3",
    "assemble": "#ffedd5",
    "evaluate": "#f3e8ff",
    "synthesize": "#fee2e2",
}


class DAGVisualizer:
    """Save a retrieval DAG as a PNG whose name ends with the current time."""

    def __init__(self, directory: str | Path = _DEFAULT_DIRECTORY) -> None:
        self._directory = Path(directory)

    def save_dag_visual(
        self,
        nodes: list[QueryPlanNode],
        directory: str | Path | None = None,
    ) -> Path:
        """Draw ``nodes`` and save a PNG named with the current time."""
        output_dir = self._directory if directory is None else Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = output_dir / f"query_dag_{stamp}.png"
        self._write_png(nodes, path)
        return path

    def _write_png(self, nodes: list[QueryPlanNode], path: Path) -> None:
        """Render the plan top to bottom, with arrows from each dependency."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch

        layers = self._layers(nodes)
        positions = self._positions(nodes, layers)
        xs = [point[0] for point in positions.values()]
        ys = [point[1] for point in positions.values()]
        span = max(xs) - min(xs)

        figure, axes = plt.subplots(
            figsize=(max(6.0, (span + 2.0) * 1.9), max(3.5, (max(layers.values()) + 1) * 2.4))
        )
        axes.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
        axes.set_ylim(min(ys) - 0.55, max(ys) + 0.55)
        axes.set_axis_off()
        axes.set_aspect("equal")

        extents = {node.id: self._box_extent(node) for node in nodes}
        for node in nodes:
            target_x, target_y = positions[node.id]
            _, target_half_height = extents[node.id]
            for dependency in node.depends_on:
                source_x, source_y = positions[dependency]
                _, source_half_height = extents[dependency]
                axes.add_patch(
                    FancyArrowPatch(
                        (source_x, source_y - source_half_height - 0.02),
                        (target_x, target_y + target_half_height + 0.02),
                        arrowstyle="-|>",
                        mutation_scale=12,
                        linewidth=1.1,
                        color="#64748b",
                        connectionstyle="arc3,rad=0.0",
                    )
                )

        for node in nodes:
            self._draw_node(axes, node, *positions[node.id])

        figure.tight_layout()
        figure.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(figure)

    def _box_extent(self, node: QueryPlanNode) -> tuple[float, float]:
        """Half-width and half-height of the box drawn for ``node``."""
        if node.is_result_node:
            return 0.78, 0.42
        return 0.72, 0.32

    def _draw_node(self, axes, node: QueryPlanNode, center_x: float, center_y: float) -> None:
        from matplotlib.patches import FancyBboxPatch

        fill = _NODE_COLORS.get(node.node_type, "#f8fafc")
        summary = textwrap.fill(node.subquery.strip(), width=28)
        if summary.count("\n") > 2:
            summary = "\n".join(summary.splitlines()[:3]).rstrip(".") + "..."
        if node.is_result_node:
            axes.add_patch(
                FancyBboxPatch(
                    (center_x - 0.78, center_y - 0.40),
                    1.56,
                    0.82,
                    boxstyle="round,pad=0.02,rounding_size=0.08",
                    facecolor=_RESULT_FRAME,
                    edgecolor=_RESULT_FRAME,
                    linewidth=1.0,
                )
            )
            axes.text(
                center_x,
                center_y + 0.33,
                "RESULT",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="#ffffff",
            )
            body_origin = (center_x - 0.72, center_y - 0.34)
            body_size = (1.44, 0.58)
        else:
            body_origin = (center_x - 0.72, center_y - 0.32)
            body_size = (1.44, 0.64)
        axes.add_patch(
            FancyBboxPatch(
                body_origin,
                *body_size,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=fill,
                edgecolor="#334155",
                linewidth=1.0,
            )
        )
        axes.text(
            center_x,
            center_y + 0.08,
            f"{node.id}. {node.node_type}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#0f172a",
        )
        axes.text(
            center_x,
            center_y - 0.12,
            summary,
            ha="center",
            va="center",
            fontsize=6.5,
            color="#334155",
        )

    def _positions(
        self,
        nodes: list[QueryPlanNode],
        layers: dict[int, int],
    ) -> dict[int, tuple[float, float]]:
        """Seat each node under the nodes it depends on, without overlapping boxes."""
        x_gap = 1.9
        columns: dict[int, list[QueryPlanNode]] = {}
        for node in sorted(nodes, key=lambda item: item.id):
            columns.setdefault(layers[node.id], []).append(node)

        horizontal: dict[int, float] = {}
        for layer in sorted(columns):
            column = columns[layer]
            if layer == 0:
                for index, node in enumerate(column):
                    horizontal[node.id] = index * x_gap
            else:
                for node in column:
                    parents = [horizontal[dependency] for dependency in node.depends_on]
                    horizontal[node.id] = sum(parents) / len(parents)
            ordered = sorted(column, key=lambda item: (horizontal[item.id], item.id))
            for index in range(1, len(ordered)):
                previous = ordered[index - 1].id
                current = ordered[index].id
                if horizontal[current] < horizontal[previous] + x_gap:
                    horizontal[current] = horizontal[previous] + x_gap

        return {node.id: (horizontal[node.id], -layers[node.id]) for node in nodes}

    def _layers(self, nodes: list[QueryPlanNode]) -> dict[int, int]:
        """Place each node one row below the deepest node it depends on."""
        by_id = {node.id: node for node in nodes}
        layers: dict[int, int] = {}

        def layer_of(node: QueryPlanNode) -> int:
            if node.id in layers:
                return layers[node.id]
            if not node.depends_on:
                layers[node.id] = 0
                return 0
            layers[node.id] = 1 + max(layer_of(by_id[dependency]) for dependency in node.depends_on)
            return layers[node.id]

        for node in nodes:
            layer_of(node)
        return layers
