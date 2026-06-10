"""Trajectory visualization and failure analysis for embodied agent evaluation.

Generates top-down trajectory plots, episode summary images, and a
categorised failure analysis report in markdown format.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.decision.trace import EpisodeTrace


# ---------------------------------------------------------------------------
# Colour / style constants
# ---------------------------------------------------------------------------

_COLORS = {
    "rule":      "#2196F3",   # blue
    "rule+vlm":  "#FF9800",   # orange
    "llm":       "#4CAF50",   # green
}
_SUCCESS_GREEN = "#4CAF50"
_FAIL_RED = "#F44336"
_TARGET_GOLD = "#FFD700"
_SPAWN_CYAN = "#00BCD4"
_REACHABLE_GRAY = "#E0E0E0"


# ---------------------------------------------------------------------------
# Trajectory plot
# ---------------------------------------------------------------------------

def plot_trajectory(trace: EpisodeTrace, output_path: str) -> str:
    """Plot a top-down (XZ) trajectory for a single episode.

    Returns *output_path* on success.
    """
    color = _COLORS.get(trace.policy, "#333333")
    status_color = _SUCCESS_GREEN if trace.success else _FAIL_RED
    status_text = "SUCCESS" if trace.success else "FAIL"

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")

    # --- Reachable positions (background) ---
    if trace.reachable_positions:
        rx, rz = zip(*trace.reachable_positions)
        ax.scatter(rx, rz, c=_REACHABLE_GRAY, s=2, alpha=0.5, zorder=0,
                   label="reachable")

    # --- Spawn point ---
    ax.scatter(trace.spawn_x, trace.spawn_z, c=_SPAWN_CYAN, s=120, marker="o",
               edgecolors="black", linewidths=0.5, zorder=4, label="spawn")

    # --- Target position ---
    if trace.target_x is not None and trace.target_z is not None:
        ax.scatter(trace.target_x, trace.target_z, c=_TARGET_GOLD, s=180,
                   marker="*", edgecolors="black", linewidths=0.5,
                   zorder=4, label=f"target: {trace.target}")

    # --- Agent trajectory ---
    if trace.steps:
        xs = [trace.spawn_x] + [s.x for s in trace.steps]
        zs = [trace.spawn_z] + [s.z for s in trace.steps]
        n = len(xs)
        # Colour gradient from blue (start) to red (end)
        for i in range(n - 1):
            t_val = i / max(n - 1, 1)
            seg_color = (t_val, 0.3, 1.0 - t_val)  # blue → red
            ax.plot(xs[i:i+2], zs[i:i+2], color=seg_color, linewidth=1.5,
                    alpha=0.8, zorder=2)

        # Start / end markers
        ax.scatter(xs[0], zs[0], c=_SPAWN_CYAN, s=80, marker="s", zorder=5)
        ax.scatter(xs[-1], zs[-1], c=status_color, s=100, marker="X",
                   edgecolors="black", linewidths=0.5, zorder=5,
                   label="end")

        # Collision markers (small red x)
        for s in trace.steps:
            if s.is_colliding:
                ax.scatter(s.x, s.z, c="red", s=20, marker="x", alpha=0.6,
                           zorder=3)

        # Direction arrows every N steps
        arrow_every = max(1, n // 8)
        for i in range(0, n - 1, arrow_every):
            if i < len(trace.steps):
                dx = xs[i+1] - xs[i]
                dz = zs[i+1] - zs[i]
                dist = (dx**2 + dz**2) ** 0.5
                if dist > 0.01:
                    dx, dz = dx / dist * 0.15, dz / dist * 0.15
                    ax.arrow(xs[i], zs[i], dx, dz, head_width=0.08,
                             head_length=0.08, fc=color, ec=color,
                             alpha=0.5, zorder=3)

    # --- Labels ---
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title(
        f"[{trace.policy}] {trace.task}\n"
        f"{status_text} | {trace.total_steps} steps | "
        f"{trace.elapsed_s:.1f}s | SPL={trace.spl:.3f}\n"
        f"{trace.final_reason[:100]}",
        fontsize=9, color=status_color,
    )
    ax.legend(loc="upper right", fontsize=7, markerscale=0.6)
    ax.grid(True, alpha=0.3)

    # Auto-zoom with padding
    all_x = [trace.spawn_x] + [s.x for s in trace.steps]
    all_z = [trace.spawn_z] + [s.z for s in trace.steps]
    if trace.target_x is not None:
        all_x.append(trace.target_x)
        all_z.append(trace.target_z)
    if all_x:
        pad = max(1.0, (max(all_x) - min(all_x)) * 0.2)
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax.set_ylim(min(all_z) - pad, max(all_z) + pad)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Episode summary (trajectory + condensed action log)
# ---------------------------------------------------------------------------

def plot_episode_summary(trace: EpisodeTrace, output_path: str) -> str:
    """Plot trajectory on the left, condensed action log on the right.

    Returns *output_path* on success.
    """
    color = _COLORS.get(trace.policy, "#333333")
    status_color = _SUCCESS_GREEN if trace.success else _FAIL_RED
    status_text = "SUCCESS" if trace.success else "FAIL"

    fig = plt.figure(figsize=(16, 8))

    # --- Left: trajectory ---
    ax_traj = fig.add_subplot(1, 2, 1)
    ax_traj.set_aspect("equal")

    if trace.reachable_positions:
        rx, rz = zip(*trace.reachable_positions)
        ax_traj.scatter(rx, rz, c=_REACHABLE_GRAY, s=2, alpha=0.5, zorder=0)

    ax_traj.scatter(trace.spawn_x, trace.spawn_z, c=_SPAWN_CYAN, s=100,
                    marker="o", edgecolors="black", linewidths=0.5, zorder=4)

    if trace.target_x is not None:
        ax_traj.scatter(trace.target_x, trace.target_z, c=_TARGET_GOLD, s=150,
                        marker="*", edgecolors="black", linewidths=0.5, zorder=4)

    if trace.steps:
        xs = [trace.spawn_x] + [s.x for s in trace.steps]
        zs = [trace.spawn_z] + [s.z for s in trace.steps]
        n = len(xs)
        for i in range(n - 1):
            t_val = i / max(n - 1, 1)
            ax_traj.plot(xs[i:i+2], zs[i:i+2],
                         color=(t_val, 0.3, 1.0 - t_val), linewidth=1.5,
                         alpha=0.8, zorder=2)
        ax_traj.scatter(xs[-1], zs[-1], c=status_color, s=80, marker="X",
                        edgecolors="black", linewidths=0.5, zorder=5)
        for s in trace.steps:
            if s.is_colliding:
                ax_traj.scatter(s.x, s.z, c="red", s=15, marker="x", alpha=0.5,
                                zorder=3)

    ax_traj.set_xlabel("X (m)"); ax_traj.set_ylabel("Z (m)")
    ax_traj.set_title(f"Trajectory — {trace.scene}", fontsize=10)
    ax_traj.grid(True, alpha=0.3)

    all_x = [trace.spawn_x] + [s.x for s in trace.steps]
    all_z = [trace.spawn_z] + [s.z for s in trace.steps]
    if trace.target_x is not None:
        all_x.append(trace.target_x); all_z.append(trace.target_z)
    if all_x:
        pad = max(1.0, (max(all_x) - min(all_x)) * 0.2)
        ax_traj.set_xlim(min(all_x) - pad, max(all_x) + pad)
        ax_traj.set_ylim(min(all_z) - pad, max(all_z) + pad)

    # --- Right: action log ---
    ax_log = fig.add_subplot(1, 2, 2)
    ax_log.axis("off")

    lines = [
        f"Scene: {trace.scene}",
        f"Policy: {trace.policy}",
        f"Task: {trace.task}",
        f"Target: {trace.target}  |  Type: {trace.task_type}",
        f"Result: {status_text}  |  Steps: {trace.total_steps}  |  "
        f"Time: {trace.elapsed_s:.1f}s",
        f"SPL: {trace.spl:.3f}  |  Optimal: {trace.optimal_steps:.0f} steps",
        f"Reason: {trace.final_reason[:100]}",
        "",
        f"{'─'*60}",
        f"{'Step':<5s} {'Action':<18s} {'Collide':<8s} {'Seen':<6s} "
        f"{'Dist':>6s}  Reason",
        f"{'─'*60}",
    ]

    for s in trace.steps:
        seen = "✓" if s.target_detected else "—"
        dist_str = f"{s.target_distance:.1f}m" if s.target_distance > 0 else "—"
        collide = "✗" if s.is_colliding else ""
        lines.append(
            f"{s.step:<5d} {s.action:<18s} {collide:<8s} {seen:<6s} "
            f"{dist_str:>6s}  {s.reason[:55]}"
        )

    log_text = "\n".join(lines)
    ax_log.text(0, 1, log_text, transform=ax_log.transAxes,
                fontfamily="monospace", fontsize=7, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    fig.suptitle(
        f"[{trace.policy}] {trace.task} — {status_text}",
        fontsize=12, color=status_color, fontweight="bold",
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Failure categorisation
# ---------------------------------------------------------------------------

def _categorise(reason: str) -> str:
    """Map a failure reason string to a category."""
    r = reason.lower()
    if "a* failed" in r or "a* navigation" in r:
        return "A* 不可达"
    if "not in spatial memory" in r or "not visible during room tour" in r:
        return "感知缺失（不在空间记忆）"
    if "not detected" in r or "target not in scene" in r:
        return "感知缺失（YOLO 未检测到）"
    if "max steps" in r:
        return "超时（Max steps）"
    if "vlm exhausted" in r:
        return "VLM 耗尽"
    if "stopped" in r and "m" in r:
        return "提前停止（距离过远）"
    if "parse error" in r:
        return "指令解析失败"
    if "blocked" in r or "stuck" in r or "collision" in r or "colliding" in r:
        return "靠近卡住（碰撞/阻塞）"
    return "其他"


def generate_failure_report(traces: list[EpisodeTrace],
                            output_path: str) -> str:
    """Generate a markdown failure-analysis report.

    Groups failures by category, includes statistics and per-episode
    details with embedded trajectory images.

    Returns *output_path*.
    """
    failures = [t for t in traces if not t.success]
    if not failures:
        # Write a minimal report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Failure Analysis Report\n\n")
            f.write(f"**Total episodes**: {len(traces)}\n\n")
            f.write("✅ No failures — all episodes succeeded.\n")
        return output_path

    # Categorise
    by_cat: dict[str, list[EpisodeTrace]] = defaultdict(list)
    for t in failures:
        cat = _categorise(t.final_reason)
        by_cat[cat].append(t)

    total = len(traces)
    fail_count = len(failures)

    lines = [
        "# Failure Analysis Report",
        "",
        f"**Total episodes**: {total}  |  **Failures**: {fail_count}  |  "
        f"**Failure rate**: {fail_count/total*100:.1f}%",
        "",
        "## Failure Distribution",
        "",
        "| Category | Count | % of Failures | % of Total |",
        "|----------|-------|---------------|------------|",
    ]

    for cat in sorted(by_cat, key=lambda c: -len(by_cat[c])):
        cnt = len(by_cat[cat])
        lines.append(f"| {cat} | {cnt} | {cnt/fail_count*100:.1f}% | "
                     f"{cnt/total*100:.1f}% |")

    lines += [
        "",
        "---",
        "",
        "## Per-Category Details",
        "",
    ]

    for cat in sorted(by_cat, key=lambda c: -len(by_cat[c])):
        cat_traces = by_cat[cat]
        lines += [
            f"### {cat} ({len(cat_traces)} episodes)",
            "",
        ]
        # Per-episode table
        lines.append(
            "| Scene | Policy | Task | Target | Type | Steps | Reason |"
        )
        lines.append(
            "|-------|--------|------|--------|------|-------|--------|"
        )
        for t in cat_traces[:15]:  # limit per category
            reason_short = t.final_reason[:70]
            lines.append(
                f"| {t.scene} | {t.policy} | {t.task} | {t.target} | "
                f"{t.task_type} | {t.total_steps} | {reason_short} |"
            )
        if len(cat_traces) > 15:
            lines.append(f"| ... | | {len(cat_traces)-15} more episodes | | | | |")
        lines.append("")

        # Typical trajectory — pick the one with median steps
        sorted_traces = sorted(cat_traces, key=lambda t: t.total_steps)
        median_trace = sorted_traces[len(sorted_traces) // 2]
        trace_dir = os.path.dirname(output_path)
        png_name = _safe_filename(median_trace) + "_summary.png"
        png_path = os.path.join(trace_dir, png_name)
        if os.path.exists(png_path):
            rel = os.path.relpath(png_path, os.path.dirname(output_path))
            lines.append(f"![{median_trace.task}]({rel})")
        lines.append("")

    lines += [
        "---",
        "",
        "## Recommendations",
        "",
    ]
    # Auto-generate recommendations based on top categories
    top_cats = sorted(by_cat, key=lambda c: -len(by_cat[c]))[:3]
    recs = {
        "A* 不可达":
            "- 检查空间记忆中目标位置的精度，A* 邻居搜索范围从 0.5m 扩大到 1.0m\n"
            "- 对 A* 失败的目标，尝试用 VLM 做纯视觉导航",
        "感知缺失（不在空间记忆）":
            "- Room tour 增加采样点或降低检测阈值\n"
            "- 对不在空间记忆的目标，启用 VLM 语义搜索",
        "感知缺失（YOLO 未检测到）":
            "- 继续微调 YOLO 模型提升小物体检测能力\n"
            "- 考虑用 CLIP 零样本检测补充 YOLO 漏检类别",
        "靠近卡住（碰撞/阻塞）":
            "- 扩大绕行步数或增加 MOVE_BACK 退避策略\n"
            "- 对无法物理到达的物体（柜台内），A* 到达 1.5m 即判定导航成功",
        "提前停止（距离过远）":
            "- 调大导航成功阈值或增加靠近步数上限\n"
            "- 检查 A* 是否到达了错误的位置（空间记忆坐标不准）",
        "超时（Max steps）":
            "- 增加 max_steps 或优化决策效率\n"
            "- 检查是否陷入反复碰撞→恢复循环",
        "VLM 耗尽":
            "- 增加 VLM 探索步数（15→20）或允许更多轮次（3→5）\n"
            "- VLM prompt 增加场景布局描述辅助决策",
    }
    for cat in top_cats:
        if cat in recs:
            lines.append(f"**{cat}**:")
            lines.append(recs[cat])
            lines.append("")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_filename(trace: EpisodeTrace) -> str:
    """Generate a filesystem-safe name for an episode trace."""
    task_slug = re.sub(r"[^a-zA-Z0-9]+", "_", trace.task).strip("_")[:30]
    return f"{trace.scene}_{trace.policy}_{task_slug}"
