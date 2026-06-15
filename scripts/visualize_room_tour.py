#!/usr/bin/env python3
"""
Room Tour 可视化脚本 —— 展示机器人如何进行预探索（巡视房间）。

功能：
  1. 显示场景的俯视地图（可达位置 + 巡游路径）
  2. 在每个巡游点保存 360° 扫描的标注帧
  3. 输出空间记忆摘要（哪些物体被找到，在什么位置）
  4. 生成巡游过程的 MP4 视频

用法::

    # 默认场景
    python scripts/visualize_room_tour.py

    # 指定场景和巡游点数
    python scripts/visualize_room_tour.py --scene FloorPlan3 --stops 5

    # 使用微调模型 + YOLO-World 混合检测
    python scripts/visualize_room_tour.py --scene FloorPlan10 --detector hybrid

    # 生成更多输出（轨迹图 + 检测帧）
    python scripts/visualize_room_tour.py --scene FloorPlan1 --export-frames --export-video

输出::

    outputs/room_tour_<场景>_<时间戳>/
      ├── tour_map.png          ← 俯视巡游路径图
      ├── spatial_memory.txt    ← 空间记忆文本摘要
      ├── stop_01/              ← 巡游点 1 的扫描帧
      │     ├── scan_000.png
      │     ├── scan_001.png
      │     └── ...
      ├── stop_02/
      ├── tour_trajectory.mp4   ← 巡游轨迹视频（如果 --export-video）
      └── tour_frames/          ← 全部标注帧（如果 --export-frames）
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller.thor import ThorController
from src.perception.detector import create_hybrid, create_detector, YOLODetector
from src.perception.depth import HeuristicDepth
from src.perception.pipeline import PerceptionPipeline
from src.perception.prior import ScenePrior
from src.perception.spatial_memory import SpatialMemory
from src.common.types import Vec3
from src.common.logger import setup_logger
from src.common.utils import draw_annotations

logger = setup_logger("room_tour_viz")

# ===========================================================================
# 俯视地图绘制
# ===========================================================================

def draw_tour_map(
    reachable: list[Vec3],
    stops: list[Vec3],
    tour_path: list[tuple[float, float]],
    memory: SpatialMemory,
    scene_name: str,
    output_path: str,
    agent_spawn: Vec3 | None = None,
) -> None:
    """绘制巡游俯视地图。

    - 灰色点：可达位置
    - 绿色星标：巡游停靠点
    - 蓝色线：巡游路径
    - 红色圆点：空间记忆中物体位置
    - 橙色三角：出生点
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))

    # 可达位置（灰色小点，采样以加速渲染）
    if reachable:
        sample = reachable if len(reachable) <= 5000 else reachable[::max(1, len(reachable) // 5000)]
        rx = [p.x for p in sample]
        rz = [p.z for p in sample]
        ax.scatter(rx, rz, c="#e0e0e0", s=2, alpha=0.5, label="Reachable positions", zorder=1)

    # 巡游路径
    if len(tour_path) >= 2:
        px = [p[0] for p in tour_path]
        pz = [p[1] for p in tour_path]
        ax.plot(px, pz, c="#2196F3", linewidth=2, alpha=0.8, label="Tour path", zorder=3)
        # 起点
        ax.scatter([px[0]], [pz[0]], c="#FF9800", s=180, marker="^",
                   edgecolors="black", linewidth=0.8, label="Start (spawn)", zorder=5)

    # 停靠点
    if stops:
        sx = [p.x for p in stops]
        sz = [p.z for p in stops]
        ax.scatter(sx, sz, c="#4CAF50", s=150, marker="*",
                   edgecolors="black", linewidth=0.8, label=f"Tour stops ({len(stops)})", zorder=4)
        for i, s in enumerate(stops):
            ax.annotate(f" {i+1}", (s.x, s.z), fontsize=9, fontweight="bold",
                        color="#2E7D32", zorder=6)

    # 空间记忆中的物体位置
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    color_idx = 0
    for label, positions in sorted(memory._index.items()):
        if positions:
            c = colors[color_idx % len(colors)]
            color_idx += 1
            ox = [p.x for p in positions]
            oz = [p.z for p in positions]
            ax.scatter(ox, oz, c=[c], s=30, alpha=0.7, zorder=2)
            # 只标注第一个（最佳位置）
            ax.annotate(label, (ox[0], oz[0]), fontsize=6, alpha=0.8,
                        xytext=(3, 3), textcoords="offset points", zorder=6)

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Z (meters)")
    ax.set_title(f"Room Tour — {scene_name}\n{len(stops)} stops, {len(memory)} objects in spatial memory")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # 自动调整视图范围
    if reachable:
        all_x = [p.x for p in reachable]
        all_z = [p.z for p in reachable]
        margin = 1.0
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_z) - margin, max(all_z) + margin)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Tour map saved → {output_path}")


# ===========================================================================
# 巡游核心逻辑
# ===========================================================================

def _pick_tour_stops(ctrl: ThorController, num_stops: int = 4) -> list[Vec3]:
    """从可达位置中选出分散的巡游停靠点。"""
    positions = ctrl._get_reachable_positions_raw()
    if len(positions) < num_stops + 1:
        return list(positions[:max(1, len(positions))])

    # 按距质心距离排序
    cx = sum(p.x for p in positions) / len(positions)
    cz = sum(p.z for p in positions) / len(positions)
    ranked = sorted(positions, key=lambda p: -(p.x - cx) ** 2 - (p.z - cz) ** 2)

    stops = []
    for p in ranked:
        if len(stops) >= num_stops:
            break
        if not any((p.x - s.x) ** 2 + (p.z - s.z) ** 2 < 1.5 for s in stops):
            stops.append(p)
    return stops


def _build_spatial_memory_full(
    ctrl: ThorController,
    pipeline: PerceptionPipeline,
    stops: list[Vec3],
    *,
    output_dir: str,
    export_frames: bool = False,
    verbose: bool = True,
) -> tuple[SpatialMemory, list[tuple[float, float]]]:
    """执行完整巡游：走到每个停靠点 → 360° 扫描 → 建立空间记忆。

    Returns:
        (spatial_memory, tour_path) — 空间记忆和巡游路径。
    """
    mem = SpatialMemory()
    tour_path: list[tuple[float, float]] = []

    # 记录出生点
    spawn = ctrl.agent_state.position
    tour_path.append((spawn.x, spawn.z))

    for stop_idx, stop_pos in enumerate(stops):
        if verbose:
            print(f"\n  {'─'*50}")
            print(f"  📍 Stop {stop_idx + 1}/{len(stops)}: target ({stop_pos.x:.2f}, {stop_pos.z:.2f})")

        # A* 导航到停靠点，沿途记录路径
        nav_steps = 0
        nav_ok = False
        try:
            for result in ctrl.navigate_to(stop_pos):
                nav_steps += 1
                p = result.agent_state.position
                tour_path.append((p.x, p.z))
                # 小延迟降低 AI2-THOR 连续渲染的崩溃概率
                if nav_steps % 4 == 0:
                    time.sleep(0.02)
            nav_ok = True
        except Exception as e:
            if verbose:
                print(f"     ⚠ Navigation interrupted: {e}")
            # 尝试恢复：做一次 Pass 让 AI2-THOR 重新同步
            try:
                ctrl.step("MOVE_BACK")
                time.sleep(0.1)
            except Exception:
                pass

        # 记录导航终点 vs 目标停靠点的偏差
        current = ctrl.agent_state.position
        dist_to_stop = math.sqrt(
            (current.x - stop_pos.x) ** 2 + (current.z - stop_pos.z) ** 2
        )
        if verbose:
            status = "OK" if nav_ok else "(partial)"
            print(f"     A* navigation: {nav_steps} steps to stop {stop_idx + 1} {status}")
            if dist_to_stop > 0.05:
                print(f"     ↳ Arrived at ({current.x:.2f}, {current.z:.2f}), "
                      f"{dist_to_stop:.2f}m from target stop ({stop_pos.x:.2f}, {stop_pos.z:.2f})")

        if not nav_ok and nav_steps == 0:
            # 完全无法到达此停靠点，跳过
            if verbose:
                print(f"     ⚠ Skipping stop {stop_idx + 1} — unreachable")
            tour_path.append((current.x, current.z))
            continue

        # 用停靠点坐标覆盖路径终点，确保地图上蓝色路径线精确穿过绿色星标
        # （A* 导航的物理落点可能和网格目标有 <0.25m 的浮点偏差）
        if len(tour_path) > 0:
            tour_path[-1] = (stop_pos.x, stop_pos.z)

        # 创建该停靠点的帧输出目录
        stop_dir = ""
        if export_frames:
            stop_dir = os.path.join(output_dir, f"stop_{stop_idx + 1:02d}")
            os.makedirs(stop_dir, exist_ok=True)

        # 360° 扫描（每次转 30°，共 12 步覆盖 360°）
        scan_angles = 12
        stop_labels: set[str] = set()
        scan_errors = 0

        for scan_i in range(scan_angles):
            try:
                view = ctrl.get_current_view()
                rgb = view.sensor_data.rgb
                dets = pipeline.process(rgb, controller=ctrl)

                # 统计检测到的标签
                for d in dets:
                    stop_labels.add(d.label.lower())

                # 注入空间记忆
                s = view.agent_state
                mem.ingest_scan(
                    dets, view.sensor_data.depth,
                    s.position.x, s.position.y, s.position.z,
                    s.heading_deg, s.horizon_deg,
                    min_conf=0.2,
                )

                # 保存标注帧
                if export_frames:
                    annotated = draw_annotations(rgb, dets)
                    # 叠加状态信息
                    h, w = annotated.shape[:2]
                    overlay = annotated.copy()
                    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
                    info_lines = [
                        f"Stop {stop_idx + 1}/{len(stops)} | Scan {scan_i + 1}/{scan_angles}",
                        f"Heading: {s.heading_deg:.0f} deg | Pos: ({s.position.x:.2f}, {s.position.z:.2f})",
                        f"Detected: {len(dets)} objects | Memory: {len(mem)} types",
                    ]
                    for li, line in enumerate(info_lines):
                        cv2.putText(annotated, line, (10, 20 + li * 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    fname = os.path.join(stop_dir, f"scan_{scan_i:03d}.png")
                    cv2.imwrite(fname, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

                # 转动 30°
                if scan_i < scan_angles - 1:
                    ctrl.step("TURN_LEFT_SMALL")
                    time.sleep(0.01)

            except Exception as e:
                scan_errors += 1
                if verbose and scan_errors <= 2:
                    print(f"     ⚠ Scan step {scan_i} error: {e}")
                # 尝试恢复
                try:
                    time.sleep(0.05)
                except Exception:
                    pass
                continue

        if verbose:
            print(f"     Scan: {len(stop_labels)} unique labels detected"
                  + (f" ({scan_errors} errors)" if scan_errors else ""))
            print(f"     Labels: {', '.join(sorted(stop_labels)[:15])}"
                  + (f" ... (+{len(stop_labels) - 15} more)" if len(stop_labels) > 15 else ""))

        # 保存最后一帧的原始画面
        if export_frames:
            try:
                view = ctrl.get_current_view()
                rgb = view.sensor_data.rgb
                dets = pipeline.process(rgb, controller=ctrl)
                annotated = draw_annotations(rgb, dets)
                cv2.imwrite(os.path.join(stop_dir, "overview.png"),
                            cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            except Exception:
                pass

    # 返回出生点
    if verbose:
        print(f"\n  {'─'*50}")
        print(f"  Returning to spawn...")
    try:
        for _ in ctrl.navigate_to(spawn):
            pass
        tour_path.append((spawn.x, spawn.z))
    except Exception as e:
        if verbose:
            print(f"  ⚠ Could not return to spawn: {e}")
        current = ctrl.agent_state.position
        tour_path.append((current.x, current.z))

    return mem, tour_path


# ===========================================================================
# 空间记忆摘要
# ===========================================================================

def _save_memory_summary(mem: SpatialMemory, output_path: str, scene_name: str) -> None:
    """保存空间记忆文本摘要。"""
    lines = [
        f"Spatial Memory Summary",
        f"{'='*60}",
        f"Scene: {scene_name}",
        f"Objects indexed: {len(mem)}",
        f"",
        f"{'Label':<22s} {'Count':>6s}  {'Best Position (X, Y, Z)':>30s}",
        f"{'-'*22} {'-'*6}  {'-'*30}",
    ]
    for label in sorted(mem._index.keys()):
        positions = mem._index[label]
        best = positions[0]  # 第一个是距离质心最近的
        lines.append(
            f"{label:<22s} {len(positions):>6d}  "
            f"({best.x:>8.2f}, {best.y:>6.2f}, {best.z:>8.2f})"
        )
    lines.append("")
    lines.append("Note: positions are vision-based (YOLO + depth), not AI2-THOR metadata.")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Memory summary → {output_path}")


# ===========================================================================
# 主函数
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description="Room Tour 可视化 — 展示机器人预探索（巡视房间）全过程")
    p.add_argument("--scene", default="FloorPlan1",
                   help="场景名称 (FloorPlan1 ~ FloorPlan30)")
    p.add_argument("--stops", type=int, default=4,
                   help="巡游停靠点数量 (默认 4)")
    p.add_argument("--detector", default="hybrid",
                   choices=["hybrid", "finetuned", "yolo-world"],
                   help="检测器类型")
    p.add_argument("--model", default="runs/detect/runs/train/weights/best.pt",
                   help="微调 YOLO 模型路径")
    p.add_argument("--confidence", type=float, default=0.3,
                   help="YOLO 置信度阈值")
    p.add_argument("--tour-confidence", type=float, default=0.2,
                   help="巡游时使用的置信度阈值（较低以捕获小物体）")
    p.add_argument("--export-frames", action="store_true", default=True,
                   help="导出每个停靠点的扫描帧 PNG")
    p.add_argument("--no-frames", action="store_true",
                   help="不导出扫描帧")
    p.add_argument("--export-video", action="store_true",
                   help="导出巡游轨迹 MP4 视频")
    p.add_argument("--width", type=int, default=800,
                   help="画面宽度")
    p.add_argument("--height", type=int, default=600,
                   help="画面高度")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子")
    args = p.parse_args()

    export_frames = not args.no_frames

    # 输出目录
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"room_tour_{args.scene}_{ts}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  ROOM TOUR VISUALIZATION")
    print(f"  Scene: {args.scene}  |  Stops: {args.stops}")
    print(f"  Detector: {args.detector}  |  Output: {output_dir}")
    print(f"{'='*60}")

    # ── 创建控制器和感知管道 ──
    print(f"\n  Loading scene '{args.scene}'...")

    with ThorController(width=args.width, height=args.height,
                        render_depth=True) as ctrl:
        ctrl.load_scene(args.scene, seed=args.seed)

        # 获取可达位置（后续画地图用）
        reachable = ctrl._get_reachable_positions_raw()
        print(f"  Reachable positions: {len(reachable)} grid points")

        # 创建检测器
        if args.detector == "hybrid":
            detector = create_hybrid(
                finetuned_model=args.model,
                finetuned_conf=args.confidence,
                world_model="yolov8s-worldv2.pt",
                world_conf=0.15,
                controller=ctrl,
            )
        elif args.detector == "yolo-world":
            detector = create_detector(
                model_name="yolov8s-worldv2.pt",
                confidence=args.confidence,
                classes_path="config/classes.yaml",
            )
            if hasattr(detector, "set_classes_from_scene"):
                detector.set_classes_from_scene(ctrl)
        else:
            detector = YOLODetector(model_name=args.model, confidence=args.confidence)

        # 巡游专用管道（低阈值）
        depth = HeuristicDepth()
        prior = ScenePrior(args.scene)
        if args.detector == "hybrid":
            detector_tour = create_hybrid(
                finetuned_model=args.model,
                finetuned_conf=args.tour_confidence,
                world_model="yolov8s-worldv2.pt",
                world_conf=0.10,
                controller=ctrl,
            )
        else:
            detector_tour = detector
        pipeline_tour = PerceptionPipeline(detector_tour, depth, prior=prior)

        # ── 选出巡游停靠点 ──
        stops = _pick_tour_stops(ctrl, args.stops)
        print(f"\n  Tour stops ({len(stops)}):")
        for i, s in enumerate(stops):
            print(f"    {i+1}. ({s.x:.2f}, {s.z:.2f})")

        # ── 执行巡游 ──
        print(f"\n  Starting room tour...")
        t0 = time.time()

        mem, tour_path = _build_spatial_memory_full(
            ctrl, pipeline_tour, stops,
            output_dir=output_dir,
            export_frames=export_frames,
            verbose=True,
        )

        elapsed = time.time() - t0
        print(f"\n  ✅ Room tour complete in {elapsed:.1f}s")
        print(f"  Spatial memory: {len(mem)} object types indexed")

        # ── 保存空间记忆摘要 ──
        _save_memory_summary(mem, os.path.join(output_dir, "spatial_memory.txt"),
                            args.scene)

        # ── 绘制俯视地图 ──
        spawn = ctrl.agent_state.position
        draw_tour_map(
            reachable=reachable,
            stops=stops,
            tour_path=tour_path,
            memory=mem,
            scene_name=args.scene,
            output_path=os.path.join(output_dir, "tour_map.png"),
            agent_spawn=spawn,
        )

        # ── 导出视频 ──
        if args.export_video:
            print(f"\n  Exporting trajectory video (this may take a moment)...")
            try:
                from src.recording.collector import FrameCollector
                collector = FrameCollector(scene=args.scene)
                # 重放巡游路径录制视频
                ctrl.load_scene(args.scene, seed=args.seed)
                for stop_pos in stops:
                    for result in ctrl.navigate_to(stop_pos):
                        collector.record(result)
                    for _ in range(4):
                        view = ctrl.get_current_view()
                        collector.record(view)
                        ctrl.step("TURN_LEFT")
                video_path = collector.export_video(
                    os.path.join(output_dir, "tour_trajectory.mp4"), fps=5)
                print(f"  Video saved → {video_path}")
            except Exception as e:
                print(f"  Video export failed: {e}")

    # ── 最终摘要 ──
    print(f"\n{'='*60}")
    print(f"  OUTPUT")
    print(f"  Directory: {output_dir}")
    print(f"  Files:")
    print(f"    tour_map.png        — 俯视巡游路径图")
    print(f"    spatial_memory.txt  — 空间记忆文本摘要")
    if export_frames:
        for i in range(len(stops)):
            d = os.path.join(output_dir, f"stop_{i+1:02d}")
            if os.path.isdir(d):
                count = len([f for f in os.listdir(d) if f.endswith(".png")])
                print(f"    stop_{i+1:02d}/             — {count} 张扫描帧")
    if args.export_video:
        print(f"    tour_trajectory.mp4 — 巡游轨迹视频")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
