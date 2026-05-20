# Embodied Agent — AI2-THOR 具身智能系统

基于 AI2-THOR 三维室内仿真环境的多模态具身 Agent，支持自然语言驱动的**导航**（Go to the chair）和**物体交互**（Open the fridge / Pick up the mug / Toggle on the lamp）。支持规则策略（Rule Policy）和大语言模型策略（LLM Policy）对比实验。

## 项目状态

**Phase 2 进行中** — 感知管道已完成，微调数据已采集，待训练和决策模块开发。

| Phase | 内容 | 状态 |
|-------|------|------|
| Phase 1 | 场景控制 + A\* 寻路 + RGB 采集 + 手动 Demo | ✅ 完成 |
| Phase 2 | YOLO 检测 + 距离估计 + CLIP 验证 + 微调基础设施 | ⚠️ 感知完成，微调训练待执行 |
| Phase 3 | 决策模块 (Rule Policy + LLM Policy) | ❌ 未开始 |
| Phase 4 | Episode Runner + 评估指标 (SR, SPL) + 可视化 | ❌ 未开始 |
| Phase 5 | 端到端集成 + 批量实验 | ❌ 未开始 |

详见 [`docs/开发日志.md`](docs/开发日志.md)

## 项目结构

```
embodied-agent/
├── src/
│   ├── controller/          # AI2-THOR 控制器
│   │   └── thor.py          #   ThorController — A* 寻路 + 导航 + 物体交互
│   ├── perception/          # 感知管道
│   │   ├── detector.py      #   YOLOv8 物体检测器
│   │   ├── depth.py         #   距离估计 (Heuristic + DepthAnything V2)
│   │   ├── verifier.py      #   CLIP 语义验证
│   │   ├── prior.py         #   场景先验（房间类型加权）
│   │   ├── pipeline.py      #   感知管道（组合以上模块）
│   │   ├── class_config.py  #   YAML 类别配置加载
│   │   └── finetune/        #   微调工具
│   │       ├── collect.py   #     数据采集（AI2-THOR 实例分割 → YOLO 标注）
│   │       ├── eval.py      #     模型评估（mAP / Precision / Recall）
│   │       └── train.py     #     YOLO 微调训练
│   ├── common/              # 共享
│   │   ├── types.py         #   数据类型 (Vec3, Detection, ActionResult...)
│   │   └── logger.py        #   日志
│   ├── recording/           # RGB 采集
│   │   └── collector.py     #   FrameCollector (帧记录 + 视频/PNG 导出)
│   └── cli/                 # 命令行入口
│       ├── manual.py        #   手动控制 Demo (wasd 移动 + A* 导航 + detect)
│       └── perceive.py      #   感知专用 Demo (YOLO + CLIP + 场景先验)
├── config/
│   └── classes.yaml         # 类别定义 (COCO 80 + AI2-THOR 扩展 119 类)
├── docs/
│   ├── 开发日志.md           # 开发日志
│   ├── bugfix/              #   问题修复记录
│   └── superpowers/         #   原始设计文档
├── remembr-main/            # 参考项目 (NVIDIA ReMEmbR)
├── weights/                 # 模型权重 (yolov8n.pt / yolov8m.pt)
├── outputs/                 # 运行输出
└── requirements.txt
```

## 架构概览

```
用户 NL 指令
    ↓
[TaskParser] → TaskSpec             ← Phase 3 待实现
    ↓                           task_type: "navigation" | "interaction"
AI2-THOR → Observation(rgb, position, heading)
    ↓
[YOLODetector] → detections[]
    ↓
[CLIPVerifier] + [ScenePrior] → verified detections
    ↓
[DepthEstimator] → distance_level (NEAR/MEDIUM/FAR)
    ↓
[DecisionPolicy] → ActionDecision   ← Phase 3 待实现
    ↓
[ThorController] → 执行动作
    ↓   导航: MOVE_FORWARD / TURN_LEFT / TURN_RIGHT / STOP
    ↓   交互: INTERACT_OPEN / INTERACT_PICKUP / INTERACT_TOGGLE
循环 → [EpisodeRunner] → EpisodeResult(success, spl, ...) ← Phase 4 待实现
```

### 任务类型

| 类型 | 示例指令 | 期望结果 |
|------|----------|----------|
| 导航 (navigation) | "Go to the chair", "Find the microwave" | Agent 移动到目标物体 1.5m 内并 STOP |
| 交互 (interaction) | "Open the fridge", "Pick up the mug", "Toggle on the lamp" | Agent 先导航到目标，接近后执行 Open/Pickup/Toggle |

ThorController 已实现的交互动作映射：
- `INTERACT_OPEN` → AI2-THOR `OpenObject`（如开门、开冰箱）
- `INTERACT_PICKUP` → AI2-THOR `PickupObject`（如拿起杯子）
- `INTERACT_TOGGLE` → AI2-THOR `ToggleObjectOn/Off`（如开关灯）

## 快速开始

### 环境要求

- Python 3.10+
- WSL2 (Ubuntu) + NVIDIA GPU (8GB+ VRAM)
- AI2-THOR (Unity 仿真)

### 安装

```bash
git clone <repo-url>
cd embodied-agent

# 创建虚拟环境
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 下载 YOLO 模型（首次运行自动下载，也可手动放到 weights/）
# 模型权重文件较大，已在 .gitignore 中排除

# 配置 API Key
cp .env.example .env
# 编辑 .env 填入你的 OPENAI_API_KEY
```

### 运行

```bash
# 手动控制 Demo（wasd 移动 + A* 导航 + VLM 描述）
python src/cli/manual.py --scene FloorPlan1

# 感知 Demo（YOLO 检测 + CLIP + 场景先验）
python src/cli/perceive.py --scene FloorPlan1

# 诊断：检查场景物体映射
python src/cli/perceive.py --check-classes --scene FloorPlan1

# 诊断：可视化 ground-truth 标注
python src/cli/perceive.py --verify-labels --scene FloorPlan1

# 数据采集（已完成，重新采集可覆盖）
python src/perception/finetune/collect.py --scenes 10 --steps 200

# 微调前评估
python src/perception/finetune/eval.py --model weights/yolov8n.pt --num-scenes 5

# 微调训练
python src/perception/finetune/train.py --data data/data.yaml --epochs 50
```

## 技术栈

| 模块 | 技术 | 版本 |
|------|------|------|
| 仿真环境 | ai2thor | 5.0+ |
| 目标检测 | YOLOv8 (ultralytics) | 8.0+ |
| 语义匹配 | CLIP (openai-clip) | ViT-B/32 |
| 距离估计 | HeuristicDepth / Depth Anything V2 | 可选 |
| VLM | DeepSeek-v4-pro / GPT-4V | OpenAI Vision API 兼容 |
| 图像处理 | OpenCV, Pillow | |
| 数据 | NumPy | |

## 设计原则

1. Controller 用 `with` 语句管理生命周期
2. `step()` 是唯一改变状态的方法，查询方法不改变状态
3. A* 导航通过 generator 返回，调用者可观察每一步
4. 感知管道模块化：检测/验证/先验/距离各自独立，通过 pipeline 组合
5. 类别配置完全 YAML 驱动：新增类别只需改 `config/classes.yaml`

## License

Academic project.
