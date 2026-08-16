import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, PresentationFile } from "@oai/artifact-tool";

const PRESENTATION_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const STARTER_PPTX = path.join(PRESENTATION_ROOT, "templates", "template-starter.pptx");
const FINAL_PPTX = path.join(PRESENTATION_ROOT, "output", "interview_presentation.pptx");
const RENDER_DIR = path.join(PRESENTATION_ROOT, ".build", "final-render");
const LAYOUT_DIR = path.join(PRESENTATION_ROOT, ".build", "final-layout");
const MONTAGE_PATH = path.join(PRESENTATION_ROOT, ".build", "final-montage.webp");
const INSPECT_PATH = path.join(PRESENTATION_ROOT, ".build", "final-inspect.ndjson");

function slidesFromPresentation(presentation) {
  if (Array.isArray(presentation.slides?.items)) return presentation.slides.items;
  if (Number.isInteger(presentation.slides?.count) && typeof presentation.slides.getItem === "function") {
    return Array.from({ length: presentation.slides.count }, (_, index) => presentation.slides.getItem(index));
  }
  throw new Error("Could not enumerate slides.");
}

async function writeBlob(filePath, blob) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, Buffer.from(await blob.arrayBuffer()));
}

function parseRecords(ndjson) {
  return ndjson
    .split(/\r?\n/)
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line));
}

function buildTextIndex(records) {
  const bySlide = new Map();
  for (const record of records) {
    if (!record.id?.startsWith("sh/") || !record.name || !record.slide) continue;
    if (!record.text && record.kind !== "textbox") continue;
    const slideMap = bySlide.get(record.slide) ?? new Map();
    const bucket = slideMap.get(record.name) ?? [];
    bucket.push(record);
    slideMap.set(record.name, bucket);
    bySlide.set(record.slide, slideMap);
  }

  for (const slideMap of bySlide.values()) {
    for (const bucket of slideMap.values()) {
      bucket.sort((a, b) => {
        const [al, at] = a.bbox ?? [0, 0];
        const [bl, bt] = b.bbox ?? [0, 0];
        return at - bt || al - bl;
      });
    }
  }
  return bySlide;
}

function setText(presentation, textIndex, slideNumber, name, value, occurrence = 0) {
  const record = textIndex.get(slideNumber)?.get(name)?.[occurrence];
  if (!record) throw new Error(`Missing text shape: slide ${slideNumber}, ${name}[${occurrence}]`);
  const shape = presentation.resolve(record.id);
  shape.text = value;
}

function setMany(presentation, textIndex, slideNumber, values) {
  for (const [name, value] of Object.entries(values)) {
    if (Array.isArray(value)) {
      value.forEach((item, index) => setText(presentation, textIndex, slideNumber, name, item, index));
    } else {
      setText(presentation, textIndex, slideNumber, name, value);
    }
  }
}

function sourceNotes(lines) {
  return [
    "[Sources]",
    ...lines.map((line) => `- ${line}`),
  ].join("\n");
}

function setNotes(slide, notes) {
  slide.speakerNotes.textFrame.setText(notes);
  slide.speakerNotes.setVisible(true);
}

const footer = "Source: interview-transcript.txt; docs/index.html; template.pptx";

const slideContent = [
  {
    slide: 1,
    values: {
      eyebrow: "PROJECT STORY",
      title: "自然语言数据处理 Agent\n从对话到可执行 DAG",
      subtitle: "在可视化拖拽数据处理平台上，引入 Agent 层，让用户用自然语言构建与修改可执行 JSON DAG。",
      "why-label": "为什么值得讲",
      "bullet-text": [
        "用户不再需要理解选择列、UNION、JOIN、SQL 脚本等节点参数。",
        "系统把需求拆成 planner、scheduler、worker、reducer 的闭环。",
        "评测、SFT、GRPO 和性能优化形成持续迭代的数据飞轮。"
      ],
      "pipeline-label": "核心任务",
      "input-caption": "平台画布 / 表结构",
      "prompt-box": "\"生成或修改\n数据管道\"",
      "prompt-caption": "自然语言需求",
      "output-caption": "可执行 JSON DAG",
      footer,
      "slide-num": "01"
    },
    notes: sourceNotes([
      "interview transcript.txt: project background, Agent motivation, architecture summary.",
      "template.pptx: visual template and copied source slide 1."
    ])
  },
  {
    slide: 2,
    values: {
      eyebrow: "SYSTEM DESIGN",
      title: "DAG 构建的可控闭环",
      subtitle: "opencode 管上下文与平台 API；LangGraph 将自然语言需求拆成 planner -> scheduler -> worker -> reducer 循环。",
      "methods-label": "核心模块",
      "method-badge-1": "1",
      "method-title-1": "opencode 接入层",
      "method-desc-1": "管理上下文、回滚、状态同步与取消任务。",
      "method-badge-2": "2",
      "method-title-2": "Planner",
      "method-desc-2": "选表压缩上下文，判断构造/修改并生成 DAG。",
      "method-badge-3": "3",
      "method-title-3": "Scheduler / Worker",
      "method-desc-3": "按拓扑派发任务，执行模型调用、工具校验与重试。",
      "method-badge-4": "4",
      "method-title-4": "Reducer",
      "method-desc-4": "聚合冲突与失败，必要时触发下游重试或 replan。",
      "metrics-title": "关键机制",
      "unsup-pill": "上下文",
      "unsup-metrics": "表格选择\n对话回滚\n远端同步",
      "seg-pill": "事务",
      "seg-metrics": "candidate schema\n校验后提交\n失败回滚",
      "outcomes-title": "面试主线",
      "outcomes-copy": "把用户从理解算子参数，转为自然语言描述目标。",
      "deliverables-copy": "重点讲闭环、评测和训练优化，而不是 UI 细节。",
      "timeline-label": "5 分钟路径",
      "week-num-1": "1",
      "week-text-1": "背景",
      "week-num-2": "2",
      "week-text-2": "架构",
      "week-num-3": "3",
      "week-text-3": "评测",
      "week-num-4": "4",
      "week-text-4": "SFT",
      "week-num-5": "5",
      "week-text-5": "GRPO",
      "week-num-6": "6",
      "week-text-6": "性能",
      footer,
      "slide-num": "02"
    },
    notes: sourceNotes([
      "interview transcript.txt: opencode, custom API, LangGraph planner/scheduler/worker/reducer architecture.",
      "template.pptx: visual template and copied source slide 2."
    ])
  },
  {
    slide: 3,
    values: {
      eyebrow: "EVALUATION",
      title: "评测闭环把错误变成训练信号",
      subtitle: "先用真实执行结果判断任务是否完成，再在失败时拆解 DAG 节点和拓扑，得到可归因的错误类型。",
      "methods-label": "评测设计",
      "method-badge-1": "1",
      "method-title-1": "Outcome 判定",
      "method-desc-1": "把 JSON 传回平台执行，比较最终数据集与需求。",
      "method-badge-2": "2",
      "method-title-2": "LLM-as-Judge",
      "method-desc-2": "替代人工核查，让评估可以持续、大规模运行。",
      "method-badge-3": "3",
      "method-title-3": "失败诊断",
      "method-desc-3": "逐节点看算子类型和 intent，再看整体 DAG 拓扑。",
      "method-badge-4": "4",
      "method-title-4": "训练反馈",
      "method-desc-4": "失败案例落到类别后，反哺数据构造与约束机制。",
      "metrics-title": "Judge 质量",
      "unsup-pill": "整体",
      "unsup-metrics": "Precision 0.88\nRecall 0.90\nF1 0.90",
      "seg-pill": "归因",
      "seg-metrics": "Macro-F1 83%\n约 8 成在 SQL\n其余为 DAG/算子",
      "outcomes-title": "为什么重要",
      "outcomes-copy": "从“人工看 JSON 对不对”变成可规模化、可归因的评估。",
      "deliverables-copy": "错误分布直接指向 SQL 节点微调和 checker 优化。",
      "timeline-label": "闭环",
      "week-num-1": "A",
      "week-text-1": "需求",
      "week-num-2": "B",
      "week-text-2": "JSON",
      "week-num-3": "C",
      "week-text-3": "执行",
      "week-num-4": "D",
      "week-text-4": "Judge",
      "week-num-5": "E",
      "week-text-5": "归因",
      "week-num-6": "F",
      "week-text-6": "训练",
      footer,
      "slide-num": "03"
    },
    notes: sourceNotes([
      "interview transcript.txt: LLM-as-Judge two-layer evaluation design and reported precision/recall/F1/macro-F1.",
      "template.pptx: visual template and copied source slide 2."
    ])
  },
  {
    slide: 4,
    values: {
      eyebrow: "SFT",
      title: "SFT 优化 SQL 节点瓶颈",
      subtitle: "约 70%-80% 失败集中在 SQL 脚本节点，因此先用 Qwen2.5-Coder-14B + LoRA 替换原 SQL 子任务模型。",
      "methods-label": "训练策略",
      "method-badge-1": "1",
      "method-title-1": "数据构造",
      "method-desc-1": "Spider benchmark + prompt/schema + gold SQL 成对数据。",
      "method-badge-2": "2",
      "method-title-2": "LoRA 范围",
      "method-desc-2": "注入 q/k/v 与 gate/up/down，控制容量并只跑少量 epoch。",
      "method-badge-3": "3",
      "method-title-3": "混合监督",
      "method-desc-3": "对比 gold SQL 结果监督与 trajectory 过程监督比例。",
      "method-badge-4": "4",
      "method-title-4": "网格搜索",
      "method-desc-4": "固定评测协议，扫描学习率与 LoRA rank。",
      "metrics-title": "关键结果",
      "unsup-pill": "对照",
      "unsup-metrics": "Pure-SFT 77.60%\nMixed-SFT 80.60%\n+3.00 pt",
      "seg-pill": "最佳点",
      "seg-metrics": "lr=5e-5, r=32\n3,200 gold + 1,600 traj\n约 80% accuracy",
      "outcomes-title": "结论",
      "outcomes-copy": "约三分之一 trajectory supervision 比纯 gold-only 更稳。",
      "deliverables-copy": "r=64 未带来稳定收益，保留更简洁的 r=32 配置。",
      "timeline-label": "SFT 决策",
      "week-num-1": "1",
      "week-text-1": "定位",
      "week-num-2": "2",
      "week-text-2": "数据",
      "week-num-3": "3",
      "week-text-3": "LoRA",
      "week-num-4": "4",
      "week-text-4": "混合",
      "week-num-5": "5",
      "week-text-5": "网格",
      "week-num-6": "6",
      "week-text-6": "上线",
      footer,
      "slide-num": "04"
    },
    notes: sourceNotes([
      "interview transcript.txt: SQL-node bottleneck, SFT data choice, LoRA setup, mixed supervision and reported 80% result.",
      "index.html: Pure-SFT 77.60%, Mixed-SFT 80.60%, lr=5e-5/r=32, 3,200 gold + 1,600 trajectory.",
      "template.pptx: visual template and copied source slide 2."
    ])
  },
  {
    slide: 5,
    values: {
      eyebrow: "GRPO",
      title: "GRPO 受 checker 限制",
      subtitle: "在统一 SFT-Merged policy 上训练新的 GRPO LoRA，对比 chain/tree rollout、fallback reward 和关键超参数。",
      "methods-label": "实验设计",
      "method-badge-1": "1",
      "method-title-1": "奖励定义",
      "method-desc-1": "执行结果与 gold SQL 一致则 reward=1。",
      "method-badge-2": "2",
      "method-title-2": "Chain rollout",
      "method-desc-2": "完整跑完一条轨迹，再把最终奖励折扣回传。",
      "method-badge-3": "3",
      "method-title-3": "Tree rollout",
      "method-desc-3": "每层有限分支，用 beam 保留更优路径继续探索。",
      "method-badge-4": "4",
      "method-title-4": "超参搜索",
      "method-desc-4": "KL=0.01、PPO epoch=2、T=1.0、gamma=0.9 最稳。",
      "metrics-title": "关键结果",
      "unsup-pill": "策略",
      "unsup-metrics": "S3 n=20 83.67%\nS1 n=8 83.33%\n差距 0.33 pt",
      "seg-pill": "Checker",
      "seg-metrics": "DeepSeek checker 86.00%\n耗尽轮数率 6.00%\n误判仍是瓶颈",
      "outcomes-title": "判断",
      "outcomes-copy": "tree 没有拉开清晰优势；S1 更省 rollout slots。",
      "deliverables-copy": "下一步优先校准 checker，而不是单纯扩大 rollout。",
      "timeline-label": "GRPO 变量",
      "week-num-1": "1",
      "week-text-1": "奖励",
      "week-num-2": "2",
      "week-text-2": "Chain",
      "week-num-3": "3",
      "week-text-3": "Tree",
      "week-num-4": "4",
      "week-text-4": "KL",
      "week-num-5": "5",
      "week-text-5": "采样",
      "week-num-6": "6",
      "week-text-6": "Checker",
      footer,
      "slide-num": "05"
    },
    notes: sourceNotes([
      "interview transcript.txt: GRPO reward, chain/tree rollout, fallback reward and hyperparameter narrative.",
      "index.html: S3 83.67%, S1 83.33%, S3 n=24 82.00%, checker replacement to 86.00%, KL/PPO/temperature/gamma settings.",
      "template.pptx: visual template and copied source slide 2."
    ])
  },
  {
    slide: 6,
    values: {
      eyebrow: "PERFORMANCE",
      title: "性能调优压缩瓶颈",
      subtitle: "训练效率按 Actor->vLLM 同步、policy update、rollout 和 serving 拓扑拆解，逐段做消融和补丁。",
      "methods-label": "优化动作",
      "method-badge-1": "1",
      "method-title-1": "权重同步",
      "method-desc-1": "修复 LoRA 参数识别，避免 full-summon CPU offload。",
      "method-badge-2": "2",
      "method-title-2": "Policy update",
      "method-desc-2": "扩大 micro-batch，并启用 remove padding + dynamic batching。",
      "method-badge-3": "3",
      "method-title-3": "Rollout 容量",
      "method-desc-3": "把 vLLM seqs/tokens 从 4/4096 提到 8/8192。",
      "method-badge-4": "4",
      "method-title-4": "Serving 拓扑",
      "method-desc-4": "固定 4-GPU 预算，对比 TP=1/2/4 与多实例布局。",
      "metrics-title": "关键收益",
      "unsup-pill": "训练侧",
      "unsup-metrics": "同步 44.3s -> 7.2s\nupdate 30.864s -> 3.670s\nstep 62.4s -> 32.0s",
      "seg-pill": "吞吐侧",
      "seg-metrics": "43.2 -> 116.8 token/s\nrollout 24.558s -> 18.000s\n低并发 TP=4 最优",
      "outcomes-title": "收束",
      "outcomes-copy": "最大价值来自定位真实瓶颈，再用小补丁和参数消融逐段压缩。",
      "deliverables-copy": "最终故事：系统闭环 + 数据闭环 + 性能闭环。",
      "timeline-label": "优化链路",
      "week-num-1": "1",
      "week-text-1": "同步",
      "week-num-2": "2",
      "week-text-2": "Batch",
      "week-num-3": "3",
      "week-text-3": "Padding",
      "week-num-4": "4",
      "week-text-4": "Token",
      "week-num-5": "5",
      "week-text-5": "Rollout",
      "week-num-6": "6",
      "week-text-6": "TP",
      footer,
      "slide-num": "06"
    },
    notes: sourceNotes([
      "interview transcript.txt: GRPO training efficiency optimization, LoRA sync patch, dynamic batching, empty_cache and vLLM topology narrative.",
      "index.html: 44.3s to 7.2s sync, 30.864s to 3.670s policy update, rollout 24.558s to 18.000s, step 62.4s to 32.0s, 43.2 to 116.8 token/s.",
      "template.pptx: visual template and copied source slide 2."
    ])
  }
];

async function main() {
  await fs.rm(RENDER_DIR, { recursive: true, force: true });
  await fs.rm(LAYOUT_DIR, { recursive: true, force: true });
  await fs.mkdir(RENDER_DIR, { recursive: true });
  await fs.mkdir(LAYOUT_DIR, { recursive: true });

  const presentation = await PresentationFile.importPptx(await FileBlob.load(STARTER_PPTX));
  const slides = slidesFromPresentation(presentation);
  const before = await presentation.inspect({
    kind: "slide,textbox,shape,notes",
    maxChars: 250000
  });
  const textIndex = buildTextIndex(parseRecords(before.ndjson));

  for (const item of slideContent) {
    const slide = slides[item.slide - 1];
    setMany(presentation, textIndex, item.slide, item.values);
    setNotes(slide, item.notes);
  }

  const after = await presentation.inspect({
    kind: "slide,textbox,shape,notes",
    maxChars: 250000
  });
  await fs.writeFile(INSPECT_PATH, after.ndjson, "utf8");

  for (let index = 0; index < slides.length; index += 1) {
    const slide = slides[index];
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await writeBlob(path.join(RENDER_DIR, `${stem}.png`), png);
    const layout = await slide.export({ format: "layout" });
    await writeBlob(path.join(LAYOUT_DIR, `${stem}.layout.json`), layout);
  }

  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await writeBlob(MONTAGE_PATH, montage);

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
  console.log(FINAL_PPTX);
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});
