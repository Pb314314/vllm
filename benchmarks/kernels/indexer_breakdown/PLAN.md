# 过夜实验计划：Sparse-indexer decode logits 的显存来源 + top-k kernel 的 kernel-level breakdown

> 执行者：一个独立 agent，预算约 8 小时，GB200（SM100）节点。
> 背景 PR：Pb314314/vllm#2（`cg/pr-a-v1`，CUDA graph 拆分 prefill/decode capture sizes），
> 其描述中 "Memory: three causes and their fixes → 1. Sparse-attention indexer" 一节。
> 行分块原型：commit `553747b1df`（未合并）。上游相关：vllm-project/vllm#62169（SM90/SM120 分块）、
> #60802（已关闭："Prefer DeepSelect over persistent_topk for large batches on SM100"，先读它的讨论和数据）。

---

## 0. 规则（必须遵守）

1. **不改产品代码的行为，不推上游，不开 PR。** 在分支 `cg/indexer-topk-breakdown` 上工作（已基于 `cg/pr-a-v1` 建好，本计划即 `benchmarks/kernels/indexer_breakdown/PLAN.md`），
   只新增：`benchmarks/kernels/benchmark_indexer_decode_topk_breakdown.py`、`benchmarks/kernels/indexer_breakdown/` 下的
   结果和报告。可以 push 到 fork `Pb314314/vllm` 的这个分支以免容器回收丢数据。
2. Python 一律用 `.venv/bin/python`（见 AGENTS.md），不要用系统 python / 裸 pip。
3. 开始前先用 Skill 加载 `kernel-microbenchmark`（CUPTI 计时、正确性检查、SOL 校验的规范），Phase 5 用到时再加载 `model-microbenchmark`。
4. **增量落盘**：每个配置测完立即 append 一行 JSONL 到 `results/*.jsonl`，并每 ~30 分钟 commit+push 一次。
   脚本必须支持 `--resume`（跳过 JSONL 里已有的 key）。崩溃/重启后从断点继续。
5. 维护 `indexer_breakdown/PROGRESS.md`：每完成一个 Phase 写一段（做了什么、关键数字、异常）。
6. **不编数字。** 每个结论都要能指到 JSONL 行或 allocator snapshot。测不了的写 "未测 + 原因"。
7. 时间盒：每个 Phase 有上限，超时就缩小网格（按本文 §3.3 的优先级裁剪），不要卡死在一处。

---

## 1. 必须先理解的事实（已核实，附位置）

| 事实 | 位置 |
|---|---|
| SM100 decode 路径：`fp8_fp4_paged_mqa_logits` 一次性产出 `[rows, stride≈max_model_len/compress]` 的 **BF16** logits | `vllm/model_executor/layers/sparse_attn_indexer.py` ~L880 |
| 若 `topk_backend != "deep_select"` 或 DCP>1，`as_fp32_logits` 再整份造一个 **FP32 副本**（同 stride） | 同文件 L304–L327、~L891 |
| 这不是"量化"，是 **BF16→FP32 upcast**：一次额外 kernel（读 2B 写 4B/元素）+ 一份 4B/元素 的显存 | 同上 |
| top-k dispatcher：`auto` = `cooperative`（rows ≤ 64，`AUTO_COOPERATIVE_MAX_ROWS`）→ `persistent`（topk ∈ {512,1024,2048}）→ `per_row`。`deep_select` / `flashinfer` / `torch` 只能显式指定 | `vllm/model_executor/layers/indexer_topk.py` L17–21, L236–253 |
| cooperative / persistent 只吃 FP32；deep_select 吃 BF16 或 FP32（SM100 only，topk ≤ 4096，stride 对齐要求） | 同文件 L44–56, L173–234 |
| `persistent_topk` 最后一个参数 `max_seq_len` 是 **host 端 int**，控制 sampled_topk 长行路径。在 CUDA graph 里它会被**烤进 graph**（capture 时的值） | 同文件 L330–346 |
| cooperative/persistent 用 1 MiB 的 `RADIX_TOPK_WORKSPACE_SIZE` workspace | 同文件 L15 |
| 原型 `553747b1df`：按行分块（`VLLM_SPARSE_INDEXER_MAX_DECODE_LOGITS_MB`，默认 1024），`auto` 时在该路径改用 deep_select 以免 FP32 副本 | `git show 553747b1df` |
| `vllm::sparse_attn_indexer` 在默认 splitting ops 里（`_attention_ops`） | `vllm/config/compilation.py` L768–788 |
| profiling 只给 logits 预留 `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`（512 MB）的 dummy；这个预算只用于 prefill 分块（`indexer.py` L1503），decode 路径不分块 | `sparse_attn_indexer.py` L501–506 |
| DSv4：只有 `compress_ratio == 4` 的层有 indexer（logits 宽 = max_model_len/4）；DSv4.1 有两级 candidate 选择（`_select_candidate_blocks` / `_apply_candidate_mask`，作用在 logits 上，发生在 top-k 前） | `vllm/models/deepseek_v4/attention.py` L297, L946–1000；`sparse_attn_indexer.py` ~L895–925 |
| PR #2 的实测：DSv4-Flash DSpark K=7、1M ctx、512 req×q=8=4096 rows，target FULL pool +10.525 GiB，其中 FP32 副本 8.016（"两个 4104 MiB segment"），BF16 原始 2.004 | PR #2 描述 |

现有工具可复用：`benchmarks/kernels/benchmark_persistent_topk.py`（CUPTI 计时、FP32 行、cooperative/persistent）、
`benchmarks/kernels/benchmark_sparse_indexer_topk_prefill.py`、`tests/kernels/test_top_k_per_row.py`。

---

## Phase 0 — 环境与基线（≤ 30 min）

1. `git fetch origin cg/indexer-topk-breakdown && git checkout cg/indexer-topk-breakdown`；`git fetch origin 553747b1df`（Phase 1/4 要对照）。
2. 记录环境到 `indexer_breakdown/ENV.md`：GPU 型号/SM 数/显存、驱动、CUDA、torch、DeepGEMM 版本、
   `vllm._deepselect_C` 是否可 import、flashinfer 版本、`nvidia-smi -q -d CLOCK` 结果。尝试锁频
   （`nvidia-smi -lgc`），没权限就记录"未锁频"，并在每条 JSONL 里记录测量前后的 SM clock 与温度。
3. 跑通 `benchmark_persistent_topk.py` 默认网格一次作为冒烟测试，确认 CUPTI 计时可用。
4. 读 #60802 的描述和评论，把其中已有的 deep_select vs persistent 数据摘进 `PROGRESS.md`，标出它覆盖了/没覆盖哪些区间。

---

## Phase 1 — 显存开销的机制：为什么 FULL 会、PIECEWISE 不会（≤ 1.5 h）

目标：用实测回答三个问题，每个问题给出 "结论 + 证据（snapshot/数字）"。

### 1.1 静态确认：PIECEWISE 下 indexer 是否在 graph 外
- 对每个使用 indexer 的模型路径（DSv3.2 `deepseek_v32/attention.py`、DSv4 `deepseek_v4/attention.py`、DSv4.1、GLM5next、HY-v4），
  确认 indexer 是通过哪个 custom op 被调用的（`vllm::sparse_attn_indexer` 本身，还是被包在 `vllm::deepseek_v4_attention` 里），
  以及该 op 是否在最终 `splitting_ops` 里。
- 列出**会让 indexer 被 PIECEWISE capture 进去**的配置（例如 `splitting_ops=[]`、inductor graph partition、`fuse_attn_quant` 等清空 attention splitting ops 的路径，见 `compilation.py` L1162–1280）。
- 输出：表格「模型 → indexer 所在 op → 默认是否 split → 哪些 flag 会改变这一点」。

### 1.2 动态确认：capture 时的 pool 归属
用一个小但能复现的配置（建议 DSv4-Flash，TP 按集群可用，`max_model_len` 先用 256K 以节省时间，再在 1M 上复测一次关键点）：
- A：`cudagraph_mode=PIECEWISE`；B：`FULL_DECODE_ONLY`；C：`FULL_AND_PIECEWISE`（PR #2 默认）。
- 在 graph capture 阶段开 `torch.cuda.memory._record_memory_history(max_entries=...)`，capture 完 dump snapshot。
- 按 stack frame 归因：哪些 segment 属于 graph pool、来自 `fp8_fp4_paged_mqa_logits` / `as_fp32_logits` 的分配在 A/B/C 各自是多少。
- 预期：A 中 indexer logits 不在任何 graph pool；B/C 的 FULL pool 里有，大小 ≈ `rows × stride × (2 + 4)`。若不符，记录并解释。

### 1.3 PIECEWISE / eager 的运行时峰值（风险，不是常驻）
- 在 A 模式下，跑 (a) 超出 4096-token decode 预算、退回 PIECEWISE 的大 uniform decode（如 1024 req × q=8），
  (b) 含大量 decode 行的 mixed batch。测 `torch.cuda.max_memory_allocated()` 相对 step 前的增量。
- 与 profiling 预留（512 MB dummy）对比，给出 "运行时峰值 − profiling 预留" 的差额随 rows × ctx 的表。
- 结论要说明：在默认 `gpu_memory_utilization` 下是否会 OOM（实际 KV 已占满的情况下跑一次，看是否 OOM）。

### 1.4 解释 "两个 4104 MiB segment"
- 4096 rows × 262144 × 4 B = 4096 MiB；多出的 8 MiB 是否来自 DeepGEMM 的对齐 stride（打印 `logits.stride(0)`）。
- 为什么是**两个**：候选假设 ——（a）两种 capture size 的 FP32 副本无法复用同一块；（b）DSpark/AV 的两次 forward；
  （c）BF16 释放后 FP32 分配因 caching allocator 的 block 规则无法复用而新开 segment。用 1.2 的 snapshot 的 alloc/free 时间线确认。

### 1.5 显存公式并验证
- 给出公式 `peak_bytes(rows, width) = rows × stride(width) × (2 + 4·[upcast]) + workspace(backend)`，
  以及有分块时 `chunk_rows × …`。在 rows ∈ {512, 1024, 2048, 4096}、width ∈ {64K, 256K} 上用实测 pool 增量验证（误差 < 2%）。

---

## Phase 2 — 盘点所有要覆盖的形状（≤ 30 min）

从各模型的 HF config 与 vLLM 代码中**读出**（不要凭记忆）每类 indexer 层的参数，写成 `indexer_breakdown/SHAPES.md`：

| 模型 | 层类型 | indexer heads × head_dim | `index_topk` | compress ratio | 默认/典型 max_model_len | logits 宽度 | 是否有 candidate 两级选择 | Q/KV dtype（fp8 / mxfp4） | 每层 forward 的 indexer 层数 |
|---|---|---|---|---|---|---|---|---|---|

至少覆盖：DeepSeek-V3.2、DeepSeek-V4-Flash、DeepSeek-V4（Pro，若有）、DeepSeek-V4.1-Flash、GLM5next、HY-v4。
同一模型内维度不同的层要分开列（例如 DSv4 的 compress 4 层 vs 其他层）。
**top-k kernel 只关心：rows、logits 宽度/stride、实际行长分布、topk、输入 dtype、是否先做了 candidate mask**；
heads/head_dim 只影响 logits GEMM（Phase 3.5 的组合测量要用）。

---

## Phase 3 — Kernel-level microbenchmark（≤ 3.5 h，主体）

新写 `benchmarks/kernels/benchmark_indexer_decode_topk_breakdown.py`（参考 `benchmark_persistent_topk.py` 的风格与 CUPTI 计时）。

### 3.1 被比较的"方案"（端到端从 BF16 logits 到 top-k indices）

起点统一为 **DeepGEMM 产出的 BF16 logits**（SM100 真实情况），终点为 `topk_indices`。
每个方案的计时 = 方案内所有 kernel 之和，并**分别记录每个 kernel 的时间**：

| 方案 | 步骤 |
|---|---|
| `DS-bf16` | deep_select(BF16) |
| `UP+COOP` | as_fp32_logits → cooperative_topk（仅 rows ≤ 64 合法） |
| `UP+PERS` | as_fp32_logits → persistent_topk |
| `UP+DS-fp32` | as_fp32_logits → deep_select(FP32)（用于分离 "upcast 成本" 与 "kernel 本身快慢"） |
| `UP+FI` | as_fp32_logits → flashinfer `top_k_ragged_transform`（可用时） |
| `UP+PERROW` | as_fp32_logits → `top_k_per_row_decode`（基线） |
| `COOP-fp32-only` / `PERS-fp32-only` | 只测 top-k kernel 本身（假设已有 FP32），用来回答 "如果上游 GEMM 直接出 FP32" 的反事实 |

另外单独测 `as_fp32_logits` 本身（只 upcast），以给出 upcast 的纯成本曲线。

### 3.2 每个配置要记录的指标

- **延迟**：CUPTI kernel 时间，median / p10 / p90，≥ 100 次重复 × 3 trials；trial 间 CV > 3% 则重测并标记。
- **两种执行方式都测**：(a) eager 连续调用；(b) **把整段方案 capture 进一个 CUDA graph 后测 replay**（生产里 FULL decode 就是 replay；
  小 rows 时 launch/cluster-launch 开销差别大）。
- **L2 状态**：(a) warm（logits 刚写完，模拟 GEMM 紧接 top-k）；(b) cold（测前用 ≥ 2×L2 大小的 buffer flush）。生产更接近 warm-partial，两者都报。
- **显存**：
  - 方案额外分配的峰值字节：`torch.cuda.reset_peak_memory_stats()` 后测 `max_memory_allocated` 增量（不含输入 logits 本身）。
  - **graph pool 增量**：在新的 private pool 里 capture 该方案，记录 `memory_reserved` 增量（这才是 FULL graph 下常驻的成本）。
  - 记录 deep_select 是否有内部 workspace（读它的 C++ 接口或用 snapshot 看）。
- **正确性**：每个配置和 `torch` 参考比较——index **集合**相等（顺序无关），越界位置为 -1，无重复；
  对有并列值的情况记录"集合不同但被选值的多重集合相同"的比例（并列导致的差异可以接受，但要统计）。

### 3.3 形状网格（按优先级，超时从低优先级裁剪）

**P0（必须全测）**
- `rows`：**在 cooperative 阈值附近加密**：1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,60,62,63,64,65,66,72,80,96,112,128，
  再加 160,192,256,320,384,512,640,768,1024,1536,2048,3072,4096,6144,8192。
  并标注每个 rows 对应的 (requests, q)：q ∈ {1,2,3,4,6,8}（K=0,1,2,3,5,7），即 rows = requests × q。
  rows 要额外包含 PR #2 decode grid 上真实会出现的 `n × q` 值（从 `decode_cudagraph_capture_grid` 生成）。
- `topk`：Phase 2 表里出现的每个值（预期至少 512 / 1024 / 2048）。
- logits **capacity（stride）**：Phase 2 中每个模型/层的 `max_model_len/compress`（例如 1M/4=256K、128K、160K 等），并加 64K、32K。
- **实际行长分布**（决定 kernel 走哪条路径，非常重要）：
  - `short`：所有行长 ≈ 4K；`mid`：≈ 32K；`long`：≈ 128K；`full`：= capacity；
  - `ragged`：行长在 [1K, capacity] 对数均匀；
  - spec decode 结构：同一 request 的 q 行长度依次 +1（`_row_ends` 的规则）。
- **`persistent_topk` 的 `max_seq_len` 参数**：分别用 (a) 实际 batch 最大行长，(b) capture 时会烤进 graph 的值。
  先从代码/打印确认 FULL graph capture 时 `attn_metadata_narrowed.max_seq_len` 实际是多少，再按真实值测。

**P1**
- **logits 值分布**（radix/select 类 kernel 对值分布和并列敏感）：(a) `randn`；(b) **真实 logits**：在 Phase 1 的模型跑一次 decode，
  hook 下 2–3 个 indexer 层的 BF16 logits（不同层、不同 ctx 长度），存成 `.pt`，用它们（按需平铺到目标 rows）重测 P0 中的关键点
  （rows ∈ {1,8,32,64,65,128,512,1024,4096} × 主要 capacity）。如果和 randn 结论不同，**以真实 logits 为准**。
  (c) 大量并列值（BF16 精度下很常见）：量化到少量 distinct 值的人造输入。
- **DSv4.1 candidate 路径**：带 `_apply_candidate_mask` / `_select_candidate_blocks` 之后的 logits（mask 掉大部分列）作为 top-k 输入，
  测各方案；并测 candidate 操作本身在 BF16 vs FP32 上的时间（upcast 放在 candidate 之前或之后是否有差别）。
- **行分块交互**（对照原型 `553747b1df`）：chunk budget ∈ {256, 512, 1024, 2048} MB 和不分块，测每种方案的
  "总延迟（含多次 launch + 每块 `get_paged_mqa_logits_metadata`）" 和 "峰值/pool 显存"。分块下 FP32 副本变小，
  cooperative/persistent 的显存劣势会缩小——要看分块后 deep_select 是否仍然更优。

**P2（有时间再做）**
- DCP>1 路径（强制 FP32）：只需确认方案集合和 rows/ctx 的缩放，不需要全网格。
- H100/H200（若有节点）：只有 cooperative vs persistent vs per_row（无 deep_select），rows 网格同 P0。

### 3.4 运行组织
- 方案顺序在每个形状内随机交错（A/B/C/A/B/C…），避免温漂偏向某一方案。
- 每 ~200 个配置做一次同一 "哨兵配置"（如 rows=64、256K、topk=2048、`UP+PERS`）的复测，检测时钟漂移，偏差 > 3% 时记录并重跑上一段。
- 粗估：P0 约 45 rows × 3 topk × ~5 capacity × 6 长度分布 × 6 方案 × 2 执行方式 × 2 L2 ≈ 9.7 万次测量点，
  每点 ~0.1–0.3 s → 3–8 h，**太多**。按以下方式裁剪到 ~2.5 h：
  - L2 cold 只在 rows ∈ {1,16,64,256,1024,4096} 上做；
  - eager 只在 rows ≤ 128 上做（大 rows 时 launch 开销可忽略，用 graph replay 即可）；
  - `UP+PERROW`、`UP+FI` 只在 rows ∈ {1,8,64,512,4096} 上做；
  - capacity 每个模型只取其最大值 + 64K。
  先跑一个 5 分钟的"计时探针"估算每点耗时，再定裁剪，并把裁剪决定写进 `PROGRESS.md`。

### 3.5 组合测量：一层 indexer decode 的完整成本
对 Phase 2 表中的每个主要（模型, 层）组合，在 rows ∈ {8, 64, 65, 256, 512, 1024, 4096} 上测完整流水线：
`fp8_fp4_paged_mqa_logits` → [candidate] → [upcast] → top-k，包含真实 KV cache 内容（随机 fp8/mxfp4 即可）与真实 block_table。
目的：(a) top-k 方案的差异占整层 indexer 时间的比例；(b) warm-L2 的真实效果；(c) × 该模型 indexer 层数 = 每 step 的总差异（µs/step）。

---

## Phase 4 — 分析与决策（≤ 1 h）

产出 `indexer_breakdown/REPORT.md`（中文，表为主，图可选，图用 matplotlib 存 PNG）：

1. **Phase 1 结论**：三个问题的答案 + 证据；显存公式；PIECEWISE/eager 运行时峰值 vs profiling 预留的风险表。
2. **upcast 的成本曲线**：延迟（µs）和显存（字节）随 rows × capacity。
3. **每个 (模型/层, topk, q) 的胜者表**：横轴 rows 区间，纵轴 ctx 分布，单元格 = 最快方案 + 相对当前 `auto` 的延迟差（%）+ graph pool 显存差（MiB）。
   特别回答：
   - **rows ≤ 64：`UP+COOP` 是否真的优于 `DS-bf16`？**（算上 upcast 的时间和显存后）在哪些 ctx/topk 下翻转？
   - **rows > 64：`UP+PERS` vs `DS-bf16`** 的交叉点；
   - 阈值 64 本身是否合适（cooperative 在 60–64 的表现，以及 65 处切到 persistent 是否有台阶）。
4. **推荐的 dispatch 规则**：形如 `if rows <= R1 and ctx <= C1: X elif …`，规则要简单（≤ 4 个分支），
   并给出用该规则 vs 当前 `auto` vs 原型（`auto`→deep_select）在 §3.5 组合测量上的每 step 节省（µs）和 FULL pool 节省（GiB）。
   明确写出规则在哪些形状上**不是**最优、损失多少。
5. **正确性**：各方案与参考不一致的统计；并列值的影响。
6. **未覆盖 / 不确定项**：清单。

---

## Phase 5 — 端到端抽查（≤ 1 h，若前面超时则跳过）

只在 Phase 4 推荐规则与当前 `auto` 不同时做。用 `--kernel-config '{"sparse_indexer_topk_backend": ...}'` 强制方案（不改代码），
用 `vllm bench forward`（`model-microbenchmark` skill）在 DSv4-Flash DSpark K=7 上测：
decode batch requests ∈ {1, 8, 64, 256, 512}，ctx ∈ {8K, 128K}，对比 `auto` / `cooperative`（仅小 batch）/ `persistent` / `deep_select`。
报告每 step 延迟，以及 FULL graph capture 后的 pool 大小（看启动日志里的 graph 显存行）。
**不需要**跑 GSM8K（top-k 是精确选择，Phase 3 正确性检查已覆盖）；若发现方案间 index 集合不一致率 > 0，则在报告中标红，交给人决定是否需要 eval。

---

## 交付物清单（结束时必须存在并已 push 到 `cg/indexer-topk-breakdown`）

- `benchmarks/kernels/benchmark_indexer_decode_topk_breakdown.py`（可 `--resume`，`--help` 写清所有参数）
- `benchmarks/kernels/indexer_breakdown/ENV.md`、`SHAPES.md`、`PROGRESS.md`、`REPORT.md`
- `benchmarks/kernels/indexer_breakdown/results/*.jsonl`（原始数据，每行含：git sha、GPU、clock、全部形状参数、方案、各 kernel 时间、显存、正确性）
- Phase 1 的 allocator snapshot 体积过大则不提交，只提交归因汇总（JSON）并在 REPORT 中写明 snapshot 在节点上的路径
- 最后在 `PROGRESS.md` 顶部写 10 行以内的 TL;DR：两个问题的答案 + 推荐规则 + 预计收益

## 时间表（总计 8 h）

| Phase | 上限 |
|---|---|
| 0 环境与基线 | 0:30 |
| 1 显存机制 | 1:30 |
| 2 形状盘点 | 0:30 |
| 3 microbenchmark | 3:30 |
| 4 分析与报告 | 1:00 |
| 5 端到端抽查 | 1:00（可跳过） |

若任一 Phase 超时：先保证已测数据落盘并写 PROGRESS，再进入下一 Phase。**Phase 4 的报告无论如何都要产出**，哪怕基于不完整的数据（写明缺什么）。
