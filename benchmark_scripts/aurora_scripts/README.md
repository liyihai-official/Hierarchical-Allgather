# Aurora: 每个 np 只提交一个 job

同一节点数（默认 `np = nodes × 104`）的配置在一个 PBS job 内顺序执行。
同时传入 main 和 batch 时，也按节点数合并。原有 b、k、消息大小、50 次重复不变。

```text
np=1664：1 个 job
  main：b=26 → b=52 → b=104
  batch：b=4 → b=13 → b=16 → b=26 → b=52
```

## 在项目根目录运行

```bash
make -j4
S=benchmark_scripts/aurora_scripts

# 预览全套：13 个 job（已有完整结果的配置会跳过）
DRY=1 bash "$S/submit_plan.sh" "$S/plan_main.tsv" "$S/plan_batch.tsv"

# 实际提交：每个 np 一个 job，等当前 job 完成后提交下一个
EXE=./main bash "$S/submit_plan.sh" "$S/plan_main.tsv" "$S/plan_batch.tsv"

# 仅跑 np=1664：两份清单合并后只提交 1 个 job
NODES="16" EXE=./main bash "$S/submit_plan.sh" "$S/plan_main.tsv" "$S/plan_batch.tsv"

# 手动指定时间（旧估算只作参考，不会阻止提交）
NODES="16" WALLTIME=1:00:00 EXE=./main bash "$S/submit_plan.sh" "$S/plan_main.tsv" "$S/plan_batch.tsv"
```

仅传 main 时共 13 个 job；仅传 batch 时共 10 个 job。需要全套同 np 一个 job 时，
请在同一条命令传入两份计划，不要分别启动两次。

可沿用队列过滤参数，将节点数分成两条提交循环；两组节点数不重叠：

```bash
EXE=./main bash "$S/submit_plan.sh" "$S/plan_main.tsv" "$S/plan_batch.tsv" capacity
EXE=./main bash "$S/submit_plan.sh" "$S/plan_main.tsv" "$S/plan_batch.tsv" debug-scaling
```

## 时间和队列

- 默认每节点 104 个进程，`PPN`、`BIND`、`NOGPU` 会显式传给 job。
- `debug-scaling` 默认申请 1 小时；其他队列默认将待跑配置的旧估算求和并留余量。
- `WALLTIME=H:MM:SS` 或 `--walltime H:MM:SS` 指定实际申请时间。旧估算仅提示，不强制要求跑够估算时间。
- `QUEUE=name` 或 `--queue name` 覆盖目标队列。已知队列的节点数和时间范围会在提交前检查。
- 不会为适应队列时间限制而拆成多个 job。实际超时后，用户重新提交相同命令可跳过已完成配置。
- 默认队列：1/2/4/8/16 节点为 capacity；26/32/52/64/80/96/112/128 节点为 debug-scaling。
- 队列限制参考：[ALCF Aurora Running Jobs](https://docs.alcf.anl.gov/aurora/running-jobs-aurora/)。以集群实际政策为准。

## 输出和续跑

计划表的每行现在表示 job 内的一段配置，不再对应单独 job。旧的消息大小分块仍在同一个 job 内执行。
提交时生成不可变的配置快照 `results/Aurora/plans/n<N>_*.tsv`，避免排队期间编辑原计划影响 job。

CSV、main/batch 输出目录、`merge_results.sh`、`compare_with_author.py` 的使用方式保持兼容。
成功结果必须包含全部预期 `(algorithm,k,b,np,send_count)`、每组 50 行，且 `is_correct=1`。
不完整 CSV 不会被当成完成；成功文件通过临时文件原子移动发布。
失败时保留 `.FAILED`，停止当前 job 和提交循环；重新运行同一命令即可续跑。

`DRY=1` 只预览，不要求本机有可执行文件，不调用 qsub，也不创建结果快照。
`NODES="2 4"` 过滤节点数；`SKIP_FLAGGED=1` 会明确排除旧计划中标记的配置，默认不排除。
日志为 `submit_plan_main_plan_batch.log`（队列过滤时带队列后缀）。
不要同时启动覆盖相同节点数的提交循环，以免重复提交。

```bash
bash "$S/merge_results.sh"
```

## 新增规模

新增 128 节点，默认 `np=128×104=13312`。main 沿用 64 节点的配置（b=52、104），
batch 沿用 32 节点的配置（b=4、8、13、16、26、52）。两份计划一起传入时只提交一个新 job。
现有行保持原样，新行保留参考配置的消息大小分块。耗时按节点数比例外推并标记 `estimate_linear`，仅作参考。

```bash
# 只预览新规模
NODES="128" DRY=1 bash "$S/submit_plan.sh" "$S/plan_main.tsv" "$S/plan_batch.tsv"
# 只提交新规模
NODES="128" EXE=./main bash "$S/submit_plan.sh" "$S/plan_main.tsv" "$S/plan_batch.tsv"
```

以后在 `gen_plan.py` 的 `MAIN_EXT` 和 `BATCH_EXT` 中配置新增规模：
`新节点数: 原始参考节点数`。本次为 `MAIN_EXT[128]=64`、`BATCH_EXT[128]=32`。
生成器根据作者原始 CSV 重新计算分块，分块边界可能与当前复制的边界不同；重生成前先完成当前计划。

## 64～128 节点之间的新增规模

已补充 80、96、112 节点（默认 np=8320、9984、11648）。每个新 np 均沿用
128 节点的 main/batch 参数与消息大小分块：main 的 b=52、104，batch 的 b=4、8、13、16、26、52。
main 的 b=52 和 batch 的消息为 4 B～32 KiB；main 的 b=104 为 32 B～64 KiB（按 4 字节 MPI_INT 计算）。
每个新 np 一个 job，内部 15 段配置（main 8 段、batch 7 段）。

完整计划现在为 13 种 np、最多 13 个 job，共 140 段配置；已有完整结果仍会跳过。
原有行保持原样，新增行的耗时按节点数比例外推，仅作参考。默认队列仍为 debug-scaling，申请 1 小时。

```bash
# 仅预览这次新增的 3 个 job
NODES="80 96 112" DRY=1 bash "$S/submit_plan.sh" "$S/plan_main.tsv" "$S/plan_batch.tsv"
# 仅提交这次新增的 3 个 job
NODES="80 96 112" EXE=./main bash "$S/submit_plan.sh" "$S/plan_main.tsv" "$S/plan_batch.tsv"
```

生成器已同步：MAIN_EXT 中新增 80/96/112 对 64 节点的映射，BATCH_EXT 中新增这些规模对 32 节点的映射。
参考参数与 128 节点一致，重新生成后仍包含所有新增规模。

## 补齐 80/96/112/128 节点，并把 b=104 扩展到 24k、32k（只跑 main）

`plan_main.tsv` 已用 `extend_plan_main.py` 修改（可重复运行，结果不变）：

- 80/96/112/128 节点：保留已完成的 b=52 send_count 1～16 那一行；其余改为**每个 send_count 一行**
  （b=52 的 32～8192，b=104 的 8～16384）。
- 所有节点数：b=104 新增 send_count 24576 和 32768 两行（k=2..101 步长 3，与原 b=104 一致）。
- 原有的其他行一字不改，已完成的结果仍会被识别为 DONE。
- est_min 现在由 `calibrate_plan.py` 按你自己的结果计算（见下面“时间估计”）。

注意：不要再运行 `gen_plan.py` 或 `extend_plan_main.py`，它们会按作者的数据重写 est_min。
提交时 est_min 会自动重新计算，计划文件里的值只是起点。

```bash
S=benchmark_scripts/aurora_scripts
DRY=1 EXE=./main bash "$S/submit_plan.sh" "$S/plan_main.tsv"            # 预览：DONE / 要跑的 / 不提交的
EXE=./main nohup setsid bash "$S/submit_plan.sh" "$S/plan_main.tsv" capacity      > /dev/null 2>&1 &
EXE=./main nohup setsid bash "$S/submit_plan.sh" "$S/plan_main.tsv" debug-scaling > /dev/null 2>&1 &
```

### 执行方式（submit_plan.py / allgather_aurora_run.sh 已改）

- 每个节点数是“一个作业接一个”：
  - **作业内时间检查**：每段开始前，若 `已用时间 + est_min + 3 分钟 > walltime`，跳过这段（日志 `DEFERRED`），
    继续看后面的段，不会有段跑到一半被杀。
  - **自动续交**：作业结束后，只要完成了至少一段且还有没完成的，就自动提交下一个作业；
    一段都没完成（例如节点故障）才停下，并接着处理下一个节点数。
  - **单段超过时限的不提交**：DRY 里显示为 `WILL NOT SUBMIT`。
- `TIME_SAFETY=0.8` 按估计的 0.8 倍检查（估计偏保守时用）；`TIME_MARGIN_SEC` 改留白秒数（默认 180）。

## 时间估计（按你自己的结果自动校准）

作者的计时在 8 节点以上、大消息时是你的 3～5 倍，不能直接用；而且每次重复里没计时的
“清零接收缓冲区 + 逐个比对正确性”在大消息时占了很大一部分时间。所以现在：

```
est_min = Σ 每个 send_count [ 你的计时（已跑过就用实测；没跑过从你更小节点数/更小消息线性外推 × 1.2）
                             + 每节点清零和比对的内存读写量 / BW ] + 启动时间
```

- **BW 和启动时间从你每段的真实耗时拟合**：读 `results/Aurora/main/runs/*/*.meta`
  （开始时间 + `elapsed_sec`，旧的没有这一行就用文件修改时间）。BW 取最慢的那一段，偏保守。
  没有可用的耗时记录时用 BW=270 GB/s、启动 2 分钟：由 10 月 16～17 日 debug-scaling 作业的开始时间推出
  （该队列每人同时只跑一个作业，相邻两次开始的间隔就是前一个作业耗时的上限；52 节点 16384 ≤ 26.2 分钟）。
  文件被复制过（`cp` 没加 `-p`、下载、在别的机器上重新打包）时修改时间不是真实的，这些记录会被自动忽略。
- **`submit_plan.py` 每次提交前都会重新计算**：每跑完一个作业，就多了实测耗时，估计随之更新；
  原来判定跑不完的段如果估计降到 57 分钟以内，会在下一次自动提交。DRY 显示的也是重新计算后的值。
- 单独查看或写回计划：
  ```bash
  python3 "$S/calibrate_plan.py" "$S/plan_main.tsv"            # 只打印新旧估计
  python3 "$S/calibrate_plan.py" "$S/plan_main.tsv" --write    # 写回 est_min / walltime / 标记
  ```
- `timing_baseline.csv`（和脚本放在一起）是你 10 月 16～29 日各段的计时汇总，作为计时部分的起点；
  重新全部跑时也会用它，新结果出来后与它一起平均。`CALIB_BASELINE=0` 不用它。
- 旧结果如果挪到了别处：`CALIB_DATA=那个目录`（也可以是合并后的 results_all.csv，多个用 `:` 分隔），
  例如 `CALIB_DATA=old/main:results/Aurora/main/results_all.csv`。
- 其他开关：`AUTO_CALIB=0` 不重新计算，直接用计划里的 est_min；`BW_GBS=290` 指定还没有实测记录时用的带宽（有了实测就以实测为准）；
  `LAUNCH_MIN`、`EXTRAP_PAD`（外推放大系数，默认 1.2）。

按默认值（BW 270 GB/s）估计，仍有 12 段超过 57 分钟（都是 b=104）：52、64 节点 32768；80、96 节点 24576、32768；
112、128 节点 16384、24576、32768。实际跑过以后估计会按实测更新；仍然降不下来的段需要 `main.cpp` 支持只跑一部分 k。

## 结果文件被挪走后恢复（DONE 识别不到时）

`submit_plan.py` 判断 DONE、`merge_results.sh` 合并，都只看 `results/Aurora/main/` 下的分段文件
`results104_<N>_<b>__base.._it.._inc.._<job>.csv`。每个作业在 `runs/<job>_n.._b.._base.._it../` 里还有一份副本，
所以只要 `runs/` 还在，就可以把分段文件重建出来（只复制成功且完整的，不覆盖已有文件）：

```bash
mkdir -p results/Aurora/main && mv 旧位置/runs results/Aurora/main/     # runs/ 不在这里时先挪回来
DRY=1 python3 "$S/restore_chunks.py" results/Aurora/main                # 先看会恢复哪些
python3 "$S/restore_chunks.py" results/Aurora/main
```


### 内存

np=13312、32768 个元素/进程时，接收缓冲区 = 13312×32768×4 B ≈ 1.7 GB/进程；若还有参考结果缓冲区，
每进程约 3.5 GB，每节点 104 进程约 360 GB（节点 DDR 1 TB）。元素总数 4.4 亿，仍小于 int 上限，
但若代码里用 int 计算“字节数”(≈1.74e9) 已接近 2^31，建议检查。

### 合并

`merge_results.sh` 同一个分段若有多份结果文件（重复跑过），只取作业号最小的一份，并打印提示。

## 节点放在同一个机柜（默认开启）

Aurora 一个机柜（`tier0`）= 64 个节点 = 一个 dragonfly 组。`submit_plan.py` 默认：

| 节点数 | qsub 加的选项 |
|---|---|
| ≤ 64 | `-l place=scatter:group=tier0`（所有节点在同一个机柜） |
| > 64 | 不加（放不进一个机柜） |

- `GROUP=none`：不限制；`GROUP_MAX=56`：只对 ≤56 节点限制（64 节点要等整个机柜空闲，可能排很久）。
- `DRY=1` 预览里每个作业会显示 `placement=...`。
- 每段的 `.meta` 新增 `--- racks (nodes per rack):`，记录这次实际用了哪些机柜、各几个节点。
