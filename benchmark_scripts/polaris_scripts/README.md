# Polaris：每个 np 一个作业（和 Aurora 脚本相同的做法）

同一节点数（np = 节点数 × 32）的所有配置（不同 b、所有 send_count）在**一个 PBS 作业里依次运行**，
用同一组节点。逻辑和 Aurora 版本一致：作业内时间检查（放不下的段跳过，留给下一个作业）、自动续交、
按实测耗时自动校准估计、提交前等待同节点数或同队列的已有作业。

## 计划 `plan_main.tsv`（44 行）

| 节点数 | np | b | 队列 | 放置 |
|---|---|---|---|---|
| 1、2 | 32、64 | 8 | debug | 一个机柜（tier0） |
| 4、8 | 128、256 | 8、16 | debug-scaling | 一个机柜（tier0） |
| 16、32、48 | 512～1536 | 16、32 | prod | 一个 dragonfly 组（tier1） |
| 64 | 2048 | 32 和 64 | prod | 不限制 |
| 128、256 | 4096、8192 | 32（k 步长 4） | prod | 不限制 |

每个节点数：send_count 1～8192 一行（和作者相同），16384、24576、32768 各一行。
Polaris 一个机柜 14 个节点；GROUP=auto 时 ≤14 节点放一个机柜，≤56（GROUP_MAX）放一个 dragonfly 组。

## 运行（项目根目录，Polaris 上编译好的 ./main）

```bash
S=benchmark_scripts/polaris_scripts
DRY=1 EXE=./main bash $S/submit_plan.sh $S/plan_main.tsv          # 预览：10 个作业

EXE=./main nohup setsid bash $S/submit_plan.sh $S/plan_main.tsv prod          > /dev/null 2>&1 &
EXE=./main nohup setsid bash $S/submit_plan.sh $S/plan_main.tsv debug-scaling > /dev/null 2>&1 &
EXE=./main nohup setsid bash $S/submit_plan.sh $S/plan_main.tsv debug         > /dev/null 2>&1 &
```

- prod：6 个节点数（16～256）**同时**提交、各自排队（每个节点数一个线程；项目最多同时运行 10 个）。
- debug / debug-scaling：节点数一个接一个。
- 进度：`qstat -u $USER`，`tail submit_plan_main_prod.log`，`ls results/Polaris/main/*.csv | wc -l`（全部完成为 44）。
- 重新运行同一条命令会跳过已完成的段；重启前先 `pkill -f submit_plan.py`。

## 时间

- walltime = 该节点数所有段估计之和 × 1.5，不超过队列上限（prod：≤24 节点 3 h，25～99 节点 6 h，≥100 节点 24 h；debug 1 h）。
- 估计起点 `timing_baseline.csv` = 作者 Polaris 数据的计时汇总；带宽默认 100 GB/s（BW_GBS）。
  你的作业跑完后用实测（`.meta` 里的 `elapsed_sec`）自动校准；`python3 $S/calibrate_plan.py $S/plan_main.tsv` 查看。
- 某段估计超出时间就不提交（显示 NOT SUBMITTED），可用 `WALLTIME=H:MM:SS` 放宽。

## 合并、对比

```bash
bash $S/merge_results.sh main     # -> results/Polaris/main/merged/*.csv 和 results_all.csv（1～32K 按 send_count 顺序）
python3 $S/compare_with_author.py results/Polaris/main/merged <作者仓库>/.../all_gather/Tests/Polaris
```
