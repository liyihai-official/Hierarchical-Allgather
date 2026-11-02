# Evaluation

## Run on Aurora

```bash
# 所有命令都在项目根目录运行
make -j4

# 查看当前登录节点上的提交进程，以及 PBS 中的任务
pgrep -u "$USER" -af '[s]ubmit_plan|[q]sub'
qstat -u "$USER"

# 每个新终端 / tmux 窗口都要重新设置
S=benchmark_scripts/aurora_scripts
chmod +x $S/*.sh

pgrep -af submit_plan                   # 确认同一个队列没有已在运行的提交循环

# 预览 main + batch；DONE already 的配置会跳过
DRY=1 EXE=./main bash "$S/submit_plan.sh" "$S/plan_main.tsv" "$S/plan_batch.tsv"


# 确认同一队列没有已有提交循环后，下面两条各执行一次
# capacity：1/2/4/8/16 节点
EXE=./main nohup setsid bash "$S/submit_plan.sh" \
  "$S/plan_main.tsv" "$S/plan_batch.tsv" capacity \
  > aurora_capacity.launch.log 2>&1 < /dev/null &

EXE=./main nohup setsid $S/submit_plan.sh $S/plan_main.tsv capacity      > /dev/null 2>&1 &

# debug-scaling：26/32/52/64/80/96/112/128 节点
EXE=./main nohup setsid bash "$S/submit_plan.sh" \
  "$S/plan_main.tsv" "$S/plan_batch.tsv" debug-scaling \
  > aurora_debug-scaling.launch.log 2>&1 < /dev/null &

EXE=./main nohup setsid $S/submit_plan.sh $S/plan_main.tsv debug-scaling > /dev/null 2>&1 &

# 查看启动状态和错误；Ctrl+C 退出 tail 不会停止后台提交或 PBS 任务
# 启动日志会出现 SUBMIT qsub ...
tail -f aurora_capacity.launch.log aurora_debug-scaling.launch.log
```

跑完后：

```bash
ls results/Aurora/main/*.FAILED         # 有就重新运行上面两条命令补跑（节点故障导致，重跑即可）
$S/merge_results.sh
```

## Run: on Polaris

```bash
make -j4

S=benchmark_scripts/polaris_scripts
chmod +x $S/*.sh

# debug：1、2 节点，逐个等待执行
EXE=./main nohup setsid bash "$S/submit_plan.sh" \
  "$S/plan_main.tsv" "$S/plan_batch.tsv" debug \
  > polaris_debug.launch.log 2>&1 < /dev/null &

# prod：10 个 job，提交后直接返回
EXE=./main bash "$S/submit_plan.sh" \
  "$S/plan_main.tsv" "$S/plan_batch.tsv" prod

EXE=./main $S/submit_plan.sh $S/plan_main.tsv prod      # 16 节点及以上：一次全部提交，直接返回

EXE=./main nohup setsid $S/submit_plan.sh $S/plan_main.tsv debug-scaling > /dev/null 2>&1 &   # 4、8 节点

```

跑完后：

```bash
$S/merge_results.sh
```

## Run: on Fugaku 富岳

### 注意:

- 提交循环（nohup 或 tmux）只存在于启动它的那台登录节点上。下次登录如果分到别的节点，
  `pgrep` 和 `tmux attach` 都看不到它，要先 `ssh aurora-uan-00xx` 跳回原来的节点。
  日志文件和 `qstat` 在哪台节点上都能看。
- 同一个队列只能运行一个提交循环。启动前先 `pgrep -af submit_plan` 确认一下。
