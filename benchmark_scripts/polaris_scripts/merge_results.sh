#!/bin/bash
# Join the chunk files of each (ppn, nodes, b) into one CSV named like the author's:
#   results/Polaris/<suite>/merged/results<PPN>_<nodes>_<b>.csv
# Chunks are ordered by their first send_count; header written once.
# usage: ./merge_results.sh [suite ...]     (default: main batch)

for suite in ${@:-main batch}; do
  dir=${ROOT:-results/Polaris}/$suite
  [[ -d $dir ]] || continue
  mkdir -p $dir/merged
  for tag in $(ls $dir/results*__*.csv 2>/dev/null | xargs -n1 basename | sed 's/__.*//' | sort -u); do
    out=$dir/merged/$tag.csv
    # one file per chunk (same base/n_iter/inc): if a chunk was run twice, keep the lowest job id
    files=$(ls $dir/${tag}__*.csv | awk -F'__base' '{split($2,a,"_"); print a[1]"\t"$0}' | sort -n -k1,1 -k2,2V \
            | awk -F'\t' '{stem=$2; sub(/_[0-9]+\.csv$/,"",stem); if (!(stem in seen)) {seen[stem]=1; print $2}}')
    dups=$(( $(ls $dir/${tag}__*.csv | wc -l) - $(echo $files | wc -w) ))
    (( dups > 0 )) && echo "  note: $tag has $dups duplicate chunk file(s); using one per chunk" 
    echo "algorithm_name,k,b,nprocs,send_count,time,is_correct" > $out
    for f in $files; do grep -v '^algorithm_name' "$f" >> $out; done
    echo "$out  <- $(echo $files | wc -w) chunk(s), $(( $(wc -l < $out) - 1 )) rows"
  done
  # ---- 汇总成一个表：results_all.csv（放在 merged/ 旁边，不放在里面）----
  if ls $dir/merged/results*.csv >/dev/null 2>&1; then
    all=$dir/results_all.csv
    echo "ppn,nodes,algorithm_name,k,b,nprocs,send_count,time,is_correct" > $all
    for f in $(ls $dir/merged/results*.csv | awk -F'[_.]' '{print $(NF-2)"\t"$(NF-1)"\t"$0}' | sort -k1,1n -k2,2n | cut -f3); do
      tag=$(basename $f .csv); tag=${tag#results}            # 例如 104_16_26
      ppn=${tag%%_*}; rest=${tag#*_}; nodes=${rest%%_*}
      tail -n +2 "$f" | sed "s/^/$ppn,$nodes,/" >> $all
    done
    echo "$all  <- $(ls $dir/merged/results*.csv | wc -l) file(s), $(( $(wc -l < $all) - 1 )) rows"
  fi  
done

