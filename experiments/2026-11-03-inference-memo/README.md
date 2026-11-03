# 推理引擎教学实验

本目录包含9个仅依赖 Python 标准库的教学仿真实验，以及历史输出和两个合成 Perfetto Trace。实验用于解释算法和调度机制，结果不能作为真实 GPU、推理引擎或训练框架的性能测量。

## 运行

必须先将整个实验目录复制到独立运行目录，再执行副本。以下准备命令在 inference-lab 仓库根目录执行：

```sh
run_dir=$(mktemp -d)
cp -R experiments/2026-11-03-inference-memo "$run_dir/experiment"
python3 "$run_dir/experiment/lab01_roofline_and_kvcache_calc.py"
```

其余脚本使用运行副本目录中的对应文件名。lab09 会写入副本相邻 evidence 目录的 trace_bad_sync.json 和 trace_good_overlap.json。保留仓库中的历史结果，将新输出保存在独立运行目录。

## 实验索引

| 文件前缀 | 主题 |
| --- | --- |
| lab01 | 前向形状、KV Cache 显存与 Roofline |
| lab02 | 分页块管理与连续批处理调度 |
| lab03 | Radix Tree 前缀缓存 |
| lab04 | 重叠调度与张量并行 |
| lab05 | Online Softmax 与 Split-KV |
| lab06 | 投机解码与多阶段流水线 |
| lab07 | MLA 矩阵吸收与受约束解码 |
| lab08 | Chunked Prefill、睡眠唤醒与权重更新 |
| lab09 | RL Rollout、训推差异与 Trace 诊断 |

## 结果文件

evidence 目录的9份日志和2份 JSON 是原有教学实验记录。历史日志保留当时的输出，包括旧机器路径；旧路径不代表当前运行位置。运行副本的输出单独保存，不覆盖历史记录。

上游仓库版本及对应文件以 upstream-references.json 的核验结果为准。未核实的路径明确标记，不表示已验证实现或性能。
