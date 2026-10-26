# SD1.5、R1/R2 与指标代码

本目录只含源代码和少量数值配置。不含模型权重、图像、标注或训练日志。

路径通过命令行参数或环境变量传入，目录内不写本机绝对路径。

## 环境变量

- `SD15_MODEL`：Stable Diffusion 1.5 权重目录
- `SD15_DATA`：训练图像目录（可选）
- `SD15_MANIFEST`：训练清单目录（可选）
- `SD15_CENTERLINE`：中心线目标目录（可选）

## SD1.5

`sd15/infer.py` 读取清单并写出 raw 图。清单列：`case_id`（或 `base_case_id`）、`A_path`、`B_path`。

```
python sd15/infer.py --model <sd15_dir> --lora <lora.bin> --adapter <adapter_dir> --manifest samples.csv --out out_raw
```

LoRA 训练：`sd15/train_lora_fixed.py`。Adapter 训练：`sd15/train_adapter_fixed.py`。训练参数由命令行给出。

## R1 / R2

`r1r2/parameters.json` 是报告所用参数。清单列：`view_id`、`park_id`、`rotation`、`raw_path`、`condition_path`、`site_path`、`protect_path`、`urban_path`。

```
python r1r2/run_r1r2.py --samples samples.csv --parameters r1r2/parameters.json --r2-out out_r2 --r2p-out out_r2p
```

R1 为加权最近色。R2 以该 R1 为输入。R2' 以 raw 为输入。

## 指标

```
python evaluation/run_metrics.py --data-root <data_root> --out metrics_out
```

`<data_root>` 下需有 `samples.csv`、`outputs/`、`constraints/`、`conditions/`、`sa/`。跨生成器对照放在 `pixpix/`。指标定义在 `evaluation/eval_q.py`、`eval_spatial.py`、`eval_semantic.py`。

这套脚本对 RAW 使用读出 Q。若论文 RAW 行按调色板原色统计，需在调用前自行改为原色判定。
