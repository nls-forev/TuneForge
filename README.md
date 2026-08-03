# TuneForge

QLoRA fine-tune of **Llama-3.1-8B-Instruct** on medical instructions, with an
LLM-as-judge evaluation that compares its free-text answers against the base model.

## TL;DR

After fine-tuning, the model wins a head-to-head test against base
Llama-3.1-8B-Instruct: **30.0% win / 11.5% loss** over 200 held-out medical
prompts (the rest are ties).

## What it does

- Fine-tunes `meta-llama/Llama-3.1-8B-Instruct` with **QLoRA** (4-bit, Unsloth)
  on the [`lavita/AlpaCare-MedInstruct-52k`](https://huggingface.co/datasets/lavita/AlpaCare-MedInstruct-52k)
  medical instruction dataset.
- Evaluates answer quality with a two-phase LLM-as-judge setup: generate base and
  fine-tuned answers on a held-out split, then grade them with a DeepSeek judge
  (absolute 1 to 5, pairwise win-rate vs base, ROUGE-L, BERTScore).
- Runs as a reproducible **DVC** pipeline. Training runs on a GPU box, judging runs
  locally.

## Results

Evaluated on 200 held-out medical prompts:

| Metric | Value |
|---|---|
| Pairwise vs base (Win / Tie / Loss) | 60 / 117 / 23 |
| Win % / Loss % | **30.0% / 11.5%** |
| Judge mean (1 to 5) | 4.405 |
| Eval loss | 0.664 |
| ROUGE-L | 0.424 |
| BERTScore (F1) | 0.494 |

The fine-tune wins about 2.6x more often than it loses. Most comparisons are ties,
so the win is real but modest (see [Limitations](#limitations)).

## Architecture

```
ingest  ->  transform  ->  train        ->  generate         ->  judge
(HF DL)     (chat SFT)     (QLoRA, GPU)     (base+FT answers)     (DeepSeek + metrics)
                           adapter           responses.parquet     metrics.json
                                             [GPU box]              [local]
```

The eval is split in two by dependency. `generate` needs the GPU stack, `judge`
only needs an API key and CPU. `responses.parquet` is the file passed between them.

## Key engineering choices

- **Response-masking:** `train_on_responses_only`, so loss is computed on the
  assistant's tokens only, not the prompt.
- **Quant-matched eval:** the base model is loaded in the same 4-bit weights the
  adapter trained on, so the comparison is fair.
- **Length-neutral judge:** the pairwise prompt grades medical correctness and
  ignores length/formatting, since LLM judges tend to over-reward longer answers.
- **Position-bias control:** every pairwise comparison is judged in both orders and
  averaged.

## Repo layout

```
src/
  pipeline/run_stage.py        # entrypoint: ingest|transform|train|generate|judge
  components/
    data_ingestion.py          # HF dataset download + split
    data_transformation.py     # to chat/SFT format
    model_trainer.py           # QLoRA training (Unsloth + TRL)
    evaluation/
      generate_responses.py    # phase A (GPU): base + FT answers
      judge.py                 # phase B (local): DeepSeek + ROUGE + BERTScore
  constants/ entity/ utils/ logger/
config/experiment.yaml         # seed, revisions, decoding, fixed eval IDs
config/hyperparams.yaml        # LoRA + training config
config/sweeps.yaml             # rank/LR/epoch/max-length sweep grid
dvc.yaml                       # tracked stages
Dockerfile.train  Dockerfile.evaluate
```

## Setup

Uses [`uv`](https://github.com/astral-sh/uv) with per-stage dependency groups, so
each stage installs only what it needs:

| Group | For |
|---|---|
| `train` | ingest, transform, train (unsloth, trl, datasets) |
| `evaluate` | generate (unsloth, 4-bit inference) |
| `judge` | judge (DeepSeek client, rouge, bert-score) |
| `dev` | dvc, pre-commit, detect-secrets |

```bash
uv sync --group train      # on the GPU box
uv sync --group judge      # locally
```

Secrets go in `.env` (gitignored): `DEEPSEEK_API_KEY` (judge), `HF_TOKEN` (optional).

## Run

Per stage:
```bash
PYTHONPATH=. uv run --group train    python -m src.pipeline.run_stage ingest
PYTHONPATH=. uv run --group train    python -m src.pipeline.run_stage transform
PYTHONPATH=. uv run --group train    python -m src.pipeline.run_stage train
PYTHONPATH=. uv run --group evaluate python -m src.pipeline.run_stage generate   # GPU
PYTHONPATH=. uv run --group judge    python -m src.pipeline.run_stage judge       # local
```

Or with DVC:
```bash
uv run dvc repro
```

## Config (`config/hyperparams.yaml`)

QLoRA: r=16, alpha=16, dropout=0.05, rsLoRA, all attention and MLP projections.
Training: lr=1e-4, 1 epoch, batch 4 x grad-accum 4, cosine schedule, warmup 0.05.

`config/experiment.yaml` is the reproducibility manifest: random seed, exact
dataset/model revisions, maximum sequence length, generation settings/seeds, and
optional fixed evaluation row IDs. Its resolved contents are saved beside the
adapter, generated responses, and metrics.

Queue the full rank x learning-rate x epoch x maximum-length DVC sweep with:

```bash
PYTHONPATH=. uv run python -m src.pipeline.sweep --dry-run  # inspect commands
PYTHONPATH=. uv run python -m src.pipeline.sweep            # queue experiments
dvc exp run --run-all                                       # execute queue
```

## Evaluation methodology

- **Sample:** 200 deterministically selected held-out prompts, identified by the
  original dataset `source_row_id` in `responses.parquet`.
- **Decoding:** configurable sampling parameters, evaluated over fixed generation
  seeds (42, 43, and 44 by default).
- **Absolute:** DeepSeek grades each fine-tuned answer 1 to 5.
- **Pairwise win-rate:** base vs fine-tuned, judged in both orders to cancel
  position bias, with the judge told to ignore length and grade correctness only.
- **Reference-based:** ROUGE-L and BERTScore against the dataset's reference answer.

## Infra & cost

- Trained on a single **AWS EC2 g5.xlarge** (A10G 24 GB), about 5 hours for 1 epoch.
- Adapters and metrics are pushed to S3. A watchdog terminates the GPU instance once
  the run finishes, so it never idle-bills.
- Judging runs locally (CPU + API), no GPU needed.

## Limitations

- 117 of 200 comparisons are ties, so the model matches base more often than it
  clearly beats it. The dataset (synthetic, GPT-generated instructions) is probably
  the ceiling.
- Single judge model (DeepSeek), no human eval.
- Medical-instruction domain only. Not a clinical tool, this is a research/learning
  project.

## Stack

Python, PyTorch, Unsloth, TRL, PEFT, bitsandbytes, QLoRA, DVC, uv, Docker,
AWS EC2/S3, DeepSeek (LLM-as-judge), ROUGE, BERTScore
