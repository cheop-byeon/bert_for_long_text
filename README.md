# Long-Text BERT Classification (AB / A+B)

This repository provides a long-text BERT pipeline for paragraph–commentary classification.

Current primary entry points:

- `main.py`: training + validation
- `infer.py`: checkpoint inference + prediction export

Both currently use `lib/base_model_ga.py` as the shared model loop backend.

## Project diagnosis (March 2026)

Static checks on active code paths are clean:

- `python3 -m py_compile main.py infer.py lib/*.py tfidf.py "tfidf copy.py"` passes
- no editor diagnostics in `main.py`, `infer.py`, `lib/base_model_ga.py`

Current dependency status:

- `main.py` and `infer.py` import from `lib/base_model_ga.py`
- `lib/text_preprocessors.py` depends on `lib/pooling.py`

Known caveats:

- `running_script.sh` uses `--wandb_mode False`, but `main.py` expects `--enable_wandb` as a flag
- `infer.py` defaults to a placeholder input path when `--input_file` is omitted, which will fail unless changed
- `lib/encoder.py` exists but is not used by `main.py`/`infer.py`

## What `main.py` does

- training + evaluation with early stopping and best-checkpoint saving
- K-fold training via `--kfold` (expects `train.{fold}.tsv` and `valid.{fold}.tsv`)
- model selection via `--model_type` (`AB`, `AplusB`, `AplusB21`, `SBERT`)
- optional W&B logging via `--enable_wandb`
- evaluation exports to `--eval_path` as `group.{epoch}.csv`

## Input Data Format

The training/validation TSVs must include:

- `paragraph`: Text unit from Internet-Drafts (I-D) or RFCs
- `commentary`: Comments collected from GitHub Pull Requests (PR)
- `label`: Binary label (1 = commentary is relevant to the paragraph, 0 = not relevant)
- `draft`: Internet-Draft name and version identifier
- `wg`: Working Group (WG) responsible for the I-D

## Model types

- **AB**: Separate paragraph/commentary encoding with pooling, combined by element-wise product and scored.
- **SBERT**: SBERT-style sentence-pair features (rep / diff / product) before classification.
- **AplusB**: Joint paragraph+commentary representation with pooled chunks.
- **AplusB21**: Joint paragraph+commentary packed into a single 512-token sequence.

## CLI arguments (active)

### `main.py`

- `--data_path`: folder containing `train.{fold}.tsv` and `valid.{fold}.tsv`
- `--model_type`: `AB`, `AplusB`, `AplusB21`, `SBERT`
- `--epochs`, `--batch_size`, `--patience`, `--kfold`
- `--model_path`: checkpoint output path
- `--eval_path`: folder for per-epoch evaluation CSVs
- `--max_slice`: max commentary chunks kept in AB/SBERT path
- `--pooling`: pooling strategy selector (0=mean, 1=max)
- `--max_tokens`: token limit argument (used by preprocessing settings)
- `--seed`, `--visible_gpus`, `--parallel`, `--resume`, `--verbose`
- `--enable_wandb`: enable W&B (otherwise disabled)

### `infer.py`

- `--input_file`: input TSV for inference (recommended)
- `--model_path`: trained checkpoint path
- `--model_type`: `AB`, `AplusB`, `AplusB21`
- `--max_slice`, `--pooling`, `--max_tokens`: preprocessing/chunking controls
- `--batch_size`, `--epochs`, `--patience`, `--seed`, `--visible_gpus`
- outputs `*.predictions.tsv` next to input file

## Minimal commands

Training:

`python3 main.py --data_path ./wg-github/ --epochs 10 --model_type AB --model_path ./models/best-model-parameters.pt --patience 5`

Inference:

`python3 infer.py --input_file ./overview-update.tsv --model_path ./AB/short/bmp.pt --model_type AB`

## Library layout (lib/)

- `architecture.py`: BERT and SBERT classification architectures, pooling, and similarity functions.
- `base_model_ga.py`: active base loop used by `main.py` and `infer.py`.
- `custom_datasets.py`: Tokenized datasets and collate fns for single-text and AB inputs.
- `custom_datasets_ab.py`: Dataset/collate for A+B combined inputs.
- `text_preprocessors.py`: Tokenizers and pooled chunking for long-text inputs.
- `pooling.py`: Chunking utilities and paragraph–commentary packing helpers.
- `linear_lr.py`: Linear learning-rate scheduler.
- `encoder.py`: legacy encoder module (currently unused by `main.py`/`infer.py`).

Note: some legacy/local files (for example `base_model.py`, `base_model_amp.py`, TF-IDF experiments) may exist in local worktrees but are not part of the tracked main pipeline.

## Configuration

Model loading and default hyperparameters live in `config.py` (e.g., `MODEL_PATH`, `MODEL_LOAD_FROM_FILE`, and default parameter dicts).

## Python Virtual Environment

This project includes a local virtual environment at `bert_for_long_text_env/`. To recreate the environment, create and activate a venv, then install dependencies from `requirements.txt`:

- Create: `python3 -m venv bert_for_long_text_env`
- Activate: `source bert_for_long_text_env/bin/activate`
- Install: `pip install -r requirements.txt`


## Related Paper

This implementation is applied to the following paper:

```bibtex
@inproceedings{bian2024tell,
  title={Tell Me Why: Language Models Help Explain the Rationale Behind Internet Protocol Design},
  author={Bian, Jie and Welzl, Michael and Kutuzov, Andrey and Arefyev, Nikolay},
  booktitle={2024 IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN)},
  pages={447--453},
  year={2024},
  organization={IEEE}
}
```

## Figures

Below are key visualizations and results from the project:

- [Model Type A](img/modelAplus.pdf)
- [Model Type B](img/modelBplus.pdf)
- [poster](img/ICMLCN2024_poster.pdf)

## Acknowledgements

This repository builds on ideas and code patterns from:
- [pdsxsf/roberta_for_longer_texts](https://github.com/pdsxsf/roberta_for_longer_texts)
- [mim-solutions/bert_for_longer_texts](https://github.com/mim-solutions/bert_for_longer_texts)