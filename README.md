# Long-Text BERT Classification (AB / A+B)

This repository provides a long-text BERT pipeline focused on paragraph–commentary classification. The primary entry point is `main.py`, which handles training and evaluation with chunking/pooling for inputs longer than 512 tokens.

## What `main.py` does

- **Training + evaluation** with early stopping and best-checkpoint saving.
- **K-fold** training via `--kfold` (expects `train.{fold}.tsv` and `valid.{fold}.tsv`).
- **Model selection** via `--model_type` (AB, AplusB, AplusB21, SBERT).
- **Optional W&B logging** (`--enable_wandb`), with offline mode on GPU by default.
- **Evaluation exports** to `--eval_path` as `group.{epoch}.csv` containing predictions, labels, and metadata.

See [running_script.sh](running_script.sh) for training and evaluation examples.  
See [infer_script.sh](infer_script.sh) for inference examples.

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

## Key CLI arguments

- `--data_path`: Folder containing `train.{fold}.tsv` and `valid.{fold}.tsv`
- `--epochs`, `--batch_size`, `--patience`
- `--model_type`: `AB`, `AplusB`, `AplusB21`, `SBERT`
- `--model_path`: checkpoint output path
- `--eval_path`: folder for per-epoch evaluation CSVs
- `--max_slice`, `--pooling`: long-text chunking and pooling behavior
- `--enable_wandb`, `--seed`, `--kfold`

## Library layout (lib/)

- `architecture.py`: BERT and SBERT classification architectures, pooling, and similarity functions.
- `base_model.py`: Core training/evaluation loop, early stopping, metrics, and CSV export.
- `base_model_amp.py`: Same as base model with AMP (`torch.cuda.amp`) mixed-precision training.
- `base_model_ga.py`: Base model with gradient accumulation.
- `custom_datasets.py`: Tokenized datasets and collate fns for single-text and AB inputs.
- `custom_datasets_ab.py`: Dataset/collate for A+B combined inputs.
- `text_preprocessors.py`: Tokenizers and pooled chunking for long-text inputs.
- `pooling.py`: Chunking utilities and paragraph–commentary packing helpers.
- `linear_lr.py`: Linear learning-rate scheduler.
- `encoder.py`: Generic CNN/RNN/Transformer encoder with positional embeddings.
- `embed.py`: Alternative model using GloVe embeddings + `Encoder`.

## Configuration

Model loading and default hyperparameters live in `config.py` (e.g., `MODEL_PATH`, `MODEL_LOAD_FROM_FILE`, and default parameter dicts).


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