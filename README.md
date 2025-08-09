## Transformer-based Neural Machine Translation (PyTorch)

This repository contains a compact, from-scratch implementation of a Transformer encoder–decoder model for bilingual sequence-to-sequence translation using PyTorch. It trains on the Hugging Face `opus_books` dataset and uses a simple whitespace `WordLevel` tokenizer with special tokens. Training logs are written to TensorBoard and model checkpoints are saved every epoch.

### Key Features
- Standard Transformer architecture (multi-head attention, residual connections, layer normalization, position encodings)
- Hugging Face `datasets` for data loading and `tokenizers` for fast WordLevel tokenization
- TensorBoard logging (`runs/tmodel`) for loss curves
- Checkpointing to `{datasource}_weights/tmodel_{epoch}.pt`
- Config-driven hyperparameters in `Transformer/config.py`

## Project Structure
- `Transformer/main.py`: Model components and `build_transformer` factory
- `Transformer/dataset.py`: `BiLangData` dataset and `causal_mask`
- `Transformer/train.py`: End-to-end training loop, tokenization, logging, checkpointing
- `Transformer/config.py`: Centralized configuration and checkpoint path utilities
- `Transformer/tokenizer_*.json`: Persisted tokenizers (auto-created on first run if missing)
- `Transformer/runs/`: TensorBoard logs (created during training)
- `{datasource}_weights/`: Model checkpoints (created during training)

## Requirements
- Python 3.9+
- PyTorch (GPU optional but recommended)
- Additional Python packages: `datasets`, `tokenizers`, `tqdm`, `tensorboard`

Example installation (CPU PyTorch):

```bash
pip install torch torchvision torchaudio
pip install datasets tokenizers tqdm tensorboard
```

For CUDA builds of PyTorch, follow the official installation instructions and then install the remaining packages.

## Quickstart
Run training with the default English→Italian configuration:

```bash
cd Transformer
python train.py
```

Launch TensorBoard to monitor training:

```bash
tensorboard --logdir runs/tmodel
```

Checkpoints are saved each epoch to a folder derived from the data source, for example:

```text
opus_books_weights/tmodel_00.pt
opus_books_weights/tmodel_01.pt
...
```

## Configuration
Edit `Transformer/config.py` to adjust hyperparameters and paths:

- `batch_size`: Training batch size
- `num_epochs`: Number of epochs
- `lr`: Learning rate (Adam)
- `seq_len`: Fixed sequence length (inputs are padded/truncated; see dataset logic)
- `d_model`: Transformer hidden size
- `datasource`: Hugging Face dataset name (default `opus_books`)
- `lang_src`, `lang_tgt`: Source and target language codes (e.g., `en`, `it`)
- `model_folder`: Base model folder name (combined with datasource)
- `model_basename`: Checkpoint file prefix (e.g., `tmodel_`)
- `preload`: Optional epoch string to resume from (e.g., `"03"`)
- `tokenizer_file`: Pattern for tokenizer files (e.g., `tokenizer_{0}.json`)
- `experiment_name`: TensorBoard run directory (e.g., `runs/tmodel`)

Checkpoint path helper:

```python
get_weights_file_path(config, epoch)
```

resolves to:

```text
{config['datasource']}_{config['model_folder']}/{config['model_basename']}{epoch}.pt
```

Example: `opus_books_weights/tmodel_05.pt`.

## Data and Tokenization
- Dataset: `datasets.load_dataset('opus_books', f"{lang_src}-{lang_tgt}", split='train')`
- Split: 90% train / 10% test (validation loader is constructed; adjust as needed)
- Tokenizer: `WordLevel` with `Whitespace` pre-tokenizer and special tokens `[UNK]`, `[PAD]`, `[SOS]`, `[EOS]` (min frequency = 2)
- Tokenizers are trained once (if missing) and saved to `Transformer/tokenizer_{lang}.json` by default

## Model Architecture
- Encoder–decoder Transformer with positional encodings and multi-head attention
- Defaults: `N=6` layers, `h=8` heads, `d_model=512`, `d_ff=2048`, `dropout=0.1`
- Label smoothing: `0.1` in `CrossEntropyLoss`
- Padding tokens are masked in both encoder and decoder; causal mask applied to decoder self-attention

## Training Details
- Optimizer: Adam (`lr=1e-4`, `eps=1e-9`)
- Device: Automatically selects CUDA if available
- Logging: `SummaryWriter` at `experiment_name`
- Checkpointing: Saved at end of every epoch (model, optimizer, epoch, global_step)
- Resuming: Set `preload` (e.g., `"07"`) to resume training from an existing checkpoint

## Inference
An inference/decoding script (e.g., greedy or beam search) is not included yet. Contributions adding a decoding utility that loads a checkpoint, tokenizes input text, and produces translations are welcome.

## Troubleshooting
- Out-of-memory (OOM): Reduce `batch_size` or `seq_len`. Ensure mixed precision or gradient accumulation if extending the trainer.
- Dataset download issues: Verify network access and Hugging Face cache; try `pip install --upgrade datasets`.
- Tokenizer issues: Delete stale `Transformer/tokenizer_*.json` and rerun to retrain tokenizers.
- Checkpoint not found when resuming: Ensure `preload` matches an existing file suffix, e.g., `"03"` → `tmodel_03.pt`.

## Contributing
Contributions are encouraged. Improvements that would be especially valuable:
- Inference/decoding utilities and evaluation (e.g., BLEU, SacreBLEU)
- Beam search and length penalty
- Configurable learning-rate schedulers and warmup
- Proper train/validation/test splits and metrics tracking
- Mixed precision (AMP) and gradient accumulation for larger batch sizes
- Unit tests and lightweight integration tests
- Documentation examples for additional language pairs

Please open an issue to discuss significant changes before submitting a pull request. Keep edits focused, include rationale and references where helpful, and ensure style and linting pass.

## Acknowledgements
- Vaswani et al., "Attention Is All You Need"
- Hugging Face `datasets` and `tokenizers`


