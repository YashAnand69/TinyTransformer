# TinyTransformer-Lab-v2

## Intended use

Learning how a causal decoder transformer is trained, sampled, and inspected.
This is an English, character-level, narrow-domain educational model. It is not a
replacement for a broadly pretrained assistant or a reliable factual source.

## Architecture and artifacts

1,838,016 trainable parameters; four pre-normalized decoder blocks; four attention
heads; width 192; 768-wide GELU MLP; 256 positions; 101 tokens (printable ASCII,
newline, tab, and four special tokens). Learned position embeddings and tied input
and output weights. Training dropout: 0.1. Inference: float32.

`backend/checkpoints/best_model.pt` and `netlify/model/model.onnx` contain the same
learned model. Export tests compare last-token logits and every attention weight
against a PyTorch reference at a tolerance of 0.0001.

## Data

An original, synthetic curriculum authored for this repository: 58 machine-learning
and project topics, each with 27 question phrasings and one answer. Twenty-four
phrasings per topic are used for training, two for validation, and one for test.
No complete document is shared across splits. **The answers are shared**, so this
is explicitly a seen-topic paraphrase benchmark. Dataset hashes and counts are
in `backend/data/dataset_manifest.json`.

No external dataset was downloaded for this training run. The vocabulary is fixed
before evaluation. Prompts and padding are masked out of the training loss.

## Training and selection

Seed 2026, 2,000 initial updates followed by 5,000 refinement updates, batch size 24, AdamW (betas 0.9/0.95), weight decay 0.05,
peak learning rate 0.0008, floor 0.00008, 100 warmup steps, gradient norm clipped
to 1.0. Best checkpoint selected by validation answer loss. No checkpoint is
selected by test performance. Refinement weights the first 32 supervised tokens by 8.
Test cross entropy: 0.01902; perplexity: 1.01921.

The legacy checkpoint scored 4.40995 on the new test prompts. The legacy model had
less capacity, a shorter context, a different vocabulary, and much narrower
training coverage. The comparison measures the full upgrade; it cannot attribute
gains to a single change. The old validation score was affected by repeated text
across train and validation and should not be advertised as generalization.

## Free generation, not just loss

Greedy generation (up to 256 new tokens, stopping at EOS) exactly matches 57/58
canonical training answers and 26/58 answers for the reserved test format.
`canonical_evaluation.json` and `evaluation.json` contain every expected answer
and actual output. Exact match is strict; an incorrect answer can also be a fluent
explanation of a different topic. This is a serious limitation of the small model.

The first run's test format was examined during development, then moved into the
refinement validation set. A new format was reserved for the final test before
refinement. The final test was not used to choose a checkpoint. The initial stage
report is preserved in `pretraining_report.json`; it describes an earlier split
and must not be compared directly with the final report.

## Limitations

- Teacher-forced loss is easier than free generation; see `evaluation.json` for
  actual greedy outputs, including imperfections.
- New topics, alternative formatting, non-ASCII input, and long conversations can
  fail. Unknown characters map to the unknown token.
- The model may produce fluent but false text or repeat learned answers.
- Attention weights show one part of the computation, not a complete explanation.
- Sampling is reproducible within a fixed engine/environment, not across PyTorch
  and JavaScript or all hardware versions.
- The Stop button aborts waiting and display; a remote computation already started
  can continue until its function returns.
- No online learning occurs. Hosting infrastructure can retain request logs.
