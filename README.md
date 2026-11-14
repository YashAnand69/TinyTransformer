# TinyTransformer Lab

An inspectable language model trained from random weights, with a React playground,
real attention heatmaps, next-token probabilities, and measured training telemetry.
Runs locally with PyTorch or on Netlify with the same weights exported to ONNX.

## What's improved in v2

- **1,838,016 parameters**, up from 813,184; **256-character context**, up from 128.
- 58 original teaching topics with 1,392 training, 116 validation, and 58 test documents.
- Document-level splits, masked prompt/padding loss, a fixed vocabulary, a seeded
  training sampler, validation-selected checkpoints, and an explicit final test.
- Accelerated PyTorch causal attention plus an explicit, inspectable ONNX path.
- End-of-response tokens, seeded sampling, stop controls, useful errors, and
  automatic formatting for plain-text questions.
- Charts and source views generated from actual training artifacts. No fabricated
  attention matrix or canned answer when inference fails.
- Model, export, API, and frontend checks run in GitHub Actions.

## Results and limitations

| Same held-out question phrasings | Previous checkpoint | v2 checkpoint |
| --- | ---: | ---: |
| Answer-token cross entropy | 4.410 | 0.019 |
| Character perplexity | 82.27 | 1.019 |
| Parameters | 813,184 | 1,838,016 |
| Context length | 128 | 256 |

Greedy free generation exactly matches the target on **57/58 canonical training
questions**, but only **26/58 held-out question formats**. This gap matters:
the model is useful for studying learned explanations, but still has weak format
generalization. Both full output sets are committed in `docs/`.

These are **teacher-forced answer-token metrics on new phrasings of known topics**.
Answers and topics overlap with training by design. This tests narrow paraphrase
robustness, not unseen-topic knowledge or general reasoning. Context length and
vocabulary also changed; this is a whole-system comparison, not an isolated
parameter-count experiment. The original corpus repeated text across its split,
so its old validation score is not comparable to the new validation score.

The model remains a small educational demonstration. It can memorize or mix up
answers, hallucinate, and fail on unfamiliar questions. Token probability is not
factual confidence. See [model card](docs/MODEL_CARD.md),
[recorded generations](docs/evaluation.json), and
[training report](backend/checkpoints/training_history.json).

## Run locally

Requires Python 3.9–3.12 and **Node 22.12+** (the `.nvmrc` selects Node 22).

```bash
nvm install
nvm use
python3 -m venv backend/.venv
backend/.venv/bin/python -m pip install -r backend/requirements.txt
npm ci
npm --prefix frontend ci
./run.sh
```

- Dashboard: http://127.0.0.1:5173
- API and interactive docs: http://127.0.0.1:8008/docs

The launcher checks port availability and stops only processes it starts.
Try `What is attention?`, `Explain overfitting.`, or `What are your limitations?`.
For reproducible sampling, use the same prompt, settings, seed, and engine.
PyTorch and the JavaScript runtime use different random number generators;
identical seeds are not promised to generate identical sampled text across engines.

## Train and evaluate

```bash
# Stage 1: learn the initial curriculum from random weights.
backend/.venv/bin/python backend/train.py --steps 2000 --seed 2026 --device cpu --curriculum bootstrap --prefix-weight 1 --output backend/runs/v2
# Stage 2: improve question conditioning and format coverage.
backend/.venv/bin/python backend/train.py --steps 5000 --seed 2026 --init backend/runs/v2/best_model.pt --output backend/runs/v3
```

The default architecture has four layers, four heads, width 192, a 256-token
context, GELU, pre-normalization, tied embeddings, and dropout 0.1. Training uses
AdamW, warmup/cosine decay, gradient clipping, and answer-only loss. Refinement weights the first 32
supervised tokens eight times more strongly to improve answer selection. Checkpoints
are selected using validation loss. CPU, CUDA, and Apple MPS are supported.
Reproduction can vary across hardware and PyTorch versions. The initial stage took about six minutes on CPU, followed by about 198 seconds
on Apple MPS for refinement; see the report for the exact configuration and duration.

Promote a run only after reviewing its validation results:

```bash
cp backend/runs/v3/best_model.pt backend/checkpoints/best_model.pt
cp backend/runs/v3/final_model.pt backend/checkpoints/final_model.pt
cp backend/runs/v3/tokenizer.json backend/data/tokenizer.json
cp backend/runs/v3/training_history.json backend/checkpoints/training_history.json
backend/.venv/bin/python -m pip install -r backend/requirements-dev.txt
backend/.venv/bin/python scripts/export_model.py
backend/.venv/bin/python scripts/sync_artifacts.py
backend/.venv/bin/python scripts/evaluate.py
backend/.venv/bin/python scripts/evaluate.py --split canonical --output docs/canonical_evaluation.json
backend/.venv/bin/python scripts/sync_artifacts.py
```

`final_model.pt` is an alias of the validation-selected model, not necessarily the
last optimization step. Run directories are ignored by Git. The curriculum and
split hashes are versioned in `backend/data/dataset_manifest.json`.

## Verify

```bash
backend/.venv/bin/python -m unittest discover -s backend -p 'test_*.py'
npm test
npm run build
```

Tests cover causal masking, fast/explicit attention equivalence, prompt masking,
split disjointness, evaluation mode restoration, exported model parity, seeded
sampling, and API validation. The export parity fixture is regenerated from the
PyTorch checkpoint by `scripts/export_model.py`.

## Netlify

Site: https://tinytransformer-yashanand.netlify.app

```bash
npm ci
npm run build
npx --package netlify-cli netlify deploy --prod --no-build
```

Netlify publishes `frontend/dist` and routes `/api/*` to a CPU inference function.
The ONNX weights and tokenizer are packaged with that function; no local backend
is needed. Local Vite development uses port 8008. `VITE_API_BASE_URL` optionally
selects another API at build time. There are no required third-party model keys.
The existing site may require Netlify team login. GitHub CI validates commits;
production deployment remains a separate, manual step.

## Vercel

Deploy from the repository root, using Node.js 24 and the Other framework preset:

```bash
npm ci
npm test
npx vercel --prod
```

`vercel.json` builds the dashboard into `frontend/dist` and packages the trained
ONNX model, tokenizer, metrics, and WebAssembly runtime with `api/[route].ts`.
The adapter shares the validated inference handler with Netlify. The dashboard
calls the same-origin `/api` endpoints; no external model key or local backend
is required. Keep the Vercel project's Root Directory at the repository root.
Deployment requires a Vercel account with access to the target project.

## Project map

- `backend/model.py`: transformer and sampling
- `backend/curriculum.py`: original educational dataset
- `backend/train.py`: training and evaluation
- `backend/server.py`: local FastAPI endpoints
- `netlify/functions/`: portable inference and HTTP validation
- `frontend/src/`: interactive lab
- `scripts/`: ONNX export, evaluation, artifact synchronization, tests

The project uses PyTorch's documented
[scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
for the training path. The explicit attention path remains available for teaching,
inspection, and export.
