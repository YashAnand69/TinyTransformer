# TinyTransformer LM: Generative Decoder Transformer From Scratch

An autoregressive Causal Decoder Transformer built and pre-trained completely from scratch (zero external transformer dependencies) in pure PyTorch on a custom niche corpus ("The Cybernetic Alchemist's Research Log & Machine Philosophy Codex").

Includes real-time autoregressive text generation, interactive loss curves, multi-head attention weight heatmaps, next-token candidate probability distributions, and a complete dark-mode cyber aesthetic web dashboard.

---

## 🚀 Quick Start

### 1. Launch Both Backend & Frontend
```bash
cd /Users/yashanand/Desktop/Projects/TinyTransformer
./run.sh
```
- **Web Dashboard**: [http://127.0.0.1:5173](http://127.0.0.1:5173)
- **FastAPI Backend**: [http://127.0.0.1:8008](http://127.0.0.1:8008)
- **Interactive Swagger Docs**: [http://127.0.0.1:8008/docs](http://127.0.0.1:8008/docs)

---

## 🧠 Architecture Overview

Built from primitive PyTorch modules (`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.GELU`) without high-level abstractions:

| Component | Specification |
| :--- | :--- |
| **Model Type** | Causal Decoder-Only Transformer (GPT Architecture) |
| **Parameters** | **813,824** |
| **Embedding Dimension ($d_{model}$)** | 128 |
| **Transformer Blocks ($N_{layer}$)** | 4 Layers |
| **Attention Heads ($N_{head}$)** | 4 Heads (32 dimensions per head) |
| **Feed-Forward Expansion** | 4x ($128 \to 512 \to 128$) with GELU |
| **Context Window ($T_{block}$)** | 128 Tokens |
| **Vocabulary Size** | 86 Characters & Special Tokens (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`) |
| **Weight Tying** | Token Embedding Matrix tied to Output Projection Head |
| **Hardware Compute** | Apple Silicon Metal Performance Shaders (`mps`) |

---

## 📈 Training Telemetry & Loss Curves

Trained from pure random Gaussian noise ($\mu=0, \sigma=0.02$) with AdamW optimizer and cosine learning rate decay:

- **Total Steps**: 1,600 steps
- **Batch Size**: 32 sequences &bull; 128 tokens per sequence = 4,096 tokens/step
- **Total Tokens Trained**: **6,553,600 tokens**
- **Training Duration**: **40.93 seconds** (on Apple Silicon MPS)
- **Initial Loss**: 2.96 (step 50)
- **Final Validation Loss**: **0.2768** (&darr; 90.6% error reduction)
- **Final Perplexity**: **1.32** (down from 18.81)

### Training Progression (Evolution of Output)
1. **Step 0 (Pure Entropy)**: `=== LOG ENTRY?????0#00pgg!o0eNPywrr0!WP3..NPpk]](E?S...`
2. **Step 400 (Proto-Syntax)**: `=== LOG ENTRY = Thugrat by ithtion tonsthe alie of llemoby r...`
3. **Step 1000 (Grammar & Structure)**: `=== LOG ENTRY 1204: THE COponigh the trkeson is begen orad, ...`
4. **Step 1600 (Fluent Domain Knowledge)**: `=== LOG ENTRY 012: THE ANINTHE MINTANIFON == In recurent arc...`

---

## 💻 Web Dashboard Features

1. **Interactive Generation Playground**:
   - Seed prompt input with quick presets.
   - Real-time sliders: Temperature ($0.10 - 1.50$), Top-K ($1 - 50$), Top-P Nucleus ($0.20 - 1.00$), Max Tokens ($20 - 200$).
   - Streaming typewriter generation with celebration confetti.
   - Next-token candidate probability bar chart for any generated token.
   - Colorized token chips inspector.
2. **Loss Curves & Training Telemetry**:
   - High-fidelity interactive SVG charts comparing Training Loss vs. Validation Loss over 1,600 steps.
   - Perplexity reduction trajectory and cosine learning rate curve.
3. **Multi-Head Attention Heatmap Visualizer**:
   - Live interactive Query &times; Key dot-product attention scores across all 4 layers and 4 heads.
   - Mathematical explanation of lower-triangular causal masking ($A_{ij} = 0 \text{ for } j > i$).
4. **Transformer Internals**:
   - Interactive dataflow diagram from embedding to attention to residual MLP to logits.
   - Architectural deep-dive: *Why Scratch Pre-Training != Fine-Tuning*.
5. **Niche Corpus & Tokenizer Explorer**:
   - Searchable view of the training corpus and character frequency map.
6. **Codebase Inspector**:
   - In-app syntax-highlighted source code viewer for `model.py` and `train.py`.

---

## 📂 Project Structure

```
p5/
├── backend/
│   ├── checkpoints/
│   │   ├── best_model.pt             # Best checkpoint state dict
│   │   ├── final_model.pt            # Final trained model
│   │   └── training_history.json     # Complete loss curve telemetry
│   ├── data/
│   │   ├── corpus.txt                # Niche training text (31k chars)
│   │   └── tokenizer.json            # Vocabulary and token mappings (86 tokens)
│   ├── model.py                      # Pure PyTorch Causal Transformer from scratch
│   ├── corpus.py                     # Niche corpus definition
│   ├── tokenizer.py                  # Character-level tokenizer
│   ├── train.py                      # Training loop, cosine LR, validation evaluation
│   ├── server.py                     # FastAPI server (/generate, /attention, /metrics)
│   └── .venv/                        # Python virtual environment
├── frontend/
│   ├── src/
│   │   ├── App.tsx                   # Interactive dashboard component
│   │   ├── index.css                 # Custom Obsidian Cyber design system
│   │   ├── data/                     # Embedded training telemetry & codebase
│   │   └── main.tsx
│   ├── dist/                         # Production build
│   └── package.json
├── run.sh                            # Unified start script
└── README.md
```

## Netlify deployment

Production: https://tinytransformer-yashanand.netlify.app

Use Node.js 22.12 or newer. From the repository root:

```bash
npm ci
npm run build
npm test
npx --package netlify-cli netlify deploy --prod --no-build
```

Netlify serves `frontend/dist` and routes `/api/*` to the TypeScript inference
function. The function runs the original trained weights through ONNX Runtime
on CPU; it does not require the local FastAPI server. Local Vite development
continues to use `http://127.0.0.1:8008`. Set `VITE_API_BASE_URL` at build time
only when using a different backend.

The portable model is checked into `netlify/model/model.onnx`. After retraining,
install `onnx` into the backend Python environment and regenerate it with:

```bash
backend/.venv/bin/python scripts/export_model.py
npm test
```

The export script also regenerates the PyTorch reference outputs used to check
logit and attention parity. Publish only `frontend/dist`; model and corpus assets
are bundled privately with the function. This is a manual deployment; automatic
GitHub deploys have not been configured.
