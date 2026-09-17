"""Original educational examples, versioned with the project.
Splits hold out question phrasings, not facts; this measures narrow paraphrase
robustness, not general knowledge or unseen-topic reasoning.
"""
import hashlib
import json
from pathlib import Path

# Topic, canonical question, concise factual answer.
LESSONS = [
('identity', 'Who are you?', 'I am TinyTransformer, a small language model trained from scratch. I explain machine learning concepts. I can make mistakes and I am not a general-purpose assistant.'),
('architecture', 'What is your architecture?', 'I use four causal transformer layers, four attention heads, 192 embedding dimensions, and a 256-character context window. My weights are learned from random initialization.'),
('attention', 'What is attention?', 'Attention computes similarities between queries and keys, then uses normalized scores to combine value vectors. The result lets each token use relevant earlier context.'),
('generation', 'How do you generate text?', 'I predict the next character from the current context, append it, and repeat. This is autoregressive generation. I stop at an end token or the generation limit.'),
('temperature', 'What is sampling temperature?', 'Temperature divides logits before softmax. Lower values concentrate probability on likely tokens. Higher values spread probability more widely. Zero selects the most likely token.'),
('top-k', 'What is top-k sampling?', 'Top-k sampling keeps only the k most likely tokens before sampling. A small k reduces variety. Setting k to zero disables this filter.'),
('top-p', 'What is top-p sampling?', 'Top-p sampling retains the smallest ranked set whose cumulative probability reaches p. It adapts the number of candidates to the probability distribution.'),
('causal mask', 'What is the causal mask?', 'The causal mask prevents a position from attending to later positions. Future scores are set to negative infinity before softmax, so their attention probability is zero.'),
('training from scratch', 'Why train from scratch instead of fine-tuning?', 'Training from scratch learns all weights from random initialization. Fine-tuning adapts a pretrained model. Starting from scratch is useful for studying how learning works, but requires data and compute.'),
('backpropagation', 'What is backpropagation?', 'Backpropagation applies the chain rule to compute gradients of the loss with respect to parameters. An optimizer uses those gradients to update the weights. It does not guarantee a global minimum.'),
('context window', 'What is a context window?', 'The context window is the maximum number of input tokens the model can use at once. Here a token is a character. Older context is cropped when it exceeds the window.'),
('tokenization', 'What is tokenization?', 'Tokenization converts text into integer IDs. This model uses character tokens, including spaces and punctuation. Unknown characters map to a special unknown token.'),
('embeddings', 'What are embeddings?', 'An embedding table maps each discrete token ID to a learned vector. These vectors let the transformer process tokens with continuous arithmetic.'),
('position embeddings', 'What are position embeddings?', 'Position embeddings add information about a token position to its token vector. This model learns a separate vector for each position; it does not use sinusoidal position encodings.'),
('queries', 'What is a query vector?', 'A query is a learned projection of the current token representation. Its dot products with keys determine which earlier positions receive attention.'),
('keys', 'What is a key vector?', 'A key is a learned projection used for matching against queries. A high query-key score can give that position more weight after normalization.'),
('values', 'What is a value vector?', 'A value is a learned projection containing information to aggregate. Attention scores weight the values, and their weighted sum becomes the attention output.'),
('multiple heads', 'Why use multiple attention heads?', 'Each head projects tokens into a different subspace and computes its own attention weights. The head outputs are concatenated and projected back to the model dimension.'),
('scaled dot products', 'Why scale attention scores?', 'Attention divides query-key dot products by the square root of the head dimension. This helps keep score magnitudes and softmax gradients well behaved.'),
('softmax', 'What is softmax?', 'Softmax turns logits into positive probabilities that sum to one. Subtracting the largest logit before exponentiation improves numerical stability without changing the probabilities.'),
('logits', 'What are logits?', 'Logits are unnormalized scores for each possible next token. Softmax converts them into probabilities. They are scores, not probabilities themselves.'),
('cross entropy', 'What is cross-entropy loss?', 'Cross entropy is the negative log probability assigned to the correct next token. Lower loss means the model assigns more probability to observed targets.'),
('perplexity', 'What is perplexity?', 'Perplexity is the exponential of average cross-entropy loss. Lower is better on the same dataset and tokenizer. Character and word perplexities are not directly comparable.'),
('validation', 'What is a validation set?', 'Validation examples are excluded from gradient updates and used to select a checkpoint. A separate test set is used for the final evaluation.'),
('data leakage', 'What is data leakage?', 'Data leakage occurs when evaluation information enters training. Duplicated documents across splits can make validation scores look much better than real generalization.'),
('overfitting', 'What is overfitting?', 'Overfitting means learning training-specific patterns that do not transfer to new examples. A widening gap between training and validation loss can be a warning sign.'),
('dropout', 'What is dropout?', 'Dropout randomly zeros activations during training as regularization. It is disabled during evaluation and generation so the model uses its full representation.'),
('layer normalization', 'What is layer normalization?', 'Layer normalization normalizes features within each token representation using their mean and variance. It helps stabilize optimization in deep networks.'),
('residual connections', 'What are residual connections?', 'A residual connection adds a module input to its output. This creates a direct path for information and gradients through successive transformer layers.'),
('GELU', 'What is GELU?', 'GELU is a smooth nonlinear activation used in the feed-forward network. Nonlinearity allows the network to learn more than a single linear transformation.'),
('feed-forward network', 'What is the feed-forward network?', 'The feed-forward network applies a linear expansion, GELU activation, and linear projection independently to each token. Here its hidden width is four times the embedding width.'),
('AdamW', 'What is AdamW?', 'AdamW is an adaptive optimizer with decoupled weight decay. It tracks moving averages of gradients and squared gradients to scale parameter updates.'),
('learning rate', 'What is the learning rate?', 'The learning rate controls the size of optimizer updates. Too large can destabilize training; too small can slow learning. Warmup and decay help manage it over a run.'),
('warmup', 'Why use learning rate warmup?', 'Warmup gradually increases the learning rate near the start of training. This can reduce unstable updates while the model and optimizer statistics are still settling.'),
('cosine decay', 'What is cosine learning rate decay?', 'Cosine decay smoothly lowers the learning rate from a peak toward a minimum. Smaller late updates help refine the learned weights.'),
('gradient clipping', 'What is gradient clipping?', 'Gradient clipping limits the gradient norm before an optimizer update. It can prevent occasional large gradients from causing excessively large parameter changes.'),
('weight decay', 'What is weight decay?', 'Weight decay discourages large parameter values. With AdamW it is applied separately from the gradient-based adaptive update.'),
('weight tying', 'What is weight tying?', 'Weight tying shares the token embedding matrix with the output projection. It reduces parameter count and connects input token representations to output token scores.'),
('batch size', 'What is batch size?', 'Batch size is the number of examples used for one gradient estimate. Larger batches use more memory and change the noise in the gradient estimate.'),
('checkpoint', 'What is a checkpoint?', 'A checkpoint saves model weights and configuration. A resumable training checkpoint also saves optimizer state, the step, and random generator states.'),
('random seed', 'Why use a random seed?', 'A seed makes random initialization and sampling repeatable within a controlled environment. Exact results can still differ across hardware or software versions.'),
('inference', 'What is inference?', 'Inference runs a trained model to produce predictions without updating its weights. Dropout is disabled and gradients are not needed.'),
('ONNX', 'What is ONNX?', 'ONNX is a portable representation of a model computation graph and weights. This project exports its trained transformer so the website can run CPU inference without PyTorch.'),
('CPU and GPU', 'How do CPUs and GPUs differ?', 'CPUs handle diverse sequential work, while GPUs can execute many arithmetic operations in parallel. Model size, batching, and transfer overhead determine which is faster.'),
('MPS', 'What is MPS?', 'MPS is the PyTorch backend for accelerated tensor operations on supported Apple GPUs. This project can also train on CUDA or CPU.'),
('hallucination', 'What is a hallucination?', 'A hallucination is generated content that is incorrect or unsupported. A language model predicts text; high token probability does not guarantee factual accuracy.'),
('limitations', 'What are your limitations?', 'I am a small educational model trained on a narrow synthetic curriculum. I can repeat learned explanations but may fail on unfamiliar questions. I cannot browse the web or verify facts.'),
('reasoning', 'Can you solve any problem?', 'No. My training is limited to short educational examples. Fluent output is not proof of reasoning or correctness. Use the model to explore transformers, not for important decisions.'),
('privacy', 'Do you remember my messages?', 'A generation request supplies a temporary context. The model weights do not change during a conversation. Hosting infrastructure may still keep request logs.'),
('end token', 'What is an end token?', 'An end token marks the end of a training example or response. During generation it lets the model stop before reaching the maximum token budget.'),
('greedy decoding', 'What is greedy decoding?', 'Greedy decoding always selects the token with the highest score. It is deterministic for fixed weights and context, but does not guarantee the best whole sequence.'),
('attention interpretation', 'Do attention weights explain everything?', 'No. Attention weights show how one module mixes positions. Residual streams and feed-forward layers also affect predictions, so a heatmap is not a complete causal explanation.'),
('training data', 'What were you trained on?', 'I was trained on a small original curriculum of machine learning questions and answers. Evaluation holds out question phrasings, while the underlying topics overlap with training.'),
('silicon transmutation', 'What is silicon transmutation?', 'Silicon transmutation is a fictional metaphor from this project: turning electrical signals into structured computation. It is not a scientific claim about consciousness.'),
('fine-tuning', 'What is fine-tuning?', 'Fine-tuning continues training an existing pretrained model on new examples. It can adapt behavior efficiently, but still requires careful data selection and evaluation.'),
('generalization', 'What is generalization?', 'Generalization is performance on examples outside the training set. Testing a paraphrase of a known topic is easier than answering an entirely unseen topic.'),
('training mode', 'Why switch between train and eval mode?', 'Training mode enables operations such as dropout. Evaluation mode disables them. Forgetting to restore training mode after generating a sample can silently change a training run.'),
('parameter count', 'Does a larger model always work better?', 'No. More parameters increase capacity and compute cost, but data quality and evaluation matter too. A larger model can overfit a small dataset.'),
]

def build_splits():
    splits = {'train': [], 'validation': [], 'test': []}
    for topic, question, answer in LESSONS:
        questions = [question, f'Explain {topic}.', f'Tell me about {topic}.', f'Please describe {topic}.', f'I want to understand {topic}.', f'Give a short explanation of {topic}.', f'In simple terms, explain {topic}.', f'Can you explain {topic} briefly?']
        for i, q in enumerate(questions):
            split = 'train' if i < 6 else 'validation' if i == 6 else 'test'
            text = f'User: {q}\nAssistant: {answer}'
            splits[split].append({'topic':topic, 'question':q, 'answer':answer, 'text':text, 'id':hashlib.sha256(text.encode()).hexdigest()[:16]})
    return splits

def save_dataset(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    splits = build_splits()
    for split, docs in splits.items():
        (directory / f'{split}.json').write_text(json.dumps(docs, indent=2)+'\n')
    corpus = '\n\n'.join(d['text'] for d in splits['train'])
    (directory/'corpus.txt').write_text(corpus)
    manifest = {'version':2, 'source':'Original synthetic educational curriculum maintained in curriculum.py', 'evaluation_scope':'Held-out question phrasings of known topics; not unseen-topic generalization.', 'topics':len(LESSONS), 'counts':{k:len(v) for k,v in splits.items()}, 'sha256':{k:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest() for k,v in splits.items()}}
    (directory/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return splits

if __name__ == '__main__':
    save_dataset(Path(__file__).parent/'data')
