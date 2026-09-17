import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import * as ort from 'onnxruntime-web';

let systemPromise: ReturnType<typeof loadSystem> | undefined;
async function loadSystem() {
  ort.env.wasm.numThreads = 1;
  const [bytes, tokenizer, settings] = await Promise.all([
    readFile(resolve('netlify/model/model.onnx')),
    readFile(resolve('backend/data/tokenizer.json'), 'utf8'),
    readFile(resolve('netlify/model/config.json'), 'utf8'),
  ]);
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
  const { vocab, special_tokens } = JSON.parse(tokenizer) as { vocab: string[]; special_tokens: string[] };
  return { session, vocab, special_tokens, settings: JSON.parse(settings), stoi: new Map(vocab.map((v, i) => [v, i])) };
}
export function getSystem() {
  return systemPromise ??= loadSystem().catch(error => { systemPromise = undefined; throw error; });
}
export async function forward(ids: number[]) {
  const { session, settings } = await getSystem();
  const context = ids.slice(-settings.block_size);
  return session.run({ ids: new ort.Tensor('int64', BigInt64Array.from(context.map(BigInt)), [1, context.length]) });
}
export function softmax(logits: number[]) {
  const max = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - max));
  const sum = exps.reduce((a,b) => a+b, 0);
  return exps.map(x => x / sum);
}
export function display(char: string) { return char === ' ' ? '␣' : char === '\n' ? '↵' : char === '\t' ? '⇥' : char; }
export async function tokenize(text: string) {
  const { stoi } = await getSystem();
  return Array.from(text).map(char => ({ char, display: display(char), id: stoi.get(char) ?? 1 }));
}
export async function generate(input: { prompt: string; max_new_tokens: number; temperature: number; top_k: number; top_p: number; seed?: number }) {
  const start = performance.now();
  const { vocab, special_tokens } = await getSystem();
  const prompt = input.prompt || '=== LOG';
  const ids = (await tokenize(prompt)).map(t => t.id);
  const generated: number[] = [];
  const step_details = [];
  let seed = input.seed;
  const random = () => {
    if (seed === undefined) return Math.random();
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  };
  for (let i = 0; i < input.max_new_tokens; i++) {
    const { logits } = await forward(ids);
    const raw = Array.from(logits.data as Float32Array);
    const probs = softmax(raw);
    const ranked = raw.map((_, id) => id).sort((a,b) => raw[b]-raw[a]);
    let chosen = ranked[0];
    if (input.temperature > 0) {
      let candidates = input.top_k > 0 ? ranked.filter(id => raw[id] >= raw[ranked[Math.min(input.top_k, ranked.length)-1]]) : ranked;
      const scaled = candidates.map(id => raw[id]/input.temperature);
      const distribution = softmax(scaled);
      if (input.top_p > 0 && input.top_p < 1) {
        let cumulative = 0, count = 0;
        do { cumulative += distribution[count++]; } while (cumulative < input.top_p && count < candidates.length);
        candidates = candidates.slice(0, count);
      }
      const sampling = softmax(candidates.map(id => raw[id]/input.temperature));
      let draw = random();
      chosen = candidates[candidates.length-1];
      for (let j = 0; j < candidates.length; j++) { draw -= sampling[j]; if (draw <= 0) { chosen = candidates[j]; break; } }
    }
    ids.push(chosen); generated.push(chosen);
    step_details.push({ chosen_id: chosen, chosen_char: vocab[chosen], chosen_prob: +probs[chosen].toFixed(4), top_candidates: ranked.slice(0,5).map(id => ({id, char:vocab[id], display:display(vocab[id]), prob:+probs[id].toFixed(4)})) });
    if (chosen === 3) break;
  }
  const decode = (tokens: number[]) => tokens.map(id => special_tokens.includes(vocab[id]) ? '' : vocab[id]).join('');
  const latency_ms = +(performance.now()-start).toFixed(2);
  return { prompt, generated_text:decode(generated), full_text:decode(ids), tokens_generated:generated.length, latency_ms, tokens_per_sec:+(generated.length/Math.max(latency_ms/1000,0.001)).toFixed(1), step_details };
}
