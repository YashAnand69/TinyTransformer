import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { forward, generate } from '../netlify/functions/_shared/inference.ts';
import api from '../netlify/functions/api.ts';
import vercel from '../api/[route].ts';
test('Vercel fetch adapter loads the model and generates using packaged artifacts', async () => {
  const health = await vercel.fetch(new Request('https://test/api/health'));
  assert.equal(health.status, 200);
  assert.equal((await health.json()).model_loaded, true);
  const reference = JSON.parse(await readFile('scripts/reference.json', 'utf8'));
  const response = await vercel.fetch(new Request('https://test/api/generate', {
    method: 'POST',
    body: JSON.stringify({...reference.greedy, temperature: 0, top_k: 0, top_p: 1}),
  }));
  assert.equal(response.status, 200);
  assert.equal((await response.json()).generated_text, reference.greedy.text);
});
test('ONNX logits and attention match the trained PyTorch checkpoint', async () => {
  const reference = JSON.parse(await readFile('scripts/reference.json','utf8'));
  const actual = await forward(reference.ids);
  const expected = [reference.logits, reference.attention.flat(3)];
  for (const [i,name] of ['logits','attention'].entries()) {
    const values = Array.from(actual[name].data as Float32Array);
    assert.equal(values.length,expected[i].length);
    assert.ok(Math.max(...values.map((v,j)=>Math.abs(v-expected[i][j]))) < 0.0001);
  }
});
test('generation supports full context, sampling and reproducible seeds', async () => {
  const input = { prompt:'User: Who are you?\nAssistant: ',max_new_tokens:4,temperature:0.35,top_k:10,top_p:0.9,seed:42 };
  const a = await generate(input), b = await generate(input);
  assert.equal(a.generated_text,b.generated_text);
  assert.equal(a.step_details.length,4);
  const full = await generate({...input,prompt:'a'.repeat(300),max_new_tokens:2,temperature:0});
  assert.equal(full.tokens_generated,2);
});
test('API validates input and returns real causal attention including empty input', async () => {
  const call = (path:string,body:unknown) => api(new Request('https://test/api/'+path,{method:'POST',body:JSON.stringify(body)}));
  assert.equal((await call('generate',{max_new_tokens:100000})).status,422);
  for (const text of ['', 'hello']) {
    const res = await call('attention',{text}); assert.equal(res.status,200);
    const data = await res.json(); assert.equal(data.tokens.length,data.token_count);
    for (const layer of data.weights) for (const head of layer) for (const [i,row] of head.entries()) {
      assert.ok(Math.abs(row.reduce((a:number,b:number)=>a+b,0)-1)<0.0001);
      assert.ok(row.slice(i+1).every((v:number)=>v===0));
    }
  }
});
test('export supports boundary lengths and matches PyTorch greedy decoding', async () => {
  const reference = JSON.parse(await readFile('scripts/reference.json','utf8'));
  for (const item of reference.boundary_cases) {
    const actual = await forward(item.ids);
    const differences = Array.from(actual.logits.data as Float32Array, (v,i)=>Math.abs(v-item.logits[i]));
    assert.ok(Math.max(...differences)<0.0001);
  }
  const result = await generate({...reference.greedy,temperature:0,top_k:0,top_p:1});
  assert.equal(result.generated_text,reference.greedy.text);
});
test('dashboard and portable model describe the same checkpoint', async () => {
  const report = JSON.parse(await readFile('frontend/src/data/training_history.json','utf8'));
  const config = JSON.parse(await readFile('netlify/model/config.json','utf8'));
  const tokenizer = JSON.parse(await readFile('backend/data/tokenizer.json','utf8'));
  assert.equal(report.summary.parameters,config.parameters);
  assert.equal(report.summary.block_size,config.block_size);
  assert.equal(tokenizer.vocab.length,config.vocab_size);
  assert.deepEqual(report,JSON.parse(await readFile('backend/checkpoints/training_history.json','utf8')));
});
