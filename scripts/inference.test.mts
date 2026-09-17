import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { forward, generate } from '../netlify/functions/_shared/inference.mts';
import api from '../netlify/functions/api.mts';
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
  const full = await generate({...input,prompt:'a'.repeat(140),max_new_tokens:2,temperature:0});
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
