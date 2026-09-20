import type { Config } from '@netlify/functions';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { getSystem, forward, generate, tokenize } from './_shared/inference.mts';

export default async (req: Request) => {
  const route = new URL(req.url).pathname.split('/').pop();
  const json = (data: unknown, status = 200) => Response.json(data, { status, headers: { 'Cache-Control': 'no-store' } });
  const gets = ['health', 'info', 'metrics', 'corpus'];
  const posts = ['generate', 'attention', 'tokenize'];
  if (![...gets, ...posts].includes(route || '')) return json({detail:'Not found'},404);
  if (req.method !== (gets.includes(route!) ? 'GET' : 'POST')) return json({detail:'Method not allowed'},405);
  try {
    let input: Record<string, unknown> = {};
    if (req.method === 'POST') {
      const body = await req.text();
      if (body.length > 16384) return json({detail:'Request too large'},413);
      try { input = JSON.parse(body); } catch { return json({detail:'Invalid JSON'},400); }
      if (!input || typeof input !== 'object' || Array.isArray(input)) return json({detail:'Expected an object'},422);
    }
    const string = (key: string, fallback: string) => {
      const value = input[key] ?? fallback;
      if (typeof value !== 'string' || value.length > 4096) throw new RangeError(`Invalid ${key}`);
      return value;
    };
    const number = (key: string, fallback: number, min: number, max: number, integer = false) => {
      const value = input[key] ?? fallback;
      if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max || (integer && !Number.isInteger(value))) throw new RangeError(`Invalid ${key}`);
      return value;
    };
    if (route === 'generate') return json(await generate({prompt:string('prompt','=== LOG ENTRY'), max_new_tokens:number('max_new_tokens',80,1,256,true), temperature:number('temperature',0.8,0,2), top_k:number('top_k',20,0,100,true), top_p:number('top_p',0.9,0,1), seed:input.seed == null ? undefined : number('seed',0,0,4294967295,true)}));
    if (route === 'tokenize') {
      const text = string('text',''); const tokens = await tokenize(text);
      return json({text,token_count:tokens.length,tokens,ids:tokens.map(t=>t.id)});
    }
    if (route === 'metrics') return json(JSON.parse(await readFile(resolve('backend/checkpoints/training_history.json'),'utf8')));
    if (route === 'corpus') {
      const text = await readFile(resolve('backend/data/corpus.txt'),'utf8');
      return json({total_characters:Array.from(text).length,total_words:text.trim().split(/\s+/).length,sample_excerpts:text.split('=== ').filter(s=>s.trim()).slice(0,8).map(s=>'=== '+s.trim())});
    }
    const { settings, vocab } = await getSystem();
    if (route === 'health') return json({status:'online',device:'cpu',model_loaded:true,tokenizer_loaded:true,params:settings.parameters});
    if (route === 'info') return json({...settings, model_name:'TinyTransformer-Lab-v2',architecture:'Causal Decoder-Only Transformer (GPT Architecture)',device:'cpu',head_dim:settings.d_model/settings.n_head});
    const tokens = await tokenize(Array.from(string('text','=== LOG ENTRY: The silicon lattice')).slice(0,32).join(''));
    const ids = tokens.length ? tokens.map(t=>t.id) : [2];
    const { attention } = await forward(ids);
    const values = attention.data as Float32Array; let offset=0;
    const weights = Array.from({length:settings.n_layer},()=>Array.from({length:settings.n_head},()=>Array.from({length:ids.length},()=>Array.from({length:ids.length},()=>values[offset++]))));
    return json({text:tokens.map(t=>t.char).join(''),token_count:ids.length,tokens:tokens.length?tokens.map(t=>t.display):[vocab[2]],token_ids:ids,n_layer:settings.n_layer,n_head:settings.n_head,num_layers:settings.n_layer,num_heads:settings.n_head,weights,attention_weights:weights});
  } catch (error) {
    if (error instanceof RangeError) return json({detail:error.message},422);
    console.error('Inference failed',error);
    return json({detail:'Model inference unavailable. Please retry.'},503);
  }
};
export const config: Config = { path: '/api/*' };
