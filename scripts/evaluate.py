"""Evaluate any checkpoint on the fixed test split; optional baseline comparison."""
import argparse
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from train import checkpoint_eval
from model import TinyTransformerLM,TinyTransformerConfig
from tokenizer import CharTokenizer
p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,default=ROOT/'backend/checkpoints/best_model.pt')
p.add_argument('--tokenizer',type=Path,default=ROOT/'backend/data/tokenizer.json')
p.add_argument('--split',choices=['test','canonical'],default='test')
p.add_argument('--output',type=Path,default=ROOT/'docs/evaluation.json')
args=p.parse_args();torch.set_num_threads(4)
if args.split == 'canonical':
    from curriculum import LESSONS
    docs=[{'question':q,'answer':a,'text':f'User: {q}\nAssistant: {a}'} for _,q,a in LESSONS]
else:
    docs=json.loads((ROOT/'backend/data/test.json').read_text())
report=checkpoint_eval(args.checkpoint,args.tokenizer,docs,'cpu')
ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
model=TinyTransformerLM(TinyTransformerConfig(**ckpt['config'])).eval();model.load_state_dict(ckpt['model_state_dict'])
tok=CharTokenizer.load(str(args.tokenizer));samples=[]
for doc in docs:
    prompt=f"User: {doc['question']}\nAssistant: "
    x=torch.tensor([tok.encode(prompt)])
    y,_=model.generate(x,256,temperature=0,eos_token_id=tok.eos_token_id)
    samples.append({'prompt':prompt,'expected':doc['answer'],'generated':tok.decode(y[0,len(x[0]):].tolist())})
report['exact_matches']=sum(s['generated']==s['expected'] for s in samples)
report['total_prompts']=len(samples)
report['exact_match_rate']=report['exact_matches']/len(samples)
report['samples']=samples
report['scope']='Canonical training questions; a memorization check, not generalization.' if args.split=='canonical' else 'Held-out question phrasings of seen facts. Teacher-forced answer loss does not measure free-generation factual accuracy.'
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='samples'}))
