"""Reproducible document-based training with explicit validation and test splits."""
import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
import torch
from model import TinyTransformerLM, TinyTransformerConfig
from tokenizer import CharTokenizer
from curriculum import save_dataset

ROOT = Path(__file__).resolve().parent

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    progress = min(1.0, max(0.0, (step-warmup_steps)/max(1,max_steps-warmup_steps)))
    return min_lr + .5*(max_lr-min_lr)*(1+math.cos(math.pi*progress))

def examples(docs, tokenizer, block_size):
    """Window each document independently; mask prompts and padding in the loss."""
    xs, ys = [], []
    for doc in docs:
        prefix = f"User: {doc['question']}\nAssistant: "
        ids = tokenizer.encode(doc['text']) + [tokenizer.eos_token_id]
        # Non-overlapping targets, with preceding context for each continuation.
        for target_start in range(len(prefix), len(ids), block_size//2):
            target_end = min(target_start+block_size//2, len(ids))
            start = max(0, target_end-block_size-1)
            x = ids[start:target_end-1]
            y = ids[start+1:target_end]
            y = [-1 if start+1+i < target_start else tid for i,tid in enumerate(y)]
            xs.append(x+[tokenizer.pad_token_id]*(block_size-len(x)))
            ys.append(y+[-1]*(block_size-len(y)))
    return torch.tensor(xs), torch.tensor(ys)

@torch.no_grad()
def evaluate(model, data, device, batch_size=16):
    was_training = model.training
    model.eval()
    total, count = 0., 0
    for start in range(0,len(data[0]),batch_size):
        x,y = (t[start:start+batch_size].to(device) for t in data)
        _,loss,_ = model(x,targets=y)
        n = int((y != -1).sum())
        total += loss.item()*n; count += n
    model.train(was_training)
    return total/max(count,1)

def checkpoint_eval(path, tok_path, docs, device):
    ckpt = torch.load(path,map_location='cpu',weights_only=True)
    model = TinyTransformerLM(TinyTransformerConfig(**ckpt['config'])).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    tok = CharTokenizer.load(str(tok_path))
    loss = evaluate(model,examples(docs,tok,model.config.block_size),device)
    return {'loss':loss,'perplexity':math.exp(min(loss,20)),'parameters':model.get_num_params(),'block_size':model.config.block_size}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=2400);p.add_argument('--batch-size',type=int,default=24)
    p.add_argument('--width',type=int,default=192);p.add_argument('--context',type=int,default=256)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--eval-every',type=int,default=100)
    p.add_argument('--device',default='auto');p.add_argument('--output',type=Path,default=ROOT/'runs/v2')
    args=p.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.context < 8 or args.width%4: p.error('Invalid training dimensions')
    random.seed(args.seed);torch.manual_seed(args.seed)
    device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if args.device=='auto' else args.device
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=True)
    splits=save_dataset(ROOT/'data')
    # Fixed printable ASCII vocabulary avoids learning the evaluation vocabulary.
    tok=CharTokenizer([chr(i) for i in range(32,127)]+['\n','\t'])
    tok.save(str(args.output/'tokenizer.json'))
    cfg=TinyTransformerConfig(block_size=args.context,vocab_size=tok.vocab_size,n_layer=4,n_head=4,d_model=args.width,dropout=.1,bias=False)
    model=TinyTransformerLM(cfg).to(device)
    datasets={key:examples(docs,tok,cfg.block_size) for key,docs in splits.items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=.05,betas=(.9,.95))
    batch_rng=torch.Generator().manual_seed(args.seed)
    history=[]; samples=[]; best=float('inf');start=time.time(); supervised_tokens=0
    prompt='User: What is attention?\nAssistant: '
    ctx=torch.tensor([tok.encode(prompt)],device=device)
    print(f'{model.get_num_params():,} params; {device}; {len(splits["train"])} train documents; seed {args.seed}',flush=True)
    for step in range(args.steps+1):
        lr=get_lr(step,min(100,args.steps//10),args.steps,8e-4,8e-5)
        if step:
            model.train()  # Samples and evaluation must never silently disable dropout.
            idx=torch.randint(len(datasets['train'][0]),(args.batch_size,),generator=batch_rng)
            x,y=(t[idx].to(device) for t in datasets['train'])
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            _,loss,_=model(x,targets=y)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            supervised_tokens += int((y!=-1).sum())
        if step%args.eval_every==0 or step==args.steps:
            train_loss=evaluate(model,tuple(t[:96] for t in datasets['train']),device)
            val_loss=evaluate(model,datasets['validation'],device)
            point={'step':step,'train_loss':round(train_loss,5),'val_loss':round(val_loss,5),'perplexity':round(math.exp(min(val_loss,20)),3),'lr':lr,'elapsed_sec':round(time.time()-start,2)}
            history.append(point); print(json.dumps(point),flush=True)
            if val_loss<best:
                best=val_loss
                torch.save({'step':step,'model_state_dict':model.state_dict(),'config':asdict(cfg),'best_val_loss':best,'seed':args.seed},args.output/'best_model.pt')
            (args.output/'progress.json').write_text(json.dumps(history,indent=2))
        if step in {0,args.steps//4,args.steps//2,3*args.steps//4,args.steps}:
            out,_=model.generate(ctx,120,temperature=0,eos_token_id=tok.eos_token_id)
            samples.append({'step':step,'sample':tok.decode(out[0].tolist()),'note':'Measured greedy generation from this training step'})
    checkpoint=torch.load(args.output/'best_model.pt',map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss=evaluate(model,datasets['test'],device)
    baseline_path=ROOT/'runs/baseline/model.pt'
    baseline=checkpoint_eval(baseline_path,ROOT/'runs/baseline/tokenizer.json',splits['test'],device) if baseline_path.exists() else None
    summary={'model_name':'TinyTransformer-Lab-v2','parameters':model.get_num_params(),'vocab_size':tok.vocab_size,'block_size':cfg.block_size,'layers':cfg.n_layer,'heads':cfg.n_head,'d_model':cfg.d_model,'total_steps':args.steps,'best_step':checkpoint['step'],'batch_size':args.batch_size,'tokens_per_step':args.batch_size*cfg.block_size,'total_tokens_trained':args.steps*args.batch_size*cfg.block_size,'supervised_tokens':supervised_tokens,'training_duration_sec':round(time.time()-start,2),'device':device,'seed':args.seed,'best_val_loss':round(best,5),'final_train_loss':history[-1]['train_loss'],'final_val_loss':history[-1]['val_loss'],'final_perplexity':history[-1]['perplexity'],'test_loss':test_loss,'test_perplexity':math.exp(min(test_loss,20))}
    report={'summary':summary,'history':history,'sample_generations':samples,'evaluation':{'scope':'Answer-token loss on held-out phrasings of known topics. Not a general intelligence benchmark. Different context sizes are part of the comparison.','baseline':baseline,'candidate':{'loss':test_loss,'perplexity':math.exp(min(test_loss,20))}}}
    (args.output/'training_history.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['evaluation']),flush=True)
    torch.save({'model_state_dict':model.state_dict(),'config':asdict(cfg),'step':checkpoint['step']},args.output/'final_model.pt')

if __name__=='__main__':main()
