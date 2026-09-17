import unittest
import torch
from model import TinyTransformerLM, TinyTransformerConfig
from curriculum import build_splits
from tokenizer import CharTokenizer
from train import examples, evaluate, get_lr

class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model=TinyTransformerLM(TinyTransformerConfig(block_size=16,vocab_size=20,d_model=16,n_head=4,n_layer=2,dropout=.2)).eval()
    def test_causal_prefix_and_attention_paths_match(self):
        ids=torch.randint(0,20,(1,12))
        with torch.no_grad():
            fast,_,_=self.model(ids)
            explicit,_,attn=self.model(ids,return_attention=True)
            prefix,_,_=self.model(ids[:,:6])
        torch.testing.assert_close(fast,explicit,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(fast[:,:6],prefix,atol=1e-6,rtol=1e-5)
        for layer in attn:
            self.assertEqual(torch.triu(layer,diagonal=1).count_nonzero(),0)
            torch.testing.assert_close(layer.sum(-1),torch.ones_like(layer.sum(-1)))
    def test_evaluation_restores_training_mode(self):
        self.model.train()
        evaluate(self.model,(torch.ones(2,16,dtype=torch.long),torch.ones(2,16,dtype=torch.long)),'cpu')
        self.assertTrue(self.model.training)
    def test_dataset_has_no_identical_documents_across_splits(self):
        splits=build_splits(); ids=[{d['id'] for d in docs} for docs in splits.values()]
        for i,a in enumerate(ids):
            for b in ids[i+1:]:self.assertFalse(a&b)
    def test_supervision_covers_answer_once_and_ignores_prompt_padding(self):
        tok=CharTokenizer(list('User: Q?\nAssistant: abcdefghijklmnopqrstuvwxyz'))
        doc={'question':'Q?','text':'User: Q?\nAssistant: abcdefghijklmnopqrstuvwxyz'}
        x,y=examples([doc],tok,16)
        actual=y[y!=-1].tolist()
        self.assertEqual(actual,tok.encode('abcdefghijklmnopqrstuvwxyz')+[tok.eos_token_id])
        self.assertEqual(x.shape,y.shape)
    def test_learning_rate_endpoints(self):
        self.assertAlmostEqual(get_lr(100,10,100,.001,.0001),.0001)
        self.assertAlmostEqual(get_lr(10,10,100,.001,.0001),.001)

if __name__=='__main__':unittest.main()
