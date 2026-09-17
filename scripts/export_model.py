"""Export the trusted local checkpoint for portable Netlify CPU inference.
Run: backend/.venv/bin/python scripts/export_model.py (requires onnx).
"""
import sys, json
from pathlib import Path
import torch
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'backend'))
from model import TinyTransformerLM, TinyTransformerConfig
from tokenizer import CharTokenizer
checkpoint = torch.load(root / 'backend/checkpoints/best_model.pt', map_location='cpu', weights_only=True)
model = TinyTransformerLM(TinyTransformerConfig(**checkpoint['config'])).eval()
model.load_state_dict(checkpoint['model_state_dict'])
class Export(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model
    def forward(self, ids):
        logits, _, attention = self.model(ids, return_attention=True)
        return logits[:, -1, :], torch.stack(attention)[:, 0]
tok = CharTokenizer.load(str(root / 'backend/data/tokenizer.json'))
ids = torch.tensor([tok.encode('User: Who are you?')], dtype=torch.long)
torch.onnx.export(Export().eval(), (ids,), str(root / 'netlify/model/model.onnx'), input_names=['ids'], output_names=['logits', 'attention'], dynamic_axes={'ids':{1:'sequence'}, 'attention':{2:'sequence',3:'sequence'}}, opset_version=17, dynamo=False)
(root / 'netlify/model/config.json').write_text(json.dumps({**checkpoint['config'], 'parameters':model.get_num_params()}))
model.eval()
with torch.no_grad():
    logits, attention = Export().eval()(ids)
(root / 'scripts/reference.json').write_text(json.dumps({'ids':ids[0].tolist(), 'logits':logits[0].tolist(), 'attention':attention.tolist()}))
print('Exported trained weights and PyTorch reference outputs.')
