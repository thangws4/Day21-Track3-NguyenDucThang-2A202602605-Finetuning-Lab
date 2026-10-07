import sys, json; sys.path.insert(0, 'src')
from labkit import generate
from labkit.config import get_tier
from peft import PeftModel
T = get_tier('T4')
rows = [json.loads(l) for l in open('data/eval_target.jsonl', encoding='utf-8') if l.strip()]
model, tok = generate.load_base(T)
model = PeftModel.from_pretrained(model, 'adapters/correct', adapter_name='correct')
avail = ['correct']
for k in ('attn_only', 'qlora'):
    model.load_adapter('adapters/' + k, adapter_name=k); avail.append(k)
print('adapter dang nap:', avail)
print('ticket:', rows[0]['input'])
for n in avail:
    model.set_adapter(n); out, _ = generate.generate_batch(model, tok, [rows[0]['input']], system=generate.NAIVE_PROMPT); print('[' + n + '] ->', out[0][:160])
