import torch
from transformers import BertModel, BertTokenizer, BertForMaskedLM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained_path = './tmp/'
# pretrained_path = '../model_hub/chinese-bert-wwm-ext/'
tokenizer = BertTokenizer.from_pretrained(pretrained_path)

bertModel = BertForMaskedLM.from_pretrained(pretrained_path)
bertModel = bertModel.to(device)
bertModel.eval()


text = '中国保险资产管理业协会积极推进保险私[MASK]基金登记制改革落地实施'
input = tokenizer.encode_plus(text=text,
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors='pt', )
tokens = tokenizer.convert_ids_to_tokens(input['input_ids'][0])
input_token = input['input_ids'].to(device)
segment_ids = input['token_type_ids'].to(device)
attention_mask = input['attention_mask'].to(device)
for i,token in enumerate(tokens):
  if token == '[MASK]':
    with torch.no_grad():
      outputs = bertModel(input_token, segment_ids, attention_mask)
      outputs = outputs[0] # torch.Size([1, 512, 21128])
      outputs = outputs[0][i, :]
      outputs = torch.argmax(outputs, dim=-1)
      words = tokenizer.convert_ids_to_tokens(outputs.item())
      print(words)