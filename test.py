import torch
from transformers import Trainer, TrainingArguments, AutoModelForSeq2SeqLM, BartphoTokenizer, AutoTokenizer

path_model = 'vinai/bartpho-word'
tokenizer = AutoTokenizer.from_pretrained(path_model)
model = AutoModelForSeq2SeqLM.from_pretrained("./kaggle/working/checkpoint/checkpoint-7500")
model.to('cuda')

sentence = "Chất xo hoa tan giúp lam cham qua trinh tiêu hoa va hap thu glucose (duong) vao mau, ho tro kiem soat duong huyet."
print(sentence)
encoding = tokenizer(sentence, return_tensors="pt")
input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")
outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    max_length=1024,
)
for output in outputs:
    line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(line)