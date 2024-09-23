from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
from tqdm import tqdm

import pandas as pd

prompt = """
Analyze the given news item to determine its authenticity and give your choice:

News Content: \
{news_content}

Evidences: \
{evidences}

Verdict: Is the news item true or false?

[Options]:

True
False

Answer:

"""

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser("LLM inference test")
parser.add_argument("--path", type=str, default="/root/lmy/ECFEND/formatted_data/declare/PolitiFact/mapped_data/5fold/")
parser.add_argument("--file_name", type=str, default="test_0.tsv")
parser.add_argument("--LLM_type", type=str, default="llama3-13b")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--dataset_name", type=str, default="PolitiFact")
parser.add_argument("--empty_span", type=str, default="")

path = parser.parse_args().path
file_name = parser.parse_args().file_name
test_datas = pd.read_csv(path + file_name, sep='\t')
llm_type = parser.parse_args().LLM_type
dataset_name = parser.parse_args().dataset_name

if(llm_type == "llama3-13b"):
    model_id = "/root/lmy/llama/llama3/"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='cuda:0', torch_dtype=torch.float16)
else:
    model_id = "/root/lmy/llama/llama2-7b/"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='cuda:0', torch_dtype=torch.float16)

result_datas = pd.DataFrame(columns=['id', 'claim_text', 'label', 'prediction', 'output'], index=[])

left_ids = test_datas['id_left'].unique()

pbar = tqdm(left_ids)
acc = 0

for (i, left_id) in enumerate(pbar):
    label = str(test_datas[test_datas['id_left'] == left_id]['cred_label'].values[0])

    claim_text = str(test_datas[test_datas['id_left'] == left_id]['claim_text'].values[0])
    
    evidences = (test_datas[test_datas['id_left'] == left_id]['evidence'].values.tolist())

    news_content = claim_text
    evidences = "\n".join(evidences)
    
    ev_inputs = tokenizer(evidences, return_tensors="pt", max_length=2048, truncation=True).to("cuda:0")
    evidences = tokenizer.decode(ev_inputs['input_ids'][0], skip_special_tokens=True, clean_up_tokenization_spaces=True, pad_token_id=tokenizer.eos_token_id)
    
    input_data = prompt.format(news_content=news_content, evidences=evidences)
    model_input = tokenizer(input_data, return_tensors="pt").to("cuda:0")
    predicted_label = -1
    while predicted_label == -1:
        with torch.no_grad():
            generated_text = model.generate(**model_input, max_new_tokens=100)[0]
        output = tokenizer.decode(generated_text, skip_special_tokens=True, clean_up_tokenization_spaces=True, pad_token_id=tokenizer.eos_token_id)
        clean_output = output[output.find("Answer:"):].lower()

        index_true = clean_output.find('true')
        index_false = clean_output.find('false')

        if (index_true != -1 and (index_false == -1 or index_true < index_false)):
            predicted_label = 'TRUE'
        elif (index_false != -1 and (index_true == -1 or index_false < index_true)):
            predicted_label = 'FALSE'
    label = label.lower()
    predicted_label = predicted_label.lower()
    if(label == predicted_label):
        acc += 1
    result_datas.loc[left_id] = [left_id, claim_text, label, predicted_label, clean_output]
    
    pbar.set_postfix({'Acc': acc / (i + 1)})
with open(path + llm_type + '_' + dataset_name + '_' + file_name.replace('.tsv', '_super_prompt_result.tsv'), 'w') as f:
    result_datas.to_csv(f, sep='\t', index=False)
    f.close()