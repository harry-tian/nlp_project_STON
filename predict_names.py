from pickle import FALSE
from transformers import pipeline
import csv
import torch
# import sys
# import time
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import warnings
warnings.filterwarnings("ignore")

def eval(model, texts,targets, verbose=False):
    total = 0
    correct = 0
    errors = []

    if "roberta" in model:
        texts = [text.replace("[MASK]", "<mask>") for text in texts]
    elif "gpt" in model:
        texts = [text.replace(" [MASK] .", "") for text in texts]

    if "bert" in model:
        predict = pipeline('fill-mask', model=model, device=0)
        for text, target in tqdm(zip(texts, targets)):
            total += 1
            pred = predict(text)[0]
            pred = pred["token_str"]
            if pred.strip().lower() == target.strip().lower():
                correct += 1
            else:
                errors.append((pred, target))
            
            # if verbose: print(text, pred, target)

    elif "gpt" in model:
        tokenizer = GPT2Tokenizer.from_pretrained(model)
        model = GPT2LMHeadModel.from_pretrained(model, pad_token_id=tokenizer.eos_token_id).cuda()
        for text, target in tqdm(zip(texts, targets)):
            total += 1
            preds = generate(model, text, tokenizer, n=30)
            if target.strip().lower() in preds.lower():
            # target = tokenizer.encode(target)[0]
            # if target in preds:
                correct += 1
            else:
                errors.append(target)
            
            # if verbose: print(target, preds)

    return correct/total, errors

def generate(model, text, tokenizer, n=10):
    # text = 'his name is Henry , her name is Mary , my name is Twyla and your name is Geneva . his name is'
    input_ids = torch.tensor([tokenizer.encode(text)]).cuda()
    output = model.generate(
        input_ids,
        do_sample=True, 
        max_length=len(input_ids[0])+1, 
        top_k=50, 
        top_p=0.95, 
        num_return_sequences=n
    )
    answers = tokenizer.decode(output[:,-1], skip_special_tokens=True)
    # print(answers)
    # quit()
    # print("Output:\n" + 100 * '-')
    # print(text)
    # print(tokenizer.decode(output[0]))
    # quit()
    return answers

def get_data(data_dir):
    f = open(data_dir)
    reader = csv.DictReader(f)

    targets = []
    sentences = []
    for row in reader:
        sentence = row['sentence']
        target = row['answer']
        
        targets.append(target)
        sentences.append(sentence)

    return sentences, targets

def main():
    models = ['bert-base-uncased', 
    "bert-large-uncased", 
    "roberta-base", 
    "roberta-large", 
    "gpt2", 
    "gpt2-medium",] 
    # "gpt2-large",]
    #, "gpt2-xl"]
    fieldnames = ["model",0,1,2,3]
    results = []

    data_dir = 'cloze/simple_SVO/names/'
    out_dir = data_dir + "results.csv"

    for model in models:
        print(f"{model}:\n" + 100 * '-')
        result = dict.fromkeys(fieldnames)
        result["model"] = model
        for n in [0,1,2,3]:
            print(f"n={n}:\n" + 100 * '-')
            f = data_dir + f"num_attractors={n}.csv"
            texts,targets = get_data(f)
            acc, err = eval(model, texts, targets, verbose=False)
            result[n] = acc
        results.append(result)

    print(results)

    with open(out_dir, 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def test():
    model = 'bert-base-uncased'
    models = ['bert-base-uncased', 
    "bert-large-uncased", 
    "roberta-base", 
    "roberta-large", 
    "gpt2", 
    "gpt2-medium",] 
    models = ['bert-base-uncased'] 
    data_dir = 'cloze/simple_SVO/names/my_name/'
    
    f = data_dir + f"num_attractors=1.csv"

    for model in models:
        texts,targets = get_data(f)

        # print(eval(model, texts[:20], targets[:20], verbose=True))
        acc, err = eval(model, texts, targets, verbose=False)
        print(f"model: {acc}")
        print(err[:20])
    
test()