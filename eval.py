import torch
import csv
import sys
import time
from collections import Counter
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch.nn.functional as F
from tqdm import tqdm

# def eval(model, tokenizer, texts, candidates, targets, num_attractors):
#     pred_correct = [0] * len(texts)

#     for i, (text, cand, target) in tqdm(enumerate(zip(texts, candidates, targets))):
#         pred, _ = predict_masked(model, tokenizer, text, cand)
#         if pred.lower() == target.lower():
#             pred_correct[i] = 1
            
#     accuracy_0attractor = []
#     accuracy_1attractor = []
#     accuracy_2attractor = []
#     accuracy_3attractor = []

#     for i in range(len(num_attractors)):
#         n = int(num_attractors[i])
#         if n == 0:
#             accuracy_0attractor += [pred_correct[i]]
#         elif n == 1:
#             accuracy_1attractor += [pred_correct[i]]
#         elif n == 2:
#             accuracy_2attractor += [pred_correct[i]]
#         elif n == 3:
#             accuracy_3attractor += [pred_correct[i]]
#         else:
#             print("Instance {}: more attractor than 3?".format(i))
            
#     accuracy_0attractor = sum(accuracy_0attractor) / len(accuracy_0attractor)
#     accuracy_1attractor = sum(accuracy_1attractor) / len(accuracy_1attractor)
#     accuracy_2attractor = sum(accuracy_2attractor) / len(accuracy_2attractor)
#     accuracy_3attractor = sum(accuracy_3attractor) / len(accuracy_3attractor)

#     return accuracy_0attractor, accuracy_1attractor, accuracy_2attractor, accuracy_3attractor

def eval(model, tokenizer, texts, candidates, targets):
    total = 0
    correct = 0

    for text, cand, target in tqdm(zip(texts, candidates, targets)):
        total += 1
        pred, _ = predict_masked(model, tokenizer, text, cand)
        if pred.lower() == target.lower():
            correct += 1
        print(text, target, pred)
            
    return correct/total

def predict_masked(model, tokenizer, text, candidates, verbose=False):
    """
    Input:
        text: a prepared instance of a sentence in the data.
        candidates: candidate words for which to calculate probabilities.
        verbose: whether to print text along with predicted probabilities
    Output:
        prediction: one of the candidates with highest predicted probability.
        probs: a tensor of predicted probailities of each candidate.
    """
    
    cand_probs = []
    
    if verbose:
        print(text)
    tokenized_text = tokenizer.tokenize(text)
    if "[MASK]" in tokenized_text:
        masked_index = tokenized_text.index("[MASK]")
    elif "[mask]" in tokenized_text:
        masked_index = tokenized_text.index("[mask]")
    else:
        print("No masks found.")
        return -1, torch.ones(len(candidates)) * (-99)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensors = torch.tensor([indexed_tokens]).cuda()
    
    with torch.no_grad():
        outputs = model(tokens_tensors)
        predictions = outputs[0]
        probs = F.softmax(predictions[0, masked_index], dim=-1)
        
    
    for cand in candidates:
        cand_id = [tokenizer.convert_tokens_to_ids(cand)]
        token_weight = probs[cand_id].float().item()
        if verbose:
            print(f"    {cand} | weights: {token_weight:.4f}")
        cand_probs.append(token_weight)
        
    cand_probs = torch.tensor(cand_probs)
    prediction = candidates[cand_probs.argmax().item()]
    
    return prediction, cand_probs

def ordered_items_to_list(candidate):
    return candidate.strip('[').strip(']').replace("'",'').replace(' ','').split(',')

# def get_data(data_dir):
#     f = open(data_dir)
#     reader = csv.DictReader(f, delimiter='\t')

#     ct = 0
#     targets = []
#     sentences = []
#     candidates = []
#     num_attractors = []
#     pre_pred = []

#     for row in reader:
#         sentence = row['sentence']
#         target = row['target_occupation']
#         candidate = row['ordered_items']
#         n_attractors = row['count_attractors']
#         rel_rank = float(row['relative_rank'])
        
#         targets.append(target)
#         sentences.append(sentence)
#         candidates.append(ordered_items_to_list(candidate))
#         num_attractors.append(n_attractors)
#         if rel_rank == 1:
#             pre_pred.append(1)
#         else:
#             pre_pred.append(0)

#     return sentences, candidates, targets, pre_pred, num_attractors

def get_data(data_dir, candidate):
    f = open(data_dir)
    reader = csv.DictReader(f)

    targets = []
    sentences = []
    candidates = []
    for row in reader:
        sentence = row['sentence']
        target = row['answer']
        
        targets.append(target)
        sentences.append(sentence)
        candidates.append(candidate)

    return sentences, candidates, targets

def prepare_text(text, model):
    res = []
    if model == 'BERT':
        # res.append("[CLS]")
        res += text.strip().split()        
        if "[mask]" in res:
            res[res.index("[mask]")] = "[MASK]"
        # period_index = [ind for ind, tok in enumerate(res) if tok == '.']
        # for i, ind in enumerate(period_index):
        #     res.insert(ind + 1 + i, "[SEP]")
        # res.append("[SEP]")
    return " ".join(res)

def main():
    # data_dir = './data/combined_data/multiple_entity_distractor/BertBase/complete_data_For_MultipleEntityObjectDistractorAccuracyBertBase.csv'
    # sentences, candidates, targets, num_attractors = get_data(data_dir)




    data_dir = 'raw_data/simple_SVO/emotions/'
    candidate = ["happy", "sad", "angry", "bored", "scared", "confused", "anxious"]
    for n in [0]:
        f = data_dir + f"num_attractors={n}.csv"
        sentences, candidates, targets = get_data(f, candidate)



    texts = [prepare_text(text, "BERT") for text in sentences]
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased').to("cuda")
    model.eval()

    print(eval(model, tokenizer, texts, candidates, targets))

    # accuracy_0attractor, accuracy_1attractor, accuracy_2attractor, accuracy_3attractor = eval(model, tokenizer, texts, candidates, targets, num_attractors)

    # print(f"Accuracy for 0 attractor(s): {accuracy_0attractor}")
    # print(f"Accuracy for 1 attractor(s): {accuracy_1attractor}")
    # print(f"Accuracy for 2 attractor(s): {accuracy_2attractor}")
    # print(f"Accuracy for 3 attractor(s): {accuracy_3attractor}")


main()