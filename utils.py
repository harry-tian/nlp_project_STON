import csv
import numpy as np
choice = np.random.choice

def to_csv(sents, answers, f_name):
    with open(f_name, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["sentence","answer"])
        for s,a in zip(sents, answers):
            writer.writerow([s,a])

def subset(cloze, sample=10000):
    sentences, answers = cloze
    idx = choice(np.arange(len(sentences[0])), sample)
    return np.array(sentences)[idx], np.array(answers)[idx]