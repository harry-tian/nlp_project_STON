import csv
import utils

reader = csv.DictReader(open("baby-names.csv"))
male_names = []
female_names = []
for row in reader:
    if row['sex'] == "boy": male_names.append(row['name'])
    else: female_names.append(row['name'])
subject_pool = ["his name is","her name is", "my name is", "your name is"]

def name_0():
    sentences = []
    answers = []

    for n1, n2 in zip(male_names, female_names):
        sentence = []
        context = [subject_pool[0], n1, "."]
        sentence.extend(context)
        
        question = ["So", subject_pool[0], "[MASK]", "."]
        sentence.extend(question)

        sentence = " ".join(sentence)
        sentences.append(sentence)
        answers.append(n1)
    return sentences, answers

def name_1():
    sentences = []
    answers = []

    for n1, n2 in zip(male_names, female_names):
        sentence = []
        context = [subject_pool[0], n1]
        sentence.extend(context)

        sent = ["and", subject_pool[1], n2, "."]
        sentence.extend(sent)
        
        question = ["So", subject_pool[0], "[MASK]", "."]
        sentence.extend(question)

        sentence = " ".join(sentence)
        sentences.append(sentence)
        answers.append(n1)
    return sentences, answers

def name_2():
    all_names = male_names + female_names
    my_names = choice(all_names, len(male_names))
    sentences = []
    answers = []

    for n1, n2, n3 in zip(my_names):
        sentence = []
        context = [subject_pool[0], n1]
        sentence.extend(context)

        sent = [",", subject_pool[1], n2]
        sentence.extend(sent)

        sent = ["and", subject_pool[2], n3, "."]
        sentence.extend(sent)

        question = ["So", subject_pool[0], "[MASK]", "."]
        sentence.extend(question)

        sentences.append(" ".join(sentence))
        answers.append(n1)
    return sentences, answers

def name_3():
    all_names = male_names + female_names
    my_names = choice(all_names, len(male_names))
    your_names = choice(all_names, len(male_names))
    sentences = []
    answers = []

    for n1, n2, n3, n4 in zip(male_names, female_names, my_names, your_names):
        sentence = []
        context = [subject_pool[0], n1]
        sentence.extend(context)

        sent = [",", subject_pool[1], n2]
        sentence.extend(sent)

        sent = [",", subject_pool[2], n3]
        sentence.extend(sent)

        sent = ["and", subject_pool[3], n4, "."]
        sentence.extend(sent)

        question = ["So", subject_pool[0], "[MASK]", "."]
        sentence.extend(question)
        
        sentences.append(" ".join(sentence))
        answers.append(n1)
    return sentences, answers

sentence, anwsers = utils.subset(name_0())
utils.to_csv(sentence, anwsers,"raw_data/simple_SVO/names/num_attractors=0.csv")

sentence, anwsers = utils.subset(name_1())
utils.to_csv(sentence, anwsers,"raw_data/simple_SVO/names/num_attractors=1.csv")

sentence, anwsers = utils.subset(name_2())
utils.to_csv(sentence, anwsers,"raw_data/simple_SVO/names/num_attractors=2.csv")

sentence, anwsers = utils.subset(name_3())
utils.to_csv(sentence, anwsers,"raw_data/simple_SVO/names/num_attractors=3.csv")