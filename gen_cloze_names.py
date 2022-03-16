import csv
import utils
import numpy as np
choice = np.random.choice

male_names = []
female_names = []
for row in csv.DictReader(open("raw_data/most_common_female_names.csv")):
    female_names.append(row['name'])
for row in csv.DictReader(open("raw_data/most_common_male_names.csv")):
    male_names.append(row['name'])
subject_pool = ["his name is","her name is", "my name is", "your name is"]
his_name, her_name, my_name, your_name = subject_pool[0],subject_pool[1],subject_pool[2],subject_pool[3]
exclude = ['Ewan', 'Alastair', 'Euan', 'Calum', 'Alasdair', 'Greig', 'Martyn', 'Kieran', 'Kristopher', 'Keiran', 'Ciaran', 'Finlay', 'Arran', 'Keir', 'Kian', 'Lennon', 'Kenzie', 'Alfie', 'Jayden', 'Zak', 'Kayden', 'Kaiden', 'Ruaridh', 'Olly', 'Callan', 'Jaxon', 'Lachlan', 'Arlo', 'Innes', 'Ruairidh', 'Struan', 'Lorna', 'Kirsty', 'Shona', 'Catriona', 'Morag', 'Kirsten', 'Kirsteen', 'Lynsey', 'Aileen', 'Arlene', 'Mhairi', 'Gayle', 'Leanne', 'Lyndsey', 'Lyndsay', 'Charlene', 'Linsey', 'Eilidh', 'Hayley', 'Alana', 'Siobhan', 'Rachael', 'Ashleigh', 'Kayleigh', 'Jemma', 'Linzi', 'Jodie', 'Michaela', 'Sinead', 'Kerri', 'Kirstie', 'Nicolle', 'Rebekah', 'Hollie', 'Chantelle', 'Abbie', 'Niamh', 'Rhiannon', 'Caitlyn', 'Kaitlin', 'Ciara', 'Meghan', 'Lauryn', 'Ailsa', 'Morven', 'Cerys', 'Kiera', 'Freya', 'Zara', 'Orla', 'Keira', 'Neve', 'Abi', 'Abbi', 'Alisha', 'Mya', 'Maisie', 'Imogen', 'Nieve', 'Miley', 'Mollie', 'Laila', 'Mirren', 'Ayla', 'Mila', 'Esme', 'Arianna', 'Thea', 'Ariana', 'Lillie', 'Hallie', 'Aila', 'Myla', 'Aoife', 'Lottie', 'Lyla', 'Remi', 'Maeve', 'Ayda', 'Arabella']
exclude.extend(['Iain', 'Graeme', 'Alistair', 'Roderick', 'Gregor', 'Callum', 'Niall', 'Barrie', 'Antony', 'Declan', 'Aidan', 'Rhys', 'Reece', 'Hamish', 'Conner', 'Ronan', 'Aiden', 'Mackenzie', 'Brodie', 'Luca', 'Ollie', 'Reuben', 'Brody', 'Zachary', 'Jax', 'Lyle', 'Finley', 'Myles', 'Gillian', 'Jacqueline', 'Lesley', 'Pauline', 'Lorraine', 'Tracey', 'Lynne', 'Yvonne', 'Joanne', 'Gail', 'Joanna', 'Maureen', 'Mandy', 'Jillian', 'Vicky', 'Stacey', 'Gemma', 'Kimberley', 'Adele', 'Kylie', 'Robyn', 'Caitlin', 'Aimee', 'Cara', 'Demi', 'Bethany', 'Toni', 'Abigail', 'Iona', 'Isla', 'Kelsey', 'Carla', 'Kaitlyn', 'Jasmine', 'Skye', 'Rosie', 'Kayla', 'Elle', 'Ella', 'Millie', 'Ava', 'Evie', 'Alyssa', 'Poppy', 'Isabella', 'Charley', 'Layla', 'Libby', 'Lexi', 'Amelie', 'Phoebe', 'Lexie', 'Lucie', 'Sienna', 'Gracie', 'Rowan', 'Sofia', 'Lacey', 'Emilia', 'Lola', 'Darcy', 'Aria', 'Matilda', 'Elsie', 'Georgie', 'Sadie', 'Arya', 'Callie', 'Penelope', 'Cora', 'Evelyn', 'Alba'])
for word in exclude:
    try:  male_names.remove(word)
    except ValueError as e: pass
    try:  female_names.remove(word)
    except ValueError as e: pass

all_names = male_names + female_names
n = len(male_names)

def name_0():
    sentences = []
    answers = []

    for name in male_names:
        sentence = []
        context = [his_name, name, "."]
        sentence.extend(context)
        
        question = ["So", his_name, "[MASK]", "."]
        sentence.extend(question)

        sentence = " ".join(sentence)
        sentences.append(sentence)
        answers.append(name)

    for name in female_names:
        sentence = []
        context = [her_name, name, "."]
        sentence.extend(context)
        
        question = ["So", her_name, "[MASK]", "."]
        sentence.extend(question)

        sentence = " ".join(sentence)
        sentences.append(sentence)
        answers.append(name)

    for name in all_names:
        sentence = []
        context = [my_name, name, "."]
        sentence.extend(context)
        
        question = ["So", my_name, "[MASK]", "."]
        sentence.extend(question)

        sentence = " ".join(sentence)
        sentences.append(sentence)
        answers.append(name)
        
    for name in all_names:
        sentence = []
        context = [your_name, name, "."]
        sentence.extend(context)
        
        question = ["So", your_name, "[MASK]", "."]
        sentence.extend(question)

        sentence = " ".join(sentence)
        sentences.append(sentence)
        answers.append(name)

    return sentences, answers

def name_1():
    sentences = []
    answers = []

    for n1, n2 in zip(male_names, female_names[:n]):
        if n1 in exclude: continue
        sentence = []
        context = [her_name, n2]
        sentence.extend(context)

        sent = ["and", his_name, n1, "."]
        sentence.extend(sent)
        
        question = ["So", her_name, "[MASK]", "."]
        sentence.extend(question)

        sentence = " ".join(sentence)
        sentences.append(sentence)
        answers.append(n2)

    return sentences, answers

def name_2():
    my_names = choice(all_names, n)
    sentences = []
    answers = []

    for n1, n2, n3 in zip(male_names, female_names[:n], my_names):
        if n1 in exclude: continue
        sentence = []
        context = [his_name, n1]
        sentence.extend(context)

        sent = [",", her_name, n2]
        sentence.extend(sent)

        sent = ["and", my_name, n3, "."]
        sentence.extend(sent)

        question = ["So", his_name, "[MASK]", "."]
        sentence.extend(question)

        sentences.append(" ".join(sentence))
        answers.append(n1)
    return sentences, answers

def name_3():
    all_names = male_names + female_names
    my_names = choice(all_names, n)
    your_names = choice(all_names, n)
    sentences = []
    answers = []

    for n1, n2, n3, n4 in zip(male_names, female_names[:n], my_names, your_names):
        if n1 in exclude: continue
        sentence = []
        context = [his_name, n1]
        sentence.extend(context)

        sent = [",", her_name, n2]
        sentence.extend(sent)

        sent = [",", my_name, n3]
        sentence.extend(sent)

        sent = ["and", your_name, n4, "."]
        sentence.extend(sent)

        question = ["So", his_name, "[MASK]", "."]
        sentence.extend(question)
        
        sentences.append(" ".join(sentence))
        answers.append(n1)
    return sentences, answers

# sentence, anwsers = name_0()
# utils.to_csv(sentence, anwsers,"cloze/simple_SVO/names/num_attractors=0.csv")

sentence, anwsers = name_1()
utils.to_csv(sentence, anwsers,"cloze/simple_SVO/names/num_attractors=1.csv")

sentence, anwsers = name_2()
utils.to_csv(sentence, anwsers,"cloze/simple_SVO/names/num_attractors=2.csv")

sentence, anwsers = name_3()
utils.to_csv(sentence, anwsers,"cloze/simple_SVO/names/num_attractors=3.csv")