import csv
import utils
import numpy as np
choice = np.random.choice

male_names = []
female_names = []
for row in csv.DictReader(open("most_common_female_names.csv")):
    female_names.append(row['name'])
for row in csv.DictReader(open("most_common_male_names.csv")):
    male_names.append(row['name'])
subject_pool = ["his name is","her name is", "my name is", "your name is"]
n = len(male_names)
exclude = ['Ewan', 'Alastair', 'Euan', 'Calum', 'Alasdair', 'Greig', 'Martyn', 'Kieran', 'Kristopher', 'Keiran', 'Ciaran', 'Finlay', 'Arran', 'Keir', 'Kian', 'Lennon', 'Kenzie', 'Alfie', 'Jayden', 'Zak', 'Kayden', 'Kaiden', 'Ruaridh', 'Olly', 'Callan', 'Jaxon', 'Lachlan', 'Arlo', 'Innes', 'Ruairidh', 'Struan', 'Lorna', 'Kirsty', 'Shona', 'Catriona', 'Morag', 'Kirsten', 'Kirsteen', 'Lynsey', 'Aileen', 'Arlene', 'Mhairi', 'Gayle', 'Leanne', 'Lyndsey', 'Lyndsay', 'Charlene', 'Linsey', 'Eilidh', 'Hayley', 'Alana', 'Siobhan', 'Rachael', 'Ashleigh', 'Kayleigh', 'Jemma', 'Linzi', 'Jodie', 'Michaela', 'Sinead', 'Kerri', 'Kirstie', 'Nicolle', 'Rebekah', 'Hollie', 'Chantelle', 'Abbie', 'Niamh', 'Rhiannon', 'Caitlyn', 'Kaitlin', 'Ciara', 'Meghan', 'Lauryn', 'Ailsa', 'Morven', 'Cerys', 'Kiera', 'Freya', 'Zara', 'Orla', 'Keira', 'Neve', 'Abi', 'Abbi', 'Alisha', 'Mya', 'Maisie', 'Imogen', 'Nieve', 'Miley', 'Mollie', 'Laila', 'Mirren', 'Ayla', 'Mila', 'Esme', 'Arianna', 'Thea', 'Ariana', 'Lillie', 'Hallie', 'Aila', 'Myla', 'Aoife', 'Lottie', 'Lyla', 'Remi', 'Maeve', 'Ayda', 'Arabella']
exclude.extend(['Iain', 'Graeme', 'Alistair', 'Roderick', 'Gregor', 'Callum', 'Niall', 'Barrie', 'Antony', 'Declan', 'Aidan', 'Rhys', 'Reece', 'Hamish', 'Conner', 'Ronan', 'Aiden', 'Mackenzie', 'Brodie', 'Luca', 'Ollie', 'Reuben', 'Brody', 'Zachary', 'Jax', 'Lyle', 'Finley', 'Myles', 'Gillian', 'Jacqueline', 'Lesley', 'Pauline', 'Lorraine', 'Tracey', 'Lynne', 'Yvonne', 'Joanne', 'Gail', 'Joanna', 'Maureen', 'Mandy', 'Jillian', 'Vicky', 'Stacey', 'Gemma', 'Kimberley', 'Adele', 'Kylie', 'Robyn', 'Caitlin', 'Aimee', 'Cara', 'Demi', 'Bethany', 'Toni', 'Abigail', 'Iona', 'Isla', 'Kelsey', 'Carla', 'Kaitlyn', 'Jasmine', 'Skye', 'Rosie', 'Kayla', 'Elle', 'Ella', 'Millie', 'Ava', 'Evie', 'Alyssa', 'Poppy', 'Isabella', 'Charley', 'Layla', 'Libby', 'Lexi', 'Amelie', 'Phoebe', 'Lexie', 'Lucie', 'Sienna', 'Gracie', 'Rowan', 'Sofia', 'Lacey', 'Emilia', 'Lola', 'Darcy', 'Aria', 'Matilda', 'Elsie', 'Georgie', 'Sadie', 'Arya', 'Callie', 'Penelope', 'Cora', 'Evelyn', 'Alba'])


def name_0():
    sentences = []
    answers = []

    for name in male_names:
        if name in exclude: continue
        sentence = []
        context = [subject_pool[0], name, "."]
        sentence.extend(context)
        
        question = ["So", subject_pool[0], "[MASK]", "."]
        sentence.extend(question)

        sentence = " ".join(sentence)
        sentences.append(sentence)
        answers.append(name)

    for name in female_names:
        if name in exclude: continue
        sentence = []
        context = [subject_pool[1], name, "."]
        sentence.extend(context)
        
        question = ["So", subject_pool[1], "[MASK]", "."]
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
    my_names = choice(all_names, n)
    sentences = []
    answers = []

    for n1, n2, n3 in zip(male_names, female_names[:n], my_names):
        if n1 in exclude: continue
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
    my_names = choice(all_names, n)
    your_names = choice(all_names, n)
    sentences = []
    answers = []

    for n1, n2, n3, n4 in zip(male_names, female_names[:n], my_names, your_names):
        if n1 in exclude: continue
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

sentence, anwsers = name_0()
utils.to_csv(sentence, anwsers,"raw_data/simple_SVO/names/num_attractors=0.csv")

sentence, anwsers = name_1()
utils.to_csv(sentence, anwsers,"raw_data/simple_SVO/names/num_attractors=1.csv")

sentence, anwsers = name_2()
utils.to_csv(sentence, anwsers,"raw_data/simple_SVO/names/num_attractors=2.csv")

sentence, anwsers = name_3()
utils.to_csv(sentence, anwsers,"raw_data/simple_SVO/names/num_attractors=3.csv")