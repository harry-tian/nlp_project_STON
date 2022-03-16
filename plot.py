import re
import matplotlib.pyplot as plt
import numpy as np
import csv

SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 20
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

color_list = ['k','y','m','g','c','r','b','lime']
marker_list = ['o','s','^','x','d','p','*','8']
linestyle_list = ['solid','dashed','dashdot','dotted','solid','dashed','dashdot','dotted']

temp = "complete_data_For_MultipleEntityObjectDistractorAccuracy"
count_distractors = [0,1,2,3]

def plot_cloze_names(title=None):
    plt.rcParams.update({'legend.fontsize':12})
    fig,ax =  plt.subplots(2,2, figsize=(16,12), sharey=True, sharex=True)
    for i, temp in enumerate(["his_name","her_name","my_name","your_name"]):
        if i == 0: idx = (0,0)
        elif i == 1: idx = (0,1)
        elif i == 2: idx = (1,0)
        elif i == 3: idx = (1,1)
        data_dir = f"cloze/simple_SVO/names/{temp}/results.csv"
        
        reader = csv.DictReader(open(data_dir))
        results = {}
        for row in reader:
            model = row['model']
            results[model] = [1, float(row['1'].strip()),float(row['2'].strip()),float(row['3'].strip())]
            
        for j, (model, acc) in enumerate(results.items()):
            ax[idx[0]][idx[1]].plot([0, 1,2,3], acc,linewidth=2.0, color=color_list[j],
                markersize=7,linestyle=linestyle_list[j],label=model,marker=marker_list[j])

        ax[idx[0]][idx[1]].set_xticks([0, 1,2,3])
        ax[idx[0]][idx[1]].set_yticks(np.arange(0,1.1,0.1))
        
        ax[idx[0]][idx[1]].set_xlabel('number of attractors', labelpad=1)
        ax[idx[0]][idx[1]].set_ylabel('accuracy')
        temp = temp.split("_")
        subtitle = f"\"{temp[0]} {temp[1]} is [MASK]\""
        ax[idx[0]][idx[1]].set_title(subtitle)
        
    plt.legend(loc='upper right', bbox_to_anchor=(0.9, -0.1),fancybox=True, shadow=True, ncol=8)
    if not title: title = "cloze task: names"
    fig.suptitle(title, fontsize=25)
    plt.show()
    
def plot_cloze(results, title=None, legend=False):
    f = open(results)
    reader = csv.DictReader(f)
    results = {}
    for row in reader:
        model = row['model']
        results[model] = [1, float(row['1'].strip()),float(row['2'].strip()),float(row['3'].strip())]
        
    plt.figure(figsize=(8,6))
    for i, val in enumerate(results.values()):
        plt.plot([0, 1,2,3], val,linewidth=2.0, color=color_list[i],
            markersize=7,linestyle=linestyle_list[i],label=model,marker=marker_list[i])
    plt.xticks([0, 1,2,3])
    plt.yticks(np.arange(0,1.1,0.1))
    if legend: plt.legend(["BertBase", "BertLarge", "RobertaBase","RobertaLarge","GPT2Small","GPT2Medium"])
    plt.xlabel('number of attractors', labelpad=1)
    plt.ylabel('accuracy')
    if not title: title = "cloze task: names"
    plt.title(title)

def plot_acc(data_dir,title):
    count_distractors = [0,1,2,3]
    model_list = ['BertBase','BertLarge','RobertaBase','RobertaLarge','GPT2Small','GPT2Medium','GPT2Large','GPT2XL']

    plt.rcParams.update({'legend.fontsize':10})
    fig,ax =  plt.subplots(1,2, figsize=(14,6), sharey=True, sharex=True)
    for index, model in enumerate(model_list):
        count = 0
        
        count_0_attractor = 0
        count_1_attractor = 0
        count_2_attractor = 0
        count_3_attractor = 0

        correct_0_attractor = 0
        correct_1_attractor = 0
        correct_2_attractor = 0
        correct_3_attractor = 0
        correct = 0

        acc_count_attractor = []
        file = f'data/combined_data/multiple_entity_{data_dir}/{model}/{temp}{model}.csv'
        f = open(file,'r')
        reader = csv.DictReader(f,delimiter='\t')
        for row in reader:
            item_list = re.sub(r'[^\w]', ' ', row['ordered_items'])
            item_list = item_list.split()
            
            if int(row['count_attractors']) == 1:
                if row['target_occupation'].lower() == item_list[0].lower():
                    correct_1_attractor+=1
                count_1_attractor+=1
            if int(row['count_attractors']) == 2:
                if row['target_occupation'].lower() == item_list[0].lower():
                    correct_2_attractor+=1
                count_2_attractor+=1

            if int(row['count_attractors']) == 3:
                if row['target_occupation'].lower() == item_list[0].lower():
                    correct_3_attractor+=1
                count_3_attractor+=1
            if int(row['count_attractors']) == 0:
                if row['target_occupation'].lower() == item_list[0].lower():
                    correct_0_attractor+=1
                count_0_attractor+=1

        accuracy_attractor = correct_0_attractor/float(count_0_attractor)
        acc_count_attractor.append(accuracy_attractor)

        accuracy_attractor = correct_1_attractor/float(count_1_attractor)
        acc_count_attractor.append(accuracy_attractor)

        accuracy_attractor = correct_2_attractor/float(count_2_attractor)
        acc_count_attractor.append(accuracy_attractor)

        accuracy_attractor = correct_3_attractor/float(count_3_attractor)
        acc_count_attractor.append(accuracy_attractor)
        ##############################################################end of multi#################################################

        count_1_attractor_single_entity=0
        count_2_attractor_single_entity = 0
        count_0_attractor_single_entity=0
        count_3_attractor_single_entity = 0

        correct_1_attractor_single_entity = 0
        correct_2_attractor_single_entity = 0
        correct_0_attractor_single_entity = 0
        correct_3_attractor_single_entity = 0

        acc_count_attractor_single_entity = []
        file = f'data/combined_data/single_entity_{data_dir}/{model}/{temp}{model}.csv'
        f = open(file,'r')
        reader = csv.DictReader(f,delimiter='\t')
        for row in reader:
            item_list = re.sub(r'[^\w]', ' ', row['ordered_items'])
            item_list = item_list.split()

            if int(row['count_attractors']) == 1:
                if row['target_occupation'].lower() == item_list[0].lower():
                    correct_1_attractor_single_entity+=1
                    correct+=1

                count_1_attractor_single_entity+=1
                count+=1

            if int(row['count_attractors']) == 0:
                if row['target_occupation'].lower() == item_list[0].lower():
                    correct_0_attractor_single_entity+=1
                    correct+=1
                count_0_attractor_single_entity+=1
                count+=1
            if int(row['count_attractors']) == 2:
                if row['target_occupation'].lower() == item_list[0].lower():
                    correct_2_attractor_single_entity+=1
                    correct+=1
                count_2_attractor_single_entity+=1
                count+=1

            if int(row['count_attractors']) == 3:
                if row['target_occupation'].lower() == item_list[0].lower():
                    correct_3_attractor_single_entity+=1
                    correct+=1
                count_3_attractor_single_entity+=1
                count+=1

        accuracy_attractor_single_entity = correct_0_attractor_single_entity/float(count_0_attractor_single_entity)
        acc_count_attractor_single_entity.append(accuracy_attractor_single_entity)

        accuracy_attractor_single_entity = correct_1_attractor_single_entity/float(count_1_attractor_single_entity)
        acc_count_attractor_single_entity.append(accuracy_attractor_single_entity)

        accuracy_attractor_single_entity = correct_2_attractor_single_entity/float(count_2_attractor_single_entity)
        acc_count_attractor_single_entity.append(accuracy_attractor_single_entity)

        accuracy_attractor_single_entity = correct_3_attractor_single_entity/float(count_3_attractor_single_entity)
        acc_count_attractor_single_entity.append(accuracy_attractor_single_entity)

        ax[0].plot(count_distractors, acc_count_attractor, color=color_list[index],\
            markersize=7,linestyle=linestyle_list[index],label=model,marker=marker_list[index])
        
        ax[0].set_xlabel('number of attractors', labelpad=1)
        ax[0].set_ylabel('accuracy')
        ax[0].set_title("Multiple Entity")
        max_x = max(count_distractors)
        ax[0].set_xticks(np.arange(0, max_x+1, 1))
        ax[0].set_yticks(np.arange(0, 1.1, 0.1)) 

        ax[1].plot(count_distractors, acc_count_attractor_single_entity, color=color_list[index],markersize=7,linestyle=linestyle_list[index],label=model,marker=marker_list[index])
        ax[1].set_xlabel('number of attractors', labelpad=1)
        ax[1].set_ylabel('accuracy')
        ax[1].set_title("Single Entity")
        max_x = max(count_distractors)
        ax[1].set_xticks(np.arange(0, max_x+1, 1)) 
        ax[1].set_yticks(np.arange(0, 1.1, 0.1)) 

    plt.legend(loc='upper right', bbox_to_anchor=(1, -0.1),fancybox=True, shadow=True, ncol=8)
    fig.suptitle(title, fontsize=25)

    plt.show()

def plot_relative_prob(data_dir,title):
    model_list = ['BertBase','BertLarge','RobertaBase','RobertaLarge','GPT2Small','GPT2Medium','GPT2Large','GPT2XL']
    plt.rcParams.update({'legend.fontsize':10})

    fig,ax =  plt.subplots(1,2, figsize=(14,6), sharey=True, sharex=True)
    for index, model in enumerate(model_list):
        count = 0
        
        count_0_attractor = 0
        count_1_attractor = 0
        count_2_attractor = 0
        count_3_attractor = 0

        correct_0_attractor = 0
        correct_1_attractor = 0
        correct_2_attractor = 0
        correct_3_attractor = 0

        acc_count_attractor = []

        file = f'data/combined_data/multiple_entity_{data_dir}/{model}/{temp}{model}.csv'
        f = open(file,'r')
        reader = csv.DictReader(f,delimiter='\t')
        for row in reader:
            item_list = re.sub(r'[^\w]', ' ', row['ordered_items'])
            item_list = item_list.split()

            if int(row['count_attractors']) == 1:
                correct_1_attractor+=float(row['relative_prob'])
                count_1_attractor+=1
            if int(row['count_attractors']) == 2:
                correct_2_attractor+=float(row['relative_prob'])
                count_2_attractor+=1

            if int(row['count_attractors']) == 3:
                correct_3_attractor+=float(row['relative_prob'])
                count_3_attractor+=1

            if int(row['count_attractors']) == 0:
                correct_0_attractor+=float(row['relative_prob'])
                count_0_attractor+=1


        accuracy_attractor = correct_0_attractor/float(count_0_attractor)
        acc_count_attractor.append(accuracy_attractor)

        accuracy_attractor = correct_1_attractor/float(count_1_attractor)
        acc_count_attractor.append(accuracy_attractor)

        accuracy_attractor = correct_2_attractor/float(count_2_attractor)
        acc_count_attractor.append(accuracy_attractor)

        accuracy_attractor = correct_3_attractor/float(count_3_attractor)
        acc_count_attractor.append(accuracy_attractor)

        ##############################################################end of multi#################################################

        count_1_attractor_single_entity=0
        count_2_attractor_single_entity = 0
        count_0_attractor_single_entity=0
        count_3_attractor_single_entity = 0

        correct_1_attractor_single_entity = 0
        correct_2_attractor_single_entity = 0
        correct_0_attractor_single_entity = 0
        correct_3_attractor_single_entity = 0

        acc_count_attractor_single_entity = []
        file = f'data/combined_data/single_entity_{data_dir}/{model}/{temp}{model}.csv'
        f = open(file,'r')
        reader = csv.DictReader(f,delimiter='\t')
        for row in reader:
            item_list = re.sub(r'[^\w]', ' ', row['ordered_items'])
            item_list = item_list.split()

            if int(row['count_attractors']) == 1:
                correct_1_attractor_single_entity+=float(row['relative_prob'])
                count_1_attractor_single_entity+=1
                count+=1

            if int(row['count_attractors']) == 0:
                correct_0_attractor_single_entity+=float(row['relative_prob'])

                count_0_attractor_single_entity+=1
                count+=1
            if int(row['count_attractors']) == 2:
                correct_2_attractor_single_entity+=float(row['relative_prob'])

                count_2_attractor_single_entity+=1
                count+=1

            if int(row['count_attractors']) == 3:
                correct_3_attractor_single_entity+=float(row['relative_prob'])
                count_3_attractor_single_entity+=1
                count+=1

        accuracy_attractor_single_entity = correct_0_attractor_single_entity/float(count_0_attractor_single_entity)
        acc_count_attractor_single_entity.append(accuracy_attractor_single_entity)

        accuracy_attractor_single_entity = correct_1_attractor_single_entity/float(count_1_attractor_single_entity)
        acc_count_attractor_single_entity.append(accuracy_attractor_single_entity)

        accuracy_attractor_single_entity = correct_2_attractor_single_entity/float(count_2_attractor_single_entity)
        acc_count_attractor_single_entity.append(accuracy_attractor_single_entity)

        accuracy_attractor_single_entity = correct_3_attractor_single_entity/float(count_3_attractor_single_entity)
        acc_count_attractor_single_entity.append(accuracy_attractor_single_entity)

        ax[0].plot(count_distractors, acc_count_attractor,color=color_list[index],\
            markersize=7,linestyle=linestyle_list[index],label=model,marker=marker_list[index])
        
        ax[0].set_xlabel('number of attractors')
        ax[0].set_ylabel('Relative_Prob')
        ax[0].set_title("Multiple Entity")
        max_x = max(count_distractors)
        ax[0].set_xticks(np.arange(0, max_x+1, 1))
        ax[0].set_yticks(np.arange(0, 1.1, 0.1)) 


        ax[1].plot(count_distractors, acc_count_attractor_single_entity,color=color_list[index],markersize=7,linestyle=linestyle_list[index],label=model,marker=marker_list[index])
        ax[1].set_xlabel('number of attractors')
        ax[1].set_ylabel('Relative_Prob')
        ax[1].set_title("Single Entity")
        max_x = max(count_distractors)
        ax[1].set_xticks(np.arange(0, max_x+1, 1)) 
        ax[1].set_yticks(np.arange(0, 1.1, 0.1)) 

    plt.legend(loc='upper right', bbox_to_anchor=(1, -0.1),fancybox=True, shadow=True, ncol=8)
    fig.suptitle(title, fontsize=25)

    plt.show()