import csv
import re
import sys
import matplotlib.pyplot as plt
import matplotlib
'''matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})'''
import numpy as np


count_distractors = [0,1,2,3]

#fig = plt.figure()
#fig.set_size_inches(w=8,h=6) 
plt.rcParams["legend.loc"] = 'upper right'
#print(plt.rcParams)

plt.rcParams.update({'font.size': 14.5})
#plt.rcParams.update({'legend.fontsize':10})
plt.rcParams.update({'legend.fontsize':11.6})
#plt.rcParams["font."]
color_list = ['k','y','m','g','c','r','b','lime']
marker_list = ['o','s','^','x','d','p','*','8']
linestyle_list = ['solid','dashed','dashdot','dotted','solid','dashed','dashdot','dotted']

fig,ax =  plt.subplots(1,2)

	ax[0].plot(count_distractors, acc_count_attractor, '-ok',color=color_list[index],\
		markersize=7,linestyle=linestyle_list[index],label=model,marker=marker_list[index])
	
	ax[0].set_xlabel('number of attractors', labelpad=1)
	ax[0].set_ylabel('accuracy')
	ax[0].set_title("Multiple Entity")
	max_x = max(count_distractors)
	ax[0].set_xticks(np.arange(0, max_x+1, 1))
	ax[0].set_yticks(np.arange(0, 1.1, 0.1)) 


	ax[1].plot(count_distractors, acc_count_attractor_single_entity, '-ok',color=color_list[index],markersize=7,linestyle=linestyle_list[index],label=model,marker=marker_list[index])
	ax[1].set_xlabel('number of attractors', labelpad=1)
	ax[1].set_ylabel('accuracy')
	ax[1].set_title("Single Entity")
	max_x = max(count_distractors)
	ax[1].set_xticks(np.arange(0, max_x+1, 1)) 
	ax[1].set_yticks(np.arange(0, 1.1, 0.1)) 
	#ax[1].set_xlabel('number of attractors')
	#print(correct/float(count))
	print(model)
	print(acc_count_attractor)
	print(acc_count_attractor_single_entity)
	index+=1

#plt.legend(loc='upper right', bbox_to_anchor=(1.1, -0.08),fancybox=True, shadow=True, ncol=8)
#plt.legend(loc='center left', bbox_to_anchor=(1.1, -0.08),fancybox=True, shadow=True, ncol=8)
plt.legend(loc='upper right', bbox_to_anchor=(1, -0.08),fancybox=True, shadow=True, ncol=8)
#plt.legend(loc='upper right', bbox_to_anchor=(1.3, -0.08),fancybox=True, shadow=True, ncol=8, borderpad=0.09)
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1),fancybox=True, shadow=True, ncol=1)
plt.show()
#plt.savefig('BType.png',bbox_inches = 'tight')


