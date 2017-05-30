
from __future__ import division # In Python 3.0, // is the floor division operator and / the true division operator. The true_divide(x1, x2) function is equivalent to true division in Python.
import os
from shutil import copy
import numpy
import mcnmf
import classification as cl
import math

import ssmcnmf


dir_data="/home/yifeng/research/mf/mvmf_v1_1/data/"
dir_save="/home/yifeng/research/mf/mvmf_v1_1/results/"

a_active_W=10
b_active_W=1e5
a_active_H=10
b_active_H=1

prefix="simulated_data" + "_a_active_W="+str(a_active_W) + "_b_active_W="+str(b_active_W) + "_a_active_H="+str(a_active_H) + "_b_active_H="+str(b_active_H)

data=numpy.loadtxt(dir_data+prefix+"_X.txt",dtype=float,delimiter="\t")
features=numpy.loadtxt(dir_data+prefix+"_Features.txt",dtype=str,delimiter="\t")
feature_patterns=numpy.loadtxt(dir_data+prefix+"_Feature_Patterns.txt",dtype=str,delimiter="\t")
feature_patterns_matrix=numpy.loadtxt(dir_data+prefix+"_Feature_Patterns_Matrix.txt",dtype=bool,delimiter="\t")
classes=numpy.loadtxt(dir_data+prefix+"_Classes.txt",dtype=str,delimiter="\t")

print data.shape
print classes.shape

#classes_str=classes
unique_class_names,classes=cl.change_class_labels(classes)

prefix="simulated_data_stability_selection" + "_a_active_W="+str(a_active_W) + "_b_active_W="+str(b_active_W) + "_a_active_H="+str(a_active_H) + "_b_active_H="+str(b_active_H)
z=3
a_0s=[1e3,1e2,5e1,1e1,5,1,0.8,0.6,0.4,0.2,0.1,5e-2,1e-2,5e-3,1e-3]
#a_0s=[0.6]
b_0s=[10,1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7] # not used, if tied
a_larges=[1e3,1e2,5e1,1e1,5,1,0.8,0.6,0.4,0.2,0.1,5e-2,1e-2,5e-3,1e-3]
#a_larges=[0.4]
b_larges=[1e-2] # not used, if tied
a_small=1e2
b_small=1e-32
ab_tied=True
num_samplings=[20,40,60,80,100,150,200,400]
#num_samplings=[5,10]
prob_empiricals=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99]
#prob_empiricals=[0.8,0.9]
max_iter=1000
threshold_F=0.1
maxbc=12
rank_method="Wilcoxon_rank_sum_test"
key_feature_mean_feature_value_threshold=100
key_feature_neglog10pval_threshold=10
max_num_feature_each_pattern=3
compute_variational_lower_bound=True
variational_lower_bound_min_rate=1e-4
if_plot_lower_bound=False
if_plot_heatmap=False
a_H_test=0.1
b_H_test=1e-10
verb=False
rng=numpy.random.RandomState(100)
mat_ns=numpy.zeros(shape=(len(num_samplings),len(prob_empiricals)),dtype=float)
mat_pe=numpy.zeros(shape=(len(num_samplings),len(prob_empiricals)),dtype=float)
acc_es=numpy.zeros(shape=(len(num_samplings),len(prob_empiricals)),dtype=float)
acc_rs=numpy.zeros(shape=(len(num_samplings),len(prob_empiricals)),dtype=float)

for i in range(len(num_samplings)):
    ns=num_samplings[i]
    
    mss=ssmcnmf.ssmcnmf(X=data,y=classes,features=features)

    _,_,_,_=mss.stability_selection(z=z,a_0s=a_0s,b_0s=b_0s,a_larges=a_larges,b_larges=b_larges,a_small=a_small,b_small=b_small,ab_tied=ab_tied,mean_W=None,mean_H_large=1,mean_H_small=1e-32,num_samplings=ns,max_iter=max_iter, threshold_F=threshold_F, rank_method=rank_method, maxbc=12, compute_variational_lower_bound=compute_variational_lower_bound,variational_lower_bound_min_rate=variational_lower_bound_min_rate,if_plot_lower_bound=if_plot_lower_bound,if_plot_heatmap=if_plot_heatmap, a_H_test=a_H_test, b_H_test=b_H_test, dir_save=dir_save, prefix=prefix, verb=verb, rng=rng)

    #mss.save_mf_result(dir_save=dir_save,prefix=prefix)

    for j in range(len(prob_empiricals)):
        pe=prob_empiricals[j]
        print "num_sampling: {0}, prob_empirical: {1}".format(ns,pe)
        _,_=mss.sel_feat(prob_empirical=pe,rank_method=rank_method,maxbc=maxbc,key_feature_mean_feature_value_threshold=key_feature_mean_feature_value_threshold,key_feature_neglog10pval_threshold=key_feature_neglog10pval_threshold,max_num_feature_each_pattern=max_num_feature_each_pattern,header=unique_class_names)

        acc_e,acc_r=mss.compute_acc(feature_patterns_matrix)
        mat_ns[i,j]=ns
        mat_pe[i,j]=pe
        acc_es[i,j]=acc_e
        acc_rs[i,j]=acc_r

        #mss.save_feat(dir_save=dir_save,prefix=prefix,rank_method=rank_method,neglog10pval_threshold=neglog10pval_threshold,if_each_pattern_per_file=if_each_pattern_per_file)
        setting="num_sampling_"+str(ns)+"_prob_empirical_"+str(pe)
        mss.plot_heatmap(dir_save=dir_save,prefix=prefix+"_"+setting,pattern="All",rank_method="mean_feature_value",unique_class_names=unique_class_names, width=10, height=10, fontsize=8, fmt="png", dpi=600, colormap="hot", clrs=None, rng=rng)

        mss.plot_F_mean(dir_save=dir_save,prefix=prefix+"_"+setting,normalize=False,rank_method="mean_feature_value",unique_class_names=unique_class_names, width=10, height=10, fontsize=8, fmt="pdf", dpi=600, colormap="hot", clrs=None, rng=rng)

#plot 3D surface plot
cl.plot_3D_surface(numpy.array(mat_ns,dtype=int),mat_pe,acc_es,dir_save=dir_save,prefix=prefix+"_num_sampling_prob_empirical_vs_acc_e",figwidth=6,figheight=6,xlab="Number of Samplings",ylab="Empirical Probability",zlab="Accuracy",xyzlabel_fontsize=8,xyztick_fontsize=8,fmt="pdf",dpi=600)
cl.plot_3D_surface(numpy.array(mat_ns,dtype=int),mat_pe,acc_rs,dir_save=dir_save,prefix=prefix+"_num_sampling_prob_empirical_vs_acc_r",figwidth=6,figheight=6,xlab="Number of Samplings",ylab="Empirical Probability",zlab="Accuracy",xyzlabel_fontsize=8,xyztick_fontsize=8,fmt="pdf",dpi=600)

# save the acc in matrix
filename=dir_save+prefix+"_num_sampling_prob_empirical_vs_acc_e.txt"
numpy.savetxt(filename,acc_es,fmt="%.4f",delimiter="\t")
filename=dir_save+prefix+"_num_sampling_prob_empirical_vs_acc_r.txt"
numpy.savetxt(filename,acc_rs,fmt="%.4f",delimiter="\t")







