from __future__ import division # In Python 3.0, // is the floor division operator and / the true division operator. The true_divide(x1, x2) function is equivalent to true division in Python.
import os
from shutil import copy
import numpy
import mvnmf
import classification as cl

# change the following path to your own
dir_work="/home/yifeng/research/mf/mvmf_v1_1/"
dir_data="/home/yifeng/research/mf/mvmf_v1_1/data/"

rng=numpy.random.RandomState(100)

# load simulated data
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

# change class labels from {1,2,3} to {0,1,2}
#classes_str=classes
unique_class_names,classes=cl.change_class_labels(classes)

#print classes
#print unique_class_names
#data,classes,classes_str=cl.sort_classes(numpy.transpose(data),classes,classes_str)
#print classes
#print data.shape

z=3

if isinstance(z,(list,tuple,numpy.ndarray)):
    z_str="".join(numpy.asarray(z,dtype=str))
else:
    z_str=str(z)
a_0=0.6 
b_0=a_0*numpy.mean(data) # tie b_0
a_large=0.4
a_small=100 
A=(a_large,a_small)
b_large=a_large # tie b_large
b_small=1e-62
B=(b_large,b_small)

# create a folder for saving the results
dir_save=dir_work+"results/"+ prefix + "_a_0=" + str(a_0) + "_b_0=" + str(b_0) + "_a_large=" + str(a_large) + "_a_small=" + str(a_small) + "_b_large=" + str(b_large) + "_b_small=" + str(b_small) + "_z=" + z_str +"/"
if not os.path.exists(dir_save):
    os.makedirs(dir_save)
# make a copy of the script in the directory
scr=dir_work+"main_mvnmf_sim.py"
copy(scr, dir_save)

max_iter=200

# initialize the mvnmf object
model_fs=mvnmf.mvnmf(data,classes,features)
# factorization
_,_,training_time,training_time_L=model_fs.factorize(z=z,a_0=a_0,b_0=b_0,A=A,B=B,max_iter=max_iter,compute_variational_lower_bound=True,variational_lower_bound_min_rate=1e-4,if_plot_lower_bound=True,dir_save=dir_save,prefix=prefix,verb=False,rng=rng)
trim_nonzero_portion=0.01
# trim zero factors
#model_fs.trim(trim_nonzero_portion=trim_nonzero_portion,alpha=0.01,threshold_E_W=None,threshold_E_H=None)
threshold_F=0.1
# find feature patterns
_,fs_time=model_fs.sel_feat(threshold_F=threshold_F,rank_method="Wilcoxon_rank_sum_test",maxbc=12,key_feature_mean_feature_value_threshold=1,key_feature_neglog10pval_threshold=10,max_num_feature_each_pattern=3,header=unique_class_names)
# save feature patterns
model_fs.save_feat(dir_save,prefix,rank_method="Wilcoxon_rank_sum_test",neglog10pval_threshold=20)
# save factorization results
model_fs.save_mf_result(dir_save,prefix)
# compute RWM and EWM accuracies
acc_e,acc_r=model_fs.compute_acc(feature_patterns_matrix)
#acc_r=numpy.sum(feature_patterns==model_fs.F_str)/model_fs.M
#acc_e=numpy.sum(feature_patterns_matrix==model_fs.F)/(model_fs.M*(model_fs.V+1))
#print "The row-wise-match accuracy of features are:{0}".format(acc_r)
#print "The element-wise-match accuracy of features are:{0}".format(acc_e)
# draw heatmap
model_fs.plot_heatmap(dir_save, prefix, pattern="All",rank_method="mean_basis_value", unique_class_names=unique_class_names, width=10, height=10, fontsize=6, fmt="png",colormap="hot")











