
import os
from shutil import copy
import numpy
import ssmcnmf
import classification as cl

dir_work="/home/yifeng/research/mf/mvmf_v1_1/"
dir_data="/home/yifeng/research/mf/mvmf_v1_1/data/"

rng=numpy.random.RandomState(100)

cancers="brca_coad_gbm_hnsc_kirc_lgg_lihc_luad_lusc_ov_prad_normal"

data=numpy.loadtxt(dir_data+"tcga_mrnaseq_"+cancers+"_data_normalized.txt",dtype=float,delimiter="\t")
features=numpy.loadtxt(dir_data+"tcga_mrnaseq_"+cancers+"_features.txt",dtype=str,delimiter="\t")
classes=numpy.loadtxt(dir_data+"tcga_mrnaseq_"+cancers+"_classes.txt",dtype=str,delimiter="\t")

print data.shape
print classes.shape

classes_str=classes
unique_class_names,classes=cl.change_class_labels(classes)
print classes
print unique_class_names
data,classes,classes_str=cl.sort_classes(numpy.transpose(data),classes,classes_str)
print classes
print data.shape

# split data
train_set_x,train_set_y,train_set_ystr,valid_set_x,valid_set_y,valid_set_ystr,test_set_x,test_set_y,test_set_ystr=cl.partition_train_valid_test2(data, classes, classes_str, ratio=(2,0,1), rng=rng)
# put samples of the same class together
train_set_x,train_set_y,train_set_ystr=cl.sort_classes(train_set_x,train_set_y,train_set_ystr)
train_set_x=numpy.transpose(train_set_x)
test_set_x=numpy.transpose(test_set_x)

################################### feature selection ########################################
#z=[-1,-1,-1,0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5]
z=5
if isinstance(z,(list,tuple,numpy.ndarray)):
    z_str="".join(numpy.asarray(z,dtype=str))
else:
    z_str=str(z)
a_0s=[1e3,1e2,5e1,1e1,5,1,0.8,0.6,0.4,0.2,0.1,5e-2,1e-2,5e-3,1e-3]
b_0s=[10,1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7] # not used, if ab_tied
a_larges=[1e3,1e2,5e1,1e1,5,1,0.8,0.6,0.4,0.2,0.1,5e-2,1e-2,5e-3,1e-3]
b_larges=[1e-2] # not used, if ab_tied
a_small=1e2
b_small=1e-32
ab_tied=True
num_sampling=100
prob_empirical=0.85
max_iter=200
threshold_F=0.1
maxbc=10
patterns_for_feat_sel="view_specific"
allow_compute_pval=True
rank_method="Wilcoxon_rank_sum_test"
key_feature_mean_feature_value_threshold=100
key_feature_neglog10pval_threshold=20
neglog10pval_threshold=20
max_num_feature_each_pattern_s=[1,2,3,4,5,6,7,8,9,10]
compute_variational_lower_bound=True
variational_lower_bound_min_rate=1e-4
if_plot_lower_bound=False
if_plot_heatmap=False
a_H_test=0.1
b_H_test=1e-10
verb=False
rng=numpy.random.RandomState(100)
if_each_pattern_per_file=False

prefix="ssmcnmf_rnaseq_"+"_".join(unique_class_names)

# create a folder for the results
dir_save=dir_work+"results/"+"prefix/"
if not os.path.exists(dir_save):
    os.makedirs(dir_save)
# make a copy of the script in the directory
scr=dir_work+"main_ssmcnmf_rnaseq.py"
copy(scr, dir_save)

mss=ssmcnmf.ssmcnmf(X=train_set_x,y=train_set_y,features=features)

_,_,training_time,training_time_L=mss.stability_selection(z=z,a_0s=a_0s,b_0s=b_0s,a_larges=a_larges,b_larges=b_larges,a_small=a_small,b_small=b_small,ab_tied=ab_tied,mean_W=None,mean_H_large=1,mean_H_small=1e-32,num_samplings=num_sampling,max_iter=max_iter, threshold_F=threshold_F, rank_method=rank_method, maxbc=maxbc, compute_variational_lower_bound=compute_variational_lower_bound,variational_lower_bound_min_rate=variational_lower_bound_min_rate,if_plot_lower_bound=if_plot_lower_bound,if_plot_heatmap=if_plot_heatmap, a_H_test=a_H_test, b_H_test=b_H_test, dir_save=dir_save, prefix=prefix, verb=verb, rng=rng)

#mss.trim(trim_nonzero_portion=0.001,alpha=0.05,threshold_E_W=None,threshold_E_H=None)
use_previous_scores=False
for max_num_feature_each_pattern in max_num_feature_each_pattern_s:
    _,fs_time=mss.sel_feat(prob_empirical=prob_empirical,use_previous_scores=use_previous_scores,allow_compute_pval=allow_compute_pval,rank_method=rank_method,maxbc=maxbc,patterns_for_feat_sel=patterns_for_feat_sel,key_feature_mean_feature_value_threshold=key_feature_mean_feature_value_threshold,key_feature_neglog10pval_threshold=key_feature_neglog10pval_threshold,max_num_feature_each_pattern=max_num_feature_each_pattern,header=unique_class_names)

    mss.save_feat(dir_save=dir_save,prefix=prefix+"_max_num_feature_each_pattern_"+str(max_num_feature_each_pattern),rank_method=rank_method,neglog10pval_threshold=neglog10pval_threshold,if_each_pattern_per_file=if_each_pattern_per_file)
    use_previous_scores=True

    #mss.save_mf_result(dir_save=dir_save,prefix=prefix)

    ####################################### test ################################
    feature_selection=True
    _,test_time,test_time_L=mss.learn_H_given_X_test_and_E_W(X_test=test_set_x,a_H_test=a_H_test,b_H_test=b_H_test,feature_selection=feature_selection,max_iter=max_iter,compute_variational_lower_bound=compute_variational_lower_bound,variational_lower_bound_min_rate=variational_lower_bound_min_rate,if_plot_lower_bound=if_plot_lower_bound,dir_save=dir_save,prefix=prefix,verb=verb,rng=rng)


    # classification using selected features
    classify_method="mlp"
    feature_selection=True
    method_param={"feature_extraction":False,"learning_rate":0.1, "alpha":0.01, "lambda_reg":0.00001, "alpha_reg":0.5, "n_hidden":None, "n_epochs":1000, "batch_size":100, "activation_func":"relu"}
    test_set_y_pred,test_set_y_prob,classification_training_time,classification_test_time=mss.classify(feature_selection=feature_selection,method=classify_method,method_param=method_param,rng=rng)
    #print test_set_y_prob
    perf,conf_mat=cl.perform(test_set_y,test_set_y_pred,numpy.unique(train_set_y))
    print perf
    print conf_mat
    # save classification result
    filename=prefix+"_classification_performance_"+ classify_method  + "_feat_sel_"+str(feature_selection)+"_max_num_feature_each_pattern_"+str(max_num_feature_each_pattern)+"_num_feat_sel_"+str(len(mss.numeric_ind_key_feat_for_classification))+".txt"
    cl.save_perform(dir_save,filename,perf=perf,std=None,conf_mat=conf_mat,classes_unique=unique_class_names,training_time=classification_training_time,test_time=classification_test_time)

mss.plot_heatmap(dir_save=dir_save,prefix=prefix+"_300dpi",pattern="All",rank_method="mean_basis_value",unique_class_names=unique_class_names, width=10, height=10, fontsize=4, fmt="png", dpi=300, colormap="hot", clrs=None, rng=rng)













