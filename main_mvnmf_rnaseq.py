
import os
from shutil import copy
import numpy
import mvnmf
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
#data=numpy.transpose(data)
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

a_0=0.1 
b_0=1e-10 
a_large=10
a_small=10 
A=(a_large,a_small)
b_large=0.1 
b_small=1e-30
B=(b_large,b_small)

# parameters for test 
a_H_test=0.1 
b_H_test=1e-5

prefix="mvnmf_classification_rnaseq_"+"_".join(unique_class_names)

# create a folder for the results
dir_save=dir_work+"/results/"+ prefix + "_a_0=" + str(a_0) + "_b_0=" + str(b_0) + "_a_large=" + str(a_large) + "_a_small=" + str(a_small) + "_b_large=" + str(b_large) + "_b_small=" + str(b_small) + "_z=" + z_str + "_a_H_test="+ str(a_H_test) + "_b_H_test="+ str(b_H_test) +"/"
if not os.path.exists(dir_save):
    os.makedirs(dir_save)
# make a copy of the script in the directory
scr=dir_work+"main_mvnmf_rnaseq.py"
copy(scr, dir_save)

max_iter=200

model_fs=mvnmf.mvnmf(train_set_x,train_set_y,features)
_,_,training_time,training_time_L=model_fs.factorize(z=z,a_0=a_0,b_0=b_0,A=A,B=B,max_iter=max_iter,compute_variational_lower_bound=True,variational_lower_bound_min_rate=1e-4,if_plot_lower_bound=True,dir_save=dir_save,prefix=prefix,rng=rng)
#trim_nonzero_portion=0.01
#model_fs.trim(trim_nonzero_portion=trim_nonzero_portion,alpha=0.01,threshold_E_W=None,threshold_E_H=None)
threshold_F=1e-1
_,fs_time=model_fs.sel_feat(threshold_F=threshold_F,rank_method="Wilcoxon_rank_sum_test",key_feature_mean_feature_value_threshold=100,key_feature_neglog10pval_threshold=20,max_num_feature_each_pattern=10,header=unique_class_names)
model_fs.save_feat(dir_save,prefix,rank_method="Wilcoxon_rank_sum_test",neglog10pval_threshold=20)
model_fs.save_mf_result(dir_save,prefix)
model_fs.plot_heatmap(dir_save, prefix, pattern="All",rank_method="mean_basis_value", unique_class_names=unique_class_names, width=10, height=10, fontsize=3, fmt="png",colormap="hot")

####################################### test (Bayesian non-negative regression) ################################
_,test_time,test_time_L=model_fs.learn_H_given_X_test_and_E_W(X_test=test_set_x,a_H_test=a_H_test,b_H_test=b_H_test,feature_selection=True,max_iter=max_iter,compute_variational_lower_bound=True,variational_lower_bound_min_rate=1e-4,if_plot_lower_bound=True,dir_save=dir_save,prefix=prefix,verb=True,rng=rng)
model_fs.save_mf_test_result(dir_save,prefix)

###################################### classification #########################################
# classification using coefficients
classify_method="coef_sum"
feature_selection=True
test_set_y_pred,test_set_y_prob,classification_training_time,classification_test_time=model_fs.classify(feature_selection=feature_selection,method=classify_method,method_param=None,rng=rng)
#print test_set_y_prob
perf,conf_mat=cl.perform(test_set_y,test_set_y_pred,numpy.unique(train_set_y))
print perf
print conf_mat
# save classification result
filename=prefix+"_classification_performance_"+ classify_method  + "_feat_sel_"+str(feature_selection)+".txt"
cl.save_perform(dir_save,filename,perf=perf,std=None,conf_mat=conf_mat,classes_unique=unique_class_names,training_time=classification_training_time,test_time=classification_test_time)

# classification using the class-wise regression residuals
classify_method="regression_residual"
feature_selection=True
test_set_y_pred,test_set_y_prob,classification_training_time,classification_test_time=model_fs.classify(feature_selection=feature_selection,method=classify_method,method_param=None,rng=rng)
#print test_set_y_prob
perf,conf_mat=cl.perform(test_set_y,test_set_y_pred,numpy.unique(train_set_y))
print perf
print conf_mat
# save classification result
filename=prefix+"_classification_performance_"+ classify_method  + "_feat_sel_"+str(feature_selection)+".txt"
cl.save_perform(dir_save,filename,perf=perf,std=None,conf_mat=conf_mat,classes_unique=unique_class_names,training_time=classification_training_time,test_time=classification_test_time)

# classification using selected features
classify_method="mlp"
feature_selection=True
method_param={"feature_extraction":False,"learning_rate":0.1, "alpha":0.01, "lambda_reg":0.00001, "alpha_reg":0.5, "n_hidden":None, "n_epochs":1000, "batch_size":100, "activation_func":"relu"}
test_set_y_pred,test_set_y_prob,classification_training_time,classification_test_time=model_fs.classify(feature_selection=feature_selection,method=classify_method,method_param=method_param,rng=rng)
#print test_set_y_prob
perf,conf_mat=cl.perform(test_set_y,test_set_y_pred,numpy.unique(train_set_y))
print perf
print conf_mat
# save classification result
filename=prefix+"_classification_performance_"+ classify_method  + "_feat_sel_"+str(feature_selection)+".txt"
cl.save_perform(dir_save,filename,perf=perf,std=None,conf_mat=conf_mat,classes_unique=unique_class_names,training_time=classification_training_time,test_time=classification_test_time)

# classification using extracted new features
classify_method="mlp"
feature_selection=True
method_param={"feature_extraction":True,"learning_rate":0.1, "alpha":0.01, "lambda_reg":0.00001, "alpha_reg":0.5, "n_hidden":None, "n_epochs":1000, "batch_size":100, "activation_func":"relu"}
test_set_y_pred,test_set_y_prob,classification_training_time,classification_test_time=model_fs.classify(feature_selection=feature_selection,method=classify_method,method_param=method_param,rng=rng)
#print test_set_y_prob
perf,conf_mat=cl.perform(test_set_y,test_set_y_pred,numpy.unique(train_set_y))
print perf
print conf_mat
# save classification result
filename=prefix+"_classification_performance_"+ classify_method  + "_feat_sel_"+str(feature_selection)+"_feature_extraction.txt"
cl.save_perform(dir_save,filename,perf=perf,std=None,conf_mat=conf_mat,classes_unique=unique_class_names,training_time=classification_training_time,test_time=classification_test_time)

# classification using extracted new features
classify_method="mlp"
feature_selection=False
test_set_y_pred,test_set_y_prob,classification_training_time,classification_test_time=model_fs.classify(feature_selection=feature_selection,method=classify_method,method_param={"feature_extraction":True,"learning_rate":0.1, "alpha":0.01, "lambda_reg":0.00001, "alpha_reg":0.5, "n_hidden":None, "n_epochs":1000, "batch_size":100, "activation_func":"relu"},rng=rng)
#print test_set_y_prob
perf,conf_mat=cl.perform(test_set_y,test_set_y_pred,numpy.unique(train_set_y))
print perf
print conf_mat
# save classification result
filename=prefix+"_classification_performance_"+ classify_method  + "_feat_sel_"+str(feature_selection)+"_feature_extraction.txt"
cl.save_perform(dir_save,filename,perf=perf,std=None,conf_mat=conf_mat,classes_unique=unique_class_names,training_time=classification_training_time,test_time=classification_test_time)


