# SS-MV-NMF
from __future__ import division # In Python 3.0, // is the floor division operator and / the true division operator. The true_divide(x1, x2) function is equivalent to true division in Python.
import os
from shutil import copy
import numpy
import mvnmf
import classification as cl
import math
from scipy.stats import pearsonr
from scipy.stats import spearmanr

class ssmvnmf:
    def __init__(self,X,y,features):
        """
        X: numpy matrix, float type, each row is a feature, each column is a sample.
        y: numpy vector, integer type, the class labels.
        features: numpy vector, string type, the feature names.
        """
        self.X=X # data, each column is a sample
        self.y=y # class labels
        self.features=features
    
        
    def stability_selection(self,z=3,a_0s=[1e2,1e1,1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,1e-1,1e-2,1e-3],b_0s=[1e2,1e1,1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10],a_larges=[1e2,1e1,1,1e-1,1e-2],b_larges=[1e2,1e1,1,1e-1,1e-2],a_small=1e2,b_small=1e-30,ab_tied=True,mean_W=None,mean_H_large=1,mean_H_small=1e-32,num_samplings=1000,max_iter=200,
                            threshold_F=0.1,rank_method="Wilcoxon_rank_sum_test", maxbc=12, key_feature_mean_feature_value_threshold=1,key_feature_neglog10pval_threshold=10,max_num_feature_each_pattern=3,
                         compute_variational_lower_bound=False,variational_lower_bound_min_rate=1e-4,if_plot_lower_bound=False,if_plot_heatmap=False,
                         a_H_test=0.1,b_H_test=1e-10,dir_save="./",prefix="MVNMF_stability_selection",verb=False,rng=numpy.random.RandomState(1000)):
        """
        SS-MV-NMF to obtain the empirical probability matrix.
        
        INPUTS:
        z: integer, tuple of length V+1, list/numpy vector of size (K,), the labels of the factors (columns) in W. If z is a scalar, it means each view (including the ubi view) has z factors. If z is a tuple, e.g. (3,3,3,3), z[u] means the the u-th view has z[u] factors. If z is a list or numpy vector,e.g. [-1-1-1,0,0,1,1,1,2,2,2] where -1 means ubi view, z[k] means the k-th factor has label z[k].
        a_0s,b_0s, list of numpy vector, the predefined sets of the shape and rate parameters of Gamma for generating lambda of the exponential distribution of W.
        a_larges,b_larges: list or numpy vector, the predefined sets of the shape and rate parameters of Gamma for generating lambda of the exponential distribution of non-zero H blocks.
        a_small,b_small: float scalar, shape and rate parameters of Gamma for generating lambda of the exponential distribution of zero H blocks.
        ab_tied: bool, whether tie rate a to b. If True, b given in the input of this function is disregarded, and set b_0=mean_W*a_0, b_large=self.mean_H_large*a_large, b_small=self.mean_H_small*a_small.  
        mean_W: float scalar, the estimated mean value of W; If None, set it to mean(X).
        mean_H, float scalar, the estimated mean value of non-zero blocks in H; If None, set it to 1.
        mean_H_small: float scalar, the estimated eman value of zero blocks in H; If None, set it to 1e-32.
        num_samplings: integer, the number of samplings, i.e. the number of independent runs of MV-NMF.
        max_iter: integer, the maximal number of iterations allowed in MV-NMF.
        threshold_F: positive float scalar, used to generate the feature activity indicator matrix: F= F_mean >= (threshold_F * mean(F_mean)).
        rank_method: string, the method to rank the features within a pattern, can be one of {"mean_basis_value","mean_feature_value","Wilcoxon_rank_sum_test"}.
        maxbc: positive integer, the maximal number of views allowed to generate all possible binary codes as feature patterns.
        key_feature_mean_feature_value_threshold: positive integer, the lowest limit of the mean feature values when selecting key features.
        key_feature_neglog10pval_threshold: positive integer, the lower limit of the negative log10(pval) when selecting key features.
        max_num_feature_each_pattern: positive integer, the maximal number of key features allowed to be selected in each pattern. 
        compute_variational_lower_bound: bool, whether compute variational lower bounds.
        variational_lower_bound_min_rate: float, a tiny positive number, the threshold of the local mean change rate, below which the algorithm will terminate.
        if_plot_lower_bound: bool, whether plot the variational lower bound plot. 
        dir_save: string, path, e.g. "/home/yifeng/research/mf/mvmf_v1_1/results/", path to save the lower bound plot.
        perfix: string, prefix of the saved file name.
        verb: bool, whether plot the information for each iteration, including lower bound.
        rng: random number generator.

        OUTPUTS:
        E_W: numpy matrix, the expected basis matrix W.
        E_H: numpy matrix, the expected coefficient matrix H.
        training_time: total time spent including time of computing lower bound.
        training_time_L: time spent only for computing lower bound.
        """

        if isinstance(z,(list,tuple,numpy.ndarray)):
            self.z_str="".join(numpy.asarray(z,dtype=str))
        else:
            self.z_str=str(z)
        self.z=z

        self.a_0s=a_0s
        self.b_0s=b_0s
        self.a_larges=a_larges
        self.b_larges=b_larges
        self.a_small=a_small
        self.b_small=b_small
        self.max_iter=max_iter
        self.threshold_F=threshold_F
        self.compute_variational_lower_bound=compute_variational_lower_bound
        self.variational_lower_bound_min_rate=variational_lower_bound_min_rate
        self.if_plot_lower_bound=if_plot_lower_bound
        self.verb=verb
        
	self.training_times=[] # factorization time
        self.fs_times=[] # feature selection time
        self.training_times_L=[] # time computing lower bounds
        self.settings=[] # parameter settings
        self.E_W=0
        self.L_W=0
        self.F_mean=0

        all_combinations=[]
        if ab_tied:
            if mean_W is None:
                self.mean_X=numpy.mean(self.X)
                self.mean_W=self.mean_X
            else:
                self.mean_W=mean_W
            if mean_H_large is None:
                self.mean_H_large=1
            else:
                self.mean_H_large=mean_H_large
            if mean_H_small is None:
                self.mean_H_small=1e-32
            else:
                self.mean_H_small=mean_H_small
            b_smal=self.mean_H_small*a_small
            for a_0 in a_0s:
                for a_large in a_larges:
                    all_combinations.append((a_0,self.mean_W*a_0,a_large,self.mean_H_large*a_large))
        else:
            for a_0 in a_0s:
                for b_0 in b_0s:
                    for a_large in a_larges:
                        for b_large in b_larges:
                            all_combinations.append((a_0,b_0,a_large,b_large))
        
        for ns in range(num_samplings):
            print "The {0}-th run of MV-NMF...".format(ns)
            rng_ns=numpy.random.RandomState(ns)
            # sample data                    
            ind_cv=cl.kfold_cross_validation(self.y,k=2,shuffle=True,rng=rng_ns)
            ind_subsample=ind_cv==1
            X=self.X[:,ind_subsample]
            y=self.y[ind_subsample]
            # reorder the sampled data
            X,y,_=cl.sort_classes(numpy.transpose(X),y)
            X=numpy.transpose(X)
            # sample setting
            ind_setting=rng_ns.choice(len(all_combinations), size=1, replace=False)
            a_0,b_0,a_large,b_large=all_combinations[ind_setting]


            #a_small=a_small # when a_small>1, larger a_small, more symmetric lambda
            A=(a_large,a_small)
            #b_large=0.01 # when a>1 fixed, control rate of lambda, smaller b_large, larger lambda, wider lambda, narrower w
            #b_small=b_small # when a_small>1 fixed, control rate of lambda, smaller b_small, larger lambda, wider lambda, smaller h, narrower h
            B=(b_large,b_small)

            # current setting
            setting="a_0="+str(a_0)+"_b_0="+str(b_0)+"_a_large="+str(a_large)+"_b_large="+str(b_large)
            self.settings.append(setting)
            print "The current setting: " + setting

            # run the model
            self.model_fs=mvnmf.mvnmf(X,y,self.features)
            _,_,training_time,training_time_L=self.model_fs.factorize(z=z,a_0=a_0,b_0=b_0,A=A,B=B,max_iter=max_iter,compute_variational_lower_bound=compute_variational_lower_bound,variational_lower_bound_min_rate=variational_lower_bound_min_rate,if_plot_lower_bound=if_plot_lower_bound,dir_save=dir_save,prefix=prefix+"_"+setting,verb=verb,rng=rng)
            #trim_nonzero_portion=0.01
            #model_fs.trim(trim_nonzero_portion=trim_nonzero_portion,alpha=0.01,threshold_E_W=None,threshold_E_H=None)
            # feature selection, this procedure only need to have F, so no need to run the scoring procedures
            _,fs_time=self.model_fs.sel_feat(called_by_ssmvnmf_loop=True,threshold_F=threshold_F,rank_method=rank_method,maxbc=maxbc,key_feature_mean_feature_value_threshold=key_feature_mean_feature_value_threshold,key_feature_neglog10pval_threshold=key_feature_neglog10pval_threshold,max_num_feature_each_pattern=max_num_feature_each_pattern,header=numpy.unique(self.y),rng=rng)

            # record time
            self.training_times.append(round(training_time,2))
            self.fs_times.append(round(fs_time,2))
            self.training_times_L.append(round(training_time_L,2))

            # update E_W and F
            self.E_W=self.E_W + self.model_fs.E_W
            self.L_W=self.L_W + self.model_fs.L_W
            self.F_mean=self.F_mean + self.model_fs.F

            if if_plot_heatmap:
                self.model_fs.plot_heatmap(dir_save, prefix+"_"+setting, pattern="All",rank_method="mean_basis_value", unique_class_names=numpy.unique(self.y), width=10, height=10, fontsize=6, fmt="png",colormap="hot")
            
        # get average result
        self.E_W=self.E_W/num_samplings
        self.L_W=self.L_W/num_samplings
        # get empirical probability
        self.F_mean=self.F_mean/num_samplings
        # update the corresponding variables in model_fs
        self.model_fs.E_W=self.E_W
        self.model_fs.L_W=self.L_W
        self.model_fs.X=self.X
        self.model_fs.y=self.y
        self.model_fs.Y,self.model_fs.y_unique=cl.membership_vector_to_indicator_matrix(self.y)
        self.model_fs.N=len(self.y)     
        # final update E_H
        self.model_fs.learn_H_given_X_test_and_E_W(self.X,a_H_test=a_H_test,b_H_test=b_H_test,feature_selection=False,max_iter=max_iter,compute_variational_lower_bound=compute_variational_lower_bound,variational_lower_bound_min_rate=variational_lower_bound_min_rate,if_plot_lower_bound=if_plot_lower_bound,dir_save=dir_save,prefix=prefix+"_final_update_E_H",verb=verb,rng=rng)
        self.model_fs.E_H=self.model_fs.E_H_test
        self.model_fs.E_H_test=None
        self.E_H=self.model_fs.E_H
        # update time record
        self.model_fs.training_time=numpy.sum(self.training_times)+self.model_fs.test_time
        self.model_fs.training_time_L=numpy.sum(self.training_times_L)+self.model_fs.test_time_L
        self.model_fs.test_time=0
        self.model_fs.test_time_L=0
        
        #self.trim(trim_nonzero_portion=0.01,alpha=0.05,threshold_E_W=None,threshold_E_H=None) # trim if necessary

        # store settings
        self.settings=numpy.array(self.settings)
        print "Finished stability selection :)"
        return self.E_W,self.E_H,self.model_fs.training_time,self.model_fs.training_time_L
        

    def trim(self,trim_nonzero_portion=0.01,alpha=0.05,threshold_E_W=None,threshold_E_H=None):
        """
        Trim zero columns of E_W and corresponding rows of E_H.
        Still under investigation. Leave for future improvement.
        """
        self.model_fs.trim(trim_nonzero_portion=trim_nonzero_portion,alpha=alpha,threshold_E_W=threshold_E_W,threshold_E_H=threshold_E_H)


    def save_mf_result(self,dir_save="./",prefix="SSMVNMF"):
        """
        Save matrix factorization results for the training procedure of SS-MV-NMF.

        INPUTS:
        dir_save: string, path to save results, e.g. "/home/yifeng/research/mf/mvmf_v1_1/".
        prefix: string, prefix of file names.
        
        OUTPUTS:
        This function does not explicitly return any variable.
        """
        self.model_fs.save_mf_result(dir_save,prefix)
        # save times and settings
        training_times=numpy.array(self.training_times,dtype=str)
        training_times.shape=(len(self.training_times),1)
        training_times_L=numpy.array(self.training_times_L,dtype=str)
        training_times_L.shape=(len(self.training_times_L),1)
        settings=numpy.array(self.settings,dtype=str)
        settings.shape=(len(self.settings),1)
        settings_training_times=numpy.hstack((settings,training_times,training_times_L))
        filename=dir_save + prefix + "_settings_training_times_training_time_L.txt"
        numpy.savetxt(filename,settings_training_times,fmt='%s', delimiter='\t', newline='\n',header="setting\ttraing_time\ttraining_time_L")


    def sel_feat(self,prob_empirical=0.8,use_previous_scores=False,allow_compute_pval=True,rank_method="Wilcoxon_rank_sum_test",maxbc=12,patterns_for_feat_sel="all",key_feature_mean_feature_value_threshold=100,key_feature_neglog10pval_threshold=10,max_num_feature_each_pattern=10,max_num_feature=None,header=None,rng=numpy.random.RandomState(1000)):
        """
        Discover feature patterns using SS-MV-NMF.

        INPUTS:
        prob_empirical: float in (0,1], threshold on the empirical probability matrix, above which active feature is defined.
        use_previous_scores: bool, whether use the scores already computed in a previous call of sel_feat().
        allow_compute_pval: bool, whether computer the Wilcoxon rank sum test score. 
        rank_method: string, the method to rank the features within a pattern, can be one of {"mean_basis_value","mean_feature_value","Wilcoxon_rank_sum_test"}. 
        maxbc: positive integer, the maximal number of views allowed to generate all possible binary codes as feature patterns.
        patterns_for_feat_sel, string, in which views to select key features for building up predictive model, can be one of {"all","view-specific","top_from_all"}.
        key_feature_mean_feature_value_threshold: positive integer, the lowest limit of the mean feature values when selecting key features.
        key_feature_neglog10pval_threshold: positive integer, the lower limit of the negative log10(pval) when selecting key features.
        max_num_feature_each_pattern: positive integer, the maximal number of key features allowed to be selected in each pattern. 
        max_num_feature: positive integer, the maximal number of total key features allowed to be selected. If exceeded, randomly pick up max_num_feature key features.
        header: names of the views, including the ubiquitous view. 
        rng: random number generator.

        OUTPUTS:
        numeric_ind_key_feat_for_classification: list, the numeric indices of selected key features for building up predictive models.
        fs_time: time taken to detect feature patterns.
        """
        # final feature selection
        if header is None:
            header=numpy.unique(self.y)
        self.F=self.F_mean>prob_empirical
        self.model_fs.F_mean=self.F_mean
        self.numeric_ind_key_feat_for_classification,self.fs_time=self.model_fs.sel_feat(F=self.F,threshold_F=None,use_previous_scores=use_previous_scores,allow_compute_pval=allow_compute_pval,rank_method=rank_method,maxbc=maxbc,patterns_for_feat_sel=patterns_for_feat_sel,key_feature_mean_feature_value_threshold=key_feature_mean_feature_value_threshold,key_feature_neglog10pval_threshold=key_feature_neglog10pval_threshold,max_num_feature_each_pattern=max_num_feature_each_pattern,max_num_feature=max_num_feature,header=header,rng=rng)
        return self.numeric_ind_key_feat_for_classification,self.fs_time

    
    def save_feat(self,dir_save="./",prefix="SSMVNMF",rank_method="Wilcoxon-rank-sum-test",neglog10pval_threshold=10,if_each_pattern_per_file=False):
        """
        Save feature patterns.
        
        INPUTS: 
        dir_save: string, path to save related files, e.g. "/home/yifeng/research/mf/mvmf_v1_1/".
        prefix: string, prefix of file names.
        rank_method: string, the ranking method for sort features in each feature pattern, can be one of {"mean_feature_value","mean_basis_value","Wilcoxon_rank_sum_test"}.
        if_each_pattern_per_file: bool scalar, indicate whether save a file for each feature pattern. If True, all the features and significant features are saved separately in two .txt files.
        neglog10pval: float scalar, only valid when if_each_pattern_per_file==True, the threshold, on Wilcoxon_rank_sum_test scores, above which a feature is defined as significant in a pattern. 
        
        OUTPUTS:
        This function does not explicitly return any variable. 
        """
        self.model_fs.save_feat(dir_save=dir_save,prefix=prefix,rank_method=rank_method,neglog10pval_threshold=neglog10pval_threshold,if_each_pattern_per_file=if_each_pattern_per_file)
        
        
    def plot_heatmap(self,dir_save="./",prefix="SSMVNMF",pattern="All",rank_method="mean_basis_value",unique_class_names=None, width=10, height=10, fontsize=6, fmt="png", dpi=600, colormap="hot", clrs=None, rng=numpy.random.RandomState(1000)):
        """
        Plot heatmaps of the decomposition result, given a pattern or all patterns. 

        INPUTS:
        dir_save: string, path to save the plotted heatmap, e.g. "/home/yifeng/research/mf/mvmf_v1_1/".
        prefix: string, the prefix of the file name.  
        pattern: string, specify the feature pattern to be ploted. Suppose there are 4 classes/views, pattern can be "All", "1_0_0_0", "1_1_0_1", ..., "1_1_1_1".
        rank_method: string, the ranking method to sort feature within a feature pattern. It can be one of {"mean_feature_value","mean_basis_value","Wilcoxon_rank_sum_test"}.
        unique_class_names: numpy vector of strings, the names of all classes. If None, [0,1,2,...,V-1] will be used.
        width: flot scalar, the width of figure in inches.
        height: float scalar, the height of figure in inches.
        fontsize: the font size of text.
        fmt: string, the format of the figure file to be saved, can be one of {"png","pdf"}, or other formats acceptable by fig.savefig().
        dpi: integer scalar, the DPI of the figure to be saved.
        colormap: string, the color map used in the heatmaps.
        clrs: list of numpy vector, the colors used for individual classes.
        rng: random number generator.

        OUTPUTS:
        This function does not explicitly return any variable.
        """
        self.model_fs.plot_heatmap(dir_save=dir_save, prefix=prefix, pattern=pattern,rank_method=rank_method, unique_class_names=unique_class_names, width=width, height=height, fontsize=fontsize, fmt=fmt, dpi=dpi, colormap=colormap,clrs=clrs,rng=rng)

    def plot_F_mean(self,dir_save="./",prefix="SSMVNMF",normalize=False,rank_method="mean_basis_value",unique_class_names=None, width=6, height=6, fontsize=6, fmt="png", dpi=600, colormap="hot", clrs=None, rng=numpy.random.RandomState(1000)):
        """
        Plot the feature activity matrix F_mean which is particularly informative for MV-NMV with stability selection (SS-MV-NMF).
        
        INPUTS:
        dir_save: string, path to save the file, e.g. "/home/yifeng/research/mf/mvmf_v1_1/".
        prefix: string, prefix of the file to be saved.
        normalize: bool scalar, whether normalize the matrix for better visualization.
        rank_method: string, ranking method to sort features within each pattern, can be one of {"mean_feature_value","mean_basis_value","Wilcoxon_rank_sum_test"}.
        unique_class_names: list or numpy vector of strings, the class names.
        width: float scalar, width of figure to be plotted.
        height: float scalar, height of figure to be plotted.
        fontsize: positive integer, font size of text.
        fmt: string, the format of the figure file to be saved, can be one of {"png","pdf"}, or other formats acceptable by fig.savefig().
        dpi: positive integer, DPI of the figure.
        colormap: string, color map. 
        clrs: list of numpy vector, colors of the classes.
        rng: random number generator.

        OUTPUTS:
        This function does not explicitly return any variable.
        """
        self.model_fs.plot_F_mean(dir_save=dir_save, prefix=prefix, normalize=normalize,rank_method=rank_method, unique_class_names=unique_class_names, width=width, height=height, fontsize=fontsize, fmt=fmt, dpi=dpi, colormap=colormap,clrs=clrs,rng=rng)
       
    def compute_acc(self,feature_patterns_matrix):
        """
        Compute element-wise-match accuracy and row-wise-match accuracy given the real pattern of feature patterns in a matrix.
        
        INPUTS:
        feature_patterns_matrix: numpy matrix of integers, the actural feature patterns.

        OUTPUTS:
        acc_e: float, element-wise match accuracy.
        acc_r: float, row-wise match accuracy.
        """
        return self.model_fs.compute_acc(feature_patterns_matrix)

    def classify(self,feature_selection=True,method="coef_sum",method_param={"feature_extraction":False,"learning_rate":0.1, "alpha":0.01, "lambda_reg":0.00001, "alpha_reg":0.5, "n_hidden":None, "n_epochs":1000, "batch_size":200, "activation_func":"relu"},rng=numpy.random.RandomState(1000)):
        """
        Predict the class labels of provided test samples.
        
        INPUTS:
        feature_selection: bool, indicate whether feature selection is applied before classification.  
        method: string, method used for classification, can be "coef_sum","regression_residual", or "mlp".
        method_param: dict, parameters for specific classification method; if method_param["feature_extraction"]==True: the extracted new features are used, instead of original features, for classification in MLP; if method_param["n_hidden"]==None, one hidden layer with the same number of units as the number of input features is used; [] for zero hidden layer.
        rng: random number generator.
        
        OUTPUTS:
        test_classes_predicted: numpy vector, the predicted class labels coded by {0,1,2,...,V-1}.
        test_classes_prob_like: numpy vector, the maximal posterial probabilities.
        classification_training_time: float, time in seconds spent in the training of classification. 
        classification_test_time: float, time in seconds spent in the test of classification.
        """
        self.test_classes_predicted,self.test_classes_prob_like,self.classification_training_time,self.classification_test_time=self.model_fs.classify(feature_selection=feature_selection,method=method,method_param=method_param,rng=rng)
        return self.test_classes_predicted,self.test_classes_prob_like,self.classification_training_time,self.classification_test_time

    def learn_H_given_X_test_and_E_W(self,X_test,a_H_test=0.1,b_H_test=1e-10,feature_selection=True,max_iter=200,compute_variational_lower_bound=True,variational_lower_bound_min_rate=1e-4,if_plot_lower_bound=True,dir_save="./",prefix="SSMVNMF",verb=True,rng=numpy.random.RandomState(1000)):
        """
        Learn the coefficient matrix given test data X_test and learned E_W. This function is useful for MV-NMF based feature extraction or classification. Not working well with SS-MV-NMF, because E_W is a combined matrix.
        
        INPUTS:
        X_test: numpy matrix, each column is a test sample, each row is a feature.
        a_H_test, b_H_test: floats, shape of rates of Gamma distribution applied to generating lambda for the exponential distrution of H_test.
        feature_selection: bool, whether the input features are reduced first before this learning. If True, the selected features from the training of MV-NMF are used only.
        max_iter: integer, the maximal number of iterations.
        compute_variational_lower_bound: whether compute the variational lower bound.
        variational_lower_bound_min_rate: tiny positive float, the threshold on the local mean change rate in order to terminate the algorithm.
        if_plot_lower_bound: bool, whether plot and save the variational lower bound.
        dir_save: string, path to save the lower bound plot, e.g. "/home/yifeng/research/mf/mvmf_v1_1/".
        prefix: string, prefix of the file name to save the lower bound plot.
        verb: bool, whether print information for each iteration.
        rng: random number genrator.

        OUTPUTS:
        E_H_test: numpy matrix, the mean coefficient matrix of the test data.
        test_time: float, the overall test time in seconds, including time for lower bound.
        test_time_L: float, the time in seconds only for lower bound.
        """
        self.E_H,self.test_time,self.test_time_L=self.model_fs.learn_H_given_X_test_and_E_W(X_test=X_test,a_H_test=a_H_test,b_H_test=b_H_test,feature_selection=feature_selection,max_iter=max_iter,compute_variational_lower_bound=compute_variational_lower_bound,variational_lower_bound_min_rate=variational_lower_bound_min_rate,if_plot_lower_bound=if_plot_lower_bound,dir_save=dir_save,prefix=prefix,verb=verb,rng=rng)
        return self.E_H,self.test_time,self.test_time_L

