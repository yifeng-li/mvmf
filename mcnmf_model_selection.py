
from __future__ import division # In Python 3.0, // is the floor division operator and / the true division operator. The true_divide(x1, x2) function is equivalent to true division in Python.
import os
from shutil import copy
import numpy
import mcnmf
import classification as cl
import math
from scipy.stats import pearsonr
from scipy.stats import spearmanr

class mcnmf_model_selection:
    def __init__(self,X,y,features,feature_patterns=None,feature_patterns_matrix=None):
        
        self.X=X # data, each column is a sample
        self.y=y # class labels
        self.features=features
        self.feature_patterns=feature_patterns
        self.feature_patterns_matrix=feature_patterns_matrix
    
        
    def search(self,z=3,a_0s=[1e2,1e1,1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,1e-1,1e-2,1e-3],b_0s=[1e2,1e1,1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10],a_larges=[1e2,1e1,1,1e-1,1e-2],b_larges=[1e2,1e1,1,1e-1,1e-2],a_small=1e2,b_small=1e-30,ab_tied=True,mean_W=None,mean_H_large=1,mean_H_small=1e-32,max_iter=200,threshold_F=0.1,compute_variational_lower_bound=False,variational_lower_bound_min_rate=1e-4,if_plot_lower_bound=False,if_plot_heatmap=False,dir_save="./",prefix="mcnmf_model_search",verb=False,rng=numpy.random.RandomState(1000)):

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
        
        self.acc_best=0
        self.a_0_best=0
        self.b_0_best=0
        self.a_large_best=0
        self.b_large_best=0
        self.setting_best=""
	self.times_fact_fs=[] # factorization and feature selection time
        self.settings=[] # parameter settings
        self.accs_r=[] # row-match accuracy
        self.accs_e=[] # element-match accuracy
        self.klds=[] # KL divergences

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

        for a_0,b_0,a_large,b_large in all_combinations:
            #a_small=a_small # when a_small>1, larger a_small, more symmetric lambda
            A=(a_large,a_small)
            #b_large=0.01 # when a>1 fixed, control rate of lambda, smaller b_large, larger lambda, wider lambda, narrower w
            #b_small=b_small # when a_small>1 fixed, control rate of lambda, smaller b_small, larger lambda, wider lambda, smaller h, narrower h
            B=(b_large,b_small)

            # current setting
            setting="a_0="+str(a_0)+"_b_0="+str(b_0)+"_a_large="+str(a_large)+"_b_large="+str(b_large)
            self.settings.append(setting)

            # run the model
            model_fs=mcnmf.mcnmf(self.X,self.y,self.features)
            _,_,training_time,training_time_L=model_fs.factorize(z=z,a_0=a_0,b_0=b_0,A=A,B=B,max_iter=max_iter,compute_variational_lower_bound=compute_variational_lower_bound,variational_lower_bound_min_rate=variational_lower_bound_min_rate,if_plot_lower_bound=if_plot_lower_bound,dir_save=dir_save,prefix=prefix+"_"+setting,verb=verb,rng=rng)
            trim_nonzero_portion=0.01
            model_fs.trim(trim_nonzero_portion=trim_nonzero_portion,alpha=0.01,threshold_E_W=None,threshold_E_H=None)
            _,fs_time=model_fs.sel_feat(threshold_F=threshold_F,rank_method="Wilcoxon_rank_sum_test",key_feature_mean_feature_value_threshold=1,key_feature_neglog10pval_threshold=10,max_num_feature_each_pattern=3,header=numpy.unique(self.y))
	    
	    # record time
	    self.times_fact_fs.append(round(training_time+fs_time,2))
            
            # accuracies
            if self.feature_patterns is not None and self.feature_patterns_matrix is not None:
                acc_r=numpy.sum(self.feature_patterns==model_fs.F_str)/model_fs.M
                acc_e=numpy.sum(self.feature_patterns_matrix==model_fs.F)/(model_fs.M*(model_fs.V+1))
                print "The row match accuracy of features is: {0:.4f}".format(acc_r)
                print "The element match accuracy of features is: {0:.4f}".format(acc_e)
                if acc_e>self.acc_best:
                    self.acc_best=acc_e
                    self.a_0_best=a_0
                    self.b_0_best=b_0
                    self.a_large_best=a_large
                    self.b_large_best=b_large
                    self.setting_best=setting
                    print "The best element match acccuracy so far is:{0:.4f}, with setting {1}.".format(self.acc_best,self.setting_best)
                self.accs_e.append(round(acc_e,4))
                self.accs_r.append(round(acc_r,4))

            # compute KL divergence
            E_X=numpy.dot(model_fs.E_W,model_fs.E_H)
            log2_X=numpy.log2(self.X+1)
            log2_E_X=numpy.log2(E_X+1)
            kld=numpy.sum(self.X*log2_X-self.X*log2_E_X)
            print "The KL divergence is: {0:.1e}".format(kld)
            self.klds.append(round(kld,0))

            if if_plot_heatmap:
                model_fs.plot_heatmap(dir_save, prefix+"_"+setting, pattern="All",rank_method="mean_basis_value", unique_class_names=numpy.unique(self.y), width=10, height=10, fontsize=6, fmt="png",colormap="hot")
                        
        # convert to numpy array
        self.accs_e=numpy.array(self.accs_e)
        self.accs_r=numpy.array(self.accs_r)
        self.klds=numpy.array(self.klds)
        self.settings=numpy.array(self.settings)


    def sort_kld_acc(self,sort_by="acc_e"):
        # Sort the KL divergences and accuracies.
        
        # sort
        if self.feature_patterns is not None and self.feature_patterns_matrix is not None and sort_by=="acc_e":
            ind=numpy.argsort(self.accs_e)
        elif sort_by=="kld":
            ind=numpy.argsort(self.klds)
            
        ind=ind[::-1] # decreasing order

        # reorder/sorting
        self.settings=self.settings[ind]
        
        if self.feature_patterns is not None and self.feature_patterns_matrix is not None:
            self.accs_e=self.accs_e[ind]
            self.accs_r=self.accs_r[ind]


    def compute_cor_acc_kld(self):
        # compute correlation between acc and KL divergence
        if self.feature_patterns is not None and self.feature_patterns_matrix is not None:
            # correlation between accs_e and klds
            self.coef_pearson,self.pval_pearson=pearsonr(self.accs_e,self.klds)
            print "The Pearson correlation coefficient between the element match accuracies and KL divergences is {0:.4f}, the corresponding pval is {1:.2e}".format(self.coef_pearson,self.pval_pearson)
            self.coef_spearman,self.pval_spearman=spearmanr(self.accs_e,self.klds)
            print "The Spearman rank-order correlation coefficient between the element match accuracies and KL divergences is {0:.4f}, the corresponding pval is {1:.2e}".format(self.coef_spearman,self.pval_spearman)
            return self.coef_pearson,self.pval_pearson,self.coef_spearman,self.pval_spearman

    
    def plot_kld_acc(self,dir_save="./",prefix="mcnmf_model_selection",if_log2kld=False,figwidth=5,figheight=3):
        import matplotlib as mpl
        mpl.use("pdf")
        import matplotlib.pyplot as plt

        iters=range(1,len(self.klds)+1)
        # plot KL divergence
        fig=plt.figure(num=1,figsize=(figwidth,figheight))
        ax=fig.add_subplot(1,1,1)
        if if_log2kld:
            klds=numpy.log2(self.klds)
            ylab_kld="Log2(KL Divergence)"
        else:
            klds=self.klds
            ylab_kld="KL Divergence"
        ax.plot(iters,klds,"b-",linewidth=1)
        ax.set_xlabel("Setting",fontsize=10)
        ax.set_ylabel(ylab_kld,color="b",fontsize=10)
        ax.set_xlim([1,iters[-1]])
        ax.set_ylim([numpy.min(klds),numpy.max(klds)])
        for tl in ax.get_yticklabels():
            tl.set_color("b")
        
        # plot acc_e
        if self.feature_patterns is not None and self.feature_patterns_matrix is not None:
            ax2=ax.twinx()
            ax2.plot(iters,self.accs_e,"r-",linewidth=1)
            ax2.plot(iters,self.accs_r,"g-",linewidth=1)
            ax2.set_ylabel("Element-Wise-Match Accuracy", color="r",fontsize=10)
            ymax=numpy.max(self.accs_e)
            if ymax>1:
                ymax=1
            ymin=numpy.min(numpy.vstack((self.accs_e,self.accs_r)))
            ax2.set_xlim([1,iters[-1]])
            ax2.set_ylim([ymin,ymax])
            for tl in ax2.get_yticklabels():
                tl.set_color("r")
            ax3=ax.twinx()
            ax3.set_ylabel("Row-Wise-Match Accuracy", color="g",fontsize=10)
            ax3.tick_params(axis='y', which='both', bottom='off', top='off', left="off",right="off", labelright='off')
            ax3.set_xlim([1,iters[-1]])
            ax3.set_ylim([ymin,ymax])
            ax3.yaxis.labelpad = 44

        # save
        filename=dir_save+prefix+"_total_"+str(len(self.klds))+"_kld_acc.pdf"
        plt.tight_layout()
        fig.savefig(filename,bbox_inches='tight',format="pdf",dpi=600)
        plt.close(fig)
        
    def save_result(self,dir_save="./",prefix="mcnmf_model_selection"):
        # make a copy of the main script
        #scr="/home/yifeng/research/tcga/mf/main_meta_mf_sim_search.py"
        #copy(scr, dir_save)

        #filename=dir_save + prefix+"_total_"+str(len(self.a_0s)*len(self.b_0s)*len(self.a_larges)*len(self.b_larges))+"_search_results.txt"
        filename=dir_save + prefix+"_total_"+str(len(self.klds))+"_search_results.txt"
        # save searching result
        klds=numpy.array(self.klds,dtype=str)
        klds.shape=(len(self.klds),1)
        times_fact_fs=numpy.array(self.times_fact_fs,dtype=str)
        times_fact_fs.shape=(len(self.times_fact_fs),1)	
        if self.feature_patterns is not None and self.feature_patterns_matrix is not None:
            accs_r=numpy.array(self.accs_r,dtype=str)
            accs_r.shape=(len(self.accs_r),1)
            accs_e=numpy.array(self.accs_e,dtype=str)
            accs_e.shape=(len(self.accs_e),1)

        settings=numpy.array(self.settings,dtype=str)
        settings.shape=(len(self.settings),1)
        if self.feature_patterns is not None and self.feature_patterns_matrix is not None:
            settings_accs_klds=numpy.hstack((settings,accs_e,accs_r,klds,times_fact_fs))
            numpy.savetxt(filename,settings_accs_klds,fmt='%s', delimiter='\t', newline='\n',header="setting\tacc_e\tacc_r\tkld\ttime")
        else:
            settings_klds=numpy.hstack((settings,klds,times_fact_fs))
            numpy.savetxt(filename,settings_klds,fmt='%s', delimiter='\t', newline='\n',header="setting\tkld\ttime")







