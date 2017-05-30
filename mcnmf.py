# The multi-view NMF class.

from __future__ import division # In Python 3.0, // is the floor division operator and / the true division operator. The true_divide(x1, x2) function is equivalent to true division in Python.
import scipy.special
import scipy.stats
import numpy
import math
import time
import classification as cl # in DECRES
import mlp # in DECRES
import unique_binary_code
import utility # in DECRES

class mcnmf:
    def __init__(self, X=None, y=None, features=None): 
        """
        X: numpy matrix, float type, each row is a feature, each column is a sample.
        y: numpy vector, integer type, the class labels.
        features: numpy vector, string type, the feature names.
        """
        #initialize
        self.X=X # data
        self.y=y # class labels
        self.features=features # features
        #self.y_unique=numpy.unique(self.y) # unique class labels [0,1,2,...,V-1]
        self.Y,self.y_unique=cl.membership_vector_to_indicator_matrix(self.y)
        self.M=self.X.shape[0] # number of features
        self.N=self.X.shape[1] # number of samples
        self.V=len(self.y_unique) # number of views
        self.tol=1e-10
        self.clrs_classes=None
        self.clrs_feature_groups=None

    def make_A(self,A):
        """
        Make matrix A of size V+1 by V.

        INPUTS:
        A: tuple of length 2, must not be a list or numpy array type. A[0] is a_large, A[1] is a_small.

        OUTPUTs:
        A: numpy array of size V+1 by V.
        """
        a_large=A[0]
        a_small=A[1]
        A=numpy.ones(shape=(self.V,self.V),dtype=float)*a_small
        numpy.fill_diagonal(A,a_large)
        a=numpy.ones(shape=(1,self.V),dtype=float)*a_large
        A=numpy.vstack((a,A))
        #print "Matrix A or B:"
        #print A
        return A

    def make_B(self,B):
        """
        Make matrix B of size V+1 by V.

        INPUTS:
        B: tuple of length 2, must not be a list or numpy array type. B[0] is b_large, B[1] is b_small.

        OUTPUTS:
        B: numpy array of size V+1 by V.
        """
        b_large,b_small=B
        B=self.make_A((b_large,b_small))
        return B

    def factor_sizes_to_factor_labels(self,z,start=-1):
        """
        Convert the factor sizes to factor labels.

        INPUTS:
        z: a tuple or list, e.g. z=[3,4,2].
        start: integer, the smallest label.

        OUTPUTS:
        labels: a list of factor labels, e.g. labels=[-1,-1,-1,0,0,0,0,1,1]
        """
        labels=[]
        for i in z:
            labels.extend([start]*i)
            start=start+1
        #print labels
        return labels
            
    def factorize(self, z=None, a_0=None, b_0=None, A=None, B=None, max_iter=200,compute_variational_lower_bound=True,variational_lower_bound_min_rate=1e-4,if_plot_lower_bound=True,dir_save="./",prefix="MVNMF",verb=True,rng=numpy.random.RandomState(1000)):
        """
        Factorize X to W and H.

        INPUTS:
        z: integer, tuple of length V+1, list/numpy vector of size (K,), the labels of the factors (columns) in W. If z is a scalar, it means each view (including the ubi view) has z factors. If z is a tuple, e.g. (3,3,3,3), z[u] means the the u-th view has z[u] factors. If z is a list or numpy vector,e.g. [-1-1-1,0,0,1,1,1,2,2,2] where -1 means ubi view, z[k] means the k-th factor has label z[k].
        a_0, b_0: shape and rate hyperparameters.
        A, B: shape and rate hyperparameterss, if list or array, just use it. If tuple, they represent A=(a_large,a_small) and  B=(b_large,b_small).
        max_iter: integer, the maximal number of iterations.
        compute_variational_lower_bound: bool, whether compute the variational lower bound.
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
        start_time=time.clock()
        self.training_time_L=0

        if isinstance(z,(list,numpy.ndarray)):
            self.z=numpy.array(z,dtype=int) # e.g. [-1,-1,-1,0,0,0,1,1,2,2,2], -1: common factors accross all classes, 0,...,V-1: class labels.
        elif isinstance(z,tuple):
            self.z=self.factor_sizes_to_factor_labels(z) # e.g. (3,3,3,3) to [-1,-1,-1,0,0,0,1,1,2,2,2]
        else:
            self.z=self.factor_sizes_to_factor_labels([z]*(self.V+1)) # e.g. 3 to [-1,-1,-1,0,0,0,1,1,2,2,2]
        #print self.z
        self.K=len(self.z) # number of latent factors 
        self.Z,self.z_unique=cl.membership_vector_to_indicator_matrix(self.z) # binary, size K by V+1, self.Z[k,u]=1 indicates the k-th factor in class u.
        self.a_0=a_0
        self.b_0=b_0
        if isinstance(A,(list,numpy.ndarray)):
            self.A=numpy.array(A,dtype=float)
        else:
            self.A=self.make_A(A)
        if isinstance(B,(list,numpy.ndarray)):
            self.B=numpy.array(B,dtype=float)
        else:
            self.B=self.make_B(B)
        self.max_iter=max_iter
        self.rng=rng
        
        #print self.Z
        #print self.Y

        #initiate
        #sample E_Lambda_W and E_Lambda_H from Gamma prior distributions
        E_Lambda_W=self.rng.gamma(shape=self.a_0, scale=1/self.b_0, size=(self.M,self.V+1))
        E_Lambda_H=numpy.zeros(shape=(self.V+1,self.V))
        for u in range(self.V+1):
            for v in range(self.V):
                E_Lambda_H[u,v]=self.rng.gamma(shape=self.A[u,v],scale=1/self.B[u,v])
        #compute A_W, B_W, A_H, and B_H using prior expoential distribution(1,lambda)
        A_W=numpy.ones(shape=(self.M,self.K),dtype=float)
        B_W=numpy.dot(E_Lambda_W, numpy.transpose(self.Z)) + self.tol
        #print B_W
        A_H=numpy.ones(shape=(self.K,self.N),dtype=float)
        B_H=self.Z.dot(E_Lambda_H).dot(self.Y.transpose()) + self.tol
        #print B_H
        #compute E_W and E_H using A_W, B_W, A_H, and B_H
        #print B_W
        #print B_H
        #E_W=A_W/B_W
        #E_H=A_H/B_H
        E_W=self.rng.gamma(shape=A_W,scale=1/B_W)
        E_H=self.rng.gamma(shape=A_H,scale=1/B_H)
        #compute L_W and L_H using A_W and A_H
        #L_W=numpy.exp(scipy.special.psi(A_W))/B_W
        #L_H=numpy.exp(scipy.special.psi(A_H))/B_H
        L_W=E_W
        L_H=E_H
        L_W_L_H=L_W.dot(L_H)

        if compute_variational_lower_bound:
            neg_gammaln_X_plus_1=-scipy.special.gammaln(self.X+1)
            Ls=[] # lower bound, start from iteration 1
            Ls_rates=[] # change rate, start from iteration 2
            Ls_rates_means=[] # local mean change rate, start from iteration 2
            mean_over=5 # compute the mean rate over this number of iterations

        ONES_M_N=numpy.ones_like(self.X)
        ONES_M_K=numpy.ones_like(E_W)
        ONES_K_N=numpy.ones_like(E_H)
        num_iter=1
        while num_iter<=self.max_iter:
            if verb:
                print "iteration: {0}".format(num_iter)
            #compute Sigma_W and Sigma_H, need L_W and L_H
            X_div_L_W_L_H=self.X/L_W_L_H
            Sigma_W=L_W * X_div_L_W_L_H.dot(L_H.transpose())
            Sigma_H=L_H.transpose() * X_div_L_W_L_H.transpose().dot(L_W) 
            #print Sigma_W
            #print Sigma_H

            #compute A_W, B_W, A_H, and B_H need Sigma_W, Sigma_H, E_W, E_H, E_Lambda_W, and E_Lambda_H
            A_W=1 + Sigma_W
            B_W=E_Lambda_W.dot(self.Z.transpose()) + ONES_M_N.dot(E_H.transpose())
            E_W=A_W/B_W
            A_H=1 + Sigma_H.transpose()
            B_H=self.Z.dot(E_Lambda_H).dot(self.Y.transpose()) + E_W.transpose().dot(ONES_M_N)
            #compute E_W and E_H, need A_W, B_W, A_H, and B_H
            E_H=A_H/B_H
           
            #compute L_W and L_H, need A_W, B_W, A_H, and B_H
            #L_W=numpy.exp(scipy.special.psi(A_W))/B_W
            #L_H=numpy.exp(scipy.special.psi(A_H))/B_H
            psi_A_W=scipy.special.psi(A_W)
            exppsi_A_W=numpy.exp(psi_A_W)
            L_W=exppsi_A_W/B_W
            psi_A_H=scipy.special.psi(A_H)
            exppsi_A_H=numpy.exp(psi_A_H)
            L_H=exppsi_A_H/B_H
            L_W_L_H=L_W.dot(L_H)

            #compute E_Lambda_W and E_Lambda_H, need OLD E_W and E_H (not NEW)
            #E_Lambda_W=(self.a_0 + ONES_M_K.dot(self.Z))/(self.b_0 + E_W.dot(self.Z))
            #E_Lambda_H=(self.A + self.Z.transpose().dot(ONES_K_N).dot(self.Y))/(self.B + self.Z.transpose().dot(E_H).dot(self.Y))
            A_Lambda_W=self.a_0 + ONES_M_K.dot(self.Z)
            B_Lambda_W=self.b_0 + E_W.dot(self.Z)
            E_Lambda_W=A_Lambda_W/B_Lambda_W
            A_Lambda_H=self.A + self.Z.transpose().dot(ONES_K_N).dot(self.Y)
            B_Lambda_H=self.B + self.Z.transpose().dot(E_H).dot(self.Y)
            E_Lambda_H=A_Lambda_H/B_Lambda_H
           
            # compute variational lower bound           
            # variational lower bound
            if compute_variational_lower_bound:
                start_time_L=time.clock()

                psi_A_Lambda_W=scipy.special.psi(A_Lambda_W)
                log_B_Lambda_W=numpy.log(B_Lambda_W)
                L_Lambda_W=psi_A_Lambda_W - log_B_Lambda_W
                psi_A_Lambda_H=scipy.special.psi(A_Lambda_H)
                log_B_Lambda_H=numpy.log(B_Lambda_H)
                L_Lambda_H=psi_A_Lambda_H - log_B_Lambda_H

                L=numpy.sum( neg_gammaln_X_plus_1 + self.X*numpy.log(L_W.dot(L_H))-E_W.dot(E_H)) + numpy.sum(L_Lambda_W.dot(self.Z.transpose()) - E_Lambda_W.dot(self.Z.transpose())*E_W) + numpy.sum(self.Z.dot(L_Lambda_H).dot(self.Y.transpose()) - self.Z.dot(E_Lambda_H).dot(self.Y.transpose())*E_H) + numpy.sum(self.a_0*numpy.log(self.b_0) - scipy.special.gammaln(self.a_0) + (self.a_0-1)*L_Lambda_W - self.b_0*E_Lambda_W ) + numpy.sum(self.A*numpy.log(self.B) - scipy.special.gammaln(self.A) + (self.A-1)*L_Lambda_H - self.B*E_Lambda_H) + numpy.sum(scipy.special.gammaln(A_W) - (A_W-1)*psi_A_W - numpy.log(B_W) + A_W) + numpy.sum(scipy.special.gammaln(A_H) - (A_H-1)*psi_A_H - numpy.log(B_H) + A_H) + numpy.sum(scipy.special.gammaln(A_Lambda_W) - (A_Lambda_W-1)*psi_A_Lambda_W - log_B_Lambda_W + A_Lambda_W) + numpy.sum(scipy.special.gammaln(A_Lambda_H) - (A_Lambda_H-1)*psi_A_Lambda_H - log_B_Lambda_H + A_Lambda_H)
                #L=numpy.sum( abs(self.X-E_W.dot(E_H))) 
                Ls.append(L)
                end_time_L=time.clock()
                self.training_time_L=self.training_time_L+(end_time_L-start_time_L)
                if num_iter>=2:
                    Ls_rates.append((Ls[-1]-Ls[-2])/(-Ls[-2]))
                    Ls_rates_mean=numpy.mean(Ls_rates[-mean_over:])
                    Ls_rates_means.append(Ls_rates_mean)
                    if verb:
                        print "The variational lower bound:{0}, local mean change rate:{1}".format(L,Ls_rates_mean)
                    if Ls_rates_mean<variational_lower_bound_min_rate:
                        break
            
            num_iter=num_iter+1

        #save the result
        self.E_W=E_W
        self.E_H=E_H
        self.E_Lambda_W=E_Lambda_W
        self.E_Lambda_H=E_Lambda_H
        self.L_W=L_W
        self.L_H=L_H
        if compute_variational_lower_bound:
            self.Ls=Ls # lower bounds
            self.num_iter=num_iter-1 # actural number of iterations run
        
            # save data for plot for the lower bound, using Ls, Ls_rates_means, iters
            self.Ls=Ls[1:]
            self.Ls_rates_means=Ls_rates_means
            self.iters=range(2,(len(Ls_rates_means)+2) )
        else: 
            self.Ls=None # lower bounds
            self.num_iter=num_iter-1 # actural number of iterations run
        
            # save data for plot for the lower bound, using Ls, Ls_rates_means, iters
            self.Ls=None
            self.Ls_rates_means=None
            self.iters=None

        end_time = time.clock()
        self.training_time=end_time-start_time
        print 'traning factorization: time excluding computing variational lower bound: %f seconds' %(self.training_time-self.training_time_L)
    
        # plot lower bound
        if compute_variational_lower_bound and if_plot_lower_bound:
            print "ploting variational lower bounds ..."
            self.plot_lower_bound(dir_save=dir_save,prefix=prefix,iters=self.iters,Ls=self.Ls,Ls_rates_means=self.Ls_rates_means,figwidth=5,figheight=3)
            print 'training factorization: time for computing variational lower bound: %f seconds' %(self.training_time_L)

        print "finished fatorization!"
        return self.E_W,self.E_H,self.training_time,self.training_time_L


    def plot_lower_bound(self,dir_save,prefix="MVNMF",iters=None,Ls=None,Ls_rates_means=None,figwidth=5,figheight=3):
        """
        Plot the lower bounds and local mean change rates.
        
        INPUTS:
        dir_save: string, path to save the plotted figure, e.g. "/home/yifeng/research/mf/mvmf_v1_1/".
        prefix: string, prefix of the file name to be saved.
        iters: list or numpy vector, the numeric ids of iterations.
        Ls: list or numpy vector, the variational lower bounds.
        Ls_rates_means: list of numpy vector, the local mean change rates of the lower bounds.
        figwidth: float scalar, the width of figure in inches.
        figheight: float scalar, the height of figure in inches.

        OUTPUTS:
        This function does not explicitly return any variable. 
        """
        # 
        import matplotlib as mpl
        mpl.use("pdf")
        import matplotlib.pyplot as plt

        fig=plt.figure(num=1,figsize=(figwidth,figheight))
        ax=fig.add_subplot(1,1,1)
        ax.plot(iters,Ls,"b-",linewidth=1)
        ax.set_xlabel("Iteration",fontsize=10)
        ax.set_ylabel("Lower Bound",color="b",fontsize=10)
        for tl in ax.get_yticklabels():
            tl.set_color("b")
        
        ax2=ax.twinx()
        ax2.plot(iters,Ls_rates_means,"r-",linewidth=1)
        ax2.set_ylabel("Local Mean Change Rate", color="r",fontsize=10)
        for tl in ax2.get_yticklabels():
            tl.set_color("r")
        filename=dir_save+prefix+"_lower_bound.pdf"
        plt.tight_layout()
        fig.savefig(filename,bbox_inches='tight',format="pdf",dpi=600)
        plt.close(fig)
        
        #NOTE: MAY CONSIDER ZERO COLUMNS in E_W and ZERO ROWS in E_H
    def trim(self,trim_nonzero_portion=0.01,alpha=0.05,threshold_E_W=None,threshold_E_H=None):
        """
        Trim zero columns of E_W and corresponding rows of E_H.
        Still under investigation. Leave for future improvement.
        """
        print "triming zero factors ..."
        self.trim_nonzero_portion=trim_nonzero_portion

        self.alpha=alpha
        if threshold_E_W is None:
            threshold_E_W=self.alpha*numpy.mean(self.E_W)
        self.E_W_NZ=self.E_W>=threshold_E_W # non-zero indicator of E_W
        nonzero_portions_E_W=numpy.mean(self.E_W_NZ,axis=0)
       
        if threshold_E_H is None:
            threshold_E_H=self.alpha*numpy.mean(self.E_H)
        self.E_H_NZ=self.E_H>=threshold_E_H # non-zero indicator of E_H
        nonzero_portions_E_H=numpy.mean(self.E_H_NZ,axis=1)
        
        ind_nonzero_factors=numpy.logical_or(nonzero_portions_E_W>=self.trim_nonzero_portion,nonzero_portions_E_H>=self.trim_nonzero_portion)
        # filter zero factors
        self.E_W=self.E_W[:,ind_nonzero_factors]
        self.E_H=self.E_H[ind_nonzero_factors,:]
        self.L_W=self.L_W[:,ind_nonzero_factors]
        self.L_H=self.L_H[ind_nonzero_factors,:]
        self.z=numpy.array(self.z,dtype=int)
        self.z=self.z[ind_nonzero_factors]
        self.Z=self.Z[ind_nonzero_factors,:]
        self.K=self.E_W.shape[1]

    def sel_feat(self,called_by_ssmvnmf_loop=False,F=None,threshold_F=None,use_previous_scores=False,allow_compute_pval=True,rank_method="Wilcoxon_rank_sum_test",maxbc=10,patterns_for_feat_sel="all",key_feature_mean_feature_value_threshold=100,key_feature_neglog10pval_threshold=10,max_num_feature_each_pattern=10,max_num_feature=None,header=None,rng=numpy.random.RandomState(1000)):
        """
        Discover feature patterns based on MV-NMF.

        INPUTS:
        called_by_ssmvnmf_loop: bool, whether this function is called within SS-MV-NMF. If True, only compute the feature activity indicator matrix F_mean and then exit.
        F: matrix, the feature activity indicator matrix, usually it is not given. F is only provided when called by SS-MV-NMF in the final stage.
        threshold_F:  positive float scalar, used to generate the feature activity indicator matrix: F= F_mean >= (threshold_F * mean(F_mean)).
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

        start_time = time.clock()
        print "selecting features ..."

        self.allow_compute_pval=allow_compute_pval

        # scale W and H
        row_max_H=numpy.max(self.E_H,axis=1)
        SCALE=numpy.diag(row_max_H)
        SCALE_inv=numpy.diag(1/row_max_H)
        self.E_H_scaled=numpy.dot(SCALE_inv,self.E_H)
        self.E_W_scaled=numpy.dot(self.E_W,SCALE)

        if F is not None:
            self.F=F # F is provided when using stability selection in the final stage
        else:
            col_sum=numpy.sum(self.K,axis=0) # how many factors for each class
            self.F_mean=self.E_W_scaled.dot(self.Z/col_sum) # factor-wise mean

            # scaled, does not work so far, because the smallest number is wrongly set to zero.
            #self.F_mean_scale01,_,_=cl.normalize_row_scale01(self.F_mean,clip=False,clip_min=1e-3,clip_max=1e4) # scale the row of F_mean
            #print self.F_mean_scale01
            #if threshold_F is None:
            #    #threshold_F=0.01*numpy.mean(self.F_mean_scale01)
            #    threshold_F=0.001
            #self.F=self.F_mean_scale01>=threshold_F # indicator matrix, F of size M by V+1

            # unscaled
            if threshold_F is None:
                threshold_F=0.01
                #thres=0.01
            F_mean_mean=numpy.mean(self.F_mean)
            print "The mean of F_mean is {0}".format(F_mean_mean)
            thres=threshold_F* F_mean_mean
            #if thres>0.99:
            #    thres=0.99
            self.F=self.F_mean>=thres # indicator matrix, F of size M by V+1
	
	if called_by_ssmvnmf_loop:
	    end_time=time.clock()
            self.fs_time=end_time-start_time
	    print "called by SS-MV-NMF in its loop. Only F is needed."
            print "feature selection time: %f seconds." %(self.fs_time)
	    return self.F,self.fs_time
        
        if not use_previous_scores: # save lots of time to avoid repeating computing the scores
            self.ubc=unique_binary_code.unique_binary_code(self.V+1,maxn=maxbc) # unique binary codes
            given_range=numpy.sort(numpy.unique(self.F.sum(axis=1))) # each element is the number of 1s in a binary code, eg. given_range=[1,2,3,...V,V+1]
            #print given_range
            if given_range[0]==0:
                given_range=given_range[1:]
            #given_range=[1,2,3,4,5,6]
            self.ubc.generate_binary_code(given_range=given_range) # generate binary codes for given range
            if header is None:
                header=self.y_unique
                #classes_unique.reshape((1,len(self.y_unique)))
            header=numpy.hstack((["Ubiquitous"],header))
            # convert each row of F to a string
            self.F_str=utility.convert_each_row_of_matrix_to_a_string(numpy.array(self.F,dtype=int),sep="")
            if (self.V+1)>maxbc:
                F_str_unique,ind_Fu=numpy.unique(self.F_str,return_index=True)
                self.ubc.replace_s(self.F[ind_Fu])
            self.ubc.generate_freq_tab(self.F,header=header) # generate and save the frequency table
            self.F=numpy.asarray(self.F,dtype=int) # bool to int

            # compute feature scores
            # compute mean basis values
            self.scores_mean_basis_value=numpy.zeros((self.M,1),dtype=float)
            for m in range(self.M):
                fs=numpy.sum(self.F[m,:]) # how many active factor groups
                fms=numpy.sum(self.F_mean[m,:])
                if fs>=1: 
                    self.scores_mean_basis_value[m]=fms/fs
                else:
                    self.scores_mean_basis_value[m]=fms/(self.V+1)
            # scale scores to [0-1]
            self.scores_mean_basis_value,_,_=cl.normalize_col_scale01(self.scores_mean_basis_value,clip=False)

            # compute p values and mean read counts
            self.scores_Wilcoxon_rank_sum_test=numpy.zeros((self.M,1),dtype=float)
            self.scores_mean_feature_value=numpy.zeros((self.M,1),dtype=float)
            # compute the p values if allowed. For large data, computing it maybe time consuming, thus allow the user to switch it off.
            for m in range(self.M):
                fm=self.F[m,1:]
                fm=fm.astype(bool)# indices of active classes
                ya=self.y_unique[fm] # active class labels
                yia=self.y_unique[~fm] # inactive class labels
                if self.allow_compute_pval and (len(ya)==0 or len(yia)==0):
                    self.scores_Wilcoxon_rank_sum_test[m,0]=1
                    continue
                # obtain indices of ya in y
                ind_a=numpy.zeros((self.N,),dtype=bool)
                for c in ya:
                    ind_a=numpy.logical_or(ind_a,self.y==c)
                sample_active=self.X[m,ind_a]
                sample_inactive=self.X[m,numpy.logical_not(ind_a)]
                self.scores_mean_feature_value[m,0]=sample_active.mean()
                if self.allow_compute_pval:
                    # Wilcoxon rank-sum test
                    statistic,pval=scipy.stats.ranksums(sample_active, sample_inactive)
                    self.scores_Wilcoxon_rank_sum_test[m,0]=-numpy.log10(pval)

        # rank the features in different ways
        # convert each row of s to a string
        self.s_str=utility.convert_each_row_of_matrix_to_a_string(self.ubc.s,sep="")
        self.ind_mean_basis_value=[]
        self.ind_mean_feature_value=[]
        self.ind_Wilcoxon_rank_sum_test=[]

        numbers=numpy.array(range(self.M),dtype=int)
        for i in range(len(self.s_str)):
            c=self.s_str[i]
            #print "The current code:{0}".format(c)
            ind_log=(self.F_str==c)
	    if not any(ind_log):
	        continue
            #print "Length of ind_log:{0}".format(len(ind_log))
            ind_c=numbers[ind_log] # numeric indices for the i-th code c or cv
            
            # sort the indices based on different corresponding feature's score
            #"mean_basis_value"
            ind_mean_basis_value_c_de=numpy.argsort(self.scores_mean_basis_value[ind_log,0])[::-1]
            ind_mean_basis_value_c=ind_c[ind_mean_basis_value_c_de]
            self.ind_mean_basis_value.extend(ind_mean_basis_value_c)

            #"mean_feature_value" i.e. mean read count for RNA-seq data
            ind_mean_feature_value_c_de=numpy.argsort(self.scores_mean_feature_value[ind_log,0])[::-1]
            ind_mean_feature_value_c=ind_c[ind_mean_feature_value_c_de]
            self.ind_mean_feature_value.extend(ind_mean_feature_value_c)
            if self.allow_compute_pval:
                #"Wilcoxon_rank_sum_test"
                ind_Wilcoxon_rank_sum_test_c_de=numpy.argsort(self.scores_Wilcoxon_rank_sum_test[ind_log,0])[::-1]
                ind_Wilcoxon_rank_sum_test_c=ind_c[ind_Wilcoxon_rank_sum_test_c_de]
                self.ind_Wilcoxon_rank_sum_test.extend(ind_Wilcoxon_rank_sum_test_c)
            
        # obtain key features for classification purpose
        #ubc_specific=unique_binary_code.unique_binary_code(self.V+1) # unique binary codes
        #given_range=numpy.array([2,self.V+1],dtype=int) # each element is the number of 1s in a binary code, eg. given_range=[0,1,2,3,...K]
        #ubc_specific.generate_binary_code(given_range=given_range) # generate binary codes for given range
        #s_str_specific=utility.convert_each_row_of_matrix_to_a_string(ubc_specific.s,sep="")

        # obtain key features with small Wilcoxon test p values for each pattern
        # key features: (1) the mean read count of active views in a feature should be greater than a minimal threshold;
        #               (2) Should be ranked based on p-values

        #reorder features
        if rank_method=="mean_feature_value":
            ind=self.ind_mean_feature_value
        elif rank_method=="mean_basis_value":
            ind=self.ind_mean_basis_value
        elif rank_method=="Wilcoxon_rank_sum_test":
            if self.allow_compute_pval:
                ind=self.ind_Wilcoxon_rank_sum_test
            else:
                print "You choose to reorder the features by Wilcoxon_rank_sum_test' pvals, but computing pvals is not allowed(allow_compute_pval=False). I decide to reorder the features by mean_basis_value instead."
                ind=self.ind_mean_basis_value
        else:
            print "Error! Please select the right rank method."
        
        #E_W_sorted=self.E_W[ind,:]
        #X_sorted=self.X[ind,:]
        #F_sorted=self.F[ind,:]
        F_str_sorted=self.F_str[ind]
        features_sorted=self.features[ind]
        scores_mean_feature_value_sorted=self.scores_mean_feature_value[ind]
        scores_mean_basis_value_sorted=self.scores_mean_basis_value[ind]
        scores_Wilcoxon_rank_sum_test_sorted=self.scores_Wilcoxon_rank_sum_test[ind]
        numeric_ind_sorted=numpy.array(ind,dtype=int)

	if patterns_for_feat_sel=="all":
	    patterns=self.s_str
	elif patterns_for_feat_sel=="view_specific":
	    zero_V_1=numpy.zeros(shape=(self.V,1),dtype=int)
	    one_V_1=numpy.ones(shape=(self.V,1),dtype=int)
	    I_V_V=numpy.eye(self.V,k=0,dtype=int)
	    patterns=numpy.vstack( (numpy.hstack((zero_V_1,I_V_V)),numpy.hstack((one_V_1,I_V_V))) ) # 0100, ...
	    patterns_not=numpy.array(numpy.logical_not(patterns),dtype=int) # 1011, ...
            patterns=utility.convert_each_row_of_matrix_to_a_string(numpy.vstack((patterns,patterns_not)),sep="") # all specific patterns
	    print patterns
        elif patterns_for_feat_sel=="top_from_all":
            patterns=self.s_str

        ind_key_feat=[]
        primary_scores=[]
        for pattern in patterns:
            ind=F_str_sorted==pattern # logical ind to the sorted
            num_ind=numeric_ind_sorted[ind] # sorted numeric ind to the original
            scores_mean_feature_value=scores_mean_feature_value_sorted[ind]
            ind_v=scores_mean_feature_value>=key_feature_mean_feature_value_threshold
            if self.allow_compute_pval:
                scores_neglog10pval=scores_Wilcoxon_rank_sum_test_sorted[ind] # sorted Wilcoxon scores
                primary_score=scores_neglog10pval
                ind_p=scores_neglog10pval>=key_feature_neglog10pval_threshold
                ind_vp=numpy.logical_and(ind_v,ind_p)
            else:
                #print "Please be aware of that allow_compute_pval is switched off, so key_feature_neglog10pval_threshold is not considered."
                primary_score=scores_mean_feature_value
                ind_vp=ind_v
            ind_vp.shape=(ind_vp.shape[0],)
            # if the number of significant feature is larger than a given threshold, only take the top ones.
            num_ind=num_ind[ind_vp]
            primary_score=primary_score[ind_vp]
            if ind_vp.sum()>max_num_feature_each_pattern:
                num_ind=num_ind[0:max_num_feature_each_pattern]
                primary_score=primary_score[0:max_num_feature_each_pattern]
            # numerical indices for key features
            ind_key_feat.extend(num_ind)
            primary_scores.extend(primary_score)
        # store the numeric indices of key features
        self.numeric_ind_key_feat_for_classification=numpy.array(ind_key_feat,dtype=int)
        self.num_features_selected=len(self.numeric_ind_key_feat_for_classification)
        self.primary_scores_key_feat_for_classification=numpy.array(primary_scores,dtype=float)

        if max_num_feature is not None:
            if max_num_feature>self.num_features_selected:
                max_num_feature=self.num_features_selected
            ind_perm=rng.permutation(self.num_features_selected) # permute the order of the indices and scores
            self.numeric_ind_key_feat_for_classification=self.numeric_ind_key_feat_for_classification[ind_perm]
            self.primary_scores_key_feat_for_classification=self.primary_scores_key_feat_for_classification[ind_perm]
            self.numeric_ind_key_feat_for_classification=self.numeric_ind_key_feat_for_classification[0:max_num_feature] # randomly take top features
            self.primary_scores_key_feat_for_classification=self.primary_scores_key_feat_for_classification[0:max_num_feature]
            self.num_features_selected=max_num_feature

        end_time = time.clock()
        self.fs_time=end_time-start_time
        print "feature selection time: %f seconds." %(self.fs_time)
        print "number of features selected:{}".format(self.num_features_selected)
        
        return self.numeric_ind_key_feat_for_classification,self.fs_time


    def save_feat(self,dir_save="./",prefix="MVNMF",rank_method="Wilcoxon_rank_sum_test",neglog10pval_threshold=10,if_each_pattern_per_file=False):
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
        # alpha: Threshold for the -log10(pval) scores.
        print "saving features ..."
        filename=dir_save+ prefix +"_frequency_table.txt"
        self.ubc.save_freq_tab(filename)

        #reorder features
        if rank_method=="mean_feature_value":
            ind=self.ind_mean_feature_value
        elif rank_method=="mean_basis_value":
            ind=self.ind_mean_basis_value
        elif rank_method=="Wilcoxon_rank_sum_test":
            if self.allow_compute_pval:
                ind=self.ind_Wilcoxon_rank_sum_test
            else:
                print "You choose to reorder the features by Wilcoxon_rank_sum_test' pvals, but computing pvals is not allowed(allow_compute_pval=False). I decide to reorder the features by mean_basis_value instead."
                ind=self.ind_mean_basis_value
        else:
            print "Error! Please select the right rank method."
        
        #E_W_sorted=self.E_W[ind,:]
        #X_sorted=self.X[ind,:]
        #F_sorted=self.F[ind,:]
        F_mean_sorted=self.F_mean[ind,:]
        F_str_sorted=self.F_str[ind]
        features_sorted=self.features[ind]
        scores_mean_basis_value_sorted=self.scores_mean_basis_value[ind]
        scores_mean_feature_value_sorted=self.scores_mean_feature_value[ind]
        scores_Wilcoxon_rank_sum_test_sorted=self.scores_Wilcoxon_rank_sum_test[ind]
        numeric_ind_sorted=numpy.array(ind,dtype=int)
        
        self.features.shape=(self.M,1)
        self.F_str.shape=(self.M,1)
        #self.scores_mean_basis_value.shape=(self.M,1)
        #self.scores_mean_feature_value.shape=(self.M,1)
        #self.scores_Wilcoxon_rank_sum_test.shape=(self.M,1)
        features_sorted.shape=(self.M,1)
        F_str_sorted.shape=(self.M,1)
        scores_mean_feature_value_sorted.shape=(self.M,1)
        scores_mean_basis_value_sorted.shape=(self.M,1)
        scores_Wilcoxon_rank_sum_test_sorted.shape=(self.M,1)
        #self.scores_sorted,_,_=cl.normalize_col_scale01(self.scores_sorted,clip=False)

        # save class specific genes
        #indsig=self.scores_Wilcoxon_rank_sum_test_sorted>-numpy.log10(alpha)
        #for v in range(1,self.V+1):
        #    pattern=numpy.zeros(self.V+1,dtype=str)
        #    pattern[0]=1
        #    pattern[v]=1
        #    pattern="".join(pattern)
        #    ind=self.F_str_sorted==pattern
        #    filename=dir_save+ prefix +"_features_" + pattern + ".txt"
        #    numpy.savetxt(filename, self.features_sorted[ind], fmt='%s', delimiter='\t')
        #    filename=dir_save+ prefix +"_features_" + pattern + "_sig.txt"
        #    numpy.savetxt(filename, self.features_sorted[numpy.logical_and(ind,indsig)], fmt='%s', delimiter='\t')
        # save ubiquitous genes
        #pattern=numpy.ones(self.V+1,dtype=str)
        #pattern="".join(pattern)
        #ind=self.F_str_sorted==pattern
        #filename=dir_save+ prefix +"_features_" + pattern + ".txt"
        #numpy.savetxt(filename, self.features_sorted[ind], fmt='%s', delimiter='\t')
        #filename=dir_save+ prefix +"_features_" + pattern + "_sig.txt"
        #numpy.savetxt(filename, self.features_sorted[numpy.logical_and(ind,indsig)], fmt='%s', delimiter='\t')

        # save gene list for each pattern
        if if_each_pattern_per_file:
            if self.allow_compute_pval:
                indsig=scores_Wilcoxon_rank_sum_test_sorted>neglog10pval_threshold
                for pattern in self.s_str:
                    ind=F_str_sorted==pattern
                    if any(ind):
                        filename=dir_save+ prefix +"_features_" + pattern + "_sorted.txt"
                        numpy.savetxt(filename, features_sorted[ind], fmt='%s', delimiter='\t',header='feature')
                        filename=dir_save+ prefix +"_features_" + pattern + "_sig_"+str(neglog10pval_threshold)+".txt"
                        numpy.savetxt(filename, features_sorted[numpy.logical_and(ind,indsig)], fmt='%s', delimiter='\t',header='feature')
            else:
                for pattern in self.s_str:
                    ind=F_str_sorted==pattern
                    if any(ind):
                        filename=dir_save+ prefix +"_features_" + pattern + "_sorted.txt"
                        numpy.savetxt(filename, features_sorted[ind], fmt='%s', delimiter='\t',header='feature')
        # save the whole gene list and data
        filename=dir_save+ prefix +"_features_patterns_scores.txt"
        numpy.savetxt(filename, numpy.concatenate((self.features,self.F_str,self.scores_mean_basis_value,self.scores_mean_feature_value,self.scores_Wilcoxon_rank_sum_test),axis=1), fmt='%s', delimiter='\t',header='feature\tpattern\tmean_basis_value\tmean_feature_value\tpval')
        filename=dir_save+ prefix +"_features_patterns_scores_sorted.txt"
        numpy.savetxt(filename, numpy.concatenate((features_sorted,F_str_sorted,scores_mean_basis_value_sorted,scores_mean_feature_value_sorted,scores_Wilcoxon_rank_sum_test_sorted),axis=1), fmt='%s', delimiter='\t',header='feature\tpattern\tmean_basis_value\tmean_feature_value\tpval')
        
        # save selected key features for classification
        filename=dir_save+ prefix +"_key_features_for_classification.txt"
        numpy.savetxt(filename, numpy.concatenate((self.features[self.numeric_ind_key_feat_for_classification],self.F_str[self.numeric_ind_key_feat_for_classification],self.scores_mean_basis_value[self.numeric_ind_key_feat_for_classification],self.scores_mean_feature_value[self.numeric_ind_key_feat_for_classification],self.scores_Wilcoxon_rank_sum_test[self.numeric_ind_key_feat_for_classification]),axis=1), fmt='%s', delimiter='\t',header='feature\tpattern\tmean_basis_value\tmean_feature_value\tpval')

        # save F_mean and F_mean_sorted
        filename=dir_save + prefix +"_F_mean.txt"
        numpy.savetxt(filename, self.F_mean, fmt='%.4f', delimiter='\t')
        filename=dir_save + prefix +"_F_mean_sorted.txt"
        numpy.savetxt(filename, F_mean_sorted, fmt='%.4f', delimiter='\t')

        self.features.shape=(self.M,)
        self.F_str.shape=(self.M,)
        #self.scores_mean_basis_value.shape=(self.M,)
        #self.scores_mean_feature_value.shape=(self.M,)
        #self.scores_Wilcoxon_rank_sum_test.shape=(self.M,)
        features_sorted.shape=(self.M,)
        F_str_sorted.shape=(self.M,)
        scores_mean_feature_value_sorted.shape=(self.M,)
        scores_mean_basis_value_sorted.shape=(self.M,)
        scores_Wilcoxon_rank_sum_test_sorted.shape=(self.M,)

    
    def colorbar_classes(self,classes,ax=None,hv="horizontal",clrs=None,unique_classes=None,unique_class_names=None,fontsize=6,ylabel_right=False,rotation=0):
        """
        Plot the color bar for classes. classes: 0,1,2,...,C-1. This function is called inside plot_heatmap.

        INPUTS:
        hv: either "horizontal" or "vertical" color bar.
        clrs: numpy vector of strings, the color names.
        unique_class_names: number vector of strings, the names of classes. If None, numpy array [0,1,2,...,V-1] will be used.
        fontsize: integer scalar, the font size of text.
        ylabel_right: bool scalar, whether give the class labels at the right-hand-side of the color bar.
        rotation: float scalar, angles to rotate the ticks (class labels). 
        
        OUTPUTS:
        This function does not explicitly return any variable.
        """

        num_samples=len(classes)
        if unique_classes is None:
            unique_classes=numpy.unique(classes)
        if unique_class_names is None:
            unique_class_names=unique_classes
        else:
            unique_class_names=numpy.array(unique_class_names)
        num_classes=len(unique_classes)
        num_samples_per_class=[numpy.sum(classes==unique_classes[c]) for c in range(num_classes)]
        #xranges=[(,num_samples_per_class[c]) for c in unique_classes]
        yrange=(0,1)
        facecolors=clrs[0:num_classes]
        xranges=[]
        xticks=[]
        xmin=1
        for c in range(num_classes):
            xwidth=num_samples_per_class[c]
            xranges.append((xmin,xwidth))
            #xticks.extend([xmin+math.floor(xwidth/2)])
            xticks.extend([xmin+xwidth/2])
            xmin=xmin+xwidth
    
        if hv=="horizontal":
            ax.set_ylim(0,1)
            ax.set_xlim(1,num_samples+1)
            ax.broken_barh(xranges,yrange,facecolors=facecolors)
            ax.set_xticks(xticks)
            ax.xaxis.tick_top()
            ax.set_xticklabels(unique_class_names.astype(str),fontsize=fontsize,rotation=rotation)
            ax.set_yticklabels([])
        elif hv=="vertical":
            ax.set_ylim(1,num_samples+1)
            ax.set_xlim(0,1)
            for c in range(num_classes):
                xr=[(0,1)]
                yr=xranges[c]
                color=clrs[c]
                ax.broken_barh(xr,yr,facecolors=color)
            #ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks(xticks)
            ax.set_yticklabels(unique_class_names.astype(str), fontsize=fontsize, rotation=rotation)
            if ylabel_right:
                ax.yaxis.tick_right()
            ax.invert_yaxis() #invert y axis
        ax.tick_params(labelsize=fontsize)

        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False

    def plot_heatmap(self,dir_save="./", prefix="MCNMF", pattern="All", rank_method="mean_feature_value",unique_class_names=None, width=10, height=10, fontsize=6, fmt="png", dpi=600, colormap="hot", clrs=None, rng=numpy.random.RandomState(1000)):
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
        import matplotlib as mpl
        #mpl.use(fmt)
        import matplotlib.pyplot as plt
        from matplotlib import colors
        print "ploting heatmap ..."
        # color map
        if colormap=="Reds":
            cmap=plt.cm.Reds
        elif colormap=="Blues":
            cmap=plt.cm.Blues
        elif colormap=="hot":
            cmap=plt.cm.hot
        elif colormap=="cool":
            cmap=plt.cm.cool
        # colors of classes and feature groups
        if clrs is None:
            cnames=colors.cnames
            clrs=[name for name,hex in cnames.iteritems()]
            num_clrs=len(clrs)
            clrs=numpy.array(clrs)
        if self.clrs_classes is None:
            # permute all color names
            ind_clrs=rng.choice(num_clrs,self.V+1)
            self.clrs_classes=clrs[ind_clrs]
        if self.clrs_feature_groups is None:
            ind_clrs=rng.choice(num_clrs,2**(self.V+1))
            self.clrs_feature_groups=clrs[ind_clrs]

        if unique_class_names is not None:
            unique_class_names=numpy.hstack((["ubi"],unique_class_names))

        #reorder features
        if rank_method=="mean_feature_value":
            ind=self.ind_mean_feature_value
        elif rank_method=="mean_basis_value":
            ind=self.ind_mean_basis_value
        elif rank_method=="Wilcoxon_rank_sum_test":
            if self.allow_compute_pval:
                ind=self.ind_Wilcoxon_rank_sum_test
            else:
                print "You choose to reorder the features by Wilcoxon_rank_sum_test' pvals, but computing pvals is not allowed(allow_compute_pval=False). I decide to reorder the features by mean_basis_value instead."
                ind=self.ind_mean_basis_value
        else:
            print "Error! Please select the right rank method."
        
        E_W_sorted=self.E_W_scaled[ind,:]
        X_sorted=self.X[ind,:]
        #F_sorted=self.F[ind,:]
        F_str_sorted=self.F_str[ind]
        features_sorted=self.features[ind]
        #scores_mean_feature_value_sorted=self.scores_mean_feature_value[ind]
        #scores_mean_basis_value_sorted=self.scores_mean_basis_value[ind]
        #scores_Wilcoxon_rank_sum_test_sorted=self.scores_Wilcoxon_rank_sum_test[ind]
        #numeric_ind_sorted=numpy.array(ind,dtype=int)

        # get data first
        if pattern=="All":
            #X=numpy.log2(X_sorted+1)
            X,_,_=cl.normalize_row_scale01(X_sorted,clip=True,clip_min=0.5,clip_max=1e4)
            #X,_,_=cl.normalize_row_scale01(numpy.log2(X_sorted+1),clip=True,clip_min=0.1,clip_max=10)
            col_lab_X=numpy.asarray(self.y,dtype=str)
            row_lab_X=features_sorted
            #W=numpy.log2(E_W_sorted+1)
            W,_,_=cl.normalize_row_scale01(E_W_sorted,clip=True,clip_min=1e-1,clip_max=1e4)
            #W,_,_=cl.normalize_row_scale01(numpy.log2(E_W_sorted+1),clip=True,clip_min=0.1,clip_max=10)
            col_lab_W=numpy.asarray(self.z,dtype=str)
            row_lab_W=row_lab_X
            #H=numpy.log2(self.E_H+1)
            H,_,_=cl.normalize_col_scale01(self.E_H_scaled,clip=True,clip_min=1e-3,clip_max=1e3)
            #H=self.E_H_scaled # already scaled, because the maximum is 1 in each row
            #H,_,_=cl.normalize_col_scale01(numpy.log2(self.E_H_scaled+1),clip=True,clip_min=0.1,clip_max=10)
            col_lab_H=col_lab_X
            row_lab_H=col_lab_W
        else:
            ind_X=(F_str_sorted==pattern)
            #X=numpy.log2(X_sorted[ind_X,:]+1)
            X,_,_=cl.normalize_row_scale01(X_sorted[ind_X,:],clip=True,clip_min=0.5,clip_max=1e4)
            #X,_,_=cl.normalize_row_scale01(numpy.log2(X_sorted[ind_X,:]+1),clip=True,clip_min=0.1,clip_max=10)
            col_lab_X=numpy.asarray(self.y,dtype=str)
            row_lab_X=features_sorted[ind_X]
            #W=numpy.log2(E_W_sorted[ind_X,:]+1)
            W,_,_=cl.normalize_row_scale01(E_W_sorted[ind_X,:],clip=True,clip_min=1e-1,clip_max=1e4)
            #W,_,_=cl.normalize_row_scale01(numpy.log2(E_W_sorted[ind_X,:]+1),clip=True,clip_min=0.1,clip_max=10)
            col_lab_W=numpy.asarray(self.z,dtype=str)
            row_lab_W=row_lab_X
            #H=numpy.log2(self.E_H_scaled+1)
            H,_,_=cl.normalize_col_scale01(self.E_H_scaled,clip=True,clip_min=1e-3,clip_max=1e3)
            #H=self.E_H_scaled # already scaled, because the maximum is 1 in each row
            #H,_,_=cl.normalize_col_scale01(numpy.log2(self.E_H_scaled+1),clip=True,clip_min=0.1,clip_max=10)
            col_lab_H=col_lab_X
            row_lab_H=col_lab_W

        # plot the heatmaps
        #fig, ax = plt.subplots(1,3)
        fig=plt.figure(figsize=(width,height))
        allgrids=20
        ax0_rowspan=allgrids-2
        ax0_colspan=int(0.4*allgrids)-1
        ax0_rowstart=1
        ax0_colstart=0
        ax0ccl_rowspan=1
        ax0ccl_colspan=ax0_colspan
        ax0ccl_rowstart=0
        ax0ccl_colstart=0
        ax0c_rowspan=1
        ax0c_colspan=ax0_colspan
        ax0c_rowstart=allgrids-1
        ax0c_colstart=0
        ax0cfe_rowspan=ax0_rowspan
        ax0cfe_colspan=1
        ax0cfe_rowstart=1
        ax0cfe_colstart=ax0_colspan
        ax1_rowspan=ax0_rowspan
        ax1_colspan=int(0.2*allgrids)
        ax1_rowstart=1
        ax1_colstart=ax0_colspan+ax0cfe_colspan
        ax1cfa_rowspan=1
        ax1cfa_colspan=ax1_colspan
        ax1cfa_rowstart=0
        ax1cfa_colstart=ax1_colstart
        ax1c_rowspan=1
        ax1c_colspan=ax1_colspan
        ax1c_rowstart=allgrids-1
        ax1c_colstart=ax1_colstart
        ax2_rowspan=ax1_colspan
        ax2_colspan=ax0_colspan
        ax2_rowstart=1
        ax2_colstart=ax1_colstart+ax1_colspan+1
        ax2ccl_rowspan=1
        ax2ccl_colspan=ax2_colspan
        ax2ccl_rowstart=0
        ax2ccl_colstart=ax2_colstart
        ax2cfa_rowspan=ax2_rowspan
        ax2cfa_colspan=1
        ax2cfa_rowstart=1
        ax2cfa_colstart=ax1_colstart+ax1_colspan
        ax2c_rowspan=1
        ax2c_colspan=ax2_colspan
        ax2c_rowstart=ax2_rowspan+ax2ccl_rowspan
        ax2c_colstart=ax2_colstart
              
        # plot X
        ax0 = plt.subplot2grid((allgrids, allgrids), (ax0_rowstart,ax0_colstart), rowspan=ax0_rowspan, colspan=ax0_colspan)
        heatmap_X = ax0.pcolormesh(X, cmap=cmap)
        #ax0.set_frame_on(True)
        ax0.grid(False)
        # put the major ticks at the middle of each cell
        ax0.set_xticks(numpy.arange(X.shape[1])+0.5, minor=False)
        ax0.set_yticks(numpy.arange(X.shape[0])+0.5, minor=False)
        # want a more natural, table-like display
        ax0.invert_yaxis()
        ax0.xaxis.tick_top()
        # add labels
        #ax0.set_xticklabels(col_lab_X, minor=False,fontsize=fontsize-math.floor(math.log10(len(col_lab_X))))
        ax0.set_xticklabels([], minor=False,fontsize=fontsize-math.floor(math.log10(len(col_lab_X))))
        ax0.set_yticklabels(row_lab_X, minor=False,fontsize=fontsize-math.floor(math.log10(len(row_lab_X))))
        # turn off ticks
        for t in ax0.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax0.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        # reduce the white margin
        ax0.axis("tight")
        #ax0.set_xlim(0,X.shape[1]+1)
        #ax0.set_ylim(0,X.shape[0]+1)

        # color bar
        ax0c=plt.subplot2grid((allgrids, allgrids), (ax0c_rowstart, ax0c_colstart), rowspan=ax0c_rowspan, colspan=ax0c_colspan)
        cbar_X = plt.colorbar(heatmap_X,orientation="horizontal",spacing="proportional",cax=ax0c,use_gridspec=False)
        cbar_X.ax.tick_params(labelsize=fontsize-2)
        # color bar for classes
        ax0ccl=plt.subplot2grid((allgrids, allgrids), (ax0ccl_rowstart, ax0ccl_colstart), rowspan=ax0ccl_rowspan, colspan=ax0ccl_colspan)
        self.colorbar_classes(self.y,ax=ax0ccl,hv="horizontal",clrs=self.clrs_classes[1:],unique_class_names=unique_class_names[1:],fontsize=fontsize)
        # color bar for feature groups
        ax0cfe=plt.subplot2grid((allgrids, allgrids), (ax0cfe_rowstart, ax0cfe_colstart), rowspan=ax0cfe_rowspan, colspan=ax0cfe_colspan)
        self.colorbar_classes(F_str_sorted, ax=ax0cfe,hv="vertical",clrs=self.clrs_feature_groups,unique_classes=self.s_str,unique_class_names=None,fontsize=fontsize-2,ylabel_right=True,rotation=0)

        # plot W
        ax1 = plt.subplot2grid((allgrids, allgrids), (ax1_rowstart, ax1_colstart), rowspan=ax1_rowspan, colspan=ax1_colspan)
        heatmap_W = ax1.pcolormesh(W, cmap=cmap)
        # put the major ticks at the middle of each cell
        #ax1.set_frame_on(True)
        ax1.grid(False)
        ax1.set_xticks(numpy.arange(W.shape[1])+0.5, minor=False)
        ax1.set_yticks(numpy.arange(W.shape[0])+0.5, minor=False)
        # want a more natural, table-like display
        ax1.invert_yaxis()
        ax1.xaxis.tick_top()
        # add labels
        #ax1.set_xticklabels(col_lab_W, minor=False,fontsize=fontsize-math.floor(math.log10(len(col_lab_W))))
        #ax1.set_yticklabels(row_lab_W, minor=False,fontsize=fontsize-math.floor(math.log10(len(row_lab_W))))
        ax1.set_xticklabels([], minor=False,fontsize=fontsize-math.floor(math.log10(len(col_lab_W))))
        ax1.set_yticklabels([], minor=False,fontsize=fontsize-math.floor(math.log10(len(row_lab_W))))
        for t in ax1.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax1.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        ax1.axis("tight")
        #ax1.set_xlim(0,W.shape[1]+1)
        #ax1.set_ylim(0,W.shape[0]+1)

        # color bar
        ax1c=plt.subplot2grid((allgrids, allgrids), (ax1c_rowstart, ax1c_colstart), rowspan=ax1c_rowspan,colspan=ax1c_colspan)
        cbar_W = plt.colorbar(heatmap_W,orientation="horizontal",spacing="proportional",cax=ax1c,use_gridspec=False)
        cbar_W.ax.tick_params(labelsize=fontsize-2)
        # color bar for factors
        ax1cfa=plt.subplot2grid((allgrids, allgrids), (ax1cfa_rowstart, ax1cfa_colstart), rowspan=ax1cfa_rowspan, colspan=ax1cfa_colspan)
        self.colorbar_classes(self.z,ax=ax1cfa,hv="horizontal",clrs=self.clrs_classes,unique_class_names=unique_class_names,fontsize=fontsize)

        # plot H
        ax2 = plt.subplot2grid((allgrids, allgrids), (ax2_rowstart, ax2_colstart), rowspan=ax2_rowspan, colspan=ax2_colspan)
        heatmap_H = ax2.pcolormesh(H, cmap=cmap)
        # put the major ticks at the middle of each cell
        #ax2.set_frame_on(True)
        ax2.grid(False)
        ax2.set_xticks(numpy.arange(H.shape[1])+0.5, minor=False)
        ax2.set_yticks(numpy.arange(H.shape[0])+0.5, minor=False)
        # want a more natural, table-like display
        ax2.invert_yaxis()
        ax2.xaxis.tick_top()
        # add labels
        #ax2.set_xticklabels(col_lab_H, minor=False,fontsize=fontsize-math.floor(math.log10(len(col_lab_H))))
        #ax2.set_yticklabels(row_lab_H, minor=False,fontsize=fontsize-math.floor(math.log10(len(row_lab_H))))
        ax2.set_xticklabels([], minor=False,fontsize=fontsize-math.floor(math.log10(len(col_lab_H))))
        ax2.set_yticklabels([], minor=False,fontsize=fontsize-math.floor(math.log10(len(row_lab_H))))
        for t in ax2.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax2.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        ax2.axis("tight")
        #ax2.set_xlim(0,H.shape[1]+1)
        #ax2.set_ylim(0,H.shape[0]+1)
        
        # color bar
        ax2c=plt.subplot2grid((allgrids, allgrids), (ax2c_rowstart, ax2c_colstart) ,rowspan=ax2c_rowspan, colspan=ax2c_colspan)
        cbar_H = plt.colorbar(heatmap_H,orientation="horizontal",spacing="proportional",cax=ax2c,use_gridspec=False)
        cbar_H.ax.tick_params(labelsize=fontsize-2)
        # color bar for factors
        ax2cfa=plt.subplot2grid((allgrids, allgrids), (ax2cfa_rowstart, ax2cfa_colstart), rowspan=ax2cfa_rowspan, colspan=ax2cfa_colspan)
        self.colorbar_classes(self.z,ax=ax2cfa,hv="vertical",clrs=self.clrs_classes,unique_class_names=unique_class_names,fontsize=fontsize-2,rotation=90)
        # color bar for classes
        ax2ccl=plt.subplot2grid((allgrids, allgrids), (ax2ccl_rowstart, ax2ccl_colstart), rowspan=ax2ccl_rowspan, colspan=ax2ccl_colspan)
        self.colorbar_classes(self.y,ax=ax2ccl,hv="horizontal",clrs=self.clrs_classes[1:],unique_class_names=unique_class_names[1:],fontsize=fontsize)        

        filename=dir_save+prefix+"_"+pattern+"_ranked_by_"+rank_method+"_heatmap_XWH."+fmt
        plt.tight_layout()
        #fig.savefig(filename,bbox_inches='tight',format="png")
        #fig.savefig(filename,bbox_inches='tight',format=fmt,dpi=dpi)
        fig.savefig(filename,format=fmt,dpi=dpi)
        plt.close(fig)


    def plot_heatmap_given_XWH(self, X=None, W=None, H=None, y=None, z=None, features=None, feature_patterns=None, clrs_classes=None, clrs_feature_groups=None, dir_save="./", prefix="MCNMF", pattern="All", rank_method="original", given_feature_rank_ind=None, unique_class_names=None, width=10, height=10, fontsize=6, fmt="png", dpi=600, colormap="hot", clrs=None, rng=numpy.random.RandomState(1000)):
        """
        Plot heatmaps of the decomposition result, given a pattern or all patterns. 

        INPUTS:
        X: numpy matrix, data, each column is a sample.
        W: numpy matrix, the basis matrix.
        H: numpy matrix, the coefficient matrix.
        y: numpy 1d array, the class labels of the samples in the columns of X.
        z: integer, tuple of length V+1, list/numpy vector of size (K,), the labels of the factors (columns) in W. If z is a scalar, it means each view (including the ubi view) has z factors. If z is a tuple, e.g. (3,3,3,3), z[u] means the the u-th view has z[u] factors. If z is a list or numpy vector,e.g. [-1-1-1,0,0,1,1,1,2,2,2] where -1 means ubi view, z[k] means the k-th factor has label z[k]. 
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
        
        M=X.shape[0] # number of features
        N=X.shape[1] # number of samples
        Y,y_unique=cl.membership_vector_to_indicator_matrix(y)
        V=len(y_unique)
        # scale W and H
        row_max_H=numpy.max(H,axis=1)
        SCALE=numpy.diag(row_max_H)
        SCALE_inv=numpy.diag(1/row_max_H)
        H_scaled=numpy.dot(SCALE_inv,H)
        W_scaled=numpy.dot(W,SCALE)
        if isinstance(z,(list,numpy.ndarray)):
            z=numpy.array(z,dtype=int) # e.g. [-1,-1,-1,0,0,0,1,1,2,2,2], -1: common factors accross all classes, 0,...,V-1: class labels.
        elif isinstance(z,tuple):
            z=self.factor_sizes_to_factor_labels(z) # e.g. (3,3,3,3) to [-1,-1,-1,0,0,0,1,1,2,2,2]
        else:
            z=self.factor_sizes_to_factor_labels([z]*(V+1)) # e.g. 3 to [-1,-1,-1,0,0,0,1,1,2,2,2]
        K=len(z) # number of latent factors 
        Z,z_unique=cl.membership_vector_to_indicator_matrix(z) # binary, size K by V+1, self.Z[k,u]=1 indicates the k-th factor in class u.        
        
        #import matplotlib as mpl
        #mpl.use(fmt)
        import matplotlib.pyplot as plt
        from matplotlib import colors
        print "ploting heatmap ..."
        # color map
        if colormap=="Reds":
            cmap=plt.cm.Reds
        elif colormap=="Blues":
            cmap=plt.cm.Blues
        elif colormap=="hot":
            cmap=plt.cm.hot
        elif colormap=="cool":
            cmap=plt.cm.cool
        # colors of classes and feature groups
        if clrs is None:
            cnames=colors.cnames
            clrs=[name for name,hex in cnames.iteritems()]
            num_clrs=len(clrs)
            clrs=numpy.array(clrs)
        if clrs_classes is None:
            # permute all color names
            ind_clrs=rng.choice(num_clrs,V+1)
            clrs_classes=clrs[ind_clrs]
        if clrs_feature_groups is None:
            ind_clrs=rng.choice(num_clrs,2**(V+1))
            clrs_feature_groups=clrs[ind_clrs]
            
        if unique_class_names is None:
            unique_class_names=numpy.array(y_unique,dtype=str)
        unique_class_names=numpy.hstack((["ubi"],unique_class_names))

        #reorder features
        if rank_method=="original":
            ind=range(M)
        elif rank_method=="given":
            ind=given_feature_rank_ind
        else:
            print "You can either decide to use the original order of the features, or given an ordered list of features you want to use."

        # sort X and W        
        X_sorted=X[ind,:]
        W_sorted=W_scaled[ind,:]
        feature_patterns=numpy.array(feature_patterns)
        feature_patterns_sorted=feature_patterns[ind]
        features=numpy.array(features)
        features_sorted=features[ind]

        # get data first
        if pattern=="All":
            #X=numpy.log2(X_sorted+1)
            X,_,_=cl.normalize_row_scale01(X_sorted,clip=True,clip_min=0.5,clip_max=1e4)
            #X,_,_=cl.normalize_row_scale01(numpy.log2(X_sorted+1),clip=True,clip_min=0.1,clip_max=10)
            col_lab_X=numpy.asarray(y,dtype=str)
            row_lab_X=features_sorted
            #W=numpy.log2(E_W_sorted+1)
            W,_,_=cl.normalize_row_scale01(W_sorted,clip=True,clip_min=1e-1,clip_max=1e4)
            #W,_,_=cl.normalize_row_scale01(numpy.log2(E_W_sorted+1),clip=True,clip_min=0.1,clip_max=10)
            col_lab_W=numpy.asarray(z,dtype=str)
            row_lab_W=row_lab_X
            #H=numpy.log2(self.E_H+1)
            H,_,_=cl.normalize_col_scale01(H_scaled,clip=True,clip_min=1e-3,clip_max=1e3)
            #H=self.E_H_scaled # already scaled, because the maximum is 1 in each row
            #H,_,_=cl.normalize_col_scale01(numpy.log2(self.E_H_scaled+1),clip=True,clip_min=0.1,clip_max=10)
            col_lab_H=col_lab_X
            row_lab_H=col_lab_W
        else:
            ind_X=(feature_patterns_sorted==pattern)
            #X=numpy.log2(X_sorted[ind_X,:]+1)
            X,_,_=cl.normalize_row_scale01(X_sorted[ind_X,:],clip=True,clip_min=0.5,clip_max=1e4)
            #X,_,_=cl.normalize_row_scale01(numpy.log2(X_sorted[ind_X,:]+1),clip=True,clip_min=0.1,clip_max=10)
            col_lab_X=numpy.asarray(y,dtype=str)
            row_lab_X=features_sorted[ind_X]
            #W=numpy.log2(E_W_sorted[ind_X,:]+1)
            W,_,_=cl.normalize_row_scale01(W_sorted[ind_X,:],clip=True,clip_min=1e-1,clip_max=1e4)
            #W,_,_=cl.normalize_row_scale01(numpy.log2(E_W_sorted[ind_X,:]+1),clip=True,clip_min=0.1,clip_max=10)
            col_lab_W=numpy.asarray(z,dtype=str)
            row_lab_W=row_lab_X
            #H=numpy.log2(self.E_H_scaled+1)
            H,_,_=cl.normalize_col_scale01(H_scaled,clip=True,clip_min=1e-3,clip_max=1e3)
            #H=self.E_H_scaled # already scaled, because the maximum is 1 in each row
            #H,_,_=cl.normalize_col_scale01(numpy.log2(self.E_H_scaled+1),clip=True,clip_min=0.1,clip_max=10)
            col_lab_H=col_lab_X
            row_lab_H=col_lab_W

        # plot the heatmaps
        #fig, ax = plt.subplots(1,3)
        fig=plt.figure(figsize=(width,height))
        allgrids=20
        ax0_rowspan=allgrids-2
        ax0_colspan=int(0.4*allgrids)-1
        ax0_rowstart=1
        ax0_colstart=0
        ax0ccl_rowspan=1
        ax0ccl_colspan=ax0_colspan
        ax0ccl_rowstart=0
        ax0ccl_colstart=0
        ax0c_rowspan=1
        ax0c_colspan=ax0_colspan
        ax0c_rowstart=allgrids-1
        ax0c_colstart=0
        ax0cfe_rowspan=ax0_rowspan
        ax0cfe_colspan=1
        ax0cfe_rowstart=1
        ax0cfe_colstart=ax0_colspan
        ax1_rowspan=ax0_rowspan
        ax1_colspan=int(0.2*allgrids)
        ax1_rowstart=1
        ax1_colstart=ax0_colspan+ax0cfe_colspan
        ax1cfa_rowspan=1
        ax1cfa_colspan=ax1_colspan
        ax1cfa_rowstart=0
        ax1cfa_colstart=ax1_colstart
        ax1c_rowspan=1
        ax1c_colspan=ax1_colspan
        ax1c_rowstart=allgrids-1
        ax1c_colstart=ax1_colstart
        ax2_rowspan=ax1_colspan
        ax2_colspan=ax0_colspan
        ax2_rowstart=1
        ax2_colstart=ax1_colstart+ax1_colspan+1
        ax2ccl_rowspan=1
        ax2ccl_colspan=ax2_colspan
        ax2ccl_rowstart=0
        ax2ccl_colstart=ax2_colstart
        ax2cfa_rowspan=ax2_rowspan
        ax2cfa_colspan=1
        ax2cfa_rowstart=1
        ax2cfa_colstart=ax1_colstart+ax1_colspan
        ax2c_rowspan=1
        ax2c_colspan=ax2_colspan
        ax2c_rowstart=ax2_rowspan+ax2ccl_rowspan
        ax2c_colstart=ax2_colstart
              
        # plot X
        ax0 = plt.subplot2grid((allgrids, allgrids), (ax0_rowstart,ax0_colstart), rowspan=ax0_rowspan, colspan=ax0_colspan)
        heatmap_X = ax0.pcolormesh(X, cmap=cmap)
        #ax0.set_frame_on(True)
        ax0.grid(False)
        # put the major ticks at the middle of each cell
        ax0.set_xticks(numpy.arange(X.shape[1])+0.5, minor=False)
        ax0.set_yticks(numpy.arange(X.shape[0])+0.5, minor=False)
        # want a more natural, table-like display
        ax0.invert_yaxis()
        ax0.xaxis.tick_top()
        # add labels
        #ax0.set_xticklabels(col_lab_X, minor=False,fontsize=fontsize-math.floor(math.log10(len(col_lab_X))))
        ax0.set_xticklabels([], minor=False,fontsize=fontsize-math.floor(math.log10(len(col_lab_X))))
        ax0.set_yticklabels(row_lab_X, minor=False,fontsize=fontsize-math.floor(math.log10(len(row_lab_X))))
        # turn off ticks
        for t in ax0.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax0.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        # reduce the white margin
        ax0.axis("tight")
        #ax0.set_xlim(0,X.shape[1]+1)
        #ax0.set_ylim(0,X.shape[0]+1)

        # color bar
        ax0c=plt.subplot2grid((allgrids, allgrids), (ax0c_rowstart, ax0c_colstart), rowspan=ax0c_rowspan, colspan=ax0c_colspan)
        cbar_X = plt.colorbar(heatmap_X,orientation="horizontal",spacing="proportional",cax=ax0c,use_gridspec=False)
        cbar_X.ax.tick_params(labelsize=fontsize-2)
        # color bar for classes
        ax0ccl=plt.subplot2grid((allgrids, allgrids), (ax0ccl_rowstart, ax0ccl_colstart), rowspan=ax0ccl_rowspan, colspan=ax0ccl_colspan)
        print clrs_classes
        print unique_class_names
        self.colorbar_classes(y,ax=ax0ccl,hv="horizontal",clrs=clrs_classes[1:],unique_class_names=unique_class_names[1:],fontsize=fontsize)
        # color bar for feature groups
        ax0cfe=plt.subplot2grid((allgrids, allgrids), (ax0cfe_rowstart, ax0cfe_colstart), rowspan=ax0cfe_rowspan, colspan=ax0cfe_colspan)
        self.colorbar_classes(feature_patterns_sorted, ax=ax0cfe,hv="vertical",clrs=clrs_feature_groups,unique_classes=numpy.unique(feature_patterns_sorted),unique_class_names=None,fontsize=fontsize-2,ylabel_right=True,rotation=0)

        # plot W
        ax1 = plt.subplot2grid((allgrids, allgrids), (ax1_rowstart, ax1_colstart), rowspan=ax1_rowspan, colspan=ax1_colspan)
        heatmap_W = ax1.pcolormesh(W, cmap=cmap)
        # put the major ticks at the middle of each cell
        #ax1.set_frame_on(True)
        ax1.grid(False)
        ax1.set_xticks(numpy.arange(W.shape[1])+0.5, minor=False)
        ax1.set_yticks(numpy.arange(W.shape[0])+0.5, minor=False)
        # want a more natural, table-like display
        ax1.invert_yaxis()
        ax1.xaxis.tick_top()
        # add labels
        #ax1.set_xticklabels(col_lab_W, minor=False,fontsize=fontsize-math.floor(math.log10(len(col_lab_W))))
        #ax1.set_yticklabels(row_lab_W, minor=False,fontsize=fontsize-math.floor(math.log10(len(row_lab_W))))
        ax1.set_xticklabels([], minor=False,fontsize=fontsize-math.floor(math.log10(len(col_lab_W))))
        ax1.set_yticklabels([], minor=False,fontsize=fontsize-math.floor(math.log10(len(row_lab_W))))
        for t in ax1.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax1.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        ax1.axis("tight")
        #ax1.set_xlim(0,W.shape[1]+1)
        #ax1.set_ylim(0,W.shape[0]+1)

        # color bar
        ax1c=plt.subplot2grid((allgrids, allgrids), (ax1c_rowstart, ax1c_colstart), rowspan=ax1c_rowspan,colspan=ax1c_colspan)
        cbar_W = plt.colorbar(heatmap_W,orientation="horizontal",spacing="proportional",cax=ax1c,use_gridspec=False)
        cbar_W.ax.tick_params(labelsize=fontsize-2)
        # color bar for factors
        ax1cfa=plt.subplot2grid((allgrids, allgrids), (ax1cfa_rowstart, ax1cfa_colstart), rowspan=ax1cfa_rowspan, colspan=ax1cfa_colspan)
        self.colorbar_classes(z,ax=ax1cfa,hv="horizontal",clrs=clrs_classes,unique_class_names=unique_class_names,fontsize=fontsize)

        # plot H
        ax2 = plt.subplot2grid((allgrids, allgrids), (ax2_rowstart, ax2_colstart), rowspan=ax2_rowspan, colspan=ax2_colspan)
        heatmap_H = ax2.pcolormesh(H, cmap=cmap)
        # put the major ticks at the middle of each cell
        #ax2.set_frame_on(True)
        ax2.grid(False)
        ax2.set_xticks(numpy.arange(H.shape[1])+0.5, minor=False)
        ax2.set_yticks(numpy.arange(H.shape[0])+0.5, minor=False)
        # want a more natural, table-like display
        ax2.invert_yaxis()
        ax2.xaxis.tick_top()
        # add labels
        #ax2.set_xticklabels(col_lab_H, minor=False,fontsize=fontsize-math.floor(math.log10(len(col_lab_H))))
        #ax2.set_yticklabels(row_lab_H, minor=False,fontsize=fontsize-math.floor(math.log10(len(row_lab_H))))
        ax2.set_xticklabels([], minor=False,fontsize=fontsize-math.floor(math.log10(len(col_lab_H))))
        ax2.set_yticklabels([], minor=False,fontsize=fontsize-math.floor(math.log10(len(row_lab_H))))
        for t in ax2.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax2.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        ax2.axis("tight")
        #ax2.set_xlim(0,H.shape[1]+1)
        #ax2.set_ylim(0,H.shape[0]+1)
        
        # color bar
        ax2c=plt.subplot2grid((allgrids, allgrids), (ax2c_rowstart, ax2c_colstart) ,rowspan=ax2c_rowspan, colspan=ax2c_colspan)
        cbar_H = plt.colorbar(heatmap_H,orientation="horizontal",spacing="proportional",cax=ax2c,use_gridspec=False)
        cbar_H.ax.tick_params(labelsize=fontsize-2)
        # color bar for factors
        ax2cfa=plt.subplot2grid((allgrids, allgrids), (ax2cfa_rowstart, ax2cfa_colstart), rowspan=ax2cfa_rowspan, colspan=ax2cfa_colspan)
        self.colorbar_classes(z,ax=ax2cfa,hv="vertical",clrs=clrs_classes,unique_class_names=unique_class_names,fontsize=fontsize-2,rotation=90)
        # color bar for classes
        ax2ccl=plt.subplot2grid((allgrids, allgrids), (ax2ccl_rowstart, ax2ccl_colstart), rowspan=ax2ccl_rowspan, colspan=ax2ccl_colspan)
        self.colorbar_classes(y,ax=ax2ccl,hv="horizontal",clrs=clrs_classes[1:],unique_class_names=unique_class_names[1:],fontsize=fontsize)        

        filename=dir_save+prefix+"_"+pattern+"_ranked_by_"+rank_method+"_heatmap_XWH."+fmt
        plt.tight_layout()
        #fig.savefig(filename,bbox_inches='tight',format="png")
        #fig.savefig(filename,bbox_inches='tight',format=fmt,dpi=dpi)
        fig.savefig(filename,format=fmt,dpi=dpi)
        plt.close(fig)


    def plot_F_mean(self,dir_save="./", prefix="MVNMF", normalize=False,rank_method="mean_feature_value",unique_class_names=None, width=10, height=10, fontsize=6, fmt="png", dpi=600, colormap="hot", clrs=None, rng=numpy.random.RandomState(1000)):
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
        import matplotlib as mpl
        #mpl.use(fmt)
        import matplotlib.pyplot as plt
        from matplotlib import colors
        print "ploting heatmap ..."
        # color map
        if colormap=="Reds":
            cmap=plt.cm.Reds
        elif colormap=="Blues":
            cmap=plt.cm.Blues
        elif colormap=="hot":
            cmap=plt.cm.hot
        elif colormap=="cool":
            cmap=plt.cm.cool
        # colors of classes and feature groups
        if clrs is None:
            cnames=colors.cnames
            clrs=[name for name,hex in cnames.iteritems()]
            num_clrs=len(clrs)
            clrs=numpy.array(clrs)
        if self.clrs_classes is None:
            # permute all color names
            ind_clrs=rng.choice(num_clrs,self.V+1)
            self.clrs_classes=clrs[ind_clrs]
        if self.clrs_feature_groups is None:
            ind_clrs=rng.choice(num_clrs,2**(self.V+1))
            self.clrs_feature_groups=clrs[ind_clrs]

        if unique_class_names is not None:
            unique_class_names=numpy.hstack((["ubi"],unique_class_names))

        #reorder features
        if rank_method=="mean_feature_value":
            ind=self.ind_mean_feature_value
        elif rank_method=="mean_basis_value":
            ind=self.ind_mean_basis_value
        elif rank_method=="Wilcoxon_rank_sum_test":
            if self.allow_compute_pval:
                ind=self.ind_Wilcoxon_rank_sum_test
            else:
                print "You choose to reorder the features by Wilcoxon_rank_sum_test' pvals, but computing pvals is not allowed(allow_compute_pval=False). I decide to reorder the features by mean_basis_value instead."
                ind=self.ind_mean_basis_value
        else:
            print "Error! Please select the right rank method."
        
        
        F_mean_sorted=self.F_mean[ind,:]
        F_str_sorted=self.F_str[ind]
        features_sorted=self.features[ind]
        if normalize:
            F_mean,_,_=cl.normalize_row_scale01(X_sorted,clip=True,clip_min=1e-1,clip_max=1e4)
        self.z_unique=numpy.unique(self.z)
        col_lab=numpy.array(self.z_unique,dtype=str)
        row_lab=features_sorted

        # plot the heatmaps
        #fig, ax = plt.subplots(1,3)
        fig=plt.figure(figsize=(width,height))
        allgrids=20
        # main heatmap
        ax0_rowspan=allgrids-2
        ax0_colspan=allgrids-1
        ax0_rowstart=1
        ax0_colstart=0
        # color bar of classes/views
        ax0ccl_rowspan=1
        ax0ccl_colspan=ax0_colspan
        ax0ccl_rowstart=0
        ax0ccl_colstart=0
        # color at the bottom
        ax0c_rowspan=1
        ax0c_colspan=ax0_colspan
        ax0c_rowstart=allgrids-1
        ax0c_colstart=0
        # color bar of features
        ax0cfe_rowspan=ax0_rowspan
        ax0cfe_colspan=1
        ax0cfe_rowstart=1
        ax0cfe_colstart=ax0_colspan
              
        # plot F_mean
        ax0 = plt.subplot2grid((allgrids, allgrids), (ax0_rowstart,ax0_colstart), rowspan=ax0_rowspan, colspan=ax0_colspan)
        heatmap_F_mean_sorted = ax0.pcolormesh(F_mean_sorted, cmap=cmap)
        #ax0.set_frame_on(True)
        ax0.grid(False)
        # put the major ticks at the middle of each cell
        ax0.set_xticks(numpy.arange(F_mean_sorted.shape[1])+0.5, minor=False)
        ax0.set_yticks(numpy.arange(F_mean_sorted.shape[0])+0.5, minor=False)
        # want a more natural, table-like display
        ax0.invert_yaxis()
        ax0.xaxis.tick_top()
        # add labels
        #ax0.set_xticklabels(col_lab, minor=False,fontsize=fontsize-math.floor(math.log10(len(col_lab_X))))
        ax0.set_xticklabels([], minor=False,fontsize=fontsize-math.floor(math.log10(len(col_lab))))
        ax0.set_yticklabels(row_lab, minor=False,fontsize=fontsize-math.floor(math.log10(len(row_lab))))
        # turn off ticks
        for t in ax0.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax0.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        # reduce the white margin
        ax0.axis("tight")
        #ax0.set_xlim(0,X.shape[1]+1)
        #ax0.set_ylim(0,X.shape[0]+1)

        # color bar at the bottom
        ax0c=plt.subplot2grid((allgrids, allgrids), (ax0c_rowstart, ax0c_colstart), rowspan=ax0c_rowspan, colspan=ax0c_colspan)
        cbar_F_mean_sorted = plt.colorbar(heatmap_F_mean_sorted,orientation="horizontal",spacing="proportional",cax=ax0c,use_gridspec=False)
        cbar_F_mean_sorted.ax.tick_params(labelsize=fontsize-2)
        # color bar for classes
        ax0ccl=plt.subplot2grid((allgrids, allgrids), (ax0ccl_rowstart, ax0ccl_colstart), rowspan=ax0ccl_rowspan, colspan=ax0ccl_colspan)
        self.colorbar_classes(self.z_unique,ax=ax0ccl,hv="horizontal",clrs=self.clrs_classes,unique_class_names=unique_class_names,fontsize=fontsize)
        # color bar for feature groups
        ax0cfe=plt.subplot2grid((allgrids, allgrids), (ax0cfe_rowstart, ax0cfe_colstart), rowspan=ax0cfe_rowspan, colspan=ax0cfe_colspan)
        self.colorbar_classes(F_str_sorted, ax=ax0cfe,hv="vertical",clrs=self.clrs_feature_groups,unique_classes=self.s_str,unique_class_names=None,fontsize=fontsize-2,ylabel_right=True,rotation=0)

        filename=dir_save+prefix+"_ranked_by_"+rank_method+"_heatmap_F_mean."+fmt
        plt.tight_layout()
        #fig.savefig(filename,bbox_inches='tight',format="png")
        #fig.savefig(filename,bbox_inches='tight',format=fmt,dpi=dpi)
        fig.savefig(filename,format=fmt,dpi=dpi)
        plt.close(fig)     
        

    def learn_H_given_X_test_and_E_W(self,X_test,a_H_test=0.1,b_H_test=1e-10,feature_selection=True,max_iter=200,compute_variational_lower_bound=True,variational_lower_bound_min_rate=1e-4,if_plot_lower_bound=True,dir_save="./",prefix="MVNMF",verb=True,rng=numpy.random.RandomState(1000)):
        """
        Learn the coefficient matrix given test data X_test and learned E_W. This function is useful for MV-NMF based feature extraction or classification.
        
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
        
        start_time=time.clock()
        self.test_time_L=0

        self.X_test=X_test # store the original X_test
        self.a_H_test=a_H_test
        self.b_H_test=b_H_test
        self.N_test=self.X_test.shape[1] # number of samples
        if feature_selection:
            X_test=self.X_test[self.numeric_ind_key_feat_for_classification,:]
            E_W=self.E_W[self.numeric_ind_key_feat_for_classification,:]
            L_W=self.L_W[self.numeric_ind_key_feat_for_classification,:]
        else:
            X_test=self.X_test
            E_W=self.E_W
            L_W=self.L_W

        #initiate
        #sample E_Lambda_W and E_Lambda_H from Gamma prior distributions
        #E_Lambda_W=self.rng.gamma(shape=self.a_0, scale=1/self.b_0, size=(self.M,self.V+1))
        E_Lambda_H=self.rng.gamma(shape=self.a_H_test, scale=1/self.b_H_test, size=(self.V+1,1))
        #compute A_H, and B_H using prior expoential distribution(1,lambda)
        A_H=numpy.ones(shape=(self.K,self.N_test),dtype=float)
        ONES_1_N=numpy.ones(shape=(1,self.N_test),dtype=float)
        ONES_N_1=numpy.ones(shape=(self.N_test,1),dtype=float)
        B_H=self.Z.dot(E_Lambda_H).dot(ONES_1_N) + self.tol
        E_H=self.rng.gamma(shape=A_H,scale=1/B_H)
        L_H=E_H
        L_W_L_H=L_W.dot(L_H)

        if compute_variational_lower_bound:
            neg_gammaln_X_plus_1=-scipy.special.gammaln(X_test+1)
            Ls=[] # lower bound, start from iteration 1
            Ls_rates=[] # change rate, start from iteration 2
            Ls_rates_means=[] # local mean change rate, start from iteration 2
            mean_over=5 # compute the mean rate over this number of iterations

        ONES_M_N=numpy.ones_like(X_test)
        ONES_K_1=numpy.ones(shape=(self.K,1),dtype=float)

        num_iter=1
        while num_iter<=self.max_iter:
            if verb:
                print "iteration: {0}".format(num_iter)
            #compute Sigma_H, need L_W and L_H
            X_div_L_W_L_H=X_test/L_W_L_H
            Sigma_H=L_H.transpose() * X_div_L_W_L_H.transpose().dot(L_W) 

            #compute A_H, and B_H need Sigma_H, E_W, E_H, E_Lambda_W, and E_Lambda_H
            A_H=1 + Sigma_H.transpose()
            B_H=self.Z.dot(E_Lambda_H).dot(ONES_1_N) + E_W.transpose().dot(ONES_M_N)
            #compute E_W and E_H, need A_W, B_W, A_H, and B_H
            E_H=A_H/B_H
           
            #compute L_H, need A_H, and B_H
            psi_A_H=scipy.special.psi(A_H)
            exppsi_A_H=numpy.exp(psi_A_H)
            L_H=exppsi_A_H/B_H
            L_W_L_H=L_W.dot(L_H)

            #compute E_Lambda_H, need OLD E_H (not NEW)
            A_Lambda_H=self.a_H_test + self.Z.transpose().dot(ONES_K_1)*self.N_test
            B_Lambda_H=self.b_H_test + self.Z.transpose().dot(E_H).dot(ONES_N_1)
            E_Lambda_H=A_Lambda_H/B_Lambda_H
           
            # compute variational lower bound
            # variational lower bound
            if compute_variational_lower_bound:
                start_time_L=time.clock()
                psi_A_Lambda_H=scipy.special.psi(A_Lambda_H)
                log_B_Lambda_H=numpy.log(B_Lambda_H)
                L_Lambda_H=psi_A_Lambda_H - log_B_Lambda_H

                L=numpy.sum( neg_gammaln_X_plus_1 + X_test*numpy.log(L_W.dot(L_H))-E_W.dot(E_H)) + numpy.sum(self.Z.dot(L_Lambda_H).dot(ONES_1_N) - self.Z.dot(E_Lambda_H).dot(ONES_1_N)*E_H) + numpy.sum(self.a_H_test*numpy.log(self.b_H_test) - scipy.special.gammaln(self.a_H_test) + (self.a_H_test-1)*L_Lambda_H - self.b_H_test*E_Lambda_H) + numpy.sum(scipy.special.gammaln(A_H) - (A_H-1)*psi_A_H - numpy.log(B_H) + A_H) + numpy.sum(scipy.special.gammaln(A_Lambda_H) - (A_Lambda_H-1)*psi_A_Lambda_H - log_B_Lambda_H + A_Lambda_H)
                #L=numpy.sum( abs(X_test-E_W.dot(E_H))) # l1 norm 
                Ls.append(L)
                end_time_L=time.clock()
                self.test_time_L=self.test_time_L+(end_time_L-start_time_L)
                if num_iter>=2:
                    Ls_rates.append((Ls[-1]-Ls[-2])/(-Ls[-2]))
                    Ls_rates_mean=numpy.mean(Ls_rates[-mean_over:])
                    Ls_rates_means.append(Ls_rates_mean)
                    if verb:
                        print "The variational lower bound:{0}, local mean change rate:{1}".format(L,Ls_rates_mean)
                    if Ls_rates_mean<variational_lower_bound_min_rate:
                        break
            
            num_iter=num_iter+1

        #save the result
        self.E_H_test=E_H
        self.L_H_test=L_H
        self.E_Lambda_H_test=E_Lambda_H
        if compute_variational_lower_bound:
            self.Ls_test=Ls # lower bounds
            self.num_iter_test=num_iter-1 # actural number of iterations run
        
            # save data for plot for the lower bound, using Ls, Ls_rates_means, iters
            self.Ls_test=Ls[1:]
            self.Ls_rates_means_test=Ls_rates_means
            self.iters_test=range(2,(len(Ls_rates_means)+2) )
        else:
            self.Ls_test=None # lower bounds
            self.num_iter_test=num_iter-1 # actural number of iterations run
        
            # save data for plot for the lower bound, using Ls, Ls_rates_means, iters
            self.Ls_test=None
            self.Ls_rates_means_test=None
            self.iters_test=None


        end_time = time.clock()
        self.test_time=end_time-start_time
        print 'test factorization: time excluding computing variational lower bound: %f seconds' %(self.test_time)
        
        # plot lower bound
        if compute_variational_lower_bound and if_plot_lower_bound:
            print "ploting variational lower bounds ..."
            self.plot_lower_bound(dir_save=dir_save,prefix=prefix+"_test",iters=self.iters_test,Ls=self.Ls_test,Ls_rates_means=self.Ls_rates_means_test,figwidth=5,figheight=3)
            print 'test factorization: time for computing variational lower bound: %f seconds' %(self.test_time_L)

        print "finished fatorization for given test data!"

        return self.E_H_test,self.test_time,self.test_time_L


    def classify(self,feature_selection=True,method="coef_sum",method_param={"train_valid_ratio":(9,1,0),"feature_extraction":False,"learning_rate":0.1, "alpha":0.01, "lambda_reg":0.00001, "alpha_reg":0.5, "n_hidden":None, "n_epochs":1000, "batch_size":200, "activation_func":"relu"},rng=numpy.random.RandomState(1000)):
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
        if method=="coef_sum":
            start_time = time.clock()
            #probs=numpy.zeros(shape=(self.V,self.N_test),dtype=float)
            probs=self.Z.transpose().dot(self.E_H_test)
            probs=probs[1:,:]
            classes_predicted=numpy.argmax(probs,axis=0)
            # store prediction results
            self.test_classes_predicted=classes_predicted
            self.test_classes_prob_like,_,_=cl.normalize_col_scale01(probs,tol=1e-6,data_min=None,data_max=None,clip=False,clip_min=1e-3,clip_max=1e3)
            end_time = time.clock()
            self.classification_test_time=end_time-start_time
            self.classification_training_time=0
        elif method=="regression_residual":
            start_time = time.clock()
            if feature_selection:
                X_test=self.X_test[self.numeric_ind_key_feat_for_classification,:]
                E_W=self.E_W[self.numeric_ind_key_feat_for_classification,:]
            else:
                X_test=self.X_test
                E_W=self.E_W
            z0=self.Z[:,0]
            z0.shape=(z0.shape[0],1)
            ZV=self.Z[:,1:]+z0 # a K times V matrix
            residuals=numpy.zeros(shape=(self.V,self.N_test),dtype=float)
            for n in range(self.N_test):
                xn=X_test[:,n]
                xn.shape=(xn.shape[0],1)
                hn=self.E_H_test[:,n]
                hn.shape=(hn.shape[0],1)
                residuals[:,n]=numpy.sum(numpy.fabs(xn - E_W.dot(hn*ZV)),axis=0 ) 
            classes_predicted=numpy.argmin(residuals,axis=0)
            # store prediction results
            self.test_classes_predicted=classes_predicted
            self.test_classes_prob_like,_,_=cl.normalize_col_scale01(residuals,tol=1e-6,data_min=None,data_max=None,clip=False,clip_min=1e-3,clip_max=1e3)
            self.test_classes_prob_like=1-self.test_classes_prob_like
            end_time = time.clock()
            self.classification_test_time=end_time-start_time
            self.classification_training_time=0
        elif method=="mlp":
            # reduce dimensionality
            # if feature extraction is True, use the new features anyway.
            if method_param["feature_extraction"]:
                train_set_x=self.E_H
            elif feature_selection:
                train_set_x=self.X[self.numeric_ind_key_feat_for_classification,:]
            else:
                train_set_x=self.X

            train_set_x=numpy.transpose(train_set_x) # each row is a sample now
            train_set_y=self.y

            # split data
            train_subset_x,train_subset_y,_,valid_subset_x,valid_subset_y,_,test_subset_x,test_subset_y,_=cl.partition_train_valid_test2(train_set_x, train_set_y, others=None, ratio=method_param["train_valid_ratio"], rng=rng)
            
            # normalization
            train_subset_x,data_min,data_max=cl.normalize_col_scale01(train_subset_x,tol=1e-10)
            valid_subset_x,_,_=cl.normalize_col_scale01(valid_subset_x,tol=1e-10,data_min=data_min,data_max=data_max)

            # training
            learning_rate=method_param["learning_rate"]
            alpha=method_param["alpha"]
            lambda_reg=method_param["lambda_reg"]
            alpha_reg=method_param["alpha_reg"]
            n_hidden=method_param["n_hidden"]
            if n_hidden is None:
                n_hidden=[len(train_subset_x)] # set it the same as input units
            n_epochs=method_param["n_epochs"]
            batch_size=method_param["batch_size"]
            activation_func=method_param["activation_func"]

            classifier,self.classification_training_time=mlp.train_model(train_set_x_org=train_subset_x, train_set_y_org=train_subset_y, 
                                                     valid_set_x_org=valid_subset_x, valid_set_y_org=valid_subset_y, 
                                                     learning_rate=learning_rate, alpha=alpha, lambda_reg=lambda_reg, alpha_reg=alpha_reg, # alpha_reg from interval [0,1]
                                                     n_hidden=n_hidden, n_epochs=n_epochs, batch_size=batch_size, 
                                                     activation_func=activation_func, rng=rng,
                                                     max_num_epoch_change_learning_rate=80,max_num_epoch_change_rate=0.8,learning_rate_decay_rate=0.8)
                        
            self.classification_training_time=self.classification_training_time
            # predicting new samples
            # reduce dimensionality
            # if feature extraction is True, use the new features anyway.
            if method_param["feature_extraction"]:
                test_set_x=self.E_H_test
            elif feature_selection:
                test_set_x=self.X_test[self.numeric_ind_key_feat_for_classification,:]
            else:
                test_set_x=self.X_test

            test_set_x=numpy.transpose(test_set_x)

            # normalization
            test_set_x,_,_=cl.normalize_col_scale01(test_set_x,tol=1e-10,data_min=data_min,data_max=data_max)

            # predict/test
            self.test_classes_predicted,self.test_classes_prob_like,self.classification_test_time=mlp.test_model(classifier, test_set_x, batch_size=200)
            self.classification_test_time=self.classification_test_time

        else:
            print "Error! Please select a correct method to make the prediction!"           

        self.classification_time=self.classification_training_time-self.classification_test_time
        print 'classification time: %f seconds' %(self.classification_time)
        print 'classification training time: %f seconds' %(self.classification_training_time)
        print 'classification test time: %f seconds' %(self.classification_test_time)
        print "finished classification!"

        return self.test_classes_predicted,self.test_classes_prob_like,self.classification_training_time,self.classification_test_time
        


    def save_mf_result(self,dir_save="./",prefix="MVNMF"):
        """
        Save matrix factorization results of the training procedure.
        
        INPUTS:
        dir_save: string, path to save the factor matrices, e.g. "/home/yifeng/research/mf/mvmf_v1_1/".
        prefix: string, prefix of file names.
        OUTPUTS:
        This function does not explicitly return any variable.
        """
        print "saving factorization results ..."
        filename_E_W=dir_save+ prefix +"_E_W.txt"
        numpy.savetxt(filename_E_W, self.E_W, fmt='%.4e', delimiter='\t')
        filename_E_W=dir_save+ prefix +"_E_W_scaled.txt"
        numpy.savetxt(filename_E_W, self.E_W_scaled, fmt='%.4e', delimiter='\t')
        filename_E_H=dir_save+ prefix + "_E_H_transpose.txt"
        numpy.savetxt(filename_E_H, self.E_H.transpose(), fmt='%.4e', delimiter='\t')
        filename_E_H=dir_save+ prefix + "_E_H_scaled_transpose.txt"
        numpy.savetxt(filename_E_H, self.E_H_scaled.transpose(), fmt='%.4e', delimiter='\t')
        filename_E_Lambda_W=dir_save+ prefix +"_E_Lambda_W.txt"
        numpy.savetxt(filename_E_Lambda_W, self.E_Lambda_W, fmt='%.4e', delimiter='\t')
        filename_E_Lambda_H=dir_save+ prefix +"_E_Lambda_H.txt"
        numpy.savetxt(filename_E_Lambda_H, self.E_Lambda_H, fmt='%.4e', delimiter='\t')
        filename_time=dir_save+ prefix +"_time.txt"
        f = open(filename_time, 'w')
        f.write("Training factorization time excluding computing lower bound: "+str(self.training_time-self.training_time_L)+" seconds\n")
        f.write("Training factorization time for computing lower bound: "+str(self.training_time_L)+" seconds\n")
        f.write("Training factorization total time: "+str(self.training_time)+" seconds\n")
        f.write("Feature selection time: "+str(self.fs_time)+" seconds\n")
        f.write("Total number of iterations: "+str(self.num_iter)+"\n")
        f.close()


    def save_mf_test_result(self,dir_save="./",prefix="MVNMF"):
        """
        Save matrix factorization results for the test procedure (Bayesian non-negative regression). 
        
        INPUTS:
        dir_save: string, path to save the factor matrices, e.g. "/home/yifeng/research/mf/mvmf_v1_1/".
        prefix: string, prefix of file names.
        OUTPUTS:
        This function does not explicitly return any variable.
        """
        print "saving factorization results for test data ..."
        filename_E_H=dir_save+ prefix + "_E_H_test_transpose.txt"
        numpy.savetxt(filename_E_H, self.E_H_test.transpose(), fmt='%.4e', delimiter='\t')
        filename_E_Lambda_H=dir_save+ prefix +"_E_Lambda_H_test.txt"
        numpy.savetxt(filename_E_Lambda_H, self.E_Lambda_H_test, fmt='%.4e', delimiter='\t')
        filename_time=dir_save+ prefix +"_time_test.txt"
        f = open(filename_time, 'w')
        f.write("Test time excluding computing lower bound: "+str(self.test_time-self.test_time_L)+" seconds\n")
        f.write("Test time for computing lower bound: "+str(self.test_time_L)+" seconds\n")
        f.write("Test factorization total time: "+str(self.test_time)+" seconds\n")
        f.write("Total number of iterations: "+str(self.num_iter_test)+"\n")
        f.close()

    def compute_acc(self,feature_patterns_matrix):
        """
        Compute element-wise-match accuracy and row-wise-match accuracy given the real pattern of feature patterns in a matrix.
        
        INPUTS:
        feature_patterns_matrix: numpy matrix of integers, the actural feature patterns.

        OUTPUTS:
        acc_e: float, element-wise match accuracy.
        acc_r: float, row-wise match accuracy.
        """
        feature_patterns=utility.convert_each_row_of_matrix_to_a_string(numpy.array(feature_patterns_matrix,dtype=int),sep="")
        acc_e=numpy.sum(feature_patterns_matrix==self.F)/(self.M*(self.V+1))
        acc_r=numpy.sum(feature_patterns==self.F_str)/self.M
        print "The element-wise-match accuracy of features are: {0}".format(acc_e)
        print "The row-wise-match accuracy of features are: {0}".format(acc_r)
        return acc_e,acc_r
