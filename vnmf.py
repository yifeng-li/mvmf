# Variational NMF.

from __future__ import division # In Python 3.0, // is the floor division operator and / the true division operator. The true_divide(x1, x2) function is equivalent to true division in Python.
import scipy.special
import scipy.stats
import numpy
import math
import classification as cl
import unique_binary_code
import utility

class mvnmf:
    def __init__(self, X=None, features=None, samples=None): 
        #initialize
        self.X=X # data
        self.features=features # names of features
        self.samples=samples # names of samples
        self.M=self.X.shape[0] # number of features
        self.N=self.X.shape[1] # number of samples
        self.tol=1e-32
           
    def factorize(self, K=None, a_W=None, b_W=None, a_H=None, b_H=None, max_iter=200, compute_variational_lower_bound=True, variational_lower_bound_min_rate=1e-4, rng=numpy.random.RandomState(1000)):
        """
        Factorize the matrix to at most K factors vectors.
        """
        self.K=K # number of latent factors 
        self.a_W=a_W
        self.b_W=b_W
        self.a_H=a_H
        self.b_H=b_H
        self.max_iter=max_iter
        self.rng=rng

        #initiate
        #sample E_Lambda_W and E_Lambda_H from Gamma prior distributions
        E_Lambda_W=self.rng.gamma(shape=self.a_W, scale=1/self.b_W, size=(self.M,self.K))
        E_Lambda_H=self.rng.gamma(shape=self.a_H, scale=1/self.b_H, size=(self.K,self.N))
        
        #compute A_W, B_W, A_H, and B_H using prior expoential distribution(1,lambda)
        A_W=1 #numpy.ones(shape=(self.M,self.K),dtype=float)
        B_W=E_Lambda_W
        #print B_W
        A_H=1 #numpy.ones(shape=(self.K,self.N),dtype=float)
        B_H=E_Lambda_H
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
        #print L_W
        #print L_H
        L_W_L_H=L_W.dot(L_H)
        
        if compute_variational_lower_bound:
            neg_gammaln_X_plus_1=-scipy.special.gammaln(self.X+1)
            Ls=[] # lower bound
            Ls_rates=[] # change rate
            mean_over=5 # compute the mean rate over this number of iterations

        ONE_M_N=numpy.ones(shape=(self.M,self.N),dtype=float)

        num_iter=1
        while num_iter<=self.max_iter:
            print "iteration: {0}".format(num_iter)
            #compute Sigma_W and Sigma_H, need L_W and L_H
            X_div_L_W_L_H=self.X/L_W_L_H
            Sigma_W=L_W * X_div_L_W_L_H.dot(L_H.transpose())
            Sigma_H=L_H.transpose() * X_div_L_W_L_H.transpose().dot(L_W)
            #print Sigma_W
            #print Sigma_H

            #compute A_W, B_W, A_H, and B_H need Sigma_W, Sigma_H, E_W, E_H, E_Lambda_W, and E_Lambda_H
            A_W=1 + Sigma_W
            B_W=E_Lambda_W + ONE_M_N.dot(E_H.transpose())
            #compute E_W
            E_W=A_W/B_W 
            A_H=1 + Sigma_H.transpose()
            B_H=E_Lambda_H + E_W.transpose().dot(ONE_M_N)
            #compute E_H
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

            #compute E_Lambda_W and E_Lambda_H, need E_W and E_H
            A_Lambda_W=self.a_W+1
            B_Lambda_W=self.b_W+E_W
            E_Lambda_W=A_Lambda_W/B_Lambda_W
            A_Lambda_H=self.a_H+1
            B_Lambda_H=b_H+E_H
            E_Lambda_H=A_Lambda_H/B_Lambda_H
            
            # variational lower bound
            if compute_variational_lower_bound:
                # compute variational lower bound
                psi_A_Lambda_W=scipy.special.psi(A_Lambda_W)
                log_B_Lambda_W=numpy.log(B_Lambda_W)
                L_Lambda_W=psi_A_Lambda_W - log_B_Lambda_W
                psi_A_Lambda_H=scipy.special.psi(A_Lambda_H)
                log_B_Lambda_H=numpy.log(B_Lambda_H)
                L_Lambda_H=psi_A_Lambda_H - log_B_Lambda_H
                # bound
                L=numpy.sum( neg_gammaln_X_plus_1 + self.X*numpy.log(L_W.dot(L_H))-E_W.dot(E_H)) + numpy.sum(L_Lambda_W - E_Lambda_W*E_W) + numpy.sum(L_Lambda_H - E_Lambda_H*E_H) + numpy.sum(self.a_W*numpy.log(self.b_W) - scipy.special.gammaln(self.a_W) + (self.a_W-1)*L_Lambda_W - self.b_W*E_Lambda_W ) + numpy.sum(self.a_H*numpy.log(self.b_H) - scipy.special.gammaln(self.a_H) + (self.a_H-1)*L_Lambda_H - self.b_H*E_Lambda_H) + numpy.sum(scipy.special.gammaln(A_W) - (A_W-1)*psi_A_W - numpy.log(B_W) + A_W) + numpy.sum(scipy.special.gammaln(A_H) - (A_H-1)*psi_A_H - numpy.log(B_H) + A_H) + numpy.sum(scipy.special.gammaln(A_Lambda_W) - (A_Lambda_W-1)*psi_A_Lambda_W - log_B_Lambda_W + A_Lambda_W) + numpy.sum(scipy.special.gammaln(A_Lambda_H) - (A_Lambda_H-1)*psi_A_Lambda_H - log_B_Lambda_H + A_Lambda_H)
                #L=numpy.sum( abs(self.X-E_W.dot(E_H))) 
                Ls.append(L)
                if num_iter>=2:
                    Ls_rates.append((Ls[-1]-Ls[-2])/(-Ls[-2]))
                    Ls_rates_mean=numpy.mean(Ls_rates[-mean_over:])
                    print "The variational lower bound:{0}, change rate:{1}".format(L,Ls_rates_mean)
                    if num_iter>=10 and Ls_rates_mean<variational_lower_bound_min_rate:
                        break

            num_iter=num_iter+1

        #save the result
        self.E_W=E_W
        self.E_H=E_H
        self.E_Lambda_W=E_Lambda_W
        self.E_Lambda_H=E_Lambda_H
        self.Ls=Ls # lower bounds
        self.num_iter=num_iter # actural number of iterations run
        print "finished fatorization!"
        
        #NOTE: MAY CONSIDER ZERO COLUMNS in E_W and ZERO ROWS in E_H
    def trim(self,trim_nonzero_portion=0.01,alpha=0.05,threshold_E_W=None,threshold_E_H=None):
        # trim zero columns of E_W and corresponding rows of E_H
        print "triming zero factors ..."
        self.trim_nonzero_portion=trim_nonzero_portion
        self.alpha=alpha

        if threshold_E_W is None:
            threshold_E_W=self.alpha*numpy.mean(self.E_W)
        self.E_W_NZ=self.E_W>=threshold_E_W # non-zero indicator of E_W
        #print "threhold_E_W:"
        #print threshold_E_W
        #print "E_W_NZ:"
        #print numpy.array(self.E_W_NZ,dtype=int)
        nonzero_portions_E_W=numpy.mean(self.E_W_NZ,axis=0)
       
        if threshold_E_H is None:
            threshold_E_H=self.alpha*numpy.mean(self.E_H)
        self.E_H_NZ=self.E_H>=threshold_E_H # non-zero indicator of E_H
        nonzero_portions_E_H=numpy.mean(self.E_H_NZ,axis=1)
        
        ind_nonzero_factors=numpy.logical_and(nonzero_portions_E_W>=self.trim_nonzero_portion,nonzero_portions_E_H>=self.trim_nonzero_portion)
        # filter zero factors
        self.E_W=self.E_W[:,ind_nonzero_factors]
        self.E_H=self.E_H[ind_nonzero_factors,:]
        self.K=self.E_W.shape[1]
        
    def reorder_factors(self):
        print "reorder factors ..."
        max_coef_ind=numpy.argmax(self.E_W,axis=0)
        factor_order=numpy.argsort(max_coef_ind)
        #factor_order=factor_order[::-1]
        self.E_W=self.E_W[:,factor_order]
        self.E_H=self.E_H[factor_order,:]
        self.E_Lambda_W=self.E_Lambda_W[factor_order]
        self.E_Lambda_H=self.E_Lambda_H[factor_order]
        
    def reorder_samples(self,method="cluster_label",scores=None):
        """
        Reorder the columns of X and H based on the column order of W for better visualization.
        method: string, "cluster_label", "max_coef", "entropy", "pval", "correlation"
        """
        print "reorder samples in each cluster ..."
        # obtain the factor/cluster label of each sample
        self.y=numpy.argmax(self.E_H,axis=0) # cluster labels
        self.unique_y=numpy.unique(self.y)
        # get the sizes of each cluster
        cluster_sizes=[]
        for c in self.unique_y:
            ind_log=self.y==c
            num_samples_c=ind_log.sum()
            cluster_sizes.extend([num_samples_c])
        self.cluster_sizes=numpy.array(cluster_sizes)

        if method=="cluster_label":
            ind=numpy.argsort(self.y)
            self.scores=numpy.ones_like(self.y)
        elif method=="max_coef":
            if scores is None:
                self.scores=numpy.max(self.E_H,axis=0)
            #self.unique_y=numpy.unique(self.y)
            numbers=numpy.array(range(self.N),dtype=int)
            ind=[]
            for c in self.unique_y:
                ind_log=self.y==c
                ind_log.shape=(ind_log.size,)
                ind_c=numbers[ind_log]
                ind_sorted=numpy.argsort(self.scores[ind_log])
                ind_sorted=ind_c[ind_sorted]
                ind_sorted=ind_sorted[::-1]
                ind.extend(ind_sorted)
        elif method=="pval":
            pass
        elif method=="correlation":
            #self.scores,pval=scipy.stats.spearmanr(self.X,self.E_W, axis=0)
            #self.unique_y=numpy.unique(self.y)
            if scores is None:
                scores=[]
                pvals=[]
                for c in self.unique_y:
                    ind_log=self.y==c
                    ind_log.shape=(ind_log.size,)
                    X_c=self.X[:,ind_log]
                    factor_c=self.E_W[:,c]
                    factor_c.shape=(self.M,1)
                    num_samples_c=ind_log.sum()
                    if num_samples_c==1:
                        print X_c.shape
                        X_c.shape=(self.M,1)
                    print "computing Spearman correlation for cluster:{0}".format(c)
                    scores_c=numpy.zeros(num_samples_c)
                    pvals_c=numpy.zeros(num_samples_c)
                    for s in range(num_samples_c):
                        scores_c[s],pvals_c[s]=scipy.stats.spearmanr(X_c[:,s],self.E_W[:,c])
                        if numpy.isnan(scores_c[s]) or numpy.isinf(scores_c[s]):
                            print "Get an irregular value:{0}. Set it to 0.".format(scores_c[s])
                            scores_c[s]=0
                            pvals_c[s]=1
                    scores.extend(scores_c)
                    pvals.extend(pvals_c)
                self.scores=numpy.array(scores)
                #self.scores[numpy.isnan(self.scores)]=0
                pvals=numpy.array(pvals)
                #pvals[numpy.isnan(pvals)]=1

            numbers=numpy.array(range(self.N),dtype=int)
            ind=[]
            for c in self.unique_y:
                ind_log=self.y==c
                ind_log.shape=(ind_log.size,)
                ind_c=numbers[ind_log]
                ind_sorted=numpy.argsort(self.scores[ind_log])
                ind_sorted=ind_c[ind_sorted]
                ind_sorted=ind_sorted[::-1]
                ind.extend(ind_sorted)
        
        self.X_sorted=self.X[:,ind]
        self.E_H_sorted=self.E_H[:,ind]
        self.y_sorted=self.y[ind]
        self.scores_sorted=self.scores[ind]
        self.samples_sorted=self.samples[ind]
        self.z=numpy.array(range(self.K),dtype=int)

    def ext_feat(self):
        pass

    def colorbar_classes(self,classes,ax=None,hv="horizontal",clrs=None,unique_classes=None,unique_class_names=None,fontsize=6,ylabel_right=False,rotation=0):
    # plot the color bar for classes. classes: 0,1,2,...,C-1
    # hv: either "horizontal" or "vertical"
    # clrs: numpy vector of strings, the color names.
    # unique_class_names: if is None, it will be calcuated as numpy array of [0,1,2,...,C-1]
    # fontsize, scalar integer.

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


    def plot_heatmap(self,dir_save="./", prefix="ARDNMF", normalize="l2norm", width=10, height=10, fontsize=6, fmt="png", dpi=600, colormap="Reds", clrs=None, rng=numpy.random.RandomState(10)):
        """
        Plot heatmaps of the decomposition result. 
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

        # colors of classes
        if clrs is None:
            #cnames=colors.cnames
            #clrs=[name for name,hex in cnames.iteritems()]
            # remove light colors
            clrs=["black","grey","rosybrown","frebrick","r","darksalmon","sienna","sandybrown","tan","gold","darkkhaki","olivedrab","chartreuse","lightgreen","green","mediumseagreen","mediumaquamarine","mediumturquoise","darkslategrey","c","cadetblue","dodgerblue","slategrey","darkblue","slateblue","blueviolet","mediumorchid","purple","magenta","hotpink","k","gray","lightcoral","maroon","coral","darkorange","orange","darkgoldenrob","olive","yellowgreen","lawngreen","forestgreen","g","springgreen","aquamarine","darkslategray","aqua","mediumblue","darkslateblue","indigo","darkmagenta","orchid","dimgrey","indianred","darkred","salmon","orangered","chocolate","peru","goldenrod","y","darkolivegreen","darkseagreen","limegreen","lime","turquoise","teal","cyan","steelblue","cornflowerblue","midnightblue","blue","mediumslateblue","darkorchid","m","mediumvioletred","palevioletred","dimgray","brown","red","tomato","lightsalmon","saddlebrown","yellow","greenyellow","palegreen","darkgreen","seagreen","mediumspringgreen","lightseagreen","darkcyan","darkturquoise","deepskyblue","slategray","royalblue","navy","b","mediumpurple","darkviolet","violet","fuchsia","deeppink","crimson"]
            # permute all color names
            num_clrs=len(clrs)
            clrs=numpy.array(clrs)
            ind_clrs=rng.choice(num_clrs,self.K)
            clrs_classes=clrs[ind_clrs]
            self.clrs_classes=clrs_classes
 
        # get data first
        if normalize=="l2norm":
            X=cl.normalize_l2norm(numpy.transpose(self.X_sorted),tol=1e-32)
            X=numpy.transpose(X)
            W=cl.normalize_l2norm(numpy.transpose(self.E_W),tol=1e-32)
            W=numpy.transpose(W)
            H=cl.normalize_l2norm(numpy.transpose(self.E_H_sorted),tol=1e-32)
            H=numpy.transpose(H)
        elif normalize=="scale01":
            X,_,_=cl.normalize_col_scale01(self.X_sorted,clip=True,clip_min=0.5,clip_max=1e4)
            W,_,_=cl.normalize_col_scale01(self.E_W,clip=True,clip_min=1e-4,clip_max=1e4)
            H,_,_=cl.normalize_col_scale01(self.E_H_sorted,clip=True,clip_min=1e-4,clip_max=1e4)
        elif normalize=="log2":
            X=numpy.log2(self.X_sorted+1)
            W=numpy.log2(self.E_W_sorted+1)
            H=numpy.log2(self.E_H+1)
        elif normalize=="scale01log2":
            X,_,_=cl.normalize_row_scale01(numpy.log2(self.X_sorted+1),clip=True,clip_min=0.1,clip_max=10)
            W,_,_=cl.normalize_row_scale01(numpy.log2(self.E_W+1),clip=True,clip_min=0.1,clip_max=10)
            H,_,_=cl.normalize_col_scale01(numpy.log2(self.E_H+1),clip=True,clip_min=0.1,clip_max=10)
        else:
            print "No normalization method is specified or this normalization method is not defined yet."

        col_lab_X=self.samples_sorted
        row_lab_X=self.features
        col_lab_W=numpy.asarray(self.z,dtype=str)
        row_lab_W=row_lab_X
        col_lab_H=col_lab_X
        row_lab_H=col_lab_W

        # plot the heatmaps
        #fig, ax = plt.subplots(1,3)
        fig=plt.figure(figsize=(width,height))
        allgrids=20
        ax0_rowspan=allgrids-2
        ax0_colspan=int(0.4*allgrids)
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
        #ax0cfe_rowspan=ax0_rowspan
        #ax0cfe_colspan=1
        #ax0cfe_rowstart=1
        #ax0cfe_colstart=ax0_colspan
        ax1_rowspan=ax0_rowspan
        ax1_colspan=int(0.2*allgrids)
        ax1_rowstart=1
        ax1_colstart=ax0_colspan
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
        cbar_X.ax.tick_params(labelsize=fontsize)
        # color bar for classes
        ax0ccl=plt.subplot2grid((allgrids, allgrids), (ax0ccl_rowstart, ax0ccl_colstart), rowspan=ax0ccl_rowspan, colspan=ax0ccl_colspan)
        self.colorbar_classes(self.y_sorted,ax=ax0ccl,hv="horizontal",clrs=clrs_classes,unique_class_names=None,fontsize=fontsize)
        ## color bar for feature groups
        #ax0cfe=plt.subplot2grid((allgrids, allgrids), (ax0cfe_rowstart, ax0cfe_colstart), rowspan=ax0cfe_rowspan, colspan=ax0cfe_colspan)
        #self.colorbar_classes(self.F_str_sorted, ax=ax0cfe,hv="vertical",clrs=clrs_feature_groups,unique_classes=self.s_str,unique_class_names=None,fontsize=fontsize,ylabel_right=True,rotation=0)

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
        cbar_W.ax.tick_params(labelsize=fontsize)
        # color bar for factors
        ax1cfa=plt.subplot2grid((allgrids, allgrids), (ax1cfa_rowstart, ax1cfa_colstart), rowspan=ax1cfa_rowspan, colspan=ax1cfa_colspan)
        self.colorbar_classes(self.z,ax=ax1cfa,hv="horizontal",clrs=clrs_classes,unique_class_names=None,fontsize=fontsize)

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
        cbar_H.ax.tick_params(labelsize=fontsize)
        # color bar for factors
        ax2cfa=plt.subplot2grid((allgrids, allgrids), (ax2cfa_rowstart, ax2cfa_colstart), rowspan=ax2cfa_rowspan, colspan=ax2cfa_colspan)
        self.colorbar_classes(self.z,ax=ax2cfa,hv="vertical",clrs=clrs_classes,unique_class_names=None,fontsize=fontsize,rotation=90)
        # color bar for classes
        ax2ccl=plt.subplot2grid((allgrids, allgrids), (ax2ccl_rowstart, ax2ccl_colstart), rowspan=ax2ccl_rowspan, colspan=ax2ccl_colspan)
        self.colorbar_classes(self.y_sorted,ax=ax2ccl,hv="horizontal",clrs=clrs_classes,unique_class_names=None,fontsize=fontsize)        

        filename=dir_save+prefix+"."+fmt
        plt.tight_layout()
        #fig.savefig(filename,bbox_inches='tight',format="png")
        fig.savefig(filename,bbox_inches='tight',format=fmt,dpi=dpi)
        plt.close(fig)

    def plot_clusters(self,dir_save="./", prefix="ARDNMF_Clusters", top=100, normalize="unitl2norm",  clrs=None, rotation=90, subplot_col=2, subplot_width=3, subplot_height=2, fontsize=8, fmt="png", dpi=600, rng=numpy.random.RandomState(10)):
        """
        Plot the samples for each cluster. 
        top: float or int. If top is in (0,1], the samples correlated with the corresponding metasample with correlation >=top are ploted. If top>1, simply select the top number (correlation) of samples.
        normalize: string, "None","l2norm","scale01". 
        clrs: vector of color names.
        """

        "Plot the data for each cluster."
        import matplotlib as mpl
        #mpl.use(fmt)
        import matplotlib.pyplot as plt
        from matplotlib import colors

        print "ploting each cluster ..."
        # colors of classes
        if clrs is None:
            #cnames=colors.cnames
            #clrs=[name for name,hex in cnames.iteritems()]
            # remove light colors
            clrs=["black","grey","rosybrown","frebrick","r","darksalmon","sienna","sandybrown","tan","gold","darkkhaki","olivedrab","chartreuse","lightgreen","green","mediumseagreen","mediumaquamarine","mediumturquoise","darkslategrey","c","cadetblue","dodgerblue","slategrey","darkblue","slateblue","blueviolet","mediumorchid","purple","magenta","hotpink","k","gray","lightcoral","maroon","coral","darkorange","orange","darkgoldenrob","olive","yellowgreen","lawngreen","forestgreen","g","springgreen","aquamarine","darkslategray","aqua","mediumblue","darkslateblue","indigo","darkmagenta","orchid","dimgrey","indianred","darkred","salmon","orangered","chocolate","peru","goldenrod","y","darkolivegreen","darkseagreen","limegreen","lime","turquoise","teal","cyan","steelblue","cornflowerblue","midnightblue","blue","mediumslateblue","darkorchid","m","mediumvioletred","palevioletred","dimgray","brown","red","tomato","lightsalmon","saddlebrown","yellow","greenyellow","palegreen","darkgreen","seagreen","mediumspringgreen","lightseagreen","darkcyan","darkturquoise","deepskyblue","slategray","royalblue","navy","b","mediumpurple","darkviolet","violet","fuchsia","deeppink","crimson"]
            # permute all color names
            num_clrs=len(clrs)
            clrs=numpy.array(clrs)
            ind_clrs=rng.choice(num_clrs,self.K)
            clrs_classes=clrs[ind_clrs]
            self.clrs_classes=clrs_classes

        num_clusters=len(self.unique_y)
        subplot_row=int(math.ceil(num_clusters/subplot_col))
        fig, ax = plt.subplots(subplot_row, subplot_col)
        fig.set_size_inches(subplot_width*subplot_col, subplot_height*subplot_row)
        ax_ind=0
        for i in range(num_clusters):
            row_c=ax_ind//subplot_col
            col_c=ax_ind%subplot_col
            c=self.unique_y[i]
            ind_log=self.y_sorted==c
            ind_log.shape=(ind_log.size,)
            #print "The shape of y_sorted {0}".format(self.y_sorted.shape)
            #print "The shape of ind_log {0}".format(ind_log.shape)
            X_c=self.X_sorted[:,ind_log]
            scores_c=self.scores_sorted[ind_log]
            num_c=ind_log.sum() # the total number of cluster members
            print "i={0}, c={1}".format(i,c)
            print "Total number samples in cluster {0} is {1}".format(c,num_c)
            if top==None or top>num_c:
                top=num_c
            if top>1:
                ind_top=numpy.zeros(shape=(num_c,),dtype=bool)
                ind_top[0:top]=True
            elif top<=1 and top>0:
                ind_top=scores_c>=top
            else:
                print "Top should be larger than zero!"
                exit
            X_c_top=X_c[:,ind_top]
            # if there is no sample in this cluster fulfil the requirement, plot top 100 instead
            if X_c_top.shape[1]==0:
                ind_scores_c=numpy.argsort(scores_c)
                ind_scores_c=ind_scores_c[::-1]
                X_c_top=X_c[:,ind_scores_c[0:100]]
                print "Warning: No sample in cluster {0} fulfil the requirement, plot the top 100 instead.".foramt(c)
            # normalize for better visualization
            if normalize=="l2norm":
                X_c_top=cl.normalize_l2norm(X_c_top.transpose(),tol=1e-32)
                X_c_top=X_c_top.transpose()
            elif normalize=="scale01":
                X_c_top,_,_=cl.normalize_col_scale01(X_c_top)
            else:
                print "No normalization method is specified or this normalization method is not defined yet."
            print "Ploting cluster:{0}".format(c)
            print "Length of clrs_classes:{0},currrent value of i:{1},cluster id:{2}".format(len(self.clrs_classes),i,c)
            print X_c_top.shape
            ax[row_c,col_c].plot(X_c_top,color=self.clrs_classes[c], linewidth=0.5, linestyle="solid")
            centroid_clr="black"
            if self.clrs_classes[c]=="black" or self.clrs_classes[c]=="k":
                centroid_clr="red"
            ax[row_c,col_c].plot(numpy.mean(X_c_top,axis=1),color=centroid_clr,linewidth=2,linestyle="solid")
            ax[row_c,col_c].set_title("Cluster "+str(c))
            ax[row_c,col_c].set_xticks(range(self.M))
            ax[row_c,col_c].set_xticklabels(self.features,rotation=rotation)
            ax[row_c,col_c].set_xlim([-0.25,self.M-0.75])
            ax_ind=ax_ind+1
 
        # clear ax for unused axes
        for i in range(num_clusters,subplot_row*subplot_col):
            row_c=i//subplot_col
            col_c=i%subplot_col
            ax[row_c,col_c].set_axis_off()
        
        # save figure
        filename=dir_save+prefix+"."+fmt
        plt.tight_layout()
        #fig.savefig(filename,bbox_inches='tight',format="png")
        fig.savefig(filename,bbox_inches='tight',format=fmt,dpi=dpi)
        plt.close(fig)
        

    def learn_H_given_X_and_W(self):
        pass

    def save_mf_result(self,dir_save="./",prefix="MVNMF",transpose_E_H=True):
        # filename include full path and prefix. For example: /global/data/tcga/mf/luad
        print "saving factorization results ..."
        filename_E_W=dir_save+ prefix +"_E_W.txt"
        numpy.savetxt(filename_E_W, self.E_W, fmt='%.2e', delimiter='\t')
        
        if transpose_E_H:
            filename_X=dir_save+ prefix + "_X_sorted.txt"
            numpy.savetxt(filename_X, numpy.transpose(self.X_sorted), fmt='%.2e', delimiter='\t')
            filename_E_H=dir_save+ prefix + "_E_H.txt"
            numpy.savetxt(filename_E_H, numpy.transpose(self.E_H), fmt='%.2e', delimiter='\t')
            filename_E_H=dir_save+ prefix + "_E_H_sorted.txt"
            numpy.savetxt(filename_E_H, numpy.transpose(self.E_H_sorted), fmt='%.2e', delimiter='\t')
        else:
            filename_X=dir_save+ prefix + "_X_sorted.txt"
            numpy.savetxt(filename_X, self.X_sorted, fmt='%.2e', delimiter='\t')
            filename_E_H=dir_save+ prefix + "_E_H.txt"
            numpy.savetxt(filename_E_H, self.E_H, fmt='%.2e', delimiter='\t')
            filename_E_H=dir_save+ prefix + "_E_H_sorted.txt"
            numpy.savetxt(filename_E_H, self.E_H_sorted, fmt='%.2e', delimiter='\t')
        
        self.samples_sorted.shape=(self.N,1)
        self.y_sorted.shape=(self.N,1)
        self.scores_sorted.shape=(self.N,1)
        filename_cluster=dir_save+ prefix + "_cluster.txt"
        numpy.savetxt(filename_cluster, numpy.concatenate((self.samples_sorted,self.y_sorted,self.scores_sorted),axis=1), fmt='%s', delimiter='\t')
        self.clrs_clusters=self.clrs_classes[self.unique_y]
        self.unique_y.shape=(self.unique_y.size,1)
        self.cluster_sizes.shape=(self.cluster_sizes.size,1)
        self.clrs_clusters.shape=(self.clrs_clusters.size,1)
        filename_cluster=dir_save+ prefix + "_cluster_summary.txt"
        numpy.savetxt(filename_cluster, numpy.concatenate((self.unique_y,self.clrs_clusters,self.cluster_sizes),axis=1), fmt='%s', delimiter='\t')
        self.samples_sorted.shape=(self.N,)
        self.y_sorted.shape=(self.N,)
        self.scores_sorted.shape=(self.N,)
        self.unique_y.shape=(self.unique_y.size,)
        self.cluster_sizes.shape=(self.cluster_sizes.size,)
        self.clrs_clusters.shape=(self.clrs_clusters.size,) 

        filename_E_Lambda_W=dir_save+ prefix +"_E_Lambda_W.txt"
        numpy.savetxt(filename_E_Lambda_W, self.E_Lambda_W, fmt='%.2e', delimiter='\t')
        filename_E_Lambda_H=dir_save+ prefix +"_E_Lambda_H.txt"
        numpy.savetxt(filename_E_Lambda_H, self.E_Lambda_H, fmt='%.2e', delimiter='\t')







