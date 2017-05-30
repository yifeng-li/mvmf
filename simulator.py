# The simulator class for generating simulated data.

from __future__ import division # In Python 3.0, // is the floor division operator and / the true division operator. The true_divide(x1, x2) function is equivalent to true division in Python.
import numpy
import unique_binary_code
import utility # in DECRES
import classification as cl

class simulator:
    def __init__(self,z=3,V=3,m=3,rng=numpy.random.RandomState(1000)):
        """
        z: integer,tuple, list, or numpy.ndarray, the number of hidden factors for each view;
        V: integer, number of views.
        m: integer, number of features for each pattern.
        rng: random state.
        """
        self.V=V
        self.m=m
        self.rng=rng
        
        if isinstance(z,tuple) or isinstance(z,list):
            self.z_list=z
            self.z=cl.factor_sizes_to_factor_labels(z) # e.g. (3,3,3,3) to [-1,-1,-1,0,0,0,1,1,2,2,2]
        else:
            self.z_list=[z]*(self.V+1)
            self.z=cl.factor_sizes_to_factor_labels(self.z_list) # e.g. 3 to [-1,-1,-1,0,0,0,1,1,2,2,2]
        #print self.z
        self.K=len(self.z) # number of latent factors 
        self.Z,self.z_unique=cl.membership_vector_to_indicator_matrix(self.z) # binary, size K by V+1, self.Z[k,u]=1 indicates the k-th factor in class u.
        #print self.Z


    def make_W(self,a_active_W=10,b_active_W=1000):
        """
        Make the real basis matrix W.
        a_active_W: scalar, shape parameter of Gamma distribution.
        b_active_W: scalar, rate parameter of Gamma distribution.
        """
        self.a_active_W=a_active_W
        self.b_active_W=b_active_W
        self.ubc=unique_binary_code.unique_binary_code(self.V+1)
        self.ubc.generate_binary_code()
        self.ubc.s
        self.s_str=utility.convert_each_row_of_matrix_to_a_string(self.ubc.s,sep="")
        self.num_patterns=len(self.s_str)
        self.M=self.m*self.num_patterns
        self.W=numpy.zeros(shape=(self.M,self.K),dtype=float)
        self.Lambda_W=numpy.zeros(shape=(self.M,self.V+1),dtype=float )
        ls=self.rng.gamma(shape=self.a_active_W, scale=1/self.b_active_W, size=(self.M,self.V+1))
        mp=cl.factor_sizes_to_factor_labels([self.m]*self.num_patterns) # [3,2,4] -> [-1,-1,-1,0,0,1,1,1]
        MP,_=cl.membership_vector_to_indicator_matrix(mp)
        #print MP
        #print self.ubc.s
        self.S=numpy.dot(MP,self.ubc.s) # extend binary codes, M times V+1
        self.S=numpy.asarray(self.S,dtype=bool)
        self.Lambda_W[self.S]=ls[self.S]
        #self.features=numpy.empty(shape=(self.M,),dtype=str)
        self.features=["features"]*self.M # names of features
	self.feature_patterns=["feature_patterns"]*self.M # pattern of features
        self.feature_patterns_matrix=numpy.zeros(shape=(self.M,self.V+1),dtype=int)
        fs=range(0,self.m)*self.num_patterns # [0,1,2,0,1,2,0,1,2,...,0,1,2]
        #print self.Lambda_W
        for i in range(self.M):
            code=numpy.asarray(self.S[i,:],dtype=int)
            self.Z=numpy.asarray(self.Z,dtype=int)
            code.shape=(len(code),1) # V+1 times 1
            code_ext=self.Z.dot(code) # K times 1
            code_ext=numpy.asarray(code_ext,dtype=bool)
            code_ext.shape=(len(code_ext),)
            code.shape=(len(code),)
            self.features[i]="".join(numpy.asarray(code,dtype=str))+"_"+str(fs[i])
            self.feature_patterns[i]="".join(numpy.asarray(code,dtype=str))
            self.feature_patterns_matrix[i,:]=code
            code=numpy.asarray(code,dtype=bool)
            #num_active_views=numpy.sum(code)
            w=[]
            for v in range(self.V+1):
                if self.S[i,v]:
                    w=numpy.concatenate((w,self.rng.exponential(scale=1/self.Lambda_W[i,v],size=self.z_list[v])))
            self.W[i,code_ext]=w
            
        #print self.W
        #print self.features
        return self.W,self.features

    def make_H(self,a_active_H=10,b_active_H=0.1,n=100):
        """
        a_active_H: scalar, shape parameter of Gamma distribution.
        b_active_H: scalar, rate parameter of Gamma distribution.
        n: scalar, number of examples for each class.
        """
        self.a_active_H=a_active_H
        self.b_active_H=b_active_H
        if isinstance(n,tuple) or isinstance(n,list):
            self.n_list=n
            self.n=cl.factor_sizes_to_factor_labels(n,start=0) # e.g. (2,3,4,3) to [0,0,1,1,1,2,2,2,2,3,3,3]
        else:
            self.n_list=[n]*(self.V)
            self.n=cl.factor_sizes_to_factor_labels(self.n_list,start=0) # e.g. 3 to [0,0,0,1,1,1,2,2,2,3,3,3]
        self.N=len(self.n) # number of samples 
        self.C,_=cl.membership_vector_to_indicator_matrix(self.n) # class membership matrix, N times V
        self.C=numpy.vstack((numpy.ones(shape=(1,self.N),dtype=int),numpy.transpose(self.C))) # V+1 times N
        #KN=numpy.dot(self.Z,self.C)
        self.Lambda_H=numpy.zeros(shape=(self.V+1,self.N),dtype=float)
        ls=self.rng.gamma(shape=self.a_active_H, scale=1/self.b_active_H, size=(self.V+1,self.N))
        self.C=numpy.asarray(self.C,dtype=bool)
        self.Lambda_H[self.C]=ls[self.C]
        self.Lambda_H_ext=numpy.dot(self.Z,self.Lambda_H) # K times N
        KN=numpy.dot(self.Z,self.C)
        self.H=numpy.zeros(shape=(self.K,self.N),dtype=float)
        for k in range(self.K):
            for n in range(self.N):
                if KN[k,n]:
                    self.H[k,n]=self.rng.exponential(scale=1/self.Lambda_H_ext[k,n],size=None)

        #print self.H
        self.classes=self.n
       # print self.classes
        return self.H,self.n


    def make_X(self):
        self.X=numpy.dot(self.W,self.H)
        #print self.X
        return self.X
    
        
    def save(self,dir_save,prefix):
        """
        dir_save: string, path to save the simulated result, e.g. "/home/yifeng/research/mf/mvmf_v1_1/data/".
        prefix: string, prefix of the saved file names, e.g. "my_simulation".
        """
        print "saving simulation results ..."
        filename_W=dir_save+ prefix +"_W.txt"
        numpy.savetxt(filename_W, self.W, fmt='%.4e', delimiter='\t')
        filename_H=dir_save+ prefix +"_H.txt"
        numpy.savetxt(filename_H, self.H, fmt='%.4e', delimiter='\t')
        filename_X=dir_save+ prefix +"_X.txt"
        numpy.savetxt(filename_X, self.X, fmt='%.4e', delimiter='\t')
        filename=dir_save+ prefix +"_Features.txt"
        numpy.savetxt(filename, self.features, fmt='%s', delimiter='\t')
	filename=dir_save+ prefix +"_Feature_Patterns.txt"
        numpy.savetxt(filename, self.feature_patterns, fmt='%s', delimiter='\t')
        filename=dir_save+ prefix +"_Feature_Patterns_Matrix.txt"
        numpy.savetxt(filename, self.feature_patterns_matrix, fmt='%s', delimiter='\t')
        filename=dir_save+ prefix +"_Classes.txt"
        numpy.savetxt(filename, self.classes, fmt='%s', delimiter='\t')
        print "saved data in "+dir_save
 
# example           
#sim=simulator(z=3,V=3,m=3)
#sim.make_W(a_active_W=10,b_active_W=1e5)
#sim.make_H(a_active_H=10,b_active_H=1,n=100)
#sim.make_X()
#dir_save="/global/data/tcga/metadata/data/"
#prefix="simulated_data"
#sim.save(dir_save,prefix)







