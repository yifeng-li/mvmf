
import numpy

# define a class to generate unique binary code: for example, 000, 100, 010, 001, 110, 101, 011, 111
class unique_binary_code:
    def __init__(self,n=5,maxn=12):
        self.n=n # the length of binary code
	self.maxn=maxn
        self.s=numpy.zeros((1,self.n),dtype=int) # numpy array holding all the 2^n codes
        self.freq_tab=None
        self.inds=None
        self.header=None

    def generate_k_ones(self,k,level=1,start=0,s=None): # recursive function
        """
        Recursive function. 
        k: the number of ones in the binary code;
        level: level, initial is 1.
        start: the start position, initial value i=0
        s: the vector of 1 by n, a binary code, initial value is None 
        """

        #print "k={0},level={1}".format(k,level)

        if level==1:
            #print "level==1"
            s=numpy.zeros((1,self.n),dtype=int)

        if level==k:
            #print "base case: k={0},level={1}".format(k,level)
            for xi in range(start,self.n):
                s_copy=numpy.copy(s)
                s_copy[0,xi]=1
                self.s=numpy.vstack((self.s,s_copy))
                #print self.s
            return
            
        for xi in range(start,self.n-k+level):
            s_copy=numpy.copy(s)
            s_copy[0,xi]=1
            #print "level:{0},s_scopy:{1}".format(level,s_copy)
            self.generate_k_ones(k=k,level=level+1,start=xi+1,s=s_copy)

    def generate_binary_code(self,given_range=None):
	
        if given_range is None:
            given_range=range(1,self.n+1)
	if len(given_range)>self.maxn:
	    self.success=False
	    print "Could not generate binary codes, because it is too big!"
	    return 0
        for k in given_range:
            self.generate_k_ones(k)
	self.success=True
	print "Generate binary codes successfully :)"
	return 1
                
    def print_result(self):
        print self.s

    def generate_freq_tab(self,data,header):
        """ Generate frequency table given data."""
        #data: each row is an entry, looks like [True,False,True,False,False]
        #header: list of strings of numpy vector of strings, the name of each column in data
	if not self.success:
	    print "Error! The length of the binary code is too long. Refuse to generate the huge frequency table."
	    return 0
        self.header=numpy.array(header)
        self.freq_tab=numpy.hstack( (self.s,numpy.zeros((len(self.s),1))))
        self.freq_tab=numpy.array(self.freq_tab,dtype=int)
        self.inds=numpy.zeros( (len(self.s),len(data)), dtype=bool )
        for c in range(len(self.s)):
            for d in range(len(data)):            
                if sum(data[d]==self.s[c])==self.n:
                    self.freq_tab[c,self.n]=self.freq_tab[c,self.n]+1
                    self.inds[c,d]=True
	return 1

    def replace_s(self,s_new):
	if self.n>self.maxn and len(s_new)<=2**self.n:
	    self.s=numpy.array(s_new,dtype=int)
	    self.success=True
	    print "Replace s successfully :)"
	else:
	    print "I decide not to replace s! If s is not generated before, this may lead to termination of the program."

    
    def save_freq_tab(self,dir_save):
        # dir_save: string, the file name with full paty;
	if not self.success:
	    print "Error! The length of the binary code is too big or zero! Nothing saved! Something is wrong :("
	    return 0
        header=numpy.hstack((self.header,["Total"]))
        #print header.shape
        #print self.freq_tab.shape
        print self.freq_tab
        freq_tab=numpy.vstack((header,self.freq_tab.astype(str)))
        numpy.savetxt(dir_save,freq_tab, fmt='%s', delimiter='\t', newline='\n')
	print "Saved the frequency table successfully :)"
	return 1

# test
#ubc=unique_binary_code(5)
#ubc.generate_binary_code()
#ubc.print_result()
