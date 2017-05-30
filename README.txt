Introduction:
This is the multi-view matrix factorization (MVMF) package in Python 2.7 devloped by Yifeng Li with NRC, Ottawa.  

Version: 1.1

Depencency: numpy, scipy, matplotlib.

Installation:
(0. Install DECRES (https://github.com/yifeng-li/DECRES) for calling the implemented classifiers.)
1. Download MVMF from github.
2. Uncompress it to your local machine, say YOUR_PATH/mvmf_v1_1.
3. Add "export PYTHONPATH=$PYTHONPATH:YOUR_PATH/mvmf_v1_1" to your .bashrc.

Models:
1. Multi-class non-negative matrix factorization (MC-NMF): mcnmf.py.
2. Multi-class non-negative matrix factorization with stability selection (SS-MC-NMF): ssmcnmf.py.
3. Automatic relavent determination non-negative matrix factorization (ARD-NMF): ardnmf.py. 
4. Variational non-negative matrix factorization models (variational NMF): vnmf.py.
5. I am designing multi-view non-negative matrix factorization models for multi-feature-set data. Stay tuned!

Data:
1. Simulated data are available in ./data of this package; Data generator are available in this package too.
2. Multi-tumor RNA-seq data are too big to upload. Please request it directly from Yifeng. 

Examples:
1. main_mcnmf_sim.py: Example of how to use MC-NMF on the simulated data.
2. main_ssmcnmf_sim.py: Example of how to use SS-MC-NMF on the simulated data.
3. main_mcnmf_rnaseq.py: Example of how to use MC-NMF on the multi-tumor RNA-seq data. 
4. main_ssmcnmf_rnaseq.py: Example of how to use SS-MC-NMF on the multi-tumor RNA-seq data.

References:
[1] Yifeng Li, Youlian Pan and Ziying Liu, "Multi-class non-negative matrix factorization for feature pattern discovery," Aug. 2016, submitted.
[2] Yifeng Li, "Advances in multi-view matrix factorizations," 2016 International Joint Conference on Neural Networks (IJCNN/WCCI), Vancouver, Canada, July 2016, pp. 3793-3800.
[3] Yifeng Li, Fangxiang Wu, and Alioune Ngom, "A review on machine learning principles for multi-view biological data integration," Briefings in Bioinformatics, accepted on Oct 14, 2016.

Contact: 
Yifeng Li, PhD
Research Officer
Information and Communications Technologies
National Research Council Canada
Email_1: yifeng.li@nrc-cnrc.gc.ca
Email_2: yifeng.li.cn@gmail.com

