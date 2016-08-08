Introduction:
This is the multi-view matrix factorization (MVMF) package in Python 2.7 devloped by Yifeng Li with NRC, Ottawa.  

Version: 1.1

Depencency: numpy, scipy, matplotlib, DECRES.

Installation:
(0. Install DECRES (https://github.com/yifeng-li/DECRES) for calling the implemented classifiers.)
1. Download MVMF from github.
2. Uncompress it to your local machine, say YOUR_PATH/mvmf_v1_1.
3. Add "export PYTHONPATH=$PYTHONPATH:YOUR_PATH/mvmf_v1_1" to your .bashrc.

Models:
1. Multi-view non-negative matrix factorization (MV-NMF): mvnmf.py.
2. Multi-view non-negative matrix factorization with stability selection (SS-MV-NMF): ssmvnmf.py.
3. Automatic relavent determination non-negative matrix factorization (ARD-NMF): ardnmf.py. 
4. Variational non-negative matrix factorization models (variational NMF): vnmf.py.

Data:
1. Simulated data are available in ./data of this package; Data generator are available in this package too.
2. Multi-tumor RNA-seq data are available in ./data of this package. 

Examples:
1. main_mvnmf_sim.py: Example of how to use mvnmf on the simulated data.
2. main_ssmvnmf_sim.py: Example of how to use SS-MV-NMF on the simulated data.
3. main_mvnmf_rnaseq.py: Example of how to use MV-NMF on the multi-tumor RNA-seq data. 
4. main_ssmvnmf_rnaseq.py: Example of how to use SS-MV-NMF on the multi-tumor RNA-seq data.

References:
[1] Yifeng Li, Youlian Pan and Ziying Liu, "Multi-view non-negative matrix factorization for feature pattern discovery," Aug. 2016, submitted.

Contact: 
Yifeng Li, PhD
Research Officer
Information and Communications Technologies
National Research Council Canada
Email_1: yifeng.li@nrc-cnrc.gc.ca
Email_2: yifeng.li.cn@gmail.com

