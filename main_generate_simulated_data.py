# simulate data for testing my MV-NMF model
        
import simulator

#sim=simulator(z=3,V=3,m=3)
#sim.make_W(a_active_W=10,b_active_W=1e5)
#sim.make_H(a_active_H=10,b_active_H=1,n=100)
#sim.make_X()
#dir_save="/global/data/tcga/metadata/data/"
#prefix="simulated_data"
#sim.save(dir_save,prefix)

a_active_W=10
b_active_W=1e5
a_active_H=10
b_active_H=1
sim=simulator.simulator(z=3,V=3,m=10)
sim.make_W(a_active_W=10,b_active_W=1e5)
sim.make_H(a_active_H=10,b_active_H=1,n=1000)
sim.make_X()

# save
dir_save="/home/yifeng/research/mf/mvmf_v1_1/data/"
prefix="simulated_data" + "_a_active_W="+str(a_active_W) + "_b_active_W="+str(b_active_W) + "_a_active_H="+str(a_active_H) + "_b_active_H="+str(b_active_H) 
sim.save(dir_save,prefix)








