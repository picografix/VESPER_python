import time
from chimera import runCommand as rc

start = time.time() 

map1 = "./data/emd_8097.mrc"
map2 = "./data/ChainA_simulated_resample.mrc"
output_name_1 = "./data/emd_8097_N1.mrc"
output_name_2 = "./data/ChainA_N1.mrc"

print("step1 ")
c1 = "open #0 " + map1
c2 = "open #1 " + map2 
rc(c1)

rc(c2)
print("Reducing the resolution")
# rotate map
rc("volume #0 voxelSize 0.1428")
rc("vop gaussian #0 sdev 0.3 model #2")
rc("volume #2 voxelSize 0.1428")
rc("vop gaussian #1 sdev 0.3 model  #3")
print("Saving the mrc files")

s1 = "volume #2 save " + output_name_1
s2 = "volume #3 save "  +  output_name_2
rc( s1)
rc( s2)

print("finish")

# commands to run
# volume #0.1 voxelSize 0.3
# vop gaussian #0.1 sdev 0.3 model #2

# chimera --silent --nogui fastVEC.py

end = time.time()
print("runtime="    ,end-start)
