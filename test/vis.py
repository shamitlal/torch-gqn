import pickle 
import matplotlib.pyplot as plt 
import sys 
pfile = sys.argv[1]
path = "/Users/shamitlal/Desktop/temp/correspondence/" + pfile
p = pickle.load(open(path, "rb"))
rgb = p['rgb_camXs_raw'][0]
plt.imshow(rgb)
plt.show(block=True)
