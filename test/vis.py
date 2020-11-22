import pickle 
import matplotlib.pyplot as plt 
import sys 
import numpy as np 
# pfile = sys.argv[1]
# path = "/Users/shamitlal/Desktop/temp/correspondence/" + pfile
# p = pickle.load(open(path, "rb"))
# rgb = p['rgb_camXs_raw'][0]
# plt.imshow(rgb)
# plt.show(block=True)
import ipdb 
st = ipdb.set_trace 
p = pickle.load(open('/Users/shamitlal/Desktop/temp/cvpr21/1604971790359981.p', 'rb'))
rgb = p['rgb_camXs_raw'][0]
obj = p['object_info_s_list'][0]
rgb = rgb/255.
for key in obj:
    obji = obj[key]
    bbox = obji[4]
    smap = obji[5]
    rgbc = np.copy(rgb)
    ymin,xmin,ymax,xmax = bbox
    mask = np.zeros((256, 256, 3))
    rgbc[ymin:ymax, xmin:xmax,:] = 1.0
    
    vis = np.concatenate([rgb, rgbc], axis=0)
    plt.imshow(vis)
    plt.show(block=True)
