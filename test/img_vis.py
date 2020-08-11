import pickle
import matplotlib.pyplot as plt
from scipy.misc import imsave
import sys 
mac = sys.argv[1]
if mac == '1':
    scene_path = "/Users/shamitlal/Desktop/temp/correspondence/multi_obj_480_a_15905238370637653.p"
else:
    scene_path = "/projects/katefgroup/datasets/clevr_vqa/raw/npys/multi_obj_480_a/multi_obj_480_a_15905238370637653.p"

data = pickle.load(open(scene_path, "rb"))
import torch
images = torch.tensor(data['rgb_camXs_raw']).permute(0,3,1,2)/255.
img_save = images.permute(0,2,3,1).cpu().numpy()
print("Mean: ", img_save.mean(), img_save.sum())
plt.imshow(img_save[0])
plt.show(block=True)
if mac == '1':
    imsave("/Users/shamitlal/Desktop/temp/correspondence/mac_save_1.jpg", img_save[0])
    
else:
    imsave("/home/shamitl/tmp/mac_save.jpg", img_save[0])
