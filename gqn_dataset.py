import collections, os, io
from PIL import Image
import torch
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import Dataset
import random
import pickle
import utils_disco
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import ipdb 
st = ipdb.set_trace
Context = collections.namedtuple('Context', ['frames', 'cameras'])
Scene = collections.namedtuple('Scene', ['frames', 'cameras'])


def transform_viewpoint(v):
    w, z = torch.split(v, 3, dim=-1)
    y, p = torch.split(z, 1, dim=-1)

    # position, [yaw, pitch]
    view_vector = [w, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
    v_hat = torch.cat(view_vector, dim=-1)

    return v_hat


class GQNDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx, is_pickle=True):

        if not is_pickle:
            scene_path = os.path.join(self.root_dir, "{}.pt".format(idx))
            data = torch.load(scene_path)

            byte_to_tensor = lambda x: ToTensor()(Resize(64)((Image.open(io.BytesIO(x)))))


            images = torch.stack([byte_to_tensor(frame) for frame in data.frames])

            viewpoints = torch.from_numpy(data.cameras)
            viewpoints = viewpoints.view(-1, 5)
        else:
            scene_path = os.path.join(self.root_dir, "{}.p".format(idx))
            data = pickle.load(open(scene_path, "rb"))
            viewpoints = torch.tensor(data['camera'])
            viewpoints = viewpoints.view(-1, 5)
            
            images = torch.tensor(data['frames'])
            # images_ = torch.stack([image_] for image_ in images)
            # images = images_

        if self.transform:
            images = self.transform(images)

        if self.target_transform:
            viewpoints = self.target_transform(viewpoints)

        return images, viewpoints


def transform_viewpoint_pdisco(v):
    w, z = torch.split(v, 3, dim=-1)
    y, p = torch.split(z, 1, dim=-1)

    # position, [yaw, pitch]
    view_vector = [w, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
    v_hat = torch.cat(view_vector, dim=-1)

    return v_hat


class GQNDataset_pdisco(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None, few_shot=False):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.target_res = 64
        self.N = 10
        self.few_shot = few_shot

        self.all_files = [os.path.join(root_dir,f) for f in os.listdir(root_dir) if f.endswith('.p')]

    def __len__(self):
        # return len(os.listdir(self.root_dir))
        return len(self.all_files)

    def __getitem__(self, idx, is_pickle=True):

        scene_path = self.all_files[idx]
        # print("scene path is : ", scene_path)
        data = pickle.load(open(scene_path, "rb"))
        
        viewpoints = torch.tensor(data['origin_T_camXs_raw'])
        
        rx, ry, rz = utils_disco.rotm2eul(viewpoints)
        rx, ry, rz = rx.unsqueeze(1), ry.unsqueeze(1), rz.unsqueeze(1)
        xyz = viewpoints[:, :3, -1]

                
        view_vector = [xyz, torch.cos(rx), torch.sin(rx), torch.cos(rz), torch.sin(rz)]
        viewpoints = torch.cat(view_vector, dim=-1)
        
        images = torch.tensor(data['rgb_camXs_raw']).permute(0,3,1,2)/255.
        # print("image size 1: ", images.shape)
        _, _, H_orig, W_orig = images.shape

        # img_save = images.permute(0,2,3,1).cpu().numpy()
        # print("image size 2: ", img_save.shape)
        # plt.imsave("/home/shamitl/tmp/gqn_rgb.jpg", img_save[0])

        if not self.few_shot:
            images = F.interpolate(images, self.target_res)

        images = images.permute(0,2,3,1)
        # print("Image shape: ", images.shape)
        # img_save = images.cpu().numpy()
        # plt.imsave("/home/shamitl/tmp/gqn_rgb_resized.jpg", img_save[0])

        
        bbox_origin = data['bbox_origin']
        pix_T_cams_raw = data['pix_T_cams_raw']
        # print("Pixt camXs shape: ", pix_T_cams_raw.shape)
        if not self.few_shot:
            pix_T_cams_raw = utils_disco.scale_intrinsics(torch.tensor(pix_T_cams_raw), self.target_res/(1.*W_orig), self.target_res/(1.*H_orig))
        
        
        camR_T_origin_raw = data['camR_T_origin_raw']
        origin_T_camXs_raw = data['origin_T_camXs_raw']

        shape_name = data['shape_list']
        color_name = data['color_list']
        material_name = data['material_list']
        
        all_name = []
        all_style = []
        for index in range(len(shape_name)):
            name = shape_name[index] + "/" + color_name[index] + "_" + material_name[index]
            style_name  = color_name[index] + "_" + material_name[index]
            all_name.append(name)
            all_style.append(style_name)

        object_category = all_name
        num_boxes = bbox_origin.shape[0]
        bbox_origin = np.array(bbox_origin)
        score = np.pad(np.ones([num_boxes]),[0,self.N-num_boxes])
        bbox_origin = np.pad(bbox_origin,[[0,self.N-num_boxes],[0,0],[0,0]])
        object_category = np.pad(object_category,[[0,self.N-num_boxes]],lambda x,y,z,m: "0")

        metadata = {"object_category":list(object_category), "bbox_origin":torch.tensor(bbox_origin).cuda(), "score":torch.tensor(score.astype(np.float32)).cuda(), "pix_T_cams_raw":torch.tensor(pix_T_cams_raw).cuda(), "camR_T_origin_raw":torch.tensor(camR_T_origin_raw).cuda(), "origin_T_camXs_raw":torch.tensor(origin_T_camXs_raw).cuda()}
        # metadata = {}
        # metadata = {"bbox_origin":torch.tensor(bbox_origin), "score":torch.tensor(score.astype(np.float32)), "pix_T_cams_raw":torch.tensor(pix_T_cams_raw), "camR_T_origin_raw":torch.tensor(camR_T_origin_raw), "origin_T_camXs_raw":torch.tensor(origin_T_camXs_raw)}
        return images, viewpoints, metadata


def sample_batch(x_data, v_data, D, M=None, seed=None):
    random.seed(seed)
    
    if D == "Clevr":
        K = 5
    elif D == "Room":
        K = 5
    elif D == "Jaco":
        K = 7
    elif D == "Labyrinth":
        K = 20
    elif D == "Shepard-Metzler":
        K = 15

    # Sample number of views
    if not M:
        M = random.randint(1, K)

    context_idx = random.sample(range(x_data.size(1)), M)
    query_idx = random.randint(0, x_data.size(1)-1)
    if M==3:
        query_idx = 0

    # Sample view
    x, v = x_data[:, context_idx], v_data[:, context_idx]
    # Sample query view
    x_q, v_q = x_data[:, query_idx], v_data[:, query_idx]
    
    return x, v, x_q, v_q, context_idx, query_idx