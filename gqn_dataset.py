import collections, os, io
from PIL import Image
import torch
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import Dataset
import random
import pickle
import utils_disco
import torch.nn.functional as F
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
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.target_res = 64

        self.all_files = [os.path.join(root_dir,f) for f in os.listdir(root_dir) if f.endswith('.p')]

    def __len__(self):
        # return len(os.listdir(self.root_dir))
        return len(self.all_files)

    def __getitem__(self, idx, is_pickle=True):

        scene_path = self.all_files[idx]
        data = pickle.load(open(scene_path, "rb"))
        
        viewpoints = torch.tensor(data['origin_T_camXs_raw'])
        
        rx, ry, rz = utils_disco.rotm2eul(viewpoints)
        rx, ry, rz = rx.unsqueeze(1), ry.unsqueeze(1), rz.unsqueeze(1)
        xyz = viewpoints[:, :3, -1]

        # print("rx ry rz xyz: ", rx.shape, ry.shape, rz.shape, xyz.shape)
                
        view_vector = [xyz, torch.cos(rx), torch.sin(rx), torch.cos(rz), torch.sin(rz)]
        viewpoints = torch.cat(view_vector, dim=-1)
        
        images = torch.tensor(data['rgb_camXs_raw']).permute(0,3,1,2)/255.
        # print("Image shape: ", images.shape)
        images = F.interpolate(images, self.target_res).permute(0,2,3,1)
        # if self.transform:
        #     images = self.transform(images)

        # if self.target_transform:
        #     viewpoints = self.target_transform(viewpoints)

        return images, viewpoints


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

    # Sample view
    x, v = x_data[:, context_idx], v_data[:, context_idx]
    # Sample query view
    x_q, v_q = x_data[:, query_idx], v_data[:, query_idx]
    
    return x, v, x_q, v_q