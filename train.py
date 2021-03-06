import argparse
import datetime
import math
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from gqn_dataset import GQNDataset, Scene, transform_viewpoint, sample_batch, GQNDataset_pdisco
from scheduler import AnnealingStepLR
from model import GQN
import utils_disco
from collections import defaultdict
import ipdb 
st = ipdb.set_trace
'''
Command for CLEVR:
python train.py --pdisco_exp --run_name run_test --train_data_dir /projects/katefgroup/datasets/clevr_vqa/raw/npys/multi_obj_480_a --test_data_dir /projects/katefgroup/datasets/clevr_vqa/raw/npys/multi_obj_480_a 

python train.py --pdisco_exp --run_name run_clevr_singleobj --train_data_dir /projects/katefgroup/datasets/clevr_vqa/raw/npys/single_obj_480_all --test_data_dir /projects/katefgroup/datasets/clevr_vqa/raw/npys/single_obj_480_all
 
python train.py --pdisco_exp --run_name run_clevr_singleobj_large --train_data_dir /projects/katefgroup/datasets/clevr_vqa/raw/npys/single_obj_large_480_all --test_data_dir /projects/katefgroup/datasets/clevr_vqa/raw/npys/single_obj_large_480_all


'''


'''
python train.py --pdisco_exp --train_data_dir /projects/katefgroup/datasets/clevr_vqa/raw/npys/multi_obj_480_a --test_data_dir /projects/katefgroup/datasets/clevr_vqa/raw/npys/multi_obj_480_a --few_shot --batch_size 1 --run_name run_test_5
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generative Query Network Implementation')
    parser.add_argument('--gradient_steps', type=int, default=2*10**6, help='number of gradient steps to run (default: 2 million)')
    parser.add_argument('--batch_size', type=int, default=36, help='size of batch (default: 36)')
    # parser.add_argument('--dataset', type=str, default='Shepard-Metzler', help='dataset (dafault: Shepard-Mtzler)')
    parser.add_argument('--dataset', type=str, default='Room', help='dataset (dafault: Shepard-Mtzler)')
    parser.add_argument('--train_data_dir', type=str, help='location of training data', \
                        default="/home/shamitl/projects/torch-gqn/rooms_ring_camera-torch/train")
    parser.add_argument('--test_data_dir', type=str, help='location of test data', \
                        default="/home/shamitl/projects/torch-gqn/rooms_ring_camera-torch/test")
    parser.add_argument('--root_log_dir', type=str, help='root location of log', default='/home/shamitl/projects/torch-gqn/logs')
    parser.add_argument('--log_dir', type=str, help='log directory (default: GQN)', default='GQN')
    parser.add_argument('--log_interval', type=int, help='interval number of steps for logging', default=100)
    parser.add_argument('--save_interval', type=int, help='interval number of steps for saveing models', default=2000)
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--device_ids', type=int, nargs='+', help='list of CUDA devices (default: [0])', default=[0])
    parser.add_argument('--representation', type=str, help='representation network (default: pool)', default='pool')
    parser.add_argument('--layers', type=int, help='number of generative layers (default: 12)', default=12)
    parser.add_argument('--shared_core', type=bool, \
                        help='whether to share the weights of the cores across generation steps (default: False)', \
                        default=False)
    parser.add_argument('--seed', type=int, help='random seed (default: None)', default=None)
    parser.add_argument('--pdisco_exp', action='store_true')
    parser.add_argument('--run_name', type=str, default="run1")
    parser.add_argument('--N', type=int, default=10, help='number of objects in scene')
    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--few_shot_size', type=int, default=5, help='few shot size')
    args = parser.parse_args()

    device = f"cuda:{args.device_ids[0]}" if torch.cuda.is_available() else "cpu"
    
    # Seed
    if args.seed!=None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # Dataset directory
    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir

    # Number of workers to load data
    num_workers = args.workers

    # Log
    log_interval_num = args.log_interval
    save_interval_num = args.save_interval
    log_dir = os.path.join(args.root_log_dir, args.log_dir)
    # st()
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    if not os.path.isdir(os.path.join(log_dir, 'models')):
        os.mkdir(os.path.join(log_dir, 'models'))
    if not os.path.isdir(os.path.join(log_dir,'runs')):
        os.mkdir(os.path.join(log_dir,'runs'))

    # TensorBoardX
    writer = SummaryWriter(log_dir=os.path.join(log_dir,'runs/{}'.format(args.run_name)))

    # Dataset
    # st()
    if not args.pdisco_exp:
        train_dataset = GQNDataset(root_dir=train_data_dir, target_transform=transform_viewpoint)
        test_dataset = GQNDataset(root_dir=test_data_dir, target_transform=transform_viewpoint)
    else:
        train_dataset = GQNDataset_pdisco(root_dir=train_data_dir, target_transform=transform_viewpoint, few_shot=args.few_shot)
        test_dataset = GQNDataset_pdisco(root_dir=test_data_dir, target_transform=transform_viewpoint, few_shot=args.few_shot)
    
    D = args.dataset

    # Pixel standard-deviation
    sigma_i, sigma_f = 2.0, 0.7
    sigma = sigma_i

    # Number of scenes over which each weight update is computed
    B = args.batch_size
    __p = lambda x: utils_disco.pack_seqdim(x, B)
    __u = lambda x: utils_disco.unpack_seqdim(x, B)
    
    content_dict = defaultdict(lambda:[])
    style_dict = defaultdict(lambda:[])
    # Number of generative layers
    L =args.layers
    # st()
    # Maximum number of training steps
    S_max = args.gradient_steps

    # Define model
    model = GQN(representation=args.representation, L=L, shared_core=args.shared_core).to(device)
    if len(args.device_ids)>1:
        model = nn.DataParallel(model, device_ids=args.device_ids)
    # st()
    if args.few_shot:
        model.load_state_dict(torch.load("/home/shamitl/projects/torch-gqn/logs/GQN/models/model-40000.pt"))
    
    # model.load_state_dict(torch.load("/home/shamitl/projects/torch-gqn/logs/GQN/models/run_clevr_singleobj_large/model-14000.pt"))
    model.load_state_dict(torch.load("/home/shamitl/projects/torch-gqn/logs/GQN/models/model-60000.pt"))

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-08)
    scheduler = AnnealingStepLR(optimizer, mu_i=5e-4, mu_f=5e-5, n=1.6e6)

    # kwargs = {'num_workers':num_workers, 'pin_memory': True} if torch.cuda.is_available() else {}
    kwargs = {'num_workers':num_workers} if torch.cuda.is_available() else {}

    train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=B, shuffle=True, **kwargs)

    train_iter = iter(train_loader)
    x_data_test, v_data_test, metadata_test = next(iter(test_loader))

    few_shot_filled = False
    # Training Iterations
    for t in tqdm(range(S_max)):
        try:
            x_data, v_data, metadata = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x_data, v_data, metadata = next(train_iter)
        # st()
        # cropped_rgbs = utils_disco.get_cropped_rgb(x_data, v_data, metadata, args, __p, __u)
        x_data = x_data.to(device)
        v_data = v_data.to(device)
        if args.few_shot:
            # st()
            model.eval()
            with torch.no_grad():
                x, v, x_q, v_q, context_idx, query_idx = sample_batch(x_data, v_data, D, M=1)
                x = x.permute(0,1,4,2,3)
                x_q = x_q.permute(0,3,1,2)
                x_data_, v_data_, label_list = utils_disco.get_cropped_rgb(x, v, metadata, args, __p, __u, context_idx[0], writer, t)
                # st()
                # torch.Size([6, 256, 1, 1])
                rep = model(x_data_, v_data_, x_data_, v_data_, sigma, few_shot=True)

                if utils_disco.is_dicts_filled(content_dict, style_dict, args.few_shot_size):
                    # run evaluation code here.
                    if few_shot_filled == False:
                        few_shot_filled = True
                        style_corr = 0
                        content_corr = 0
                        style_total = 0
                        content_tot = 0
                        
                        for ckey in content_dict.keys():
                            mean_tensor = torch.mean(torch.stack(content_dict[ckey]), dim=0)
                            content_dict[ckey] = mean_tensor
                        
                        for ckey in style_dict.keys():
                            mean_tensor = torch.mean(torch.stack(style_dict[ckey]), dim=0)
                            style_dict[ckey] = mean_tensor
                        
                    
                    for i in range(rep.shape[0]):
                        content_tot += 1.0
                        rep_i = rep[i,:,0,0]
                        content_name, style_name = objcat.split('/')
                        maxi = -100000000
                        maxi_name = ""
                        for ckey in content_dict.keys():
                            # st()
                            val = content_dict[ckey]
                            dotprod = torch.sum(val*rep_i)/(val.norm() * rep_i.norm() + 1e-5)
                            if dotprod > maxi:
                                maxi = dotprod
                                maxi_name = ckey
                        
                        if content_name == maxi_name:
                            content_corr += 1.0
                    
                    
                    for i in range(rep.shape[0]):
                        style_total += 1.0
                        rep_i = rep[i,:,0,0]
                        content_name, style_name = objcat.split('/')
                        maxi = -100000000
                        maxi_name = ""
                        for ckey in style_dict.keys():
                            # st()
                            val = style_dict[ckey]
                            dotprod = torch.sum(val*rep_i)/(val.norm() * rep_i.norm() + 1e-5)
                            if dotprod > maxi:
                                maxi = dotprod
                                maxi_name = ckey
                        
                        if style_name == maxi_name:
                            style_corr += 1.0

                    print("Content precision: ", content_corr/content_tot)    
                    print("Style precision: ", style_corr/style_total)    
                    
                else:
                    # st()
                    for i in range(rep.shape[0]):
                        objcat = label_list[i]
                        rep_i = rep[i,:,0,0]
                        content_name, style_name = objcat.split('/')
                        if len(content_dict[content_name]) < args.few_shot_size:
                            content_dict[content_name].append(rep_i)

                        if len(style_dict[style_name]) < args.few_shot_size:
                            style_dict[style_name].append(rep_i)
                        




        else:
            x, v, x_q, v_q, context_idx, query_idx = sample_batch(x_data, v_data, D)
            x = x.permute(0,1,4,2,3)
            x_q = x_q.permute(0,3,1,2)
            # st()
            elbo = model(x, v, v_q, x_q, sigma)
            
            # Logs
            writer.add_scalar('train_loss', -elbo.mean(), t)
                
            with torch.no_grad():
                # Write logs to TensorBoard
                if t % log_interval_num == 0:
                    x_data_test = x_data_test.to(device)
                    v_data_test = v_data_test.to(device)

                    x_test, v_test, x_q_test, v_q_test, context_idx, query_idx = sample_batch(x_data_test, v_data_test, D, M=3, seed=0)
                    # st()
                    x_test = x_test.permute(0,1,4,2,3)
                    x_q_test = x_q_test.permute(0,3,1,2)
                    elbo_test = model(x_test, v_test, v_q_test, x_q_test, sigma)
                    
                    if len(args.device_ids)>1:
                        kl_test = model.module.kl_divergence(x_test, v_test, v_q_test, x_q_test)
                        x_q_rec_test = model.module.reconstruct(x_test, v_test, v_q_test, x_q_test)
                        x_q_hat_test = model.module.generate(x_test, v_test, v_q_test)
                    else:
                        kl_test = model.kl_divergence(x_test, v_test, v_q_test, x_q_test)
                        x_q_rec_test = model.reconstruct(x_test, v_test, v_q_test, x_q_test)
                        x_q_hat_test = model.generate(x_test, v_test, v_q_test)

                    writer.add_scalar('test_loss', -elbo_test.mean(), t)
                    writer.add_scalar('test_kl', kl_test.mean(), t)
                    writer.add_image('test_ground_truth', make_grid(x_q_test, 6, pad_value=1), t)
                    writer.add_image('test_reconstruction', make_grid(x_q_rec_test, 6, pad_value=1), t)
                    writer.add_image('test_generation', make_grid(x_q_hat_test, 6, pad_value=1), t)

                if t % save_interval_num == 0:
                    if not os.path.isdir(log_dir + "/models/{}".format(args.run_name)):
                        os.mkdir(log_dir + "/models/{}".format(args.run_name))
                    torch.save(model.state_dict(), log_dir + "/models/{}/model-{}.pt".format(args.run_name, t))

            # Compute empirical ELBO gradients
            (-elbo.mean()).backward()

            # Update parameters
            optimizer.step()
            optimizer.zero_grad()

            # Update optimizer state
            scheduler.step()

            # Pixel-variance annealing
            sigma = max(sigma_f + (sigma_i - sigma_f)*(1 - t/(2e5)), sigma_f)
            
    torch.save(model.state_dict(), log_dir + "/models/{}/model-final.pt".format(args.run_name))  
    writer.close()
