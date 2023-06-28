import os
import torch
import sys
from einops import rearrange
from builders.builders import build_dataloader
from utils.utils import instantiate_from_config

from torchvision.utils import save_image
from torchvision import transforms 
from PIL import Image

from cleanfid import fid
from lpips import LPIPS


def generate_results(opt, kid_sample, var_sample, var_pair, out_folder_kid, out_folder_var):

    ## initialization
    if not os.path.exists(out_folder_kid):
        os.makedirs(out_folder_kid)
    if not os.path.exists(out_folder_var):
        os.makedirs(out_folder_var)

    # initialize model and load state
    singraf = instantiate_from_config(opt.model_config).to(device).eval()
    singraf.load_state_dict(state_dict, strict=True)
    z_dim = opt.model_config.params.decoder_config.params.z_dim

    # load data for intrinsic camera matrix K
    data_module = build_dataloader(opt.data_config, verbose=False)
    K_current = next(iter(data_module.val_dataloader()))['K'].to(device)
    camera_params = {'K': K_current}  # no Rt as input, randomly sampled
    B = camera_params['K'].shape[0]

    ## generate and save results for KID
    curr_z = 0
    eval_z, eval_Rt = [], []
    while(curr_z < kid_sample):
        z = torch.randn(B, z_dim, device=device)
        with torch.no_grad():
            y_hat, _, Rt, K = singraf(z, camera_params=camera_params)

        eval_z.append(z)
        eval_Rt.append(Rt)

        for bc in range(B):
            idx = curr_z + bc
            curr_name = 'random_' + str(idx).zfill(3)
            if idx < kid_sample:
                save_image(y_hat[bc], os.path.join(out_folder_kid, curr_name + '.png'))
            if idx < var_sample:
                os.makedirs(os.path.join(out_folder_var, curr_name))
                save_image(y_hat[bc], os.path.join(out_folder_var, curr_name, 'random_000.png'))

        curr_z = curr_z + B

    eval_z = torch.cat(eval_z)
    eval_Rt = torch.cat(eval_Rt)

    ## generate and save additional results for Average LPIPS Distance
    for idx in range(var_sample):
        camera_params['Rt'] = eval_Rt[[idx]].repeat(B, 1, 1, 1)

        curr_z = 1  # 0 is already saved
        curr_name = 'random_' + str(idx).zfill(3)
        while(curr_z <= var_pair):
            z = torch.randn(B, z_dim, device=device)
            with torch.no_grad():
                y_hat_pair, _, _, _ = singraf(z, camera_params=camera_params)

            rgb_pair = rearrange(y_hat_pair, 'b t c h w -> (b t) c h w')
            for bc in range(B):
                idx_pair = curr_z + bc
                if idx_pair <= var_pair:
                    save_image(y_hat_pair[bc], os.path.join(out_folder_var, curr_name, 'random_' + str(idx_pair).zfill(3) + '.png'))
            curr_z = curr_z + B


def image_resize(input_folder, output_folder, target_res):

    resize_transform = transforms.Compose([transforms.Resize(target_res), transforms.ToTensor()])

    var_list = [f for f in sorted(os.listdir(input_folder)) if not f.startswith('.')]
    if len(var_list) > 0:
        os.makedirs(output_folder)

    for folder in var_list:
        if folder.endswith('.png'):
            img = resize_transform(Image.open(os.path.join(input_folder, folder))).to(device)
            save_image(img, os.path.join(output_folder, folder))
        if os.path.isdir(os.path.join(input_folder, folder)):
            os.makedirs(os.path.join(output_folder, folder))
            files = [f for f in sorted(os.listdir(os.path.join(input_folder, folder))) if f.endswith('.png')]
            for file in files:
                img = resize_transform(Image.open(os.path.join(input_folder, folder, file))).to(device)
                save_image(img, os.path.join(output_folder, folder, file))


if __name__ == '__main__':

    ## initialize
    valname = sys.argv[1]
    if valname[:5] == 'logs/':
        valname = valname[5:]
    if len(sys.argv) > 2:
        target_res = int(sys.argv[2])
    else:
        target_res = 128

    checkpoint_filename = 'logs/' + valname + '/checkpoints/last.ckpt'
    out_folder = 'evals/' + valname

    device = 'cuda'
    torch.manual_seed(0)

    # input and output setting
    checkpoint = torch.load(checkpoint_filename)
    state_dict = checkpoint['state_dict']
    out_folder = out_folder + '/epoch_' + str(checkpoint['epoch']).zfill(3)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    print(out_folder)

    # get rid of all the params which are left over from KID metric
    keys_for_deletion = []
    for key in state_dict.keys():
        if 'kid' in key:
            keys_for_deletion.append(key)
    for key in keys_for_deletion:
        del state_dict[key]

    opt = checkpoint['opt']

    # setting for evaluation
    data_folder_kid = os.path.join(opt.data_config.data_dir, 'eval')

    kid_sample = 500

    var_sample = 100
    var_pair = 19
    feature_net = 'alex'  # setting from MUNIT, need to check both vgg and alex?

    batch_size = 20  # only for result generation

    ## Generate Result Images
    out_folder_kid = out_folder + '/kid'
    out_folder_var = out_folder + '/var'
    if not (os.path.exists(out_folder_kid) or os.path.exists(out_folder_var)):
        opt.data_config.batch_size = batch_size
        print("Start Generating Results:", kid_sample, var_sample, var_pair)
        generate_results(opt, kid_sample, var_sample, var_pair, out_folder_kid, out_folder_var)
    else:
        print("Using Existing Results")

    ## Resize Images
    out_folder_kid_resize = out_folder_kid + '_' + str(target_res) if target_res is not opt.data_config.img_res else out_folder_kid
    out_folder_var_resize = out_folder_var + '_' + str(target_res) if target_res is not opt.data_config.img_res else out_folder_var
    if not os.path.exists(out_folder_kid_resize):
        print("Start Resizing Results for KID:", target_res)
        image_resize(out_folder_kid, out_folder_kid_resize, target_res)
    else:
        print("Resized Folder Exists:", out_folder_kid_resize)

    if not os.path.exists(out_folder_var_resize):
        print("Start Resizing Results for Var:", target_res)
        image_resize(out_folder_var, out_folder_var_resize, target_res)
    else:
        print("Resized Folder Exists:", out_folder_var_resize)

    data_folder_kid_resize = data_folder_kid + '_' + str(target_res)
    if not os.path.exists(data_folder_kid_resize):
        print("Start Resizing Data for KID:", target_res)
        image_resize(data_folder_kid, data_folder_kid_resize, target_res)
    else:
        print("Resized Folder Exists:", data_folder_kid_resize)

    ## Evaluation Metric Calculation
    out_filename = 'result_' + str(target_res) + '.txt'
    out_folder_kid = out_folder_kid_resize
    out_folder_var = out_folder_var_resize
    data_folder_kid = data_folder_kid_resize

    print('metric calculation using:', out_folder_kid, out_folder_var, data_folder_kid)

    to_tensor = transforms.ToTensor()

    lpips = LPIPS(net=feature_net).to(device)
    lpips_mean = 0.0
    count = 0

    var_list = [f for f in sorted(os.listdir(out_folder_var)) if not f.startswith('.')]
    for folder in var_list:
        files = [f for f in sorted(os.listdir(os.path.join(out_folder_var, folder))) if f.endswith('.png')]
        y_hat, y_hat_pair = [], []
        for file in files:
            img = to_tensor(Image.open(os.path.join(out_folder_var, folder, file))).to(device)
            y_hat.append(img)
        y_hat = torch.stack(y_hat)
        y_hat_pair = y_hat[1:, ...]
        y_hat = y_hat[0, ...].repeat(y_hat_pair.shape[0], 1, 1, 1)
        lpips_mean = lpips_mean + lpips(y_hat, y_hat_pair, normalize=True).sum()
        count = count + y_hat.shape[0]

    lpips_mean = lpips_mean / count if count > 0 else lpips_mean

    print('######')
    print('### average lpips dist: ', float(lpips_mean))
    print('######')

    kid = fid.compute_kid(out_folder_kid, data_folder_kid)
    print('######')
    print('### kid: ', float(kid))
    print('######')

    with open(os.path.join(out_folder, out_filename), 'w') as f:
        f.write(out_folder + '\n')
        f.write('kid: {} (for total data of {}) \n'.format(float(kid), kid_sample))
        f.write('average lpips dist: {} (for total {}, {} x {} pairs) \n'.format(float(lpips_mean), count, var_sample, var_pair))
    print('result saved as:', out_filename)

