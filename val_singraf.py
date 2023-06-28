import os
import torch
import sys
from einops import rearrange
from builders.builders import build_dataloader
from utils.utils import instantiate_from_config
from utils.camera_trajectory import rotate_n

from torchvision.utils import save_image


if __name__ == '__main__':

    ## initizlize
    valname = sys.argv[1]
    if valname[:5] == 'logs/':
        valname = valname[5:]
    if len(sys.argv) > 2:
        code = int(sys.argv[2])
    else:
        code = 0

    checkpoint_filename = 'logs/' + valname + '/checkpoints/last.ckpt'
    out_folder = 'vals/' + valname

    device = 'cuda'
    torch.manual_seed(0)

    # input and output setting
    checkpoint = torch.load(checkpoint_filename)
    state_dict = checkpoint['state_dict']
    out_folder = out_folder + '/epoch_' + str(checkpoint['epoch']).zfill(3)
    if code != 0:
        out_folder = out_folder + '_' + str(code).zfill(3)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    print(out_folder)

    # get rid of all the params which are leftover from KID metric
    keys_for_deletion = []
    for key in state_dict.keys():
        if 'kid' in key:
            keys_for_deletion.append(key)
    for key in keys_for_deletion:
        del state_dict[key]

    opt = checkpoint['opt']

    # initialize model and load state
    singraf = instantiate_from_config(opt.model_config).to(device).eval()
    singraf.load_state_dict(state_dict, strict=True)
    z_dim = opt.model_config.params.decoder_config.params.z_dim

    # load data for intrinsic camera matrix K
    data_module = build_dataloader(opt.data_config, verbose=False)
    K_current = next(iter(data_module.val_dataloader()))['K'].to(device)

    # initialize extrinsic camera matrix at the center of the scene
    Rt_current = torch.eye(4, device=device).view(1, 1, 4, 4)

    # sample latent codes
    z = torch.randn(100, 1, z_dim, device=device)

    ## validate variation: visualization of 100 random scenes using a fixed camera pose
    camera_params = {'K': K_current, 'Rt': Rt_current}

    rgb_random = []
    for idx in range(100):
        with torch.no_grad():
            fake_rgb, fake_depth, _, _ = singraf(z[idx], camera_params=camera_params)

        rgb_current = rearrange(fake_rgb, 'b t c h w -> (b t) c h w').cpu()
        rgb_random.append(rgb_current)

    rgb_random = torch.cat(rgb_random, dim=0)
    save_image(rgb_random, os.path.join(out_folder, 'random_100.png'), nrow=10, range=(0, 1))

    ## validate view consistency: visuzliation of camera rotation for a single scene
    rot_degree = 15
    Rt_rot = rotate_n(n=rot_degree).to(device).unsqueeze(0)

    for rot in range(0, 360, rot_degree):
        with torch.no_grad():
            fake_rgb, fake_depth, _, _ = singraf(z[code], camera_params=camera_params)

        rgb_current = rearrange(fake_rgb, 'b t c h w -> (b t c) h w').cpu()
        save_image(rgb_current, os.path.join(out_folder, 'rotate_'+str(rot).zfill(3)+'.png'), nrow=1, range=(0, 1))

        camera_params['Rt'] = torch.bmm(Rt_rot, camera_params['Rt'][0]).unsqueeze(0)

    # convert into gif file and remove temporal rotated result images
    if os.path.exists(os.path.join(out_folder, 'rotate.gif')):
        os.system("rm " + os.path.join(out_folder, 'rotate.gif'))
    os.system("convert -delay 20 -loop 0 " + os.path.join(out_folder, 'rotate_*.png') + " " + os.path.join(out_folder, 'rotate.gif'))
    if os.path.exists(os.path.join(out_folder, 'rotate.gif')):
        os.system("rm " + os.path.join(out_folder, 'rotate_???.png'))

    ## simple example of panorama visualization
    K_pano = K_current.repeat(1, 4, 1, 1)
    Rt_pano = Rt_current.repeat(1, 4, 1, 1)
    Rt_rot = torch.Tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]).to(device)
    for rot in range(Rt_pano.shape[1] - 1):
        Rt_pano[:, rot+1, :, :] = Rt_rot @ Rt_pano[:, rot, :, :]

    camera_params = {'K': K_pano, 'Rt': Rt_pano}
    with torch.no_grad():
        fake_rgb, fake_depth, _, _ = singraf.generate_cylindric(z[code], camera_params=camera_params)

    rgb_current = rearrange(fake_rgb, 'b t c h w -> (b t) c h w').cpu()
    save_image(rgb_current, os.path.join(out_folder, 'sample_pano.png'), padding=0)

