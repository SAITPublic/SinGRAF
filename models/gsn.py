import copy
import torch
from torchmetrics import KID
import pytorch_lightning as pl
from einops import rearrange, repeat
from torchmetrics.utilities.data import dim_zero_cat

from utils.utils import instantiate_from_config
from .model_utils import RenderParams, ema_accumulate

import torch.distributed as dist
from .camera_opt import CameraPoseCarrier
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig

import math
import random
import torchvision.transforms.functional as TF


class SinGRAF(pl.LightningModule):
    def __init__(
        self,
        loss_config,
        decoder_config,
        generator_config,
        texture_net_config=None,
        img_res=64,
        patch_size=None,
        lr=0.002,
        ttur_ratio=1.0,
        voxel_res=32,
        voxel_size=0.6,
        opt_pose=False,
        jitter=False,
        t_jitter=0.0,
        r_jitter=0.0,
        **kwargs,
    ):
        super().__init__()

        self.img_res = img_res
        self.patch_size = patch_size
        self.lr = lr
        self.ttur_ratio = ttur_ratio
        self.coordinate_scale = voxel_res * voxel_size
        self.z_dim = decoder_config.params.z_dim
        self.voxel_res = voxel_res
        self.opt_pose = opt_pose
        self.jitter = jitter
        self.t_jitter = t_jitter
        self.r_jitter = r_jitter

        decoder_config.params.out_res = voxel_res

        generator_config.params.img_res = img_res
        generator_config.params.global_feat_res = voxel_res
        generator_config.params.coordinate_scale = self.coordinate_scale
        generator_config.params.nerf_mlp_config.params.z_dim = decoder_config.params.out_ch

        self.generator_config = generator_config

        if not generator_config.params.feature_nerf and generator_config.params.nerf_mlp_config.params.out_channel == 3:
            self.use_upsampling = False
        else:
            self.use_upsampling = True

        if texture_net_config is not None:
            texture_net_config.params.in_channel = generator_config.params.nerf_mlp_config.params.out_channel
            texture_net_config.params.in_res = generator_config.params.nerf_out_res
            texture_net_config.params.out_res = img_res

        loss_config.params.discriminator_config.params.in_channel = 3
        loss_config.params.discriminator_config.params.in_res = img_res

        if loss_config.params.scale_disc:
            loss_config.params.discriminator_config.params.in_channel = loss_config.params.discriminator_config.params.in_channel + 1
            self.scale_disc = True
            loss_config.params.discriminator_config.params.in_res = generator_config.params.nerf_out_res  # discriminator setting for patches
            self.scale_level = loss_config.params.scale_level
            self.scale_epoch = loss_config.params.scale_epoch
            if isinstance(loss_config.params.scale_range, list) or isinstance(loss_config.params.scale_range, ListConfig):
                self.scale_var = [loss_config.params.scale_range[0] / 2.0, loss_config.params.scale_range[1] / 2.0]
            else:
                self.scale_var = [loss_config.params.scale_range / 2.0, loss_config.params.scale_range / 2.0]
            if isinstance(loss_config.params.scale_perspective, list) or isinstance(loss_config.params.scale_perspective, ListConfig):
                self.scale_perspective = loss_config.params.scale_perspective
            else:
                self.scale_perspective = [loss_config.params.scale_perspective, loss_config.params.scale_perspective]

        else:
            self.scale_disc = False

        if 'dual_disc' in loss_config.params.keys() and loss_config.params.dual_disc:
            loss_config.params.discriminator_config.params.in_channel = loss_config.params.discriminator_config.params.in_channel + 3
            self.dual_disc = True
        else:
            self.dual_disc = False

        self.decoder = instantiate_from_config(decoder_config)
        self.generator = instantiate_from_config(generator_config)
        self.loss = instantiate_from_config(loss_config)

        self.decoder_ema = copy.deepcopy(self.decoder)
        self.generator_ema = copy.deepcopy(self.generator)

        if self.use_upsampling:
            self.texture_net = instantiate_from_config(texture_net_config)
            self.texture_net_ema = copy.deepcopy(self.texture_net)

        # camera sampler setting
        self.set_trajectory_sampler()


    def set_trajectory_sampler(self, trajectory_sampler=None):
        if self.opt_pose or trajectory_sampler is None:
            self.trajectory_sampler = CameraPoseCarrier(trajectory_sampler=trajectory_sampler, lock_axes=('x', 'z'), lock_t=('y'),
                                                            random_init_params={'N': 1000},
                                                            jitter=self.jitter, t_jitter=self.t_jitter, r_jitter=self.r_jitter)
        else:
            self.trajectory_sampler = trajectory_sampler


    def set_steps_per_epoch(self, steps=1000):
        self.steps_per_epoch = float(int(steps))


    def get_random_patch_scale(self, progressive_ratio, batch_size):
        # define values for current patch scale
        scale_var_curr = self.scale_var[0] * (1.0 - progressive_ratio) + self.scale_var[1] * progressive_ratio
        scale_curr = (self.scale_level[0] - self.scale_var[0]) * (1.0 - progressive_ratio) + (self.scale_level[1] + self.scale_var[1]) * progressive_ratio

        # random patch scaling sampling
        scale_val = scale_curr + (torch.rand(batch_size) * 2.0 - 1.0) * scale_var_curr

        # additional random degree for perspective augmentation (only used for real images)
        scale_degree = self.scale_perspective[0] * (1.0 - progressive_ratio) + self.scale_perspective[1] * progressive_ratio

        return scale_val, scale_curr, scale_var_curr, scale_degree


    def generate(self, z, camera_params, dual_disc=False, scale_disc=False):
        # camera_params should be a dict with Rt and K (if Rt is not present it will be sampled)

        nerf_out_res = self.generator_config.params.nerf_out_res
        samples_per_ray = self.generator_config.params.samples_per_ray

        # use EMA weights if in eval mode
        decoder = self.decoder if self.training else self.decoder_ema
        generator = self.generator if self.training else self.generator_ema
        if self.use_upsampling:
            texture_net = self.texture_net if self.training else self.texture_net_ema
        elif not scale_disc:  # for inference full image rendering after patch-based training
            nerf_out_res = self.img_res

        # map 1D latent code z to 2D latent code w
        w = decoder(z=z)

        # camera sampling
        if 'Rt' not in camera_params.keys():
            Rt = self.trajectory_sampler.sample_trajectories(self.generator, w)
            camera_params['Rt'] = Rt

        # duplicate latent codes along the trajectory dimension
        T = camera_params['Rt'].shape[1]  # trajectory length
        w = repeat(w, 'b c h w -> b t c h w', t=T)
        w = rearrange(w, 'b t c h w -> (b t) c h w')


        # rendering progressive-scale random patches (scale_disc == True)
        if scale_disc:
            # define random scale samples
            if self.scale_epoch > 0:
                progressive_ratio = min(self.global_step / (self.steps_per_epoch * self.scale_epoch), 1.0)
            else:
                progressive_ratio = 1.0
            scale_val, scale_curr, _, _ = self.get_random_patch_scale(progressive_ratio, w.shape[0])

            render_params = RenderParams(
                Rt=None,
                K=None,
                samples_per_ray=samples_per_ray,
                near=self.generator_config.params.near,
                far=self.generator_config.params.far,
                alpha_noise_std=self.generator_config.params.alpha_noise_std,
                nerf_out_res=None,
                mask=None,
            )

            # rendering patches in available random positions
            rgb, depth, scale = [], [], []
            for idx in range(len(scale_val)):
                render_res = int(nerf_out_res / scale_val[idx])
                ii, jj = torch.meshgrid(torch.arange(render_res), torch.arange(render_res))
                rand_i, rand_j = torch.randint(render_res - nerf_out_res + 1, (2, ))
                mask = ii[rand_i:rand_i+nerf_out_res, rand_j:rand_j+nerf_out_res] * render_res + jj[rand_i:rand_i+nerf_out_res, rand_j:rand_j+nerf_out_res]
                mask = mask.view(1, -1).to(w.device)

                # camera optimization only available for early training epochs with large patch scales
                if self.opt_pose and scale_curr > 0.5:
                    render_params.Rt = rearrange(camera_params['Rt'][[idx]], 'b t h w -> (b t) h w').clone()
                else:
                    render_params.Rt = rearrange(camera_params['Rt'][[idx]], 'b t h w -> (b t) h w').detach().clone()
                render_params.K = rearrange(camera_params['K'][[idx]], 'b t h w -> (b t) h w').clone()
                render_params.nerf_out_res = int(nerf_out_res / scale_val[idx])
                render_params.mask = mask

                y_hat = generator(local_latents=w[[idx]], render_params=render_params)
                rgb.append(y_hat['rgb'])  # shape [BT, HW, C]
                depth.append(y_hat['depth'])
                scale.append(torch.ones(y_hat['rgb'].shape[0], y_hat['rgb'].shape[1], 1, dtype=y_hat['rgb'].dtype, device=y_hat['rgb'].device) * scale_val[idx])

            rgb = torch.cat(rgb, dim=0)
            depth = torch.cat(depth, dim=0)
            scale = torch.cat(scale, dim=0)

        # rendering full images (scale_disc == False)
        else:
            if self.patch_size is None:
                # compute full image in one pass
                indices_chunks = [None]
            elif nerf_out_res <= self.patch_size:
                indices_chunks = [None]
            elif nerf_out_res > self.patch_size:
                # break the whole image into manageable pieces, then compute each of those separately
                indices = torch.arange(nerf_out_res ** 2, device=z.device)
                indices_chunks = torch.chunk(indices, chunks=int(nerf_out_res ** 2 / self.patch_size ** 2))

            rgb, depth = [], []
            for indices in indices_chunks:
                render_params = RenderParams(
                    Rt=rearrange(camera_params['Rt'], 'b t h w -> (b t) h w').clone(),
                    K=rearrange(camera_params['K'], 'b t h w -> (b t) h w').clone(),
                    samples_per_ray=samples_per_ray,
                    near=self.generator_config.params.near,
                    far=self.generator_config.params.far,
                    alpha_noise_std=self.generator_config.params.alpha_noise_std,
                    nerf_out_res=nerf_out_res,
                    mask=indices,
                )

                # camera optimization only available for early training epochs
                if self.training and self.opt_pose and self.global_step <= self.steps_per_epoch * 30:
                    render_params.Rt = rearrange(camera_params['Rt'], 'b t h w -> (b t) h w').clone()
                else:
                    render_params.Rt = rearrange(camera_params['Rt'], 'b t h w -> (b t) h w').detach().clone()

                y_hat = generator(local_latents=w, render_params=render_params)
                rgb.append(y_hat['rgb'])  # shape [BT, HW, C]
                depth.append(y_hat['depth'])

            # combine image patches back into full images
            rgb = torch.cat(rgb, dim=1)
            depth = torch.cat(depth, dim=1)

        rgb = rearrange(rgb, 'b (h w) c -> b c h w', h=nerf_out_res, w=nerf_out_res)
        if dual_disc:  # only available with upsampling
            rgb_interp = torch.nn.functional.interpolate(rgb[:, :3, :, :], size=self.img_res, mode='bilinear', align_corners=False)
            rgb_final = texture_net(rgb)
            rgb_final = torch.cat([rgb_final, rgb_interp], dim=1)
            rgb = torch.cat([rgb[:, :3, :, :], rgb[:, :3, :, :]], dim=1)

        elif self.use_upsampling:
            rgb_final = texture_net(rgb)
            rgb = rgb[:, :3, :, :]

        else:
            rgb_final = rgb

        rgb = rearrange(rgb_final, '(b t) c h w -> b t c h w', t=T)
        depth = rearrange(depth, '(b t) (h w) -> b t 1 h w', t=T, h=nerf_out_res, w=nerf_out_res)

        Rt = rearrange(y_hat['Rt'], '(b t) h w -> b t h w', t=T)
        K = rearrange(y_hat['K'], '(b t) h w -> b t h w', t=T)

        if scale_disc:
            scale = rearrange(scale, 'b (h w) c -> b c h w', h=nerf_out_res, w=nerf_out_res)
            scale = rearrange(scale, '(b t) c h w -> b t c h w', t=T)
            return rgb, depth, scale, K
        else:
            return rgb, depth, Rt, K


    ## only for inference/visualization, not for training
    def generate_cylindric(self, z, camera_params):

        nerf_out_res = self.generator_config.params.nerf_out_res
        samples_per_ray = self.generator_config.params.samples_per_ray

        # use EMA weights if in eval mode
        decoder = self.decoder if self.training else self.decoder_ema
        generator = self.generator if self.training else self.generator_ema
        #texture_net = self.texture_net if self.training else self.texture_net_ema
        if self.use_upsampling:
            texture_net = self.texture_net if self.training else self.texture_net_ema
        else:
            nerf_out_res = self.img_res

        # map 1D latent code z to 2D latent code w
        w = decoder(z=z)

        # duplicate latent codes along the trajectory dimension
        T = camera_params['Rt'].shape[1]  # trajectory length
        w = repeat(w, 'b c h w -> b t c h w', t=T)
        w = rearrange(w, 'b t c h w -> (b t) c h w')

        if self.patch_size is None:
            # compute full image in one pass
            indices_chunks = [None]
        elif nerf_out_res <= self.patch_size:
            indices_chunks = [None]
        elif nerf_out_res > self.patch_size:
            # break the whole image into manageable pieces, then compute each of those separately
            indices = torch.arange(nerf_out_res ** 2, device=z.device)
            indices_chunks = torch.chunk(indices, chunks=int(nerf_out_res ** 2 / self.patch_size ** 2))

        rgb, depth = [], []
        for indices in indices_chunks:
            render_params = RenderParams(
                Rt=rearrange(camera_params['Rt'], 'b t h w -> (b t) h w').clone(),
                K=rearrange(camera_params['K'], 'b t h w -> (b t) h w').clone(),
                samples_per_ray=samples_per_ray,
                near=self.generator_config.params.near,
                far=self.generator_config.params.far,
                alpha_noise_std=self.generator_config.params.alpha_noise_std,
                nerf_out_res=nerf_out_res,
                mask=indices,
            )
            
            y_hat = generator.render(local_latents=w, render_params=render_params, mode='cylindric')  # for cylindric panorama
            rgb.append(y_hat['rgb'])  # shape [BT, HW, C]
            depth.append(y_hat['depth'])

        # combine image patches back into full images
        rgb = torch.cat(rgb, dim=1)
        depth = torch.cat(depth, dim=1)

        rgb = rearrange(rgb, 'b (h w) c -> b c h w', h=nerf_out_res, w=nerf_out_res)

        if self.use_upsampling:
            rgb_final = texture_net(rgb)
            rgb = rgb[:, :3, :, :]
        else:
            rgb_final = rgb

        rgb = rearrange(rgb_final, '(b t) c h w -> b t c h w', t=T)
        depth = rearrange(depth, '(b t) (h w) -> b t 1 h w', t=T, h=nerf_out_res, w=nerf_out_res)
        Rt = rearrange(y_hat['Rt'], '(b t) h w -> b t h w', t=T)
        K = rearrange(y_hat['K'], '(b t) h w -> b t h w', t=T)

        return rgb, depth, Rt, K

    def on_train_batch_end(self, *args, **kwargs):
        self.update_ema()
        if hasattr(self.trajectory_sampler, 'normalize'):
            self.trajectory_sampler.normalize()

    def update_ema(self, decay=0.999):
        ema_accumulate(self.decoder_ema, self.decoder, decay)
        ema_accumulate(self.generator_ema, self.generator, decay)
        if self.use_upsampling:
            ema_accumulate(self.texture_net_ema, self.texture_net, decay)

    def forward(self, z, camera_params):
        rgb, depth, Rt, K = self.generate(z, camera_params)
        return rgb, depth, Rt, K

    def training_step(self, x, batch_idx, optimizer_idx):
        B = x['rgb'].shape[0]

        # redraw latent codes until each rank has a unique one (otherwise each rank samples the exact same codes)
        rank = dist.get_rank()
        for i in range(rank + 1):
            z = torch.randn(B, self.z_dim, device=x['rgb'].device)

        y_rgb = rearrange(x['rgb'].clone(), 'b t c h w -> (b t) c h w')
        y_scale = None

        if self.dual_disc:
            nerf_out_res = self.generator_config.params.nerf_out_res
            y_rgb_interp = torch.nn.functional.interpolate(y_rgb, size=nerf_out_res, mode='bilinear', align_corners=False)
            y_rgb_interp = torch.nn.functional.interpolate(y_rgb_interp, size=y_rgb.shape[-1], mode='bilinear', align_corners=False)
            y_rgb = torch.cat([y_rgb, y_rgb_interp], dim=1)

        # sampling progressive-scale random patches with perspective augmentation from real images (scale_disc == True)
        if self.scale_disc:
            nerf_out_res = self.generator_config.params.nerf_out_res

            # define random scale samples with current degree for perspective augmentation
            if self.scale_epoch > 0:
                progressive_ratio = min(self.global_step / (self.steps_per_epoch * self.scale_epoch), 1.0)
            else:
                progressive_ratio = 1.0
            scale_val, scale_curr, scale_var_curr, scale_degree = self.get_random_patch_scale(progressive_ratio, y_rgb.shape[0])

            # random degree sampling to rotate for perspective augmentation
            if scale_degree > 0:
                ps = [[0, 0], [y_rgb.shape[3], 0], [0, y_rgb.shape[2]], [y_rgb.shape[3], y_rgb.shape[2]]]
                center = torch.Tensor([y_rgb.shape[3], y_rgb.shape[2]]).to(y_rgb.device) * 0.5
                focal = torch.Tensor([x['K'][0, 0, 0, 0], x['K'][0, 0, 1, 1]]).to(y_rgb.device)
                pt_norm = ((torch.Tensor(ps).to(y_rgb.device) - center) / focal)
                pt = pt_norm * focal + center

                # maximum possible in available range (no unknown pixels) based on patch scale
                max_scale = scale_curr + scale_var_curr
                max_theta = math.asin((max_scale*(1-2*max_scale) + (3*max_scale*max_scale-4*max_scale+2)**0.5) / (4*max_scale*max_scale-4*max_scale+2))
                max_theta = min(max_theta, math.radians(scale_degree))

                # random rotation angle sampling
                rand_theta = torch.rand(y_rgb.shape[0], device=y_rgb.device) * max_theta * 2.0 - max_theta
            else:
                rand_theta = torch.zeros(y_rgb.shape[0], device=y_rgb.device)

            # sample patches in available range with perspective augmentation
            y_rgb_patch, y_scale = [], []
            for idx in range(y_rgb.shape[0]):
                render_res = math.ceil(nerf_out_res / scale_val[idx])
                if rand_theta[idx]:  # need to be improved...
                    # perspective transformation
                    cos_theta = torch.cos(rand_theta[idx])
                    sin_theta = torch.sin(rand_theta[idx])
                    qt = torch.stack([(pt_norm[:, 0] * cos_theta - sin_theta) / (pt_norm[:, 0] * sin_theta + cos_theta), pt_norm[:, 1] / (pt_norm[:, 0] * sin_theta + cos_theta)]).transpose(0, 1)
                    qt = qt * focal + center
                    y_rgb[idx] = TF.perspective(y_rgb[idx], ps, qt.tolist())

                    # possible random offset
                    if qt[0, 0] < 0:
                        origin = torch.Tensor([pt[0, 0], center[1]])
                        other = pt[1]
                    else:
                        origin = torch.Tensor([pt[1, 0], center[1]])
                        other = pt[0]
                    a1 = qt[0, 1] - qt[1, 1]
                    b1 = qt[1, 0] - qt[0, 0]
                    c1 = qt[0, 1] * (qt[0, 0] - qt[1, 0]) - qt[0, 0] * (qt[0, 1] - qt[1, 1])
                    a2 = origin[1] - other[1]
                    b2 = other[0] - origin[0]
                    c2 = origin[1] * (origin[0] - other[0]) - origin[0] * (origin[1] - other[1])
                    intersect = torch.Tensor([(b1*c2-b2*c1)/(a1*b2-a2*b1), (c1*a2-c2*a1)/(a1*b2-a2*b1)])
                    ratio = float(render_res) / y_rgb.shape[2]
                    rand_i = torch.randint(int(intersect[1] * ratio), int((2 * origin[1] - intersect[1]) * ratio) - nerf_out_res + 1, (1, ))
                    ratio = float(render_res) / y_rgb.shape[3]
                    rand_j = torch.randint(int(min(origin[0], intersect[0]) * ratio), int(max(origin[0], intersect[0]) * ratio) - nerf_out_res + 1, (1, ))
                else:
                    rand_i, rand_j = torch.randint(render_res - nerf_out_res + 1, (2, ))
                y_rgb_patch.append(torch.nn.functional.interpolate(y_rgb[[idx]], size=render_res, mode='bilinear', align_corners=False)[:, :, rand_i:rand_i+nerf_out_res, rand_j:rand_j+nerf_out_res])
                y_scale.append(torch.ones(1, 1, nerf_out_res, nerf_out_res, device=y_rgb.device) * scale_val[idx])

            y_rgb = torch.cat(y_rgb_patch, dim=0)
            y_scale = torch.cat(y_scale, dim=0)


        y_hat_rgb, y_hat_depth, y_hat_scale, _ = self.generate(z, camera_params=x, dual_disc=self.dual_disc, scale_disc=self.scale_disc)
        y_hat_scale = rearrange(y_hat_scale, 'b t c h w -> (b t) c h w') if self.scale_disc else None

        y_hat_rgb = rearrange(y_hat_rgb, 'b t c h w -> (b t) c h w')

        loss, log_dict = self.loss(y_rgb, y_hat_rgb, y_scale, y_hat_scale, self.global_step, optimizer_idx)

        for key, value in log_dict.items():
            self.log(key, value, rank_zero_only=True, prog_bar=True)

        return loss

    def validation_step(self, x, batch_idx):
        # redraw latent codes until each rank has a unique one (otherwise each rank samples the exact same codes)
        rank = dist.get_rank()
        for i in range(rank + 1):
            z = torch.randn(x['K'].shape[0], self.z_dim, device=x['K'].device)

        rgb_fake, _, _, _ = self(z, x)
        rgb_fake = rearrange(rgb_fake, 'b t c h w -> (b t) c h w')
        rgb_real = rearrange(x['rgb'].clone(), 'b t c h w -> (b t) c h w')

        # KID
        if not hasattr(self, 'kid'):
            self.kid = KID(subset_size=100).cuda()
        elif batch_idx == 0:
            self.kid.reset()

        self.kid.update((rgb_real * 255).type(torch.uint8), real=True)
        self.kid.update((rgb_fake * 255).type(torch.uint8), real=False)

        return

    def validation_epoch_end(self, outputs):
        # KID
        # each process stores features separately, so gather them together to calculate KID over the full distribution
        real_features = dim_zero_cat(self.kid.real_features)
        real_features_list = [torch.empty_like(real_features) for _ in range(dist.get_world_size())]
        dist.all_gather(real_features_list, real_features)
        #real_features = dim_zero_cat(real_features_list)
        real_features = real_features_list

        fake_features = dim_zero_cat(self.kid.fake_features)
        fake_features_list = [torch.empty_like(fake_features) for _ in range(dist.get_world_size())]
        dist.all_gather(fake_features_list, fake_features)
        fake_features = fake_features_list

        rank = dist.get_rank()
        if rank == 0:
            self.kid.real_features = real_features
            self.kid.fake_features = fake_features
            kid, kid_std = self.kid.compute()
            print('')
            print('KID with {} samples: {}, {}'.format(len(dim_zero_cat(real_features)), kid, kid_std))
            print('')
        else:
            kid = torch.tensor([0.0], device=self.device)

        # share the result with all GPUs so that the checkpointing function doesn't crash
        dist.broadcast(tensor=kid, src=0)
        self.log('metrics/kid', kid, rank_zero_only=True)


    def configure_optimizers(self):
        list_params = list(self.decoder.parameters()) + list(self.generator.parameters())
        if self.use_upsampling:
            list_params += list(self.texture_net.parameters())
        if self.opt_pose:
            list_params += list(self.trajectory_sampler.parameters())
        opt_ae = torch.optim.RMSprop(
            list_params,
            lr=self.lr,
            alpha=0.99,
            eps=1e-8,
        )

        opt_disc = torch.optim.RMSprop(
            self.loss.discriminator.parameters(), lr=self.lr * self.ttur_ratio, alpha=0.99, eps=1e-8
        )

        warmup_epoch = 5
        lambda_warmup = lambda epoch: float(epoch+1)/float(warmup_epoch) if epoch<warmup_epoch else 1.0
        sch_ae = torch.optim.lr_scheduler.LambdaLR(opt_ae, lr_lambda=lambda_warmup)
        sch_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, lr_lambda=lambda_warmup)

        return [opt_ae, opt_disc], [sch_ae, sch_disc]

    def on_save_checkpoint(self, checkpoint):
        # save the config if its available
        try:
            checkpoint['opt'] = self.opt
        except Exception:
            pass
