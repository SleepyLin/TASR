import os
from argparse import ArgumentParser

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch.nn.functional as F

from model import ControlLDM, SwinIR, Diffusion
from utils.common import instantiate_from_config
from utils.sampler import SpacedSampler
from utils.inference import load_model_from_url
import glob
import random

import pyiqa

# random.seed(1)

def log_txt_as_img(wh, xc):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        # font = ImageFont.truetype('font/DejaVuSans.ttf', size=size)
        font = ImageFont.load_default()
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(231)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)
    print(cfg)
    # Setup an experiment folder:
    if accelerator.is_local_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")

    # Create model:
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]

    unused = cldm.load_pretrained_sd(sd)
    if accelerator.is_local_main_process:
        print(f"strictly load pretrained SD weight from {cfg.train.sd_path}\n"
              f"unused weights: {unused}")
    # load controlnet and adapter
    if "control_adapter_pretrain" in cfg.train and cfg.train.control_adapter_pretrain is not None and os.path.exists(cfg.train.control_adapter_pretrain):
        control_adapter = torch.load(cfg.train.control_adapter_pretrain, map_location="cpu")
        # load controlnet and adapter
        cldm.load_control_adapter_from_ckpt(control_adapter)
        if accelerator.is_local_main_process:
            print(f"strictly load control_adapter from checkpoint: {cfg.train.control_adapter_pretrain}")
    # only load controlnet
    elif "controlnet_pretrain" in cfg.train and cfg.train.controlnet_pretrain is not None and os.path.exists(cfg.train.controlnet_pretrain):
        controlnet = torch.load(cfg.train.controlnet_pretrain, map_location="cpu")
        cldm.load_controlnet_from_ckpt(controlnet['checkpoint'])

        if accelerator.is_local_main_process:
            print(f"strictly load controlnet from checkpoint: {cfg.train.controlnet_pretrain}")

    assert cldm.adapter_list is not None, 'cldm.adapter_list should not be not none'
    opt_controlnet = torch.optim.AdamW(cldm.controlnet.parameters(), lr=cfg.train.learning_rate)
    opt_adapter  = torch.optim.AdamW(cldm.adapter_list.parameters(), lr=cfg.train.learning_rate)
    

    epoch = 0
    global_step = 0
    resume_dir = os.path.join(cfg.train.exp_dir, 'checkpoints')
    if cfg.train.resume and os.path.exists(resume_dir):
        pthes = glob.glob(f'{resume_dir}/*.pt')
        resume_path = None
        if len(pthes) > 0:
            sorted_pthes = sorted(pthes, key=lambda s: int(s.split('/')[-1].split('.')[0]), reverse=True)
            resume_path = sorted_pthes[0]
            
        if resume_path is not None:
            resume_dict = torch.load(resume_path, map_location="cpu")
            epoch = resume_dict['epoch']
            global_step = resume_dict['global_step']
            # load ckpts
            cldm.load_control_adapter_from_ckpt(resume_dict)
            # load optimizer
            opt_controlnet.load_state_dict(resume_dict['optimizer_state_dict_controlnet'])
            opt_adapter.load_state_dict(resume_dict['optimizer_state_dict_adapter'])
            if accelerator.is_local_main_process:
                print(f"strictly resume unet from checkpoint: {resume_path}, from epoch: {epoch}, global_step: {global_step}")
            
            
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    sd = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in torch.load(cfg.train.swinir_path, map_location="cpu").items()
    }
    swinir.load_state_dict(sd, strict=True)
    for p in swinir.parameters():
        p.requires_grad = False
    if accelerator.is_local_main_process:
        print(f"load SwinIR from {cfg.train.swinir_path}")
    
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    
    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True, drop_last=True
    )
    if accelerator.is_local_main_process:
        print(f"Dataset contains {len(dataset):,}")
        
    # Prepare models for training:
    cldm.train().to(device)
    swinir.eval().to(device)
    
    # bsrnet.eval().to(device)
    diffusion.to(device)
    cldm, opt_controlnet, opt_adapter, loader = accelerator.prepare(cldm, opt_controlnet, opt_adapter, loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    
    # Variables for monitoring/logging purposes:
    max_steps = cfg.train.train_steps
    max_epochs = cfg.train.train_epochs
    # max_steps = max_epochs * 
    step_controlnet_loss = []
    step_adapter_loss = []
    epoch_controlnet_loss = []
    epoch_adapter_loss = []
    sampler = SpacedSampler(diffusion.betas)
    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_epochs} epochs...")
        for name, param in cldm.named_parameters():
            if param.requires_grad:
                print(f"Parameter name: {name}")
    
    pixel_loss_func = None
    clipiqa_loss_func = None
    clipiqa_start_range, clipiqa_end_range = 0, 201
    l1_start_range, l1_end_range  = 0, 800
    if 'pixel_loss' in cfg.train:
        l1_weight = cfg.train.pixel_loss.weight
        l1_start_range, l1_end_range = cfg.train.pixel_loss.range
        pixel_loss_func = torch.nn.L1Loss(reduction='none').to(accelerator.device)
    
    if 'clipiqa_loss' in cfg.train:
        clipiqa_weight = cfg.train.clipiqa_loss.weight
        clipiqa_start_range, clipiqa_end_range = cfg.train.clipiqa_loss.range
        clipiqa_loss_func = pyiqa.create_metric('clipiqa', as_loss=True, device=accelerator.device).to(accelerator.device)
        clipiqa_loss_func.requires_grad_(False)
    psnr_metric = pyiqa.create_metric('psnr').to(accelerator.device)
    ssim_metric = pyiqa.create_metric('ssim').to(accelerator.device)
    # while global_step < max_steps:
    while epoch < max_epochs:
        pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch", total=len(loader))
        for gt, lq, prompt in loader:
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float().to(device)
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float().to(device)
            with torch.no_grad():
                z_0 = pure_cldm.vae_encode(gt)
                clean = swinir(lq)
                # clean = bsrnet(lq)
                cond = pure_cldm.prepare_condition(clean, prompt)

            is_CLIP = global_step % 5 == 0
            # two stage training
            if is_CLIP:
                t = torch.randint(clipiqa_start_range, clipiqa_end_range, (z_0.shape[0],), device=device)   # 0-200
            else:
                t = torch.randint(0, diffusion.num_timesteps, (z_0.shape[0],), device=device)   # 0-1000

            # loss = diffusion.p_losses(cldm, z_0, t, cond)
            loss_controlnet = 0
            loss_adapter = 0
            denoise_loss, pred_x0_z = diffusion.p_losses_w_pred_x0(cldm, z_0, t, cond)
            loss_controlnet += denoise_loss
            loss_adapter += denoise_loss
            if clipiqa_loss_func is not None or pixel_loss_func is not None:
                pred_x0 = (pure_cldm.vae_decode(pred_x0_z)/2 + 0.5).clamp(0, 1)
            
            if is_CLIP and clipiqa_loss_func is not None:
                clipiqa_reward = clipiqa_loss_func(pred_x0)
                clipiqa_loss = F.relu(-clipiqa_reward + 1) * clipiqa_weight
                loss_adapter += clipiqa_loss
                
                opt_adapter.zero_grad()
                accelerator.backward(loss_adapter, retain_graph=False)
                opt_adapter.step()
                
            if not is_CLIP and pixel_loss_func is not None:
                l1_loss = pixel_loss_func(pred_x0, gt / 2 + 0.5) 
                # add l1_mask
                l1_mask = ((t < l1_end_range)&(t > l1_start_range)).to(dtype=l1_loss.dtype)
                l1_loss = (l1_mask.reshape(-1, 1, 1, 1) * l1_loss).sum()/(l1_mask.sum()*l1_loss.shape[1]*l1_loss.shape[2]*l1_loss.shape[3] + 1e-5)
                
                # l1_loss = l1_loss.sum() / (l1_loss.shape[1]*l1_loss.shape[2]*l1_loss.shape[3]+1e-5)
                l1_loss = l1_loss * l1_weight
                loss_controlnet += l1_loss

                opt_controlnet.zero_grad()
                accelerator.backward(loss_controlnet, retain_graph=False)
                opt_controlnet.step()
            
            # if is_CLIP:
            #     opt_adapter.zero_grad()
            #     accelerator.backward(loss_adapter, retain_graph=False)
            #     opt_adapter.step()
            # else:
            #     opt_controlnet.zero_grad()
            #     accelerator.backward(loss_controlnet, retain_graph=False)
            #     opt_controlnet.step()


            """
            # 获取参数名和对应参数的映射
            param_map = {id(p): n for n, p in cldm.named_parameters()}

            # 查看优化器中对应更新的参数名
            for param_group in opt_controlnet.param_groups:
                for param in param_group['params']:
                    print(f"参数名: {param_map[id(param)]}, grad:{param.grad is not None}")
            """
            accelerator.wait_for_everyone()

            global_step += 1
            step_controlnet_loss.append(loss_controlnet.item())
            step_adapter_loss.append(loss_adapter.item())
            epoch_controlnet_loss.append(loss_controlnet.item())
            epoch_adapter_loss.append(loss_adapter.item())
            pbar.update(1)
            
            loss_str = f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Denoise Loss: {denoise_loss.item():.6f}"
            if not is_CLIP and pixel_loss_func is not None:
                loss_str+= f", L1 Loss: {l1_loss.item():.6f}" 
            if is_CLIP and clipiqa_loss_func is not None:
                loss_str+= f", Clipiqa Loss: {clipiqa_loss.item():.6f}" 
            pbar.set_description(loss_str)
            # pbar.set_description(f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, L1 Loss: {l1_loss.item():.6f}, Clipiqa Loss: {clipiqa_loss.item():.6f}, Denoise Loss: {denoise_loss.item():.6f}, Loss: {loss.item():.6f}")
            # pbar.set_description(f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, L1 Loss: {l1_loss.item():.6f}, Denoise Loss: {denoise_loss.item():.6f}, Loss: {loss.item():.6f}")
            


            # Log loss values:
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                # Gather values from all processes
                avg_controlnet_loss = accelerator.gather(torch.tensor(step_controlnet_loss, device=device).unsqueeze(0)).mean().item()
                step_controlnet_loss.clear()
                avg_adapter_loss = accelerator.gather(torch.tensor(step_adapter_loss, device=device).unsqueeze(0)).mean().item()
                step_adapter_loss.clear()
                if accelerator.is_local_main_process:
                    writer.add_scalar("loss/loss_controlnet_step", avg_controlnet_loss, global_step)
                    writer.add_scalar("loss/loss_adapter_step", avg_adapter_loss, global_step)

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_local_main_process:
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}_{epoch}.pt"
                    # torch.save(checkpoint, ckpt_path)
                    torch.save({
                            'checkpoint_controlnet': pure_cldm.controlnet.state_dict(),
                            'checkpoint_adapter': pure_cldm.adapter_list.state_dict(),
                            'epoch': epoch,
                            'global_step': global_step,
                            'optimizer_state_dict_controlnet': opt_controlnet.state_dict(),
                            'optimizer_state_dict_adapter': opt_adapter.state_dict(),
                        }, ckpt_path)

            if global_step % cfg.train.image_every == 0 or global_step == 1:
                N = 12
                log_clean = clean[:N]
                log_cond = {k:v[:N] for k, v in cond.items()}
                log_gt, log_lq = gt[:N], lq[:N]
                log_prompt = prompt[:N]
                cldm.eval()
                with torch.no_grad():
                    z = sampler.sample(
                        model=cldm, device=device, steps=50, batch_size=len(log_gt), x_size=z_0.shape[1:],
                        cond=log_cond, uncond=None, cfg_scale=1.0, x_T=None,
                        progress=accelerator.is_local_main_process, progress_leave=False
                    )
                    if accelerator.is_local_main_process:
                        for tag, image in [
                            ("image/samples", (pure_cldm.vae_decode(z) + 1) / 2),
                            ("image/gt", (log_gt + 1) / 2),
                            ("image/lq", log_lq),
                            # ("image/condition", log_clean),
                            ("image/condition_decoded", (pure_cldm.vae_decode(log_cond["c_img"]) + 1) / 2),
                            # ("image/prompt", (log_txt_as_img((512, 512), log_prompt) + 1) / 2)
                        ]:
                            writer.add_image(tag, make_grid(image, nrow=4), global_step)
                        
                        pred_img = ((pure_cldm.vae_decode(z) + 1) / 2).clamp(0, 1)
                        gt_img = ((log_gt + 1) / 2).clamp(0, 1)
                        B = pred_img.shape[0]
                        score_psnr = psnr_metric(pred_img, gt_img).sum() / B
                        score_ssim = ssim_metric(pred_img, gt_img).sum() / B

                        score_clipiqa = clipiqa_loss_func(pred_img)
                        writer.add_scalar("val/psnr", score_psnr, global_step)
                        writer.add_scalar("val/ssim", score_ssim, global_step)
                        writer.add_scalar("val/clipiqa", score_clipiqa, global_step)



                cldm.train()
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break
        
        pbar.close()
        epoch += 1
        avg_epoch_controlnet_loss = accelerator.gather(torch.tensor(epoch_controlnet_loss, device=device).unsqueeze(0)).mean().item()
        epoch_controlnet_loss.clear()
        avg_epoch_adapter_loss = accelerator.gather(torch.tensor(epoch_adapter_loss, device=device).unsqueeze(0)).mean().item()
        epoch_adapter_loss.clear()

        if accelerator.is_local_main_process:
            writer.add_scalar("loss/loss_simple_epoch_controlnet", avg_epoch_controlnet_loss, global_step)
            writer.add_scalar("loss/loss_simple_epoch_adapter", avg_epoch_adapter_loss, global_step)

    if accelerator.is_local_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
