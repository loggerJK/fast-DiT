# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import os
import numpy as np
import warnings


def enable_full_determinism():
    """
    Helper function for reproducible behavior during distributed training. See
    - https://pytorch.org/docs/stable/notes/randomness.html for pytorch
    """
    #  Enable PyTorch deterministic mode. This potentially requires either the environment
    #  variable 'CUDA_LAUNCH_BLOCKING' or 'CUBLAS_WORKSPACE_CONFIG' to be set,
    # depending on the CUDA version, so we set them both here
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)

    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False

    # Numpy
    np.random.seed(0)

def main(args):
    # enable_full_determinism()
    using_cfg = args.cfg_scale > 1.0
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    if args.lsun and args.num_classes != 1:
        warnings.warn("LSUN only has one class, setting num_classes to 1.")
        args.num_classes = 1

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        register=args.register,
        save_attn=args.save_attn,
        save_final_layer_patches=args.save_final_layer_patches,
        save_activation=args.save_activation,
        zero_token=args.zero_token
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    if args.lsun:
        class_labels = [0]
        if args.num_classes != 1:
            warnings.warn("LSUN only has one class, setting num_classes to 1.")
            args.num_classes = 1
    else :
        class_labels = [207]
        # class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    if using_cfg:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([args.num_classes] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        sample_fn = model.forward_with_cfg
    else:
        model_kwargs = dict(y=y)
        sample_fn = model.forward

    # Sample images:
    samples = diffusion.p_sample_loop(
        sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    if using_cfg:
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)
    filename = f"sample_seed{args.seed}"
    if args.zero_token > 0:
        filename += f"_zero_token{args.zero_token}"
    if args.add_filename:
        filename += f"_{args.add_filename}"
    save_image(samples, os.path.join(save_folder,filename + ".png"), nrow=1, normalize=True, value_range=(-1, 1))

    if args.save_attn:
        attn_filename = filename + "_attn_maps.pt"
        torch.save(model.attention_maps_list, os.path.join(save_folder,attn_filename))

    if args.save_final_layer_patches:
        patches_filename = filename + "_final_layer_patches.pt"
        print(f"Saving final layer patches to {os.path.join(save_folder,patches_filename)}")
        torch.save(model.final_layer_patches_list, os.path.join(save_folder,patches_filename))

    if args.save_values:
        values_filename = filename + "_values.pt"
        torch.save(model.values_list, os.path.join(save_folder,values_filename))

    if args.save_key:
        key_filename = filename + "_key.pt"
        torch.save(model.key_list, os.path.join(save_folder,key_filename))

    if args.save_query:
        query_filename = filename + "_query.pt"
        torch.save(model.query_list, os.path.join(save_folder,query_filename))

    if args.save_activation:
        activations_filename = filename + "_activation.pt"
        torch.save(model.activations_list, os.path.join(save_folder,activations_filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--register", type=int, default=0)
    parser.add_argument("--lsun", action="store_true", help="Use LSUN dataset instead of ImageNet.")
    parser.add_argument("--save_attn", action="store_true", help="Save attention maps.")
    parser.add_argument("--save_final-layer-patches", action="store_true", help="Save final layer patches.")
    parser.add_argument("--save-folder", type=str, default="./", help="Folder to save samples.")
    parser.add_argument("--save-values", action="store_true", help="Save values.")
    parser.add_argument("--save-key", action="store_true", help="Save key.")
    parser.add_argument("--save-query", action="store_true", help="Save query.")
    parser.add_argument("--save-activation", action="store_true", help="Save activations.")
    parser.add_argument("--zero-token", default=0, type=int, help="Zero token for the model. Applied if > 0")
    parser.add_argument("--add-filename", default="", type=str, help="Add a string to the filename.")
    args = parser.parse_args()
    main(args)
