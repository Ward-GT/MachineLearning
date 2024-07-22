import math
from models.SR3_UNet import UNet
from SDE_tools import GaussianDiffusion

def create_model_diffusion(**kwargs):
    model = create_model(
        image_size=kwargs.get("image_size"),
        learn_sigma=kwargs.get("learn_sigma"),
        n_blocks=kwargs.get("n_blocks"),
        n_heads=kwargs.get("n_heads"),
        dim_head=kwargs.get("dim_head"),
        n_channels=kwargs.get("n_channels"),
        attention_resolutions=kwargs.get("attention_resolutions"),
        device=kwargs.get("device")
    )

    diffusion = create_diffusion(
        noise_steps=kwargs.get("noise_steps"),
        image_size=kwargs.get("image_size"),
        device=kwargs.get("device"),
        learn_sigma=kwargs.get("learn_sigma")
    )

    return model, diffusion

def create_model(
        image_size,
        learn_sigma,
        n_blocks,
        n_heads,
        dim_head,
        n_channels,
        attention_resolutions,
        device
):
    if image_size == 256:
        channel_mult = [1, 1, 2, 2, 4, 4]
    elif image_size == 128:
        channel_mult = [1, 1, 2, 3, 4]
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"Unsupported image size: {image_size}")

    attention_ds = []

    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    is_attn = [False for _ in range(len(channel_mult))]

    for res in attention_ds:
        is_attn[int(math.log2(res))] = True

    return UNet(
        input_channels=6,
        output_channels=(3 if not learn_sigma else 6),
        n_channels=n_channels,
        ch_mults=channel_mult,
        is_attn=is_attn,
        n_blocks=n_blocks,
        n_heads=n_heads,
        dim_head=dim_head
    ).to(device)

def create_diffusion(
        noise_steps,
        image_size,
        device,
        learn_sigma
):
    return GaussianDiffusion(
        noise_steps=noise_steps,
        image_size=image_size,
        device=device,
        learn_sigma=learn_sigma
    )

