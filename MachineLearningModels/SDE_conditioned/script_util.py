import math
from models.SR3_UNet import UNet
from models.FastGAN_UNet import UNet as FG_UNet
from models.SDE_SmallUNet import SmallUNet
from SDE_tools import DiffusionTools

def create_model_diffusion(device, **kwargs):
    model = create_model(
        model_name=kwargs.get("model_name"),
        image_size=kwargs.get("image_size"),
        learn_sigma=kwargs.get("learn_sigma"),
        n_blocks=kwargs.get("n_blocks"),
        n_heads=kwargs.get("n_heads"),
        dim_head=kwargs.get("dim_head"),
        n_channels=kwargs.get("n_channels"),
        attention_resolutions=kwargs.get("attention_resolutions"),
        device=device
    )

    diffusion = create_diffusion(
        noise_steps=kwargs.get("noise_steps"),
        image_size=kwargs.get("image_size"),
        device=device,
        learn_sigma=kwargs.get("learn_sigma"),
        vector_conditioning=kwargs.get("vector_conditioning")
    )

    return model, diffusion

def create_model(
        model_name,
        image_size,
        learn_sigma,
        n_blocks,
        n_heads,
        dim_head,
        n_channels,
        attention_resolutions,
        device
):

    if model_name == "FastGAN":
        return FG_UNet(
                input_channels=6,
                output_channels=(3 if not learn_sigma else 6),
                n_channels=16,
                ch_mults=[2, 2, 2, 2, 2, 2],
                n_blocks=1
            ).to(device)

    # Get appropriate channel multipliers dependend on image size
    elif model_name == "UNet" or model_name == "SmallUNet":
        if image_size == 256:
            channel_mult = [1, 2, 2, 2, 4]
        elif image_size == 128:
            channel_mult = [1, 2, 2, 4]
        elif image_size == 64:
            channel_mult = [1, 2, 2, 4]
        else:
            raise ValueError(f"Unsupported image size: {image_size}")

        attention_ds = []
        # Get places for attention layers
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

        # Initialize list with false values
        is_attn = [False for _ in range(len(channel_mult))]

        # Math log2(res) because res should be converted from resolution to index
        for res in attention_ds:
            is_attn[int(math.log2(res))] = True

        if model_name == "UNet":
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
        elif model_name == "SmallUNet":
            return SmallUNet(
                input_channels=6,
                output_channels=(3 if not learn_sigma else 6),
                n_channels=n_channels,
                ch_mults=channel_mult,
                is_attn=is_attn,
                n_blocks=n_blocks,
                n_heads=n_heads,
                dim_head=dim_head
            ).to(device)

    else:
        raise ValueError("Model Type not supported")

def create_diffusion(
        noise_steps,
        image_size,
        device,
        learn_sigma,
        vector_conditioning
):
    return DiffusionTools(
        noise_steps=noise_steps,
        img_size=image_size,
        vector_conditioning=vector_conditioning,
        learn_sigma=learn_sigma,
        device=device
    )

