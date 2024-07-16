import math
from models.SR3_UNet import UNet
def create_model(
        image_size,
        learn_sigma,
        n_blocks,
        n_heads,
        dim_head,
        n_channels,
        attention_resolutions
):
    if image_size == 256:
        channel_mult = [1, 1, 2, 2, 4, 4]
    elif image_size == 128:
        channel_mult = [1, 1, 2, 3, 4]
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
    )