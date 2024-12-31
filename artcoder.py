import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

from .src.losses import ArtCoderLoss
from .src.image_processor import image_binarize
from .src.utils import (
    add_position_pattern,
    convert_normalized_tensor_to_np_image,
    convert_pil_to_normalized_tensor,
)
def optimize_code(
    content_image: torch.Tensor,
    qrcode_image: torch.Tensor,
    style_image: torch.Tensor,
    module_size: int = 16,
    module_num: int = 37,
    iterations: int = 50000,
    soft_black_value: float = 40 / 255,
    soft_white_value: float = 220 / 255,
    error_mask_black_thres: float = 70 / 255,
    error_mask_white_thres: float = 180 / 255,
    lr: float = 0.01,
    code_weight: float = 1e12,
    content_weight: float = 1e8,
    style_weight: float = 1e15,
    display_loss: bool = True,
) -> np.ndarray:
    with torch.inference_mode(False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        content_image = content_image.to(device)
        qrcode_image = qrcode_image.to(device)
        style_image = style_image.to(device)

        x = content_image.clone().detach().to(device).requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=lr)
        objective_func = ArtCoderLoss(
            module_size=module_size,
            soft_black_value=soft_black_value,
            soft_white_value=soft_white_value,
            error_mask_black_thres=error_mask_black_thres,
            error_mask_white_thres=error_mask_white_thres,
            code_weight=code_weight,
            content_weight=content_weight,
            style_weight=style_weight,
            device=device,
        )

        for i in tqdm(range(iterations)):
            optimizer.zero_grad()
            losses = objective_func(x, qrcode_image, content_image, style_image)
            losses["total"].backward()  # 去掉 retain_graph
            optimizer.step()  # 執行優化
            x.data.clamp_(0, 1)  # 限制張量值範圍在 [0, 1]

            if display_loss:
                tqdm.write(
                    f"iterations: {i}, " +
                    ", ".join([f"{k}_loss: {v:.4f}" for k, v in losses.items()])
                )

    return add_position_pattern(
        convert_normalized_tensor_to_np_image(x),
        convert_normalized_tensor_to_np_image(qrcode_image),
        module_size=module_size,
        module_num=module_num,
    )


def generate_aesthetic_qrcode(
    qrcode_image: torch.Tensor,
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    module_size: int = 16,
    module_num: int = 37,
    iterations: int = 1000,
    soft_black_value: float = 40 / 255,
    soft_white_value: float = 220 / 255,
    error_mask_black_thres: float = 70 / 255,
    error_mask_white_thres: float = 180 / 255,
    lr: float = 0.01,
    code_weight: float = 1e12,
    content_weight: float = 1e8,
    style_weight: float = 1e15,
) -> Image.Image:
    aesthetic_qrcode = optimize_code(
        content_image,
        qrcode_image,
        style_image,
        module_size=module_size,
        module_num=module_num,
        iterations=iterations,
        soft_black_value=soft_black_value,
        soft_white_value=soft_white_value,
        error_mask_black_thres=error_mask_black_thres,
        error_mask_white_thres=error_mask_white_thres,
        lr=lr,
        code_weight=code_weight,
        content_weight=content_weight,
        style_weight=style_weight,
    )

    return torch.tensor(aesthetic_qrcode)

