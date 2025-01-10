import torch
import numpy as np
from PIL import Image
from .artcoder import *
from tqdm import tqdm
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(tensor):
    if tensor.ndimension() == 4 and tensor.shape[0] == 1: 
        tensor = tensor.squeeze(0)
    if tensor.ndimension() == 3 and tensor.shape[0] == 3:
        array = (tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    elif tensor.ndimension() == 3 and tensor.shape[2] == 3:
        array = (tensor.cpu().numpy() * 255.0).astype(np.uint8)
    elif tensor.ndimension() == 3 and tensor.shape[0] == 1:
        array = (tensor.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
    elif tensor.ndimension() == 2:
        array = (tensor.cpu().numpy() * 255.0).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")
    return Image.fromarray(array)


class ArtCoder:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content_images": ("IMAGE",), 
                "qr_image": ("IMAGE",), 
                "style_image": ("IMAGE",), 
                "content_scaling": ("FLOAT", {"default": 1.0, "min": 0, "max": 10}),
                "qr_scaling": ("FLOAT", {"default": 1.0, "min": 0, "max": 10}),
                "style_scaling": ("FLOAT", {"default": 1.0, "min": 0, "max": 10}),
                "iteration": ("INT", {"default": 500, "min": 1, "max": 2048}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_QR_images"
    CATEGORY = "Image Processing"

    def generate_QR_images(self, content_images,qr_image,style_image,content_scaling,qr_scaling,style_scaling,iteration,):

        resized_images = []
        for image_tensor in tqdm(content_images):
            qrcode_side_len = 16 * 37
            qrcode_size = (qrcode_side_len, qrcode_side_len)
            content_image = tensor2pil(image_tensor).resize(qrcode_size, Image.LANCZOS)
            qrcode_image = tensor2pil(qr_image).resize(qrcode_size, Image.LANCZOS)
            style_image = tensor2pil(style_image).resize(qrcode_size, Image.LANCZOS)

            content_image = convert_pil_to_normalized_tensor(content_image)
            qrcode_image = convert_pil_to_normalized_tensor(qrcode_image)
            qrcode_image = image_binarize(qrcode_image)
            style_image = convert_pil_to_normalized_tensor(style_image)

            result = optimize_code(
                qrcode_image=qrcode_image,
                content_image=content_image,
                style_image=style_image,
                iterations=iteration,
                code_weight=1e12 * qr_scaling,
                content_weight=1e8 * content_scaling,
                style_weight=1e15 * style_scaling,
            )

            pil_image = tensor2pil(torch.tensor(result))
            resized_images.append(pil2tensor(pil_image))            
        return (torch.cat(resized_images, dim=0),)
