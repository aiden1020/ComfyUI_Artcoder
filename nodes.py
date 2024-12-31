import torch
import numpy as np
from PIL import Image
from .artcoder import *
def pil2tensor(image):
    """Converts a PIL image to a PyTorch tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(tensor):
    """Converts a PyTorch tensor to a PIL image."""
    array = (tensor.squeeze(0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array)

class ArtCoder:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "content_images": ("IMAGE",), 
                "qr_image": ("IMAGE",), 
                "content_scaling": ("FLOAT", {"default": 1.0, "min": 0, "max": 10}),
                "qr_scaling": ("FLOAT", {"default": 1.0, "min": 0, "max": 10}),
                "resize_width": ("INT", {"default": 512, "min": 1, "max": 2048}),
                "resize_height": ("INT", {"default": 512, "min": 1, "max": 2048}),
                "iteration": ("INT", {"default": 500, "min": 1, "max": 2048}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_QR_images"
    CATEGORY = "Image Processing"

    def generate_QR_images(self, content_images,qr_image,content_scaling,qr_scaling,resize_width,resize_height,iteration,):

        resized_images = []
        for image_tensor in content_images:
            style_image_path = "custom_nodes/ComfUI_Artcoder/images/style.jpg"
            output_path = "custom_nodes/ComfUI_Artcoder/results/image.jpg"

            qrcode_side_len = 16 * 37
            qrcode_size = (qrcode_side_len, qrcode_side_len)

            content_image = tensor2pil(image_tensor).resize(qrcode_size, Image.LANCZOS)
            qrcode_image = tensor2pil(qr_image).resize(qrcode_size, Image.LANCZOS)
            style_image = Image.open(style_image_path).resize(qrcode_size, Image.LANCZOS)

            content_image = convert_pil_to_normalized_tensor(content_image)
            qrcode_image = convert_pil_to_normalized_tensor(qrcode_image)
            qrcode_image = image_binarize(qrcode_image)
            style_image = convert_pil_to_normalized_tensor(style_image)

            result = generate_aesthetic_qrcode(
                qrcode_image=qrcode_image,
                content_image=content_image,
                style_image=style_image,
                module_size=16,
                module_num=37,
                iterations=iteration,
                soft_black_value=40 / 255,
                soft_white_value=220 / 255,
                error_mask_black_thres=70 / 255,
                error_mask_white_thres=180 / 255,
                lr=0.01,
                code_weight=4e9 * qr_scaling,
                content_weight=1e12 * content_scaling,
                style_weight=0,
            )
            pil_image = tensor2pil(result)
            resized_pil_image = pil_image.resize((resize_width, resize_height), Image.LANCZOS)
            # Convert back to tensor
            resized_images.append(pil2tensor(resized_pil_image))
            
        # Stack resized images into a single tensor batch
        return (torch.cat(resized_images, dim=0),)
