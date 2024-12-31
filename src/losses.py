import torch
from torch import nn

from .feature_extractor import VGGFeatureExtractor
from .filters import QRCodeErrorExtractor, SamplingSimulationLayer, CenterPixelExtractor
from .image_processor import color_to_gray, image_binarize


class CodeLoss(nn.Module):
    def __init__(
        self,
        module_size: int,
        soft_black_value: float,
        soft_white_value: float,
        error_mask_black_thres: float,
        error_mask_white_thres: float,
    ):
        super().__init__()
        self.ss_layer = SamplingSimulationLayer(module_size=module_size)
        self.center_filter = CenterPixelExtractor(module_size=module_size)
        self.module_error_extractor = QRCodeErrorExtractor(
            module_size=module_size,
            error_mask_black_thres=error_mask_black_thres,
            error_mask_white_thres=error_mask_white_thres,
        )
        self.soft_black_value = soft_black_value
        self.soft_white_value = soft_white_value

    @staticmethod
    def _generate_soft_module(
        module_image: torch.Tensor,
        soft_black_value: int,
        soft_white_value: int
    ) -> torch.Tensor:

        assert torch.equal(
            module_image.unique(),
            torch.tensor([0., 1.], device=module_image.device),
        )
        module_image = module_image.clone()
        module_image[module_image == 0] = soft_black_value
        module_image[module_image == 1] = soft_white_value
        return module_image.repeat(1, 3, 1, 1)

    def _compute_error_mask(
        self,
        input_image: torch.Tensor,
        qrcode_image: torch.Tensor
    ) -> torch.Tensor:

        module_image = self.center_filter(image_binarize(qrcode_image))
        return self.module_error_extractor(
            color_to_gray(input_image.clone().detach()),
            module_image,
        )

    def forward(self, input_image: torch.Tensor, qrcode_image: torch.Tensor) -> torch.Tensor:
        soft_target_code = self._generate_soft_module(
            image_binarize(qrcode_image),
            self.soft_black_value,
            self.soft_white_value,
        )
        module_error_mask = self._compute_error_mask(input_image, qrcode_image)
        return nn.functional.mse_loss(
            self.ss_layer(input_image) * module_error_mask,
            self.ss_layer(soft_target_code) * module_error_mask,
        )


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = VGGFeatureExtractor()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(
            self.feature_extractor(x)[-2],
            self.feature_extractor(y)[-2],
        )


class StyleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = VGGFeatureExtractor()

    @staticmethod
    def _compute_gram_matrix(x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return torch.einsum("bchw,bdhw->bcd", x, x) / (c * h * w)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return sum(
            [
                nn.functional.mse_loss(
                    self._compute_gram_matrix(fx),
                    self._compute_gram_matrix(fy)
                )
                for fx, fy in zip(self.feature_extractor(x), self.feature_extractor(y))
            ]
        )


class ArtCoderLoss(nn.Module):
    def __init__(
        self,
        module_size: int = 16,
        soft_black_value: float = 40 / 255,
        soft_white_value: float = 220 / 255,
        error_mask_black_thres: float = 70 / 255,
        error_mask_white_thres: float = 180 / 255,
        code_weight: float = 1e12,
        content_weight: float = 1e8,
        style_weight: float = 1e15,
        pretrained_weights: str = "DEFAULT",
        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.code_weight = code_weight
        self.content_weight = content_weight
        self.style_weight = style_weight

        self.code_loss = CodeLoss(
            module_size=module_size,
            soft_black_value=soft_black_value,
            soft_white_value=soft_white_value,
            error_mask_black_thres=error_mask_black_thres,
            error_mask_white_thres=error_mask_white_thres,
        )
        self.perceptual_loss = PerceptualLoss()
        self.style_loss = StyleLoss()
        self.to(device)

    def forward(
        self,
        input_image: torch.Tensor,
        qrcode_image: torch.Tensor,
        content_image: torch.Tensor,
        style_image: torch.Tensor,
    ) -> torch.Tensor:

        code_loss = self.code_loss(input_image, qrcode_image)
        perceptual_loss = self.perceptual_loss(input_image, content_image)
        style_loss = self.style_loss(input_image, style_image)
        total_loss = (
            self.code_weight * code_loss + \
            self.content_weight * perceptual_loss + \
            self.style_weight * style_loss
        )

        return {
            "code": code_loss,
            "perceptual": perceptual_loss,
            "style": style_loss,
            "total": total_loss
        }
