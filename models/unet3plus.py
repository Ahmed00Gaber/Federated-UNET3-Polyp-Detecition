# =========================
# 📦 IMPORTS
# =========================
# torch: main PyTorch library for tensor computation and automatic differentiation
import torch

# torch.nn: contains building blocks for neural networks (layers, loss functions, etc.)
import torch.nn as nn

# torch.nn.functional: provides functions like interpolation (upsampling/downsampling)
import torch.nn.functional as F

# segmentation_models_pytorch (smp): library with pre‑trained encoders (EfficientNet, ResNet, etc.)
# We use smp.encoders.get_encoder to easily load a backbone with ImageNet weights
import segmentation_models_pytorch as smp


# =========================
# 🧱 CONV + BN + ReLU BLOCK
# =========================
class ConvBNReLU(nn.Module):
    """A simple conv block used for projection and fusion.
    Applies: Conv2d → BatchNorm2d → ReLU.
    Used to project feature maps to a fixed number of channels (cat_channels).
    """
    #Constructor
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            # bias=False because BatchNorm adds its own bias term
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),  # inplace=False to avoid potential issues with gradient hooks
        )

    def forward(self, x):
        return self.block(x)


# =========================
# 🏗️ UNET3+ MAIN MODEL
# =========================
class UNet3Plus(nn.Module):
    """
    UNet3+ implementation using a pre‑trained encoder (e.g., EfficientNet‑B0).
    
    Key features:
    - Full‑scale skip connections (every encoder level connects to every decoder level)
    - Deep supervision: outputs from all four decoder stages (d1 to d4)
    - Uses projection layers to bring all features to a common channel count (cat_channels)
    - Fusion layers combine multi‑scale features before producing segmentation maps
    """

    def __init__(
        self,
        encoder_name="efficientnet-b0",   # backbone name (supports many: resnet34, vgg16, etc.)
        encoder_weights="imagenet",       # pre‑trained weights on ImageNet
        in_channels=3,                    # RGB input
        num_classes=1,                    # binary segmentation (foreground/background)
        cat_channels=64,                  # common channel number after projection
        decoder_channels=256,             # channels inside decoder fusion layers
    ):
        super().__init__()

        self.num_classes = num_classes
        self.cat_channels = cat_channels
        self.decoder_channels = decoder_channels

        # =========================
        # 🔍 ENCODER (BACKBONE)
        # =========================
        # Load a pre‑trained encoder with 5 downsampling stages (depth=5)
        # Outputs a list of feature maps at different resolutions: [e1, e2, e3, e4, e5]
        # e1: highest resolution (shallow), e5: lowest resolution (deepest)
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )

        # Get the number of channels for each encoder stage (e.g., EfficientNet‑B0: [16, 24, 40, 112, 320])
        encoder_channels = self.encoder.out_channels[-5:]   # last 5 stages

        # =========================
        # 🔄 PROJECTION LAYERS
        # =========================
        # Project each encoder stage to cat_channels (64) so they can be concatenated
        # enc_proj[0] -> for e1, enc_proj[1] -> for e2, ..., enc_proj[4] -> for e5
        self.enc_proj = nn.ModuleList([
            ConvBNReLU(encoder_channels[0], cat_channels),
            ConvBNReLU(encoder_channels[1], cat_channels),
            ConvBNReLU(encoder_channels[2], cat_channels),
            ConvBNReLU(encoder_channels[3], cat_channels),
            ConvBNReLU(encoder_channels[4], cat_channels),
        ])

        # Projections for decoder outputs (used when upsampling to higher resolutions)
        self.dec_proj4 = ConvBNReLU(decoder_channels, cat_channels)
        self.dec_proj3 = ConvBNReLU(decoder_channels, cat_channels)
        self.dec_proj2 = ConvBNReLU(decoder_channels, cat_channels)

        # =========================
        # 🧩 FUSION LAYERS
        # =========================
        # Each fusion layer combines multiple projected features (from different scales)
        # The number of input channels = cat_channels * (number of concatenated tensors)
        # Then reduces to decoder_channels (256) using ConvBNReLU
        self.fuse4 = ConvBNReLU(cat_channels * 5, decoder_channels)   # d4 gets 5 inputs
        self.fuse3 = ConvBNReLU(cat_channels * 6, decoder_channels)   # d3 gets 6 inputs
        self.fuse2 = ConvBNReLU(cat_channels * 7, decoder_channels)   # d2 gets 7 inputs
        self.fuse1 = ConvBNReLU(cat_channels * 8, decoder_channels)   # d1 gets 8 inputs

        # =========================
        # 🎯 OUTPUT LAYERS (Deep Supervision)
        # =========================
        # Each decoder stage produces a segmentation map (1x1 conv to num_classes)
        self.out4 = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
        self.out3 = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
        self.out2 = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
        self.out1 = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

    # =========================
    # 🛠️ HELPER METHODS
    # =========================
    @staticmethod
    def _resize_to(x, ref):
        """Resize tensor x to the spatial size of reference tensor ref.
        Uses bilinear interpolation (works for both up‑ and down‑sampling).
        """
        return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)

    def _project_and_resize(self, feat, proj, ref):
        """Apply a projection layer (ConvBNReLU) to feat, then resize to match ref's spatial size."""
        feat = proj(feat)
        feat = self._resize_to(feat, ref)
        return feat

    # =========================
    # 🚀 FORWARD PASS
    # =========================
    def forward(self, x):
        """
        x: input image tensor of shape (B, 3, H, W)
        Returns: (out1, out2, out3, out4)
            out1: segmentation at full resolution (largest)
            out2: segmentation at 1/2 resolution
            out3: segmentation at 1/4 resolution
            out4: segmentation at 1/8 resolution
        """
        # ----- Encode: get 5 feature maps from the backbone -----
        feats = self.encoder(x)          # feats is a list of tensors from all stages
        e1, e2, e3, e4, e5 = feats[-5:]  # take the last 5 (highest to lowest resolution)
        # e1: (B, C1, H, W)       - full size
        # e2: (B, C2, H/2, W/2)   - half size
        # e3: (B, C3, H/4, W/4)   - quarter size
        # e4: (B, C4, H/8, W/8)   - 1/8 size
        # e5: (B, C5, H/16, W/16) - 1/16 size

        # =========================
        # 🌉 DECODER STAGE 4 (d4) – lowest resolution
        # =========================
        # d4 receives from: e1, e2, e3, e4, e5 (all projected to cat_channels and resized to e4's spatial size)
        d4_in = torch.cat([
            self._project_and_resize(e1, self.enc_proj[0], e4),   # e1 → resized down to e4
            self._project_and_resize(e2, self.enc_proj[1], e4),
            self._project_and_resize(e3, self.enc_proj[2], e4),
            self.enc_proj[3](e4),                                 # e4 stays at same resolution
            self._project_and_resize(e5, self.enc_proj[4], e4),   # e5 → resized up to e4
        ], dim=1)   # concatenate along channel dimension -> (B, 5*cat_channels, H/8, W/8)
        d4 = self.fuse4(d4_in)          # fuse to decoder_channels (256)

        # =========================
        # 🌉 DECODER STAGE 3 (d3)
        # =========================
        # d4 is upsampled and projected to cat_channels, then added to the mix
        d4_up = self._project_and_resize(d4, self.dec_proj4, e3)   # d4 → resized up to e3 size
        d3_in = torch.cat([
            self._project_and_resize(e1, self.enc_proj[0], e3),
            self._project_and_resize(e2, self.enc_proj[1], e3),
            self.enc_proj[2](e3),
            self._project_and_resize(e4, self.enc_proj[3], e3),
            self._project_and_resize(e5, self.enc_proj[4], e3),
            d4_up,                         # now includes information from d4
        ], dim=1)   # (B, 6*cat_channels, H/4, W/4)
        d3 = self.fuse3(d3_in)

        # =========================
        # 🌉 DECODER STAGE 2 (d2)
        # =========================
        d4_up2 = self._project_and_resize(d4, self.dec_proj4, e2)
        d3_up2 = self._project_and_resize(d3, self.dec_proj3, e2)
        d2_in = torch.cat([
            self._project_and_resize(e1, self.enc_proj[0], e2),
            self.enc_proj[1](e2),
            self._project_and_resize(e3, self.enc_proj[2], e2),
            self._project_and_resize(e4, self.enc_proj[3], e2),
            self._project_and_resize(e5, self.enc_proj[4], e2),
            d4_up2,
            d3_up2,
        ], dim=1)   # (B, 7*cat_channels, H/2, W/2)
        d2 = self.fuse2(d2_in)

        # =========================
        # 🌉 DECODER STAGE 1 (d1) – highest resolution (full size)
        # =========================
        d4_up1 = self._project_and_resize(d4, self.dec_proj4, e1)
        d3_up1 = self._project_and_resize(d3, self.dec_proj3, e1)
        d2_up1 = self._project_and_resize(d2, self.dec_proj2, e1)
        d1_in = torch.cat([
            self.enc_proj[0](e1),
            self._project_and_resize(e2, self.enc_proj[1], e1),
            self._project_and_resize(e3, self.enc_proj[2], e1),
            self._project_and_resize(e4, self.enc_proj[3], e1),
            self._project_and_resize(e5, self.enc_proj[4], e1),
            d4_up1,
            d3_up1,
            d2_up1,
        ], dim=1)   # (B, 8*cat_channels, H, W)
        d1 = self.fuse1(d1_in)

        # =========================
        # 🎯 DEEP SUPERVISION OUTPUTS
        # =========================
        # Each output is a segmentation map (logits) at different resolutions
        out1 = self.out1(d1)   # full resolution (H, W)
        out2 = self.out2(d2)   # 1/2 resolution (H/2, W/2)
        out3 = self.out3(d3)   # 1/4 resolution (H/4, W/4)
        out4 = self.out4(d4)   # 1/8 resolution (H/8, W/8)

        return out1, out2, out3, out4