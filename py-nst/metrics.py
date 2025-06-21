import torch
import torch.nn.functional as F
import os
import piqa
import numpy as np
from torchvision import models

from utils import utils

IMG_HEIGHT = 400

test_path = os.path.join("data", "test_images", "test1")


def to_device_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Turn an HxWxC NumPy image into a 1xCxHxW float tensor on the given device.
    Assumes img is already in [0,1] float32.
    """
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return t.to(device)

def gram(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute Gram matrix for a feature map: (1,C,H,W) → (C,C).
    """
    b, c, h, w = matrix.size()
    features = matrix.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2)) / (c * h * w)
    return G

def extract_grams(img_t: torch.Tensor, vgg, layers: list[int]) -> list[torch.Tensor]:
    """
    Pass img_t through VGG up to the highest index in layers,
    collect Gram matrices at each specified layer.
    """
    grams = []
    x = img_t
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in layers:
            grams.append(gram(x))
    return grams

def multi_crop_features(img_t: torch.Tensor,
                        fid_metric,
                        crops: int = 50,
                        crop_size: tuple[int,int] = (299, 299)
) -> torch.Tensor:
    """
    Take `crops` random crops of size `crop_size` from img_t,
    resize each to `crop_size` if needed, extract InceptionV3 features,
    and return all feature vectors stacked (shape: crops x C).
    """
    _, _, H, W = img_t.shape
    ph, pw = crop_size
    feats = []
    for _ in range(crops):
        top  = torch.randint(0, H - ph + 1, (), device=img_t.device).item()
        left = torch.randint(0, W - pw + 1, (), device=img_t.device).item()
        patch = img_t[:, :, top:top+ph, left:left+pw]
        if patch.shape[-2:] != crop_size:
            patch = F.interpolate(patch, size=crop_size,
                                  mode='bilinear', align_corners=False)
        feats.append(fid_metric.features(patch))
    return torch.cat(feats, dim=0)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    metrics_path = os.path.join(test_path, "metrics.txt")

    # — load output and content —
    output_img = utils.load_image(
        os.path.join(test_path, "output.png"), IMG_HEIGHT
    )
    h_out, w_out = output_img.shape[:2]
    content_img = utils.load_image(
        os.path.join(test_path, "content.jpg"), (h_out, w_out)
    )

    # — load style images (1 or 2) —
    style_paths = [
        os.path.join(test_path, "style.jpg"),
        os.path.join(test_path, "style2.jpg")
    ]
    style_imgs = []
    for p in style_paths:
        if os.path.exists(p):
            style_imgs.append(utils.load_image(p, (h_out, w_out)))

    # to-device tensors
    output_t  = to_device_tensor(output_img,  device)
    content_t = to_device_tensor(content_img, device)
    style_ts  = [to_device_tensor(img, device) for img in style_imgs]

    # — SSIM (output vs content) —
    ssim_val = piqa.SSIM().to(device)(output_t, content_t)

    # — LPIPS (output vs content) —
    lpips = piqa.LPIPS().to(device)
    lpips_oc = lpips(output_t, content_t)

    # — prepare FID metric —
    fid_metric = piqa.fid.FID().to(device)

    # — prepare VGG for Gram extraction —
    vgg_full = models.vgg19(pretrained=True).features.to(device).eval()
    gram_layers = [1, 6, 11, 20]  # relu1_1, relu2_1, relu3_1, relu4_1

    lines = [
        f"SSIM (output vs content): {ssim_val.item():.4f}",
        f"LPIPS (output vs content): {lpips_oc.item():.4f}",
    ]

    # — for each style image: LPIPS, FID with 50 crops, and Gram distance —
    for idx, style_t in enumerate(style_ts, start=1):
        # LPIPS
        lpips_os = lpips(output_t, style_t)

        # FID
        feats_out   = multi_crop_features(output_t, fid_metric, crops=50)
        feats_style = multi_crop_features(style_t,   fid_metric, crops=50)
        fid_val     = fid_metric(feats_out, feats_style)

        # Gram distance
        with torch.no_grad():
            grams_out   = extract_grams(output_t, vgg_full, gram_layers)
            grams_style = extract_grams(style_t,   vgg_full, gram_layers)
            gram_dist = sum(torch.norm(go - gs) for go, gs in zip(grams_out, grams_style))

        lines.append(f"LPIPS (output vs style{idx}): {lpips_os.item():.4f}")
        lines.append(f"FID (output vs style{idx}) over 50 crops: {fid_val.item():.4f}")
        lines.append(f"Gram distance (output vs style{idx}): {gram_dist.item():.4f}")

    # — print & write —
    for line in lines:
        print(line)
    with open(metrics_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nAll metrics written to: {metrics_path}")
