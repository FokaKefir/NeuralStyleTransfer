import torch
import torch.nn.functional as F
import os
import piqa
import numpy as np

from utils import utils

IMG_HEIGHT = 400

def to_device_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Turn an H×W×C NumPy image into a 1×C×H×W float tensor on the given device.
    Assumes img is already in [0,1] float32.
    """
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return t.to(device)

def multi_crop_features(img_t: torch.Tensor,
                        fid_metric,
                        crops: int = 5,
                        crop_size: tuple[int,int] = (299, 299)
) -> torch.Tensor:
    """
    Take `crops` random crops of size `crop_size` from img_t,
    resize each to `crop_size` if needed, extract InceptionV3 features,
    and return all feature vectors stacked (shape: crops × C).
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

    test_path    = os.path.join("data", "test_images", "test0")
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
        os.path.join(test_path, "style.jpg")
    ]
    style_imgs = []
    for p in style_paths:
        if os.path.exists(p):
            style_imgs.append(utils.load_image(p, (h_out, w_out)))

    # to-device tensors
    output_t  = to_device_tensor(output_img,  device)
    content_t = to_device_tensor(content_img, device)
    style_ts  = [to_device_tensor(img, device) for img in style_imgs]

    # — PSNR & SSIM (output vs content) —
    psnr_val = piqa.PSNR()(output_t, content_t)
    ssim_val = piqa.SSIM().to(device)(output_t, content_t)

    # — LPIPS (output vs content) —
    lpips = piqa.LPIPS().to(device)
    lpips_oc = lpips(output_t, content_t)

    # — prepare FID metric once —
    fid_metric = piqa.fid.FID().to(device)

    lines = [
        f"PSNR (output vs content): {psnr_val.item():.4f}",
        f"SSIM (output vs content): {ssim_val.item():.4f}",
        f"LPIPS (output vs content): {lpips_oc.item():.4f}",
    ]

    # — for each style image: LPIPS & FID —
    for idx, style_t in enumerate(style_ts, start=1):
        lpips_os = lpips(output_t, style_t)
        feats_out   = multi_crop_features(output_t, fid_metric, crops=5)
        feats_style = multi_crop_features(style_t,   fid_metric, crops=5)
        fid_val     = fid_metric(feats_out, feats_style)

        lines.append(f"LPIPS (output vs style{idx}):   {lpips_os.item():.4f}")
        lines.append(f"FID (output vs style{idx}) over 5 crops: {fid_val.item():.4f}")

    # — print & write —
    for line in lines:
        print(line)
    with open(metrics_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nAll metrics written to: {metrics_path}")
