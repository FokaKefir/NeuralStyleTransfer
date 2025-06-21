import torch
import numpy as np
import cv2 as cv

from torchvision import transforms
from torchvision.models.segmentation import (
    deeplabv3_resnet101,
    DeepLabV3_ResNet101_Weights
)

from utils import utils

PERSON_CHANNEL_INDEX = 15  # segmentation stage

IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ERROR_CODE = 1


def post_process_mask(mask):
    kernel = np.ones((13, 13), np.uint8)
    opened_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(opened_mask)
    if num_labels > 1:
        h, _ = labels.shape
        sub = labels[:int(h/10), :]
        bkg = np.argmax(np.bincount(sub.flatten()))
        areas = [(i, stats[i, cv.CC_STAT_AREA]) for i in range(num_labels)]
        areas = [a for a in sorted(areas, key=lambda x: x[1], reverse=True) if a[0] != bkg]
        person_idx = areas[0][0]
        return np.uint8((labels == person_idx) * 255)
    else:
        return opened_mask


def extract_person_mask_from_image(image_path: str, segmentation_mask_height: int = 400, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> np.ndarray:
    model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT).to(device).eval()
    
    img_bgr = utils.load_image(image_path)           # BGR uint8
    h, w = img_bgr.shape[:2]
    new_w = int(w * (segmentation_mask_height / h))
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((segmentation_mask_height, new_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1)
    ])
    img_tensor = tf(img_bgr).unsqueeze(0).to(device)  # (1,3,H,W)
    
    try:
        with torch.no_grad():
            out = model(img_tensor)['out'][0].cpu().numpy()  # (21, H, W)
    except RuntimeError as e:
        print(f"Error at segmentation: {e}")
        exit(ERROR_CODE)
    
    mask_raw = np.uint8((np.argmax(out, axis=0) == PERSON_CHANNEL_INDEX) * 255)
    mask_proc = post_process_mask(mask_raw)
    return mask_proc
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    res = extract_person_mask_from_image(
        image_path='data/content-images/downey.jpg',
        segmentation_mask_height=600,
        device=device
    )

    # save it
    out_path = 'data/output-images/seg/downey_mask.jpg'
    cv.imwrite(out_path, res)
    