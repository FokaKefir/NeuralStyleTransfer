import utils.utils as utils
from utils.video_utils import create_video_from_intermediate_results

from segmentation import extract_person_mask_from_image

import torch
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import argparse
import cv2 as cv
from tqdm import tqdm

TILE_SIZE = 500
TILE_OVERLAP = 100  # Overlap for feathering

def get_original_image_dimensions(img_path):
    """Get the height and width of the original image."""
    img = cv.imread(img_path)
    if img is None:
        raise Exception(f'Could not read image: {img_path}')
    return img.shape[:2]  # returns (height, width)


def upscale_tensor(tensor, target_height, target_width):
    """Upscale a tensor to target dimensions using bicubic interpolation."""
    return F.interpolate(tensor, size=(target_height, target_width), mode='bicubic', align_corners=False)


def create_feather_mask(height, width, overlap):
    """Create a feathering mask for smooth tile blending."""
    mask = torch.ones(1, 1, height, width)
    
    # Cap overlap to tile dimensions to avoid index errors on edge tiles
    effective_overlap_h = min(overlap, height)
    effective_overlap_w = min(overlap, width)
    
    # Create linear gradients for edges
    if overlap > 0:
        # Top edge
        for i in range(effective_overlap_h):
            mask[:, :, i, :] *= (i + 1) / effective_overlap_h
        # Bottom edge
        for i in range(effective_overlap_h):
            mask[:, :, height - 1 - i, :] *= (i + 1) / effective_overlap_h
        # Left edge
        for i in range(effective_overlap_w):
            mask[:, :, :, i] *= (i + 1) / effective_overlap_w
        # Right edge
        for i in range(effective_overlap_w):
            mask[:, :, :, width - 1 - i] *= (i + 1) / effective_overlap_w
    
    return mask


def get_tile_positions(img_height, img_width, tile_size=TILE_SIZE, overlap=TILE_OVERLAP):
    """Calculate tile positions for covering the entire image with overlap."""
    positions = []
    
    # Calculate stride (tile_size - overlap for overlapping tiles)
    stride = tile_size - overlap
    
    for y in range(0, img_height, stride):
        for x in range(0, img_width, stride):
            # Adjust last tiles to fit exactly
            y_end = min(y + tile_size, img_height)
            x_end = min(x + tile_size, img_width)
            y_start = y_end - tile_size if y_end == img_height and y_end - y > overlap else y
            x_start = x_end - tile_size if x_end == img_width and x_end - x > overlap else x
            
            # Ensure we don't go negative
            y_start = max(0, y_start)
            x_start = max(0, x_start)
            
            positions.append((y_start, y_end, x_start, x_end))
    
    return positions


def process_tiles(content_img, style_img, init_img, neural_net, content_feature_maps_index, 
                  style_feature_maps_indices, config, device, target_style_representation):
    """Process image in tiles and stitch back together."""
    _, _, img_height, img_width = init_img.shape
    tile_positions = get_tile_positions(img_height, img_width)
    
    print(f"Processing {len(tile_positions)} tiles of size {TILE_SIZE}x{TILE_SIZE}...")
    
    # Create accumulator for stitched result
    result = torch.zeros_like(init_img)
    weight_map = torch.zeros(1, 1, img_height, img_width).to(device)
    
    for idx, (y_start, y_end, x_start, x_end) in enumerate(tile_positions):
        tile_h = y_end - y_start
        tile_w = x_end - x_start
        
        print(f"  Tile {idx + 1}/{len(tile_positions)}: [{y_start}:{y_end}, {x_start}:{x_end}]")
        
        # Extract tiles
        content_tile = content_img[:, :, y_start:y_end, x_start:x_end]
        init_tile = init_img[:, :, y_start:y_end, x_start:x_end]
        
        # Use initialization from upscaled low-res result
        optimizing_tile = Variable(init_tile.clone(), requires_grad=True)
        
        # Get target representations for this tile
        content_tile_features = neural_net(content_tile)
        target_content_tile = content_tile_features[content_feature_maps_index].squeeze(axis=0)
        
        # Style representation stays the same (global style)
        target_representations = [target_content_tile, target_style_representation]
        
        # Optimize this tile
        tile_iterations = config['iterations']
        optimizer_tile = Adam((optimizing_tile,), lr=1e1)
        tuning_step_tile = make_tuning_step(neural_net, optimizer_tile, target_representations, 
                                            content_feature_maps_index, style_feature_maps_indices, config)
        
        for cnt in tqdm(range(tile_iterations), desc=f"Tile {idx+1}/{len(tile_positions)}", leave=False):
            total_loss, content_loss, style_loss, tv_loss = tuning_step_tile(optimizing_tile)
        
        # Create feather mask for this tile
        feather_mask = create_feather_mask(tile_h, tile_w, TILE_OVERLAP).to(device)
        
        # Expand mask to 3 channels
        feather_mask_3ch = feather_mask.expand(-1, 3, -1, -1)
        
        # Add to result with feathering
        with torch.no_grad():
            result[:, :, y_start:y_end, x_start:x_end] += optimizing_tile.data * feather_mask_3ch
            weight_map[:, :, y_start:y_end, x_start:x_end] += feather_mask
    
    # Normalize by weights
    weight_map_3ch = weight_map.expand(-1, 3, -1, -1)
    result = result / (weight_map_3ch + 1e-8)
    
    return result


def process_tiles_mixed(content_img, style_img_1, style_img_2, init_img, neural_net, content_feature_maps_index, 
                        style_feature_maps_indices, config, device, target_style_representation_1, target_style_representation_2):
    """Process image in tiles for mixed style transfer and stitch back together."""
    _, _, img_height, img_width = init_img.shape
    tile_positions = get_tile_positions(img_height, img_width)
    
    print(f"Processing {len(tile_positions)} tiles of size {TILE_SIZE}x{TILE_SIZE} for mixed style...")
    
    # Create accumulator for stitched result
    result = torch.zeros_like(init_img)
    weight_map = torch.zeros(1, 1, img_height, img_width).to(device)
    
    for idx, (y_start, y_end, x_start, x_end) in enumerate(tile_positions):
        tile_h = y_end - y_start
        tile_w = x_end - x_start
        
        print(f"  Tile {idx + 1}/{len(tile_positions)}: [{y_start}:{y_end}, {x_start}:{x_end}]")
        
        # Extract tiles
        content_tile = content_img[:, :, y_start:y_end, x_start:x_end]
        init_tile = init_img[:, :, y_start:y_end, x_start:x_end]
        
        # Use initialization from upscaled low-res result
        optimizing_tile = Variable(init_tile.clone(), requires_grad=True)
        
        # Get target representations for this tile
        content_tile_features = neural_net(content_tile)
        target_content_tile = content_tile_features[content_feature_maps_index].squeeze(axis=0)
        
        # Style representations stay the same (global styles)
        target_representations = [target_content_tile, target_style_representation_1, target_style_representation_2]
        
        # Optimize this tile
        tile_iterations = config['iterations']
        optimizer_tile = Adam((optimizing_tile,), lr=1e1)
        tuning_step_tile = make_tuning_step(neural_net, optimizer_tile, target_representations, 
                                            content_feature_maps_index, style_feature_maps_indices, config)
        
        for cnt in tqdm(range(tile_iterations), desc=f"Tile {idx+1}/{len(tile_positions)}", leave=False):
            total_loss, content_loss, style_loss, tv_loss = tuning_step_tile(optimizing_tile)
        
        # Create feather mask for this tile
        feather_mask = create_feather_mask(tile_h, tile_w, TILE_OVERLAP).to(device)
        
        # Expand mask to 3 channels
        feather_mask_3ch = feather_mask.expand(-1, 3, -1, -1)
        
        # Add to result with feathering
        with torch.no_grad():
            result[:, :, y_start:y_end, x_start:x_end] += optimizing_tile.data * feather_mask_3ch
            weight_map[:, :, y_start:y_end, x_start:x_end] += feather_mask
    
    # Normalize by weights
    weight_map_3ch = weight_map.expand(-1, 3, -1, -1)
    result = result / (weight_map_3ch + 1e-8)
    
    return result


def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config, str_flag=""):
    if len(target_representations) == 2:
        target_content_representation = target_representations[0]
        target_style_representation = target_representations[1]

        current_set_of_feature_maps = neural_net(optimizing_img)

        current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
        content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

        style_loss = 0.0
        current_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
        for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
            style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
        style_loss /= len(target_style_representation)

        tv_loss = utils.total_variation(optimizing_img)

        if str_flag == "":
            total_loss = config['content_weight'] * content_loss + config[f'style_weight'] * style_loss + config['tv_weight'] * tv_loss
        else:
            total_loss = config['content_weight'] * content_loss + config[f'style_{str_flag}_weight'] * style_loss + config['tv_weight'] * tv_loss
    
    elif len(target_representations) == 3:
        target_content_representation = target_representations[0]
        target_style_representation_1 = target_representations[1]
        target_style_representation_2 = target_representations[2]

        current_set_of_feature_maps = neural_net(optimizing_img)

        current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
        content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

        style_loss = 0.0
        style_loss_1 = 0.0
        style_loss_2 = 0.0
        current_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
        for gram_gt, gram_hat in zip(target_style_representation_1, current_style_representation):
            style_loss_1 += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
        style_loss_1 /= len(target_style_representation_1)

        for gram_gt, gram_hat in zip(target_style_representation_2, current_style_representation):
            style_loss_2 += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
        style_loss_2 /= len(target_style_representation_2)

        style_loss = config['alpha'] * style_loss_2 + (1 - config['alpha']) * style_loss_1

        tv_loss = utils.total_variation(optimizing_img)

        if str_flag == "":
            total_loss = config['content_weight'] * content_loss + config[f'style_weight'] * style_loss + config['tv_weight'] * tv_loss
        else:
            total_loss = config['content_weight'] * content_loss + config[f'style_{str_flag}_weight'] * style_loss + config['tv_weight'] * tv_loss

    return total_loss, content_loss, style_loss, tv_loss


def make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index, style_feature_maps_indices, config, str_flag=""):
    # Builds function that performs a step in the tuning loop
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config, str_flag=str_flag)
        # Computes gradients
        total_loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss, tv_loss

    # Returns the function that will be called inside the tuning loop
    return tuning_step


def neural_style_transfer(config):
    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])

    dump_path = config['output_img_dir']
    os.makedirs(dump_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use original size or specified height
    current_height = config['height']
    
    # Prepare images at full resolution
    content_img = utils.prepare_img(content_img_path, current_height, device)
    style_img = utils.prepare_img(style_img_path, current_height, device)

    if config['init_method'] == 'random':
        # white_noise_img = np.random.uniform(-90., 90., content_img.shape).astype(np.float32)
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img).float().to(device)
    elif config['init_method'] == 'content':
        init_img = content_img
    else:
        # init image has same dimension as content image - this is a hard constraint
        # feature maps need to be of same size for content image and init image
        style_img_resized = utils.prepare_img(style_img_path, np.asarray(content_img.shape[2:]), device)
        init_img = style_img_resized

    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config['model'], device)
    print(f'Using {config["model"]} in the optimization procedure.')

    style_img_set_of_feature_maps = neural_net(style_img)
    target_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    
    # Check if we need tiling based on actual image size
    _, _, img_height, img_width = content_img.shape
    use_tiling = img_height > TILE_SIZE or img_width > TILE_SIZE
    
    if use_tiling:
        print(f'Image size {img_height}x{img_width} requires tiling. Processing large image in {TILE_SIZE}x{TILE_SIZE} tiles...')
        
        # Process large image in tiles
        optimizing_img_data = process_tiles(
            content_img, 
            style_img, 
            init_img,
            neural_net,
            content_feature_maps_index_name[0],
            style_feature_maps_indices_names[0],
            config,
            device,
            target_style_representation
        )
        
        optimizing_img = Variable(optimizing_img_data, requires_grad=False)
    else:
        print(f'Image size {img_height}x{img_width} is small enough, running standard optimization...')
        
        # Standard optimization without tiling
        optimizing_img = Variable(init_img, requires_grad=True)
        
        content_img_set_of_feature_maps = neural_net(content_img)
        target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
        target_representations = [target_content_representation, target_style_representation]

        num_of_iterations = config['iterations']
        optimizer = Adam((optimizing_img,), lr=1e1)
        tuning_step = make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
        
        for cnt in range(num_of_iterations):
            total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
            with torch.no_grad():
                print(f'Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')

    # Save final result
    with torch.no_grad():
        img_name = utils.save_and_maybe_display(optimizing_img, dump_path, config, config['iterations']-1, config['iterations'], should_display=False)

    return img_name

def neural_style_transfer_with_segmentation(config):
    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])

    if config['style_person_img_name'] is not None:
        style_person_img_path = os.path.join(config['style_images_dir'], config['style_person_img_name'])
    if config['style_background_img_name'] is not None:
        style_background_img_path = os.path.join(config['style_images_dir'], config['style_background_img_name'])

    dump_path = config['output_img_dir']
    os.makedirs(dump_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use original size or specified height
    current_height = config['height']
    
    # Prepare images at full resolution
    content_img = utils.prepare_img(content_img_path, current_height, device)
    style_person_img = utils.prepare_img(style_person_img_path, current_height, device) if config['style_person_img_name'] is not None else torch.empty((1, 3, 32, 32), device=device)
    style_background_img = utils.prepare_img(style_background_img_path, current_height, device) if config['style_background_img_name'] is not None else torch.empty((1, 3, 32, 32), device=device)

    if config['init_method'] == 'random':
        # white_noise_img = np.random.uniform(-90., 90., content_img.shape).astype(np.float32)
        init_person_img = torch.from_numpy(
            np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        ).float().to(device)
        init_background_img = torch.from_numpy(
            np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        ).float().to(device)
    elif config['init_method'] == 'content':
        init_person_img = content_img.clone().detach()
        init_background_img = content_img.clone().detach()
    else:
        # init image has same dimension as content image - this is a hard constraint
        # feature maps need to be of same size for content image and init image
        init_person_img = utils.prepare_img(style_person_img_path, np.asarray(content_img.shape[2:]), device)
        init_background_img = utils.prepare_img(style_background_img_path, np.asarray(content_img.shape[2:]), device)


    # we are tuning optimizing_img's pixels! (that's why requires_grad=True)
    optimizing_person_img = Variable(init_person_img, requires_grad=True) if config['style_person_img_name'] is not None else Variable(content_img.clone().detach(), requires_grad=True)
    optimizing_background_img = Variable(init_background_img, requires_grad=True) if config['style_background_img_name'] is not None else Variable(content_img.clone().detach(), requires_grad=True)

    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config['model'], device)
    print(f'Using {config["model"]} in the optimization procedure.')

    content_img_set_of_feature_maps = neural_net(content_img)
    style_person_img_set_of_feature_maps = neural_net(style_person_img)
    style_background_img_set_of_feature_maps = neural_net(style_background_img)

    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_person_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_person_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_style_background_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_background_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]

    target_person_representations = [target_content_representation, target_style_person_representation]
    target_background_representations = [target_content_representation, target_style_background_representation]

    # magic numbers in general are a big no no - some things in this code are left like this by design to avoid clutter
    num_of_iterations = config['iterations']

    optimizer_person = Adam((optimizing_person_img,), lr=1e1)
    optimizer_background = Adam((optimizing_background_img,), lr=1e1)

    tuning_step_person = make_tuning_step(neural_net, optimizer_person, target_person_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config, str_flag='person')
    tuning_step_background = make_tuning_step(neural_net, optimizer_background, target_background_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config, str_flag='background')
    for cnt in range(num_of_iterations):

        # optimizing person image
        if config['style_person_img_name'] is not None:
            total_loss, content_loss, style_loss, tv_loss = tuning_step_person(optimizing_person_img)
            with torch.no_grad():
                print(f'P: Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_person_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
            
        # optimizing background image
        if config['style_background_img_name'] is not None:
            total_loss, content_loss, style_loss, tv_loss = tuning_step_background(optimizing_background_img)
            with torch.no_grad():
                print(f'B: Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_background_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
        
    # Check if tiling is needed
    _, _, img_height, img_width = content_img.shape
    use_tiling = img_height > TILE_SIZE or img_width > TILE_SIZE
    
    if use_tiling:
        print(f'\nImage size {img_height}x{img_width} requires tiling. Processing person and background in tiles...')
        
        # Process person image in tiles if needed
        if config['style_person_img_name'] is not None:
            print("\nProcessing person tiles...")
            with torch.no_grad():
                optimizing_person_img_data = process_tiles(
                    content_img,
                    style_person_img,
                    optimizing_person_img.data,
                    neural_net,
                    content_feature_maps_index_name[0],
                    style_feature_maps_indices_names[0],
                    config,
                    device,
                    target_style_person_representation
                )
            
            optimizing_person_img = Variable(optimizing_person_img_data, requires_grad=False)
        
        # Process background image in tiles if needed
        if config['style_background_img_name'] is not None:
            print("\nProcessing background tiles...")
            with torch.no_grad():
                optimizing_background_img_data = process_tiles(
                    content_img,
                    style_background_img,
                    optimizing_background_img.data,
                    neural_net,
                    content_feature_maps_index_name[0],
                    style_feature_maps_indices_names[0],
                    config,
                    device,
                    target_style_background_representation
                )
            
            optimizing_background_img = Variable(optimizing_background_img_data, requires_grad=False)
    
    # segmentation part - extract mask at the current resolution
    _, _, mask_height, mask_width = content_img.shape
    mask_seg = extract_person_mask_from_image(image_path=content_img_path, segmentation_mask_height=mask_height, device=device)

    mask_np = mask_seg.astype(np.float32) / 255.0    # most [0.0, 1.0]
    mask_t = torch.from_numpy(mask_np).to(device)    # shape: (H, W), dtype=float32, ugyanolyan eszközön
    mask_t = mask_t.unsqueeze(0).unsqueeze(0)         # (1, 1, H, W)
    mask3 = mask_t.expand(-1, 3, -1, -1)              # (1, 3, H, W)

    optimizing_img = optimizing_person_img * mask3 + optimizing_background_img * (1.0 - mask3)

    with torch.no_grad():
        img_name = utils.save_and_maybe_display(optimizing_img, dump_path, config, num_of_iterations-1, num_of_iterations, should_display=False)
        utils.save_and_maybe_display(optimizing_person_img, os.path.join(dump_path, "person"), config, num_of_iterations-1, num_of_iterations, should_display=False)
        utils.save_and_maybe_display(optimizing_background_img, os.path.join(dump_path, "background"), config, num_of_iterations-1, num_of_iterations, should_display=False)

    return img_name

def neural_style_transfer_mixed(config):
    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path_1 = os.path.join(config['style_images_dir'], config['style_img_name_1'])
    style_img_path_2 = os.path.join(config['style_images_dir'], config['style_img_name_2'])

    dump_path = config['output_img_dir']
    os.makedirs(dump_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use original size or specified height
    current_height = config['height']
    
    # Prepare images at full resolution
    content_img = utils.prepare_img(content_img_path, current_height, device)
    style_img_1 = utils.prepare_img(style_img_path_1, current_height, device)
    style_img_2 = utils.prepare_img(style_img_path_2, current_height, device)

    if config['init_method'] == 'random':
        # white_noise_img = np.random.uniform(-90., 90., content_img.shape).astype(np.float32)
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img).float().to(device)
    elif config['init_method'] == 'content':
        init_img = content_img

    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config['model'], device)
    print(f'Using {config["model"]} in the optimization procedure.')

    style_img_set_of_feature_maps_1 = neural_net(style_img_1)
    style_img_set_of_feature_maps_2 = neural_net(style_img_2)
    
    target_style_representation_1 = [utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps_1) if cnt in style_feature_maps_indices_names[0]]
    target_style_representation_2 = [utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps_2) if cnt in style_feature_maps_indices_names[0]]

    # Check if we need tiling
    _, _, img_height, img_width = content_img.shape
    use_tiling = img_height > TILE_SIZE or img_width > TILE_SIZE
    
    if use_tiling:
        print(f'Image size {img_height}x{img_width} requires tiling. Processing in {TILE_SIZE}x{TILE_SIZE} tiles...')
        
        # Process image in tiles
        optimizing_img_data = process_tiles_mixed(
            content_img,
            style_img_1,
            style_img_2,
            init_img,
            neural_net,
            content_feature_maps_index_name[0],
            style_feature_maps_indices_names[0],
            config,
            device,
            target_style_representation_1,
            target_style_representation_2
        )
        
        optimizing_img = Variable(optimizing_img_data, requires_grad=False)
    else:
        print(f'Image size {img_height}x{img_width} is small enough, running standard optimization...')
        
        # Standard optimization without tiling
        optimizing_img = Variable(init_img, requires_grad=True)
        
        content_img_set_of_feature_maps = neural_net(content_img)
        target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
        target_representations = [target_content_representation, target_style_representation_1, target_style_representation_2]

        num_of_iterations = config['iterations']
        optimizer = Adam((optimizing_img,), lr=1e1)
        tuning_step = make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
        
        for cnt in range(num_of_iterations):
            total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
            with torch.no_grad():
                print(f'Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')

    # Save final result
    with torch.no_grad():
        img_name = utils.save_and_maybe_display(optimizing_img, dump_path, config, config['iterations']-1, config['iterations'], should_display=False)

    return img_name