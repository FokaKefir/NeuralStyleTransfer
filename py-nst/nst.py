import utils.utils as utils
from utils.video_utils import create_video_from_intermediate_results

from segmentation import extract_person_mask_from_image

import torch
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
import os
import argparse


def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config, str_flag=""):
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

    content_img = utils.prepare_img(content_img_path, config['height'], device)
    style_img = utils.prepare_img(style_img_path, config['height'], device)

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

    # we are tuning optimizing_img's pixels! (that's why requires_grad=True)
    optimizing_img = Variable(init_img, requires_grad=True)

    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = utils.prepare_model(config['model'], device)
    print(f'Using {config["model"]} in the optimization procedure.')

    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)

    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations = [target_content_representation, target_style_representation]

    # magic numbers in general are a big no no - some things in this code are left like this by design to avoid clutter
    num_of_iterations = config['iterations']

    optimizer = Adam((optimizing_img,), lr=1e1)
    tuning_step = make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
    for cnt in range(num_of_iterations):
        total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
        with torch.no_grad():
            print(f'Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
            img_name = utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations, should_display=False)


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

    content_img = utils.prepare_img(content_img_path, config['height'], device)
    style_person_img = utils.prepare_img(style_person_img_path, config['height'], device) if config['style_person_img_name'] is not None else torch.empty((1, 3, 32, 32), device=device)
    style_background_img = utils.prepare_img(style_background_img_path, config['height'], device) if config['style_background_img_name'] is not None else torch.empty((1, 3, 32, 32), device=device)

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

    print(content_img.shape)
    print(optimizing_person_img.size())
    print(optimizing_background_img.size())

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
        
    
    # segmentation part
    mask_seg = extract_person_mask_from_image(image_path=content_img_path, segmentation_mask_height=config['height'], device=device)

    mask_np = mask_seg.astype(np.float32) / 255.0    # most [0.0, 1.0]
    mask_t = torch.from_numpy(mask_np).to(device)    # shape: (H, W), dtype=float32, ugyanolyan eszközön
    mask_t = mask_t.unsqueeze(0).unsqueeze(0)         # (1, 1, H, W)
    mask3 = mask_t.expand(-1, 3, -1, -1)              # (1, 3, H, W)

    optimizing_img = optimizing_person_img * mask3 + optimizing_background_img * (1.0 - mask3)

    img_name = utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations, should_display=False)
    # utils.save_and_maybe_display(optimizing_person_img, os.path.join(dump_path, "person"), config, cnt, num_of_iterations, should_display=False)
    # utils.save_and_maybe_display(optimizing_background_img, os.path.join(dump_path, "background"), config, cnt, num_of_iterations, should_display=False)


    return img_name

