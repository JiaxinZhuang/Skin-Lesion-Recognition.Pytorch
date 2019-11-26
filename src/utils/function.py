"""Function.

    Some useful functions.
"""

import os
import logging
import sys
import copy
from functools import wraps
import time

# import resource

import numpy as np
import torch
from PIL import Image


# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


sys.path.append("./src/utils")


def init_environment(seed=0, cuda_id="0"):
    """Init environment
    initialize environment including cuda, benchmark, random seed, saved model
    directory and saved logs directory
    """
    print("=> init_environment with seed: {}".format(seed))
    print("=> Use cuda: {}".format(cuda_id))

    cuda_id = str(cuda_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_id

    if seed != -1:
        print("> Use seed -{}".format(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        print("> Don't use seed")


def init_logging(output_dir, exp):
    """Init logging and return logging
        init_logging should used after init environment
    """
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(message)s",
                        datefmt="%Y%m%d-%H:%M:%S",
                        filename=os.path.join(output_dir, str(exp) + ".log"),
                        filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)
    return logging


def timethis(func, *args, **kwargs):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        elapse = time.time() - start_time
        print(">> Functoin: {} costs {:.4f}s".format(func.__name__, elapse))
        sys.stdout.flush()
        return ret
    return wrapper


def str2bool(val):
    """convert str to bool
    """
    value = None
    if val == 'True':
        value = True
    elif val == 'False':
        value = False
    else:
        raise ValueError
    return value


def str2list(val):
    """convert str to bool
    """
    value = val.split(",")
    value = [int(v.strip()) for v in value]
    return value


def preprocess_image(pil_im, mean, std, resize=512, resize_im=True,
                     device=None):
    """Process images for CNNs

    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    Returns:
        im_as_var (torch variable): Variable that contains processed
                                    float tensor
    """
    # Resize image
    if resize_im:
        pil_im.resize((resize, resize))

    im_as_arr = np.float32(pil_im)
    if len(im_as_arr.shape) == 2:
        im_as_arr = np.expand_dims(im_as_arr, axis=2)
    # Convert array to [D, W, H]
    im_as_arr = im_as_arr.transpose(2, 0, 1)
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).to(torch.float32)
    # Add one more channle to the beginning.
    im_as_ten.unsqueeze_(0)
    im_as_var = im_as_ten.clone().detach().to(device).requires_grad_(True)
    return im_as_var


def format_np_output(np_arr):
    """This is a (kind of) bandiad fix to steamline save procedure.
       It converts all the outputs to the same format which
       is 3xWxH with using successive if clauses.

    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH.
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase / Case 4: Np arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr * 255).astype(np.uint8)

    if len(np_arr.shape) == 3 and np_arr.shape[2] == 1:
        np_arr = np_arr.squeeze(axis=2)
    return np_arr


def save_image(im, path):
    """Save a numpy matrix or PIL image as an image.

    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def recreate_image(im_as_var, reverse_mean, reverse_std):
    """Recreate images from a torch variable, sort of reverse preprocessing.

    Args:
        im_as_var (torch variable): Image to recreate
    Returns:
        recreate_im (numpy arr): Recreated image in array
    """
    recreate_im = copy.copy(im_as_var)
    assert len(recreate_im.shape) == 3
    channels = recreate_im.shape[0]
    for channel in range(channels):
        recreate_im[channel] /= reverse_std[channel]
        recreate_im[channel] -= reverse_mean[channel]
    recreate_im[recreate_im > 1] = 1
    recreate_im[recreate_im < 0] = 0
    recreate_im = np.round(recreate_im * 255)

    recreate_im = np.uint8(recreate_im).transpose(1, 2, 0)
    return recreate_im


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_grad_norm(processed_images):
    """Get grad norm L2 for processed_images.
    """
    grads = processed_images.grad.clone().cpu()
    grads_norm = torch.norm(grads)
    return grads_norm


# def get_net_grad_norm(net):
#     """Get grad norm L2 for net."""
#     grad_norm =
#     for name, param in net.named_parameters():


def dataname_2_save(imgs_path, saved_dir):
    """Img path saved.
    """
    output_name = []
    for name in imgs_path:
        name = name.split("/")[-1:]
        output = os.path.join(saved_dir, *name)
        output_name.append(output)
    print(output_name)
    return output_name


def _test_image_related():
    """Test preprocess_image and save_image.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    image = Image.open("../../data/example/dd_tree.jpg").convert("RGB")
    im_as_var = preprocess_image(image, mean=mean, std=std)
    print(im_as_var.size())
    recreate_im = recreate_image(im_as_var, reverse_mean, reverse_std)
    print(recreate_im.shape)
    save_image(recreate_im, "../../saved/generated/recreate_im.jpg")


def _test_dataname2save():
    """Test dataname_2_save.
    """
    saved_dir = "/media/lincolnzjx/Disk21/interpretation/saved/generated"
    imgs_path = ["../data/CUB_200_2011/images/001.Black_footed_Albatross\
                 /Black_Footed_Albatross_0009_34.jpg",
                 "../data/CUB_200_2011/images/034.Gray_crowned_Rosy_Finch\
                 /Gray_Crowned_Rosy_Finch_0044_26976.jpg"]
    output_name = dataname_2_save(imgs_path, saved_dir)
    print("Test dataname2save")
    print(imgs_path)
    print("-"*80)
    print(output_name)


def print_loss_sometime(dicts, print_frequency=50, _print=None):
    """Print loss every epoch.
    """
    epoch = dicts["epoch"]
    n_epochs = dicts["n_epochs"]
    # batch_idx = dicts["batch_idx"]
    # batch_len = dicts["batch_len"]
    loss = dicts["loss"]
    # if batch_idx % print_frequency == 0:
    _print('=> Epoch [{}/{}], Loss: {:.4f}'.format(epoch, n_epochs, loss))


if __name__ == "__main__":
    _test_dataname2save()
