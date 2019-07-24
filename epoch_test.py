import os
from options.test_options import TestOptions
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from models import networks
from models import create_model
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from models.silnet import SilNet

def save_image(image_numpy, image_path):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def tensor2im2(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

opt = TrainOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 1
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

dataset = create_dataset(opt) # create a dataset given opt.dataset_mode and other options

device = "cuda:0" if torch.cuda.is_available() else "cpu"

with torch.no_grad():
    S = SilNet().to(device)
    S.load_state_dict(torch.load("silnet.pth"))

    for epoch in tqdm(range(5, 205, 5), "Epoch"):

        netG_A = networks.ResnetGenerator(3, 3, 64, networks.get_norm_layer('instance'), False, n_blocks=9)
        netG_B = networks.ResnetGenerator(3, 3, 64, networks.get_norm_layer('instance'), False, n_blocks=9)
        # already moved to GPU
        netG_A.load_state_dict(torch.load("checkpoints/synth2real_geocongan/{}_net_G_A.pth".format(epoch)))
        netG_B.load_state_dict(torch.load("checkpoints/synth2real_geocongan/{}_net_G_B.pth".format(epoch)))

        netG_A = netG_A.to(device)
        netG_B = netG_B.to(device)

        epoch = format(epoch, '03')
        for i, data in tqdm(enumerate(dataset), "Iter"):

            real_A = data['A'].to(device)
            real_B = data['B'].to(device)
            mask_A = S(real_A)
            mask_B = S(real_B)

            fake_B = netG_A(real_A)
            fake_A = netG_B(real_B)
            mask_fake_A = S(fake_A)
            mask_fake_B = S(fake_B)
            rec_A = netG_B(fake_B)
            rec_B = netG_A(fake_A)
            idt_A = netG_A(real_B)
            idt_B = netG_B(real_A)

            mask_A = tensor2im2(torch.sigmoid(mask_A))
            mask_B = tensor2im2(torch.sigmoid(mask_B))
            mask_fake_A = tensor2im2(torch.sigmoid(mask_fake_A))
            mask_fake_B = tensor2im2(torch.sigmoid(mask_fake_B))
            real_A = tensor2im(real_A)
            real_B = tensor2im(real_B)
            fake_A = tensor2im(fake_A)
            fake_B = tensor2im(fake_B)
            rec_A = tensor2im(rec_A)
            rec_B = tensor2im(rec_B)
            idt_A = tensor2im(idt_A)
            idt_B = tensor2im(idt_B)
            
            if not os.path.exists('feat{}'.format(i)):
                os.mkdir('feat{}'.format(i))
            save_image(real_A, "feat{}/epoch{}_real_A.png".format(i, epoch))
            save_image(real_B, "feat{}/epoch{}_real_B.png".format(i, epoch))
            save_image(mask_A, "feat{}/epoch{}_mask_A.png".format(i, epoch))
            save_image(mask_B, "feat{}/epoch{}_mask_B.png".format(i, epoch))
            save_image(fake_A, "feat{}/epoch{}_fake_A.png".format(i, epoch))
            save_image(fake_B, "feat{}/epoch{}_fake_B.png".format(i, epoch))
            save_image(mask_fake_A, "feat{}/epoch{}_mask_fake_A.png".format(i, epoch))
            save_image(mask_fake_B, "feat{}/epoch{}_mask_fake_B.png".format(i, epoch))
            save_image(rec_A, "feat{}/epoch{}_rec_A.png".format(i, epoch))
            save_image(rec_B, "feat{}/epoch{}_rec_B.png".format(i, epoch))
            save_image(idt_A, "feat{}/epoch{}_idt_A.png".format(i, epoch))
            save_image(idt_B, "feat{}/epoch{}_idt_B.png".format(i, epoch))
            if i > 9:
                break

