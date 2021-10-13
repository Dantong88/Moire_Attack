import numpy as np
import json
import os
import sys
import time
from mosaicing_demosaicing_v2 import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchattacks
from torchattacks.attack import Attack
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from image_transformer import ImageTransformer
from utils import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class Morie_attack(Attack):
    r"""
    Distance Measure : L_inf bound on sensor noise
    Arguments:
        model (nn.Module): Victim model to attack.
        steps (int): number of steps. (DEFAULT: 50)
        batch_size (int): batch size
        scale_factor (int): zoom in the images on the LCD. （DEFAULT: 3）

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    """

    def __init__(self, model, img_h, img_w, noise_budget, scale_factor, steps = 50, batch_size = 50, targeted = False):
        super(Morie_attack, self).__init__("Morie_attack", model)
        self.steps = steps
        self.targeted = targeted
        self.img_w = img_w
        self.img_h = img_h
        self.scale_factor = scale_factor
        self.noise_budget = noise_budget
        self.lr = noise_budget / steps
        noise = np.zeros([batch_size, self.img_h * self.scale_factor * 3, self.img_w * self.scale_factor * 3])
        self.noise = torch.from_numpy(noise).to(self.device)
        self.noise.requires_grad = True
        self.adv_loss = nn.CrossEntropyLoss()



    def simulate_LCD_display(self, input_img):
        """ Simulate the display of raw images on LCD screen
        Input:
            original images (tensor): batch x height x width x channel
        Output:
            LCD images (tensor): batch x (height x scale_factor)  x (width x scale_factor) x channel
        """
        input_img = np.asarray(input_img.cpu().detach())
        batch_size, h, w, c = input_img.shape

        simulate_imgs = np.zeros((batch_size, h * 3, w * 3, 3), dtype=np.float32)
        red = np.repeat(input_img[:, :, :, 0], 3, axis = 1)
        green = np.repeat(input_img[:, :, :, 1], 3, axis = 1)
        blue = np.repeat(input_img[:, :, :, 2], 3, axis = 1)

        for y in range(w):
            simulate_imgs[:, :, y * 3, 0] = red[:, :, y]
            simulate_imgs[:, :, y * 3 + 1, 1] = green[:, :, y]
            simulate_imgs[:, :, y * 3 + 2, 2] = blue[:, :, y]
        simulate_imgs = torch.from_numpy(simulate_imgs).to(self.device)

        return simulate_imgs

    def demosaic_and_denoise(self, input_img):
        """ Apply demosaicing to the images
        Input:
            images (tensor): batch x (height x scale_factor) x (width x scale_factor)
        Output:
            demosaicing images (tensor): batch x (height x scale_factor) x (width x scale_factor) x channel
        """
        demosaicing_imgs = demosaicing_CFA_Bayer_bilinear(input_img)
        return demosaicing_imgs

    def simulate_CFA(self, input_img):
        """ Simulate the raw reading of the camera sensor using bayer CFA
        Input:
            images (tensor): batch x (height x scale_factor) x (width x scale_factor) x channel
        Output:
            mosaicing images (tensor): batch x (height x scale_factor) x (width x scale_factor)
        """
        mosaicing_imgs = mosaicing_CFA_Bayer(input_img)
        return mosaicing_imgs

    def random_rotation_3(self, org_images, lcd_images):
        """ Simulate the 3D rotatation during the shooting
        Input:
            images (tensor): batch x height x width x channel
        Rotate angle:
            theta (int): (-20, 20)
            phi (int): (-20, 20)
            gamma (int): (-20, 20)
        Output:
            rotated original images (tensor): batch x height x width x channel
            rotated LCD images (tensor): batch x (height x scale_factor) x (width x scale_factor) x channel
        """
        rotate_images = np.zeros(org_images.size())
        rotate_lcd_images = np.zeros(lcd_images.size())

        for n, img in enumerate(org_images):
            Trans_org = ImageTransformer(img)
            theta, phi, gamma, rotate_img = Trans_org.rotate_along_axis(True)
            rotate_images[n, :] = rotate_img
            Trans_lcd = ImageTransformer(lcd_images[n])
            _, _, _, rotate_lcd_img = Trans_lcd.rotate_along_axis(False, theta, phi, gamma)
            rotate_lcd_images[n, :] = rotate_lcd_img

        rotate_images = torch.from_numpy(rotate_images).to(device)
        rotate_lcd_images = torch.from_numpy(rotate_lcd_images).to(device)

        return rotate_images, rotate_lcd_images

    def forward(self, org_imgs, org_labels, targeted_labels):
        r"""
        Overridden.
        """
        org_images = org_imgs.clone().detach().to(self.device)
        org_labels = org_labels.clone().detach().to(self.device)
        org_labels = self._transform_label(org_images, org_labels)

        # compute the orignal prediction
        temp_outputs = self.model(org_imgs.clone().detach().to(self.device))
        org_percentage = F.softmax(temp_outputs, dim=1) * 100
        del temp_outputs

        resize_before_lcd = F.interpolate(org_images, scale_factor = self.scale_factor, mode="bilinear")
        resize_before_lcd = resize_before_lcd.permute(0, 2, 3, 1)
        lcd_images = self.simulate_LCD_display(resize_before_lcd)

        temp_images = org_images.clone().detach().permute(0, 2, 3, 1)

        rotate_images, rotate_lcd_images = self.random_rotation_3(temp_images, lcd_images)
        rotate_images = rotate_images.to(self.device)
        rotate_lcd_images = rotate_lcd_images.to(self.device).detach()

        dim_images = adjust_contrast_and_brightness(rotate_images, beta=-60)

        ## compute the rotate prediction
        rotate_images = rotate_images.permute(0, 3, 1, 2)
        rotate_images = rotate_images.float()
        rotate_outputs = self.model(rotate_images)
        _, rotate_pre = torch.max(rotate_outputs.data, 1)
        rotate_percentage = F.softmax(rotate_outputs.clone().detach(), dim=1) * 100

        ## compute the dim prediction
        dim_images = dim_images.permute(0, 3, 1, 2)
        dim_images = dim_images.float()
        dim_outputs = self.model(dim_images)
        _, dim_pre = torch.max(dim_outputs.data, 1)
        dim_percentage = F.softmax(dim_outputs.clone().detach(), dim=1) * 100


        ## Deliver the MA
        for step in range(self.steps):
            print("Step: {}/{}".format(step, self.steps))

            cfa_img = self.simulate_CFA(rotate_lcd_images)
            cfa_img_noise = cfa_img + self.noise

            demosaic_img = self.demosaic_and_denoise(cfa_img_noise)
            demosaic_img = demosaic_img.permute(0, 3, 1, 2)

            ## Adjust the brightness
            brighter_img = adjust_contrast_and_brightness(demosaic_img, beta=20)

            at_images = F.interpolate(brighter_img, [299, 299], mode='bilinear')
            at_images = at_images.float()
            at_outputs = self.model(at_images)
            _, at_pre = torch.max(at_outputs.data, 1)

            at_percentage = F.softmax(at_outputs.clone().detach(), dim=1) * 100


            if self.targeted:
                adv_cost = self.adv_loss(at_outputs, targeted_labels.to(self.device))
            else:
                adv_cost = -1 * self.adv_loss(at_outputs, org_labels)


            total_cost = adv_cost
            print("Loss: ", total_cost, "Adv loss: ", adv_cost)

            total_cost.backward()
            gradient = self.noise.grad
            self.noise = self.noise.detach() - self.lr * torch.sign(gradient)
            self.noise = torch.clamp(self.noise, min = -self.noise_budget, max = self.noise_budget).detach()
            self.noise.requires_grad = True

        at_images = torch.clamp(at_images, min=0, max=255).detach()

        return at_images, rotate_images, dim_images, \
               at_pre, rotate_pre, dim_pre, \
               org_percentage, at_percentage, rotate_percentage, dim_percentage



class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        input = input / 255.0
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_idx = json.load(open("./data/imagenet_class_index.json"))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    class2label = [class_idx[str(k)][0] for k in range(len(class_idx))]

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(), ])

    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    model = nn.Sequential(
        norm_layer,
        models.inception_v3(pretrained=True)
    ).to(device)

    model = model.eval()

    ## Save the results of MA
    Save_results = 'False'
    if Save_results == 'True':
        savedir = './Results'
        adv_dir = os.path.join(savedir, 'adv')
        rotate_dir = os.path.join(savedir, 'rotate')
        org_dir = os.path.join(savedir, 'org')
        dim_dir = os.path.join(savedir, 'dim')
        create_dir(adv_dir)
        create_dir(rotate_dir)
        create_dir(org_dir)
        create_dir(dim_dir)

    ## deffault settings
    noise_budget = 2
    batch_size = 10
    epoch = 1#int(1000 / batch_size)
    total = 0
    suc_cnt_at = 0
    suc_cnt_dim = 0
    suc_cnt_rotate = 0



    normal_data = image_folder_custom_label(root='./dataset/incepv3_data', transform=transform,
                                            idx2label=class2label)
    normal_loader = torch.utils.data.DataLoader(normal_data, batch_size=batch_size, shuffle=False)
    normal_iter = iter(normal_loader)

    print("-" * 70)
    print("Noise_budget = ", noise_budget)
    start = time.time()
    for batch in range(epoch):

        print("-" * 70)
        org_imgs, org_labels = normal_iter.next()
        org_imgs = org_imgs * 255.0
        print('Epoch = ' + str(batch))

        targeted_labels = np.random.randint(0,999)
        targeted_labels = torch.from_numpy(targeted_labels * np.ones((batch_size), dtype = np.int))
        targeted = True


        attack = Morie_attack(model, noise_budget = noise_budget, img_w = 299, img_h = 299, scale_factor = 3, targeted = targeted, batch_size=batch_size, steps = 50)
        at_images, rotate_images, dim_images, \
        at_labels, rotate_labels, dim_labels, \
        org_percentges, at_percentages, rotate_percentages, dim_percentages = attack(org_imgs, org_labels, targeted_labels)
        org_labels = org_labels.to(device)
        targeted_labels = targeted_labels.to(device)
        rotate_labels = rotate_labels.to(device)

        ## compute the succes rate
        total += batch_size

        suc_cnt_rotate += (rotate_labels != org_labels).sum()
        suc_cnt_dim += (dim_labels != org_labels).sum()

        if targeted:
            suc_cnt_at += (at_labels == targeted_labels).sum()
        else:
            suc_cnt_at += (at_labels != org_labels).sum()


        Succ_cnt_rotate = (rotate_labels != org_labels).sum() / batch_size
        Succ_total_rotate = suc_cnt_rotate / total
        Succ_cnt_dim = (dim_labels != org_labels).sum() / batch_size
        Succ_total_dim = suc_cnt_dim / total
        if targeted:
            Succ_cnt_at = (at_labels == targeted_labels).sum() / batch_size
            Succ_total_at = suc_cnt_at / total
        else:
            Succ_cnt_at = (at_labels != org_labels).sum() / batch_size
            Succ_total_at = suc_cnt_at / total

        print("Current rotate Suc rate: ", Succ_cnt_rotate)
        print("Current dim Suc rate: ", Succ_cnt_dim)
        print("Current attack Suc rate: ", Succ_cnt_at)
        print("Total rotate Suc rate: ", Succ_total_rotate)
        print("Total dim Suc rate: ", Succ_total_dim)
        print("Total attack Suc rate: ", Succ_total_at)

        labels_np = org_labels.cpu().detach().numpy()
        rotate_labels_np = rotate_labels.cpu().detach().numpy()
        dim_labels_np = dim_labels.cpu().detach().numpy()
        at_labels_np = at_labels.cpu().detach().numpy()

        org_images_np = org_imgs.detach().cpu().numpy()
        at_images_np = at_images.detach().cpu().numpy()
        rotate_images_np = rotate_images.detach().cpu().numpy()
        dim_images_np = dim_images.detach().cpu().numpy()

        org_percentages_np = org_percentges.detach().cpu().numpy()
        at_percentages_np = at_percentages.detach().cpu().numpy()
        rotate_percentages_np = rotate_percentages.detach().cpu().numpy()
        dim_percentages_np = dim_percentages.detach().cpu().numpy()

        # save the pics
        for i in range(batch_size):
            img_org = org_images_np[i]
            img_at = at_images_np[i]
            img_dim = dim_images_np[i]
            img_rotate = rotate_images_np[i]

            img_org = np.moveaxis(img_org, 0, 2)
            img_at = np.moveaxis(img_at, 0, 2)
            img_dim = np.moveaxis(img_dim, 0, 2)
            img_rotate = np.moveaxis(img_rotate, 0, 2)

            true_class = idx2label[labels_np[i]]
            at_class = idx2label[at_labels_np[i]]
            dim_class = idx2label[dim_labels_np[i]]
            rotate_class = idx2label[rotate_labels_np[i]]

            percentage_org = org_percentages_np[i][labels_np[i]]
            percentage_at = at_percentages_np[i][at_labels_np[i]]
            percentage_rotate = rotate_percentages_np[i][rotate_labels_np[i]]
            percentage_dim = dim_percentages_np[i][dim_labels_np[i]]

            if Save_results == 'True':
                # save org_images
                img_org_name = true_class + str(percentage_org) + ".JPEG"
                img_org_path = os.path.join(org_dir, img_org_name)
                img_org_pil = Image.fromarray(img_org.astype(np.uint8))
                img_org_pil.save(img_org_path)

                # uncomment the following if you want to save the intermediate results
                # save rotated_images:
                img_rotate_name = true_class + str(percentage_org) + "_" + rotate_class + str(percentage_rotate) + ".JPEG"
                img_rotate_path = os.path.join(rotate_dir, img_rotate_name)
                img_rotate_pil = Image.fromarray(img_rotate.astype(np.uint8))
                img_rotate_pil.save(img_rotate_path)

                # save dim_images:
                img_dim_name = true_class + str(percentage_org) + "_" + dim_class + str(percentage_dim) + ".JPEG"
                img_dim_path = os.path.join(dim_dir, img_dim_name)
                img_dim_pil = Image.fromarray(img_dim.astype(np.uint8))
                img_dim_pil.save(img_dim_path)

                ##  save at_images:
                img_at_name = true_class + str(percentage_org) + "_" + at_class + str(percentage_at) + ".JPEG"
                img_at_path = os.path.join(adv_dir, img_at_name)
                img_at_pil = Image.fromarray(img_at.astype(np.uint8))
                img_at_pil.save(img_at_path)



        del attack, at_images, rotate_images, dim_images, \
            at_labels, rotate_labels, dim_labels, \
            org_percentges, at_percentages, rotate_percentages, dim_percentages, \
            org_labels, targeted_labels, \
            org_imgs



        torch.cuda.empty_cache()

    print("Rotate Success rate: ", Succ_total_rotate)
    print("dim Success rate: ", Succ_total_dim)
    print("Attack Success rate: ", Succ_total_at)
