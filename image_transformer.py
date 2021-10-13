from utils import *
import numpy as np
import cv2
from math import  pi
import torch


class ImageTransformer(object):
    """ Perspective transformation class for image
        with shape (height, width, #channels) """

    def __init__(self, img):
        self.image = img
 
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.num_channels = self.image.shape[2]

    def get_rad(self, theta, phi, gamma):
        return (self.deg_to_rad(theta),
                self.deg_to_rad(phi),
                self.deg_to_rad(gamma))

    def get_deg(self, rtheta, rphi, rgamma):
        return (self.rad_to_deg(rtheta),
                self.rad_to_deg(rphi),
                self.rad_to_deg(rgamma))

    def deg_to_rad(self, deg):
        return deg * pi / 180.0

    def rad_to_deg(self, rad):
        return rad * 180.0 / pi

    """ Wrapper of Rotating a Image """
    def rotate_along_axis(self, random_f, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):

        # Get radius of rotation along 3 axes
        if random_f:
            theta = np.random.randint(-20, 20)
            phi = np.random.randint(-20, 20)
            gamma = np.random.randint(-20, 20)
        
        rtheta, rphi, rgamma =self.get_rad(theta, phi, gamma)
        
        # Get ideal focal length on z axis
        # NOTE: Change this section to other axis if needed
        d = np.sqrt(self.height**2 + self.width**2)
        self.focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
        dz = self.focal

        # Get projection matrix
        mat = self.get_M(rtheta, rphi, rgamma, dx, dy, dz)
        # roated_img = cv2.warpPerspective(self.image.copy(), mat, (self.width, self.height))
        temp = np.array(self.image.cpu())
        rotate_img = cv2.warpPerspective(temp, mat, (self.width, self.height))
        rotate_img = torch.from_numpy(rotate_img)
        return theta, phi, gamma, rotate_img


    """ Get Perspective Projection Matrix """
    def get_M(self, theta, phi, gamma, dx, dy, dz):
        
        w = self.width
        h = self.height
        f = self.focal

        # Projection 2D -> 3D matrix
        A1 = np.array([ [1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0, 1],
                        [0, 0, 1]])
        
        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([ [1, 0, 0, 0],
                        [0, np.cos(theta), -np.sin(theta), 0],
                        [0, np.sin(theta), np.cos(theta), 0],
                        [0, 0, 0, 1]])
        
        RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                        [0, 1, 0, 0],
                        [np.sin(phi), 0, np.cos(phi), 0],
                        [0, 0, 0, 1]])
        
        RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                        [np.sin(gamma), np.cos(gamma), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation matrix
        T = np.array([  [1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([ [f, 0, w/2, 0],
                        [0, f, h/2, 0],
                        [0, 0, 1, 0]])

        # Final transformation matrix
        return np.dot(A2, np.dot(T, np.dot(R, A1)))