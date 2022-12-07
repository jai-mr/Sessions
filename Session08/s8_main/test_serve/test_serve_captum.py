import unittest

import requests
import json
from requests import Response
#import os
import torch
import numpy as np
import torchvision.transforms as T

from PIL import Image

from captum.attr import visualization as viz
#import matplotlib.pyplot as plt


class TestCaptumExplanation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = " https://09d8-44-208-162-50.ngrok.io/explanations/cifar/1.0"


        cls.image_paths = "10_deer.png"

        # convert image to base64

    def test_predict(self):
        print(f"testing: Captum explanation for 10_deer.png")

        imgpth='/home/ubuntu/s8_main/test_serve/image/'
        inp_image = Image.open(imgpth+self.image_paths)
        
        res = requests.post(self.base_url, files={'data': open(f"/home/ubuntu/s8_main/test_serve/image/{self.image_paths}", 'rb')})

        ig =  res.json()
        to_tensor = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor()
        ])
        inp_image = to_tensor(inp_image)

        inp_image = inp_image.numpy()
        attributions = np.array(ig)
        inp_image, attributions = inp_image.transpose(1, 2, 0), attributions.transpose(1, 2, 0)
        
        self.assertEqual(inp_image.shape, attributions.shape)
        print(f"done testing: ")
        

        res = viz.visualize_image_attr(attributions, inp_image, method="blended_heat_map",sign="all", show_colorbar=True, title="Overlayed Integrated Gradients")
        #res[0].savefig('/home/ubuntu/s8_main/test_serve/' + self.image_paths.split('.')[0] + "_ig.png")
        res[0].savefig('/home/ubuntu/s8_main/test_serve/' + self.image_paths.split('.')[0] + "_ig.png")
        imgcptum='/home/ubuntu/s8_main/test_serve/' + self.image_paths.split('.')[0] + "_ig.png"
        print(imgcptum)
        res[0].savefig(imgcptum)
        print()


if __name__ == '__main__':
    unittest.main()
