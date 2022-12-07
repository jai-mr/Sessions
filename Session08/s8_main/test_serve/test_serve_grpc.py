
import sys
import pyrootutils

root = pyrootutils.setup_root(sys.path[0], pythonpath=True, cwd=True)
sys.path.insert(0,'serve/ts_scripts/')
import unittest

import requests
import json
import base64
from requests import Response
from serve.ts_scripts.torchserve_grpc_client import infer, get_inference_stub

class TestFargateGradio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):


        #cls.image_paths = ['34_deer.png',  '35_airplane.png',  '40_dog.png',  '203_cat.png',  '41_bird.png',  '43_horse.png',  '62_ship.png',  '200_frog.png',  '201_automobile.png',  '202_truck.png']
        cls.image_paths = ['10_deer.png',  '30_airplane.png',  '11880_dog.png',  '203_cat.png',  '912_bird.png',  '43_horse.png',  '62_ship.png',  '200_frog.png',  '201_automobile.png',  '202_truck.png']
        cls.stub = get_inference_stub('7dbd-34-168-0-159.ngrok.io:443')
        # convert image to base64
    def test_predict(self):
        for image_path in self.image_paths:
            print(f"testing: {image_path}")

            response = infer(self.stub, 'cifar', 'test_serve/image/' + image_path)
     

            # print(f"response: {response.text}")

            data = json.loads(response)

            predicted_label = list(data)[0]
            act_label = image_path.split(".")[0].split('_')[-1]

            print(f"predicted label: {predicted_label}, actual label: {act_label}")

            self.assertEqual(act_label, predicted_label)

            print(f"done testing: {image_path}")

            print()


if __name__ == '__main__':
    unittest.main()