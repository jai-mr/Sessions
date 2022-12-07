import unittest

import requests
import json
import base64
from requests import Response


class TestFargateGradio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = "https://7dbd-34-168-0-159.ngrok.io/predictions/cifar"


        cls.image_paths = ['10_deer.png',  '30_airplane.png',  '11880_dog.png',  '203_cat.png',  '912_bird.png',  '43_horse.png',  '62_ship.png',  '200_frog.png',  '201_automobile.png',  '202_truck.png']

        # convert image to base64

    def test_predict(self):
        for image_path in self.image_paths:
            print(f"testing: {image_path}")

            data = open('test_serve/image/' + image_path, 'rb').read()
                # ext = image_path.split('.')[-1]
                # prefix = f'data:image/{ext};base64,'
                # base64_data = prefix + base64.b64encode(f.read()).decode('utf-8')

            # payload = json.dumps({
            # "data": [
            #     base64_data
            # ]
            # })

            # headers = {
            # 'Content-Type': 'application/json'
            # }

            response: Response = requests.request("POST", self.base_url, data=data, timeout=15)

            print(f"response: {response.text}")

            data = response.json()

            predicted_label = list(data)[0]
            act_label = image_path.split(".")[0].split('_')[-1]

            print(f"predicted label: {predicted_label}, actual label: {act_label}")

            self.assertEqual(act_label, predicted_label)

            print(f"done testing: {image_path}")

            print()


if __name__ == '__main__':
    unittest.main()