**Repository github url : https://github.com/jai-mr/Sessions/tree/main/Session07 <br/>
**Assignment Repository : https://github.com/jai-mr/Sessions/blob/main/Session07/README.md <br/>
**Submitted by : Jaideep R - No Partners<br/>
**Registered email id : jaideepmr@gmail.com<br/>

### Model Explanation with
  1. IG
  2. IG w/ Noise Tunnel
  3. Saliency
  4. Occlusion
  5. SHAP
  6. GradCAM
  7. GradCAM++

< Click on Cell Image to get full view of image> 

| Original Image | IG |       Noise Tunnel | Saliency | Occlusion | SHAP | GradCAM | GradCAM++ |
:----------------------------:|:--------------------------------:|:--------------------------------:|:--------------------------------:|:--------------------------------:|:--------------------------------:|:--------------------------------:|:--------------------------------:|
![](images/1-boat.jpg)| ![](output_1/1-boat_ig.jpg) | ![](output_1/1-boat_nt.jpg) | ![](output_1/1-boat_saliency.jpg) | ![](output_1/1-boat_occ.jpg) | ![](output_1/1-boat_grad_shap.jpg) | ![](output_1/1-boat_gc.jpg) | ![](output_1/1-boat_gcp.jpg)
![](images/2-car.jpg)| ![](output_1/2-car_ig.jpg) | ![](output_1/2-car_nt.jpg) | ![](output_1/2-car_saliency.jpg) | ![](output_1/2-car_occ.jpg) | ![](output_1/2-car_grad_shap.jpg) | ![](output_1/2-car_gc.jpg) | ![](output_1/2-car_gcp.jpg)
![](images/3-cat.jpg)| ![](output_1/3-cat_ig.jpg) | ![](output_1/3-cat_nt.jpg) | ![](output_1/3-cat_saliency.jpg) | ![](output_1/3-cat_occ.jpg) | ![](output_1/3-cat_grad_shap.jpg) | ![](output_1/3-cat_gc.jpg) | ![](output_1/3-cat_gcp.jpg)
![](images/4-dog.jpg)| ![](output_1/4-dog_ig.jpg) | ![](output_1/4-dog_nt.jpg) | ![](output_1/4-dog_saliency.jpg) | ![](output_1/4-dog_occ.jpg) | ![](output_1/4-dog_grad_shap.jpg) | ![](output_1/4-dog_gc.jpg) | ![](output_1/4-dog_gcp.jpg)
![](images/5-bus.jpg)| ![](output_1/5-bus_ig.jpg) | ![](output_1/5-bus_nt.jpg) | ![](output_1/5-bus_saliency.jpg) | ![](output_1/5-bus_occ.jpg) | ![](output_1/5-bus_grad_shap.jpg) | ![](output_1/5-bus_gc.jpg) | ![](output_1/5-bus_gcp.jpg)
![](images/6-truck.jpg)| ![](output_1/6-truck_ig.jpg) | ![](output_1/6-truck_nt.jpg) | ![](output_1/6-truck_saliency.jpg) | ![](output_1/6-truck_occ.jpg) | ![](output_1/6-truck_grad_shap.jpg) | ![](output_1/6-truck_gc.jpg) | ![](output_1/6-truck_gcp.jpg)
![](images/7-tiger.jpg)| ![](output_1/7-tiger_ig.jpg) | ![](output_1/7-tiger_nt.jpg) | ![](output_1/7-tiger_saliency.jpg) | ![](output_1/7-tiger_occ.jpg) | ![](output_1/7-tiger_grad_shap.jpg) | ![](output_1/7-tiger_gc.jpg) | ![](output_1/7-tiger_gcp.jpg)
![](images/8-shark.jpg)| ![](output_1/8-shark_ig.jpg) | ![](output_1/8-shark_nt.jpg) | ![](output_1/8-shark_saliency.jpg) | ![](output_1/8-shark_occ.jpg) | ![](output_1/8-shark_grad_shap.jpg) | ![](output_1/8-shark_gc.jpg) | ![](output_1/8-shark_gcp.jpg)
![](images/9-turtle.jpg)| ![](output_1/9-turtle_ig.jpg) | ![](output_1/9-turtle_nt.jpg) | ![](output_1/9-turtle_saliency.jpg) | ![](output_1/9-turtle_occ.jpg) | ![](output_1/9-turtle_grad_shap.jpg) | ![](output_1/9-turtle_gc.jpg) | ![](output_1/9-turtle_gcp.jpg)
![](images/10-alligator.jpg)| ![](output_1/10-alligator_ig.jpg) | ![](output_1/10-alligator_nt.jpg) | ![](output_1/10-alligator_saliency.jpg) | ![](output_1/10-alligator_occ.jpg) | ![](output_1/10-alligator_grad_shap.jpg) | ![](output_1/10-alligator_gc.jpg) | ![](output_1/10-alligator_gcp.jpg)

### Adversarial Attacks to predict persian cat

* Using PGD <Projected Gradient Descent> which is an iterative version of  FGSM (Fast Gradient Sign Method)that can generate adversarial examples. 
* It takes multiple gradient steps to search for an adversarial perturbation within the desired neighbor ball around the original inputs. 
* The following images have been modified so they all predict 'Persian cat' when predicted using the pretrained resnet18 model from timm library.

![](output_2/1-boat_adv.jpg) | ![](output_2/2-car_adv.jpg) | ![](output_2/3-cat_adv.jpg)
![](output_2/4-dog_adv.jpg) | ![](output_2/5-bus_adv.jpg) | ![](output_2/6-truck_adv.jpg)
![](output_2/7-tiger_adv.jpg) | ![](output_2/8-shark_adv.jpg) | ![](output_2/9-turtle_adv.jpg)
![](output_2/10-alligator_adv.jpg)
  
### Model Robustness with

  1. Pixel Dropout
  2. FGSM
  3. Random Noise
  4. Random Brightness

  
|        fgsm                    |           Gaussian Noise        | Pixel Dropout                  |           RandomBrightnesss |
|:------------------------------:|:-------------------------------:|:------------------------------:|:------------------------------:|
| ![](output_robustness/1-boat_fgsm.jpg) | ![](output_robustness/1-boat_Gaussnoise.jpg) | ![](output_robustness/1-boat_pixeldropout_.jpg) | ![](output_robustness/1-boat_RandomBrightness.jpg) |
| ![](output_robustness/2-car_fgsm.jpg) | ![](output_robustness/2-car_Gaussnoise.jpg) | ![](output_robustness/2-car_pixeldropout_.jpg) | ![](output_robustness/2-car_RandomBrightness.jpg) |
| ![](output_robustness/3-cat_fgsm.jpg) | ![](output_robustness/3-cat_Gaussnoise.jpg) | ![](output_robustness/3-cat_pixeldropout_.jpg) | ![](output_robustness/3-cat_RandomBrightness.jpg) |
| ![](output_robustness/4-dog_fgsm.jpg) | ![](output_robustness/4-dog_Gaussnoise.jpg) | ![](output_robustness/4-dog_pixeldropout_.jpg) | ![](output_robustness/4-dog_RandomBrightness.jpg) |
| ![](output_robustness/5-bus_fgsm.jpg) | ![](output_robustness/5-bus_Gaussnoise.jpg) | ![](output_robustness/5-bus_pixeldropout_.jpg) | ![](output_robustness/5-bus_RandomBrightness.jpg) |
| ![](output_robustness/6-truck_fgsm.jpg) | ![](output_robustness/6-truck_Gaussnoise.jpg) | ![](output_robustness/6-truck_pixeldropout_.jpg) | ![](output_robustness/6-truck_RandomBrightness.jpg) |
| ![](output_robustness/7-tiger_fgsm.jpg) | ![](output_robustness/7-tiger_Gaussnoise.jpg) | ![](output_robustness/7-tiger_pixeldropout_.jpg) | ![](output_robustness/7-tiger_RandomBrightness.jpg) |
| ![](output_robustness/8-shark_fgsm.jpg) | ![](output_robustness/8-shark_Gaussnoise.jpg) | ![](output_robustness/8-shark_pixeldropout_.jpg) | ![](output_robustness/8-shark_RandomBrightness.jpg) |
| ![](output_robustness/9-turtle_fgsm.jpg) | ![](output_robustness/9-turtle_Gaussnoise.jpg) | ![](output_robustness/9-turtle_pixeldropout_.jpg) | ![](output_robustness/9-turtle_RandomBrightness.jpg) |
| ![](output_robustness/10-alligator_fgsm.jpg) | ![](output_robustness/10-alligator_Gaussnoise.jpg) | ![](output_robustness/10-alligator_pixeldropout_.jpg) | ![](output_robustness/10-alligator_RandomBrightness.jpg) |


# [Link to source code](https://github.com/jai-mr/Sessions/tree/main/Session07/src)
