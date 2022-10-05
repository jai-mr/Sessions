**Repository github url : https://github.com/jai-mr/Sessions/tree/main/Session04 <br/>
**Assignment Repository : https://github.com/jai-mr/Sessions/blob/main/Session04/README.md <br/>
**Submitted by : Jaideep R - No Partners<br/>
**Registered email id : jaideepmr@gmail.com<br/>

## Objective

1. Add Demo Deployment for  trained model (scripted checkpoint) (CIFAR10 Model) & convert your model to TorchScript Scripted Model
2. It must accept image from user, and give the top 10 predictions
3. Streamlit has been used
4. The demo has been Dockerized
5. package the model (scripted only) inside the docker
6. docker image size limit is 1.2GB (uncompressed)
7. docker run -t -p 8080:8080 <image>:<tag> should start the webapp !
8. Add all the changes to your template repo, share link of the github repository
9. Push Image to DockerHub and submit link to it

### Make File

* [Github Link to Makefile](https://github.com/jai-mr/Sessions/blob/main/Session04/Makefile)
	
* Makefile Commands Used
	* Make build

* Command to invoke docker image for gradio prediction<br/>
	```
	docker run -it -p 8080:8080 jaideepmr/emlo2_s4:1.0
	```
* Output of Docker image creation and build<br/>
	<img src="images/docimg1.png" width="800"/>

* Output of Docker execution for gradio<br/>
	<img src="images/docimg11.png" width="800"/>  
	<img src="images/docimg2.png" width="800"/>
	<img src="images/docimg3.png" width="800"/>
  <img src="images/docimg4.png" width="800"/>

 * Docker Image size created after Makefile execution - 1.15GB
	
	<img src="images/docimg5.png" width="800"/>
	
* [Training Log - Click to view](https://github.com/jai-mr/Sessions/blob/main/Session04/logs/log.md)
	
* Training metrics
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test/acc          │    0.8084999918937683     │
│         test/loss         │    0.8463168144226074     │
└───────────────────────────┴───────────────────────────┘
```

### Training & Validation 
* 	[Jupyter Notebook Link](https://github.com/jai-mr/Sessions/blob/main/Session04/S4_Asgn.ipynb)
