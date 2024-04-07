<p align="center">
  <img align="left" src="./assets/gooddrag_icon.png" width="14%" /><h1 align="center">GoodDrag: Towards Good Practices for Drag Editing with Diffusion Models</h1>
  <p align="center">
    <strong>Zewei Zhang</strong>
    &nbsp;&nbsp;
    <strong>Huan Liu</strong>
    &nbsp;&nbsp;
    <a href="https://www.ece.mcmaster.ca/~junchen/"><strong>Jun Chen</strong></a>
	&nbsp;&nbsp;
    <a href="https://xuxy09.github.io/"><strong>Xiangyu Xu</strong></a>
  </p>


<div align="center">
    <img src="./assets/chess_1/original.jpg" width="20%" />
    <img src="./assets/chess_1/image_with_points.jpg" width="20%"/>
    <img src="./assets/chess_1/image_with_new_points.png" width="20%" />
    <img src="./assets/chess_1/trajectory.gif" width="20%" />
</div>

<div align="center">
    <img src="./assets/rabbit/original.jpg" width="20%" />
    <img src="./assets/rabbit/image_with_points.jpg" width="20%"/>
    <img src="./assets/rabbit/image_with_new_points.png" width="20%" />
    <img src="./assets/rabbit/trajectory.gif" width="20%" />
</div>

<div align="center">
    <img src="./assets/human_6/original.jpg" width="20%" />
    <img src="./assets/human_6/image_with_points.jpg" width="20%"/>
    <img src="./assets/human_6/image_with_new_points.png" width="20%" />
    <img src="./assets/human_6/trajectory.gif" width="20%" />
</div>


<div align="center">
    <img src="./assets/leopard/original.jpg" width="20%" />
    <img src="./assets/leopard/image_with_points.jpg" width="20%"/>
    <img src="./assets/leopard/image_with_new_points.png" width="20%" />
    <img src="./assets/leopard/trajectory.gif" width="20%" />
</div>


<div align="center">
    <img src="./assets/cat_2/original.jpg" width="20%" />
    <img src="./assets/cat_2/image_with_points.jpg" width="20%"/>
    <img src="./assets/cat_2/image_with_new_points.png" width="20%" />
    <img src="./assets/cat_2/trajectory.gif" width="20%" />
</div>

<div align="center">
    <img src="./assets/furniture_0/original.jpg" width="20%" />
    <img src="./assets/furniture_0/image_with_points.jpg" width="20%"/>
    <img src="./assets/furniture_0/image_with_new_points.png" width="20%" />
    <img src="./assets/furniture_0/trajectory.gif" width="20%" />
</div>
<p align="center">
<br>
<a href="https://colab.research.google.com/drive/1eoX-AngrcHocLKAgL5g5fyseLW_DusWl?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> 
</p>

## Start GoodDrag

Before getting started, please make sure your system is equipped with a CUDA-compatible GPU and Python 3.9 or higher. We provide three methods to directly run GoodDrag:
### 1️⃣ Automated Script for Effortless Start

- **Windows Users:** Double-click **webui.bat** to automatically set up your environment and launch the GoodDrag web UI.
- **Linux Users:** Run **webui.sh** for a similar one-step setup and launch process.

### 2️⃣ Manual Installation via pip
1.  Install the necessary dependencies:
	
	``` 
	pip install -r requirements.txt
	```
2. Launch the GoodDrag web UI:

	``` 
	python gooddrag_ui.py
	```

### 3️⃣ Start with Colab
For a quick and easy start, access GoodDrag directly through Google Colab. Click the badge below to open a pre-configured notebook that will guide you through using GoodDrag in the Colab environment: <a href="https://colab.research.google.com/drive/1eoX-AngrcHocLKAgL5g5fyseLW_DusWl?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> 

### Runtime and Memory Requirements
GoodDrag's efficiency depends on the image size and editing complexity. For a 512x512 image on an A100 GPU: the LoRA phase requires ~17 seconds, and drag editing spans 30 to 90 seconds. Memory needs are below 13GB for Linux and under 8GB for Windows.

## Parameter Description

We have predefined a set of parameters in the GoodDrag WebUI. Here are a few that you might consider adjusting based on actual results:
| Parameter Name | Description                                                  |
| -------------- | ------------------------------------------------------------ |
| Learning Rate  | Influences the speed of drag editing. Higher values lead to faster editing but may result in lower quality or instability. It is recommended to keep this value below 0.05. |
| Prompt         | The text prompt for the diffusion model. It is suggested to leave this empty. |
| End tiem step  | Specifies the length of the time step during the denoise phase of the diffusion model for drag editing. If good results are obtained early in the generated video, consider reducing this value. Conversely, if the drag editing is insufficient, increase it slightly. It is recommended to keep this value below 12. |
| Lambda         | Controls the consistency of the non-dragged regions with the original image. A higher value keeps the area outside the mask more in line with the original image. |

## Acknowledgement
Part of the code was modified based on [DragDiffusion](https://github.com/Yujun-Shi/DragDiffusion). Thanks for their great work!

## BibTeX