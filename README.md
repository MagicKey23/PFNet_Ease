# PFNet_Plus Camouflaged Object Detection
> **Authors:** 
> [Kaney Nguyen](https://github.com/MagicKey23/) &
> [Martin Navarrete]() 
<div align="center">
<figure>
    <a href="./">
        <img src="./imgs/camouflaged_sample.jpg" width="79%"/>
    </a>
    <div class = "text-align:center">
    <figcaption>Figure 1 - Camouflaged contained human head example</figcaption>
    </div>
</figure>

</div>

## Introduction
- Coming Soon
## Video Demo

- Coming Soon

## Use Case
- Coming Soon

## File Structure

<pre><code>
Yolo-Sinet
├── weights
|   ├──YoloV7
|   |  ├── best.pt
|   ├──Sinet
|   |  ├── best.pt
├── data
│   ├── Custom_Data_Name
│   │   ├── Train
│   │   │   ├── images
│   │   |   ├── gts
│   │   │   ├── labels
│   │   ├── Test
│   │   │   ├── images
│   │   │   ├── gts
|   |   |   ├── labels
│   |   |── Detect
│   │   |   ├── images
├── classification
|   ├── Custom_Data_Name
|   |   ├──found_object.txt
|   |   ├──not_found.txt
├── image_bbox
|   ├── Custom_Data_Name
|   |   ├── output_image_with_bbox.png
├── masks   
|   ├── Custom_Data_Name
|   |   ├── output_masks.png
├── test_output
|   ├── Custom_Data_Name
|   |   ├── test_output.png
├── train_output
|   ├── Custom_Data_Name
|   |   ├── YoloV7
|   |   |   ├── output.pt
|   |   ├── Sinet
|   |   |   ├── output.pt
|   |   |   |
      .
      .
      .
</code></pre>



## Installation



``` shell
# clone the git hub
git clone yolo-sinet
# go to code folder
cd yolo-net
# apt install required packages
pip install -r requirements
```

</details>

## Testing

Currently, refactoring the code

``` shell
python test.py # update soon
```

## Training

Currently, refactoring the code 

Data preparation

``` shell
python train.py # update soon
```


## Transfer learning

``` shell

```

## Inference



On image:
``` shell
python inference.py
```


## Citation

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```

```
@article{wang2022designing,
  title={Designing Network Design Strategies Through Gradient Path Analysis},
  author={Wang, Chien-Yao and Liao, Hong-Yuan Mark and Yeh, I-Hau},
  journal={arXiv preprint arXiv:2211.04800},
  year={2022}
}
```
```
 
    @article{fan2021concealed,  
     author={Fan, Deng-Ping and Ji, Ge-Peng and Cheng, Ming-Ming and Shao, Ling},  
     title={Concealed Object Detection},   
     journal={IEEE TPAMI}, 
     year={2022},  
     volume={44},  
     number={10},  
     pages={6024-6042},  
     doi={10.1109/TPAMI.2021.3085766}
    }
    
    @inproceedings{fan2020camouflaged,
      title={Camouflaged object detection},
      author={Fan, Deng-Ping and Ji, Ge-Peng and Sun, Guolei and Cheng, Ming-Ming and Shen, Jianbing and Shao, Ling},
      booktitle={IEEE CVPR},
      pages={2777--2787},
      year={2020}
    }
```


## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)
* [https://github.com/GewelsJI/SINet-V2](https://github.com/GewelsJI/SINet-V2)
</details>

## License

The source code is free for research and education use only.
