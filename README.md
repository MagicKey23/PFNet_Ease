# PFNet_Plus Camouflaged Object Detection
> **Authors:** 
> [Kaney Nguyen](https://github.com/MagicKey23/) &
> [Martin Navarrete](https://github.com/mnavarrete12) 
<div align="center">
<figure>
    <a href="./">
        <img src="./img/img4_original.png" width="79%"/>
    </a>
    <div class = "text-align:center">
    <figcaption>Figure 1 - Camouflaged with a pyramid</figcaption>
    </div>
</figure>

</div>


<div align="center">
<figure>
    <a href="./">
        <img src="./img/img4_GT.png" width="79%"/>
    </a>
    <div class = "text-align:center">
    <figcaption>Figure 2 - Ground Truth</figcaption>
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
PFNet_Plus
├── data
│   |   |
│   ├── Train
│   │   ├── image
│   │   ├── gts
│   │   |
│   ├── Test
│   │   ├── image
│   │   ├── gts
      .
      .
      .
</code></pre>



## Installation



``` shell
# clone the git hub
git clone PFNet_Plus
# go to code folder
cd PFNet_Plus
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


## Help

``` shell
  -h, --help                           show this help message and exit
  --img_size IMG_SIZE                  The size of the input image. Adjust the
                                       train scale by specifying a value, for
                                       example: 416, 704, or 1024. Default:
                                       704
  --batch_size BATCH_SIZE              The number of samples in each batch.
                                       Adjust as needed. Default: 32
  --epochs EPOCHS                      The number of epochs to train for.
                                       Adjust as needed. Default: 100
  --last_epoch LAST_EPOCH              The index of the last epoch completed.
                                       Default: 0
  --lr LR                              The learning rate for training.
                                       Default: 0.001
  --optimizer OPTIMIZER                The optimizer to use for training. For
                                       example: Adam. Default: SGD
  --weight_decay WEIGHT_DECAY          The weight decay for regularization
                                       during training. Default: 0.0005
  --snap_shot SNAP_SHOT                A snapshot of the training model to
                                       save. Leave blank if not needed.
                                       Default:
  --lr_decay LR_DECAY                  The learning rate decay factor.
  --momentum MOMENTUM                  The momentum for optimizer during
                                       training. Default: 0.9
  --poly_train POLY_TRAIN              Set to True for polynomial decay
                                       learning rate during training. Default:
                                       True
  --save_point SAVE_POINT              Epochs at which to save the model
                                       weights. Enter as a list, for example:
                                       [1,10,20,30,40,50]. Default: [1, 10,
                                       20, 30, 40, 50]
  --train_path TRAIN_PATH              The path to the training data. Default:
                                       train/
  --dataset_path DATASET_PATH          The path to the root of the dataset.
                                       Default: ./data/
  --exp_name EXP_NAME                  Set experiment name. Default: test1
  --project_name PROJECT_NAME          Set project name. Default: project1
  --ckpt_path CKPT_PATH                The location where you want to save
                                       your training result. Default: ./ckpt
  --test_path TEST_PATH                The path to the test data. Default:
                                       test/
  --result_path RESULT_PATH            The path to the results. Default:
                                       ./results/
  --load_weight LOAD_WEIGHT            The path to the pre-trained weight
                                       file. Enter in the format: Path +
                                       weight_name.pth. Default: ./best.pth
  --frame_scale FRAME_SCALE            The percentage by which to upscale or
                                       downscale the camera capture. Default:
                                       100
  --load_video LOAD_VIDEO              The path to the video file to load.
                                       Enter in the format: "./monkey.mp4".
                                       Default: videoname.mp4
  --select_camera SELECT_CAMERA        The index of the camera to use. Enter
                                       as an integer value from 0 to 4.
                                       Default: 0
  --display_accuracy DISPLAY_ACCURACY  Display TP, TN, FP, NP, Accuracy. Note:
                                       Required Proper Formatting to work.
                                       Default: False
  --display_area DISPLAY_AREA          Display area accuracy. Note: Required
                                       Proper Formatting to work. Default:
                                       False
  --device DEVICE                      The index of the camera to use for
                                       testing. For example: 0, 1, 2, or 3.
                                       Default: 0
  --save_video SAVE_VIDEO              Save result video Default: False
  --save_results SAVE_RESULTS          Save infer result Default: True
  --num_workers NUM_WORKERS            The number of worker threads to use.
                                       Default: 16

```

## Inference



On image:
``` shell
python inference.py
```


## Citation

```
@InProceedings{Mei_2021_CVPR,
    author    = {Mei, Haiyang and Ji, Ge-Peng and Wei, Ziqi and Yang, Xin and Wei, Xiaopeng and Fan, Deng-Ping},
    title     = {Camouflaged Object Segmentation With Distraction Mining},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {8772-8781}
}
```


## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/Mhaiyang/CVPR2021_PFNet)](https://github.com/AlexeyAB/darknet)](https://github.com/Mhaiyang/CVPR2021_PFNet)

</details>

## License

The source code is free for research and education use only.
