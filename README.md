# PFNet_Plus Camouflaged Object Detection
> **Authors:** 
> [Kaney Nguyen](https://github.com/MagicKey23/) &
> [Martin Navarrete](https://github.com/mnavarrete12) 
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
