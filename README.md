# Single Stage Common Object Detection

Official Pytorch implementation for the paper **"Single Stage Class Agnostic Common Object Detection"**.
This work is based on [MMDetection](https://github.com/open-mmlab/mmdetection) 1.1.0.


## Installation

```bash
conda env create -f environment.yml -n sscod
conda activate sscod

pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install -v -e .
```


## Training and Testing

Scripts for training and testing models are put in folder `scripts`:
```
scripts
├── coco
│   ├── test
│   │   ├── exp2_caseA_baseline.sh
│   │   ├── exp2_caseB_baseline.sh
│   │   └── exp2_caseB_curcon.sh
│   └── train
│       ├── exp2_caseA_baseline.sh
│       ├── exp2_caseB_baseline.sh
│       └── exp2_caseB_curcon.sh
└── voc
    ├── test
    │   ├── exp2_arcconneg.sh
    │   ├── exp2_arccon.sh
    │   ├── exp2_baseline.sh
    │   ├── exp2_curcon.sh
    │   └── exp2_focalcur.sh
    └── train
        ├── exp2_arcconneg.sh
        ├── exp2_arccon.sh
        ├── exp2_baseline.sh
        ├── exp2_curcon.sh
        └── exp2_focalcur.sh
```


## Quantitative results

#### VOC dataset

<p align="center">
    <img src="imgs/tables/table2.png" width="600">
</p>

<p align="center">
    <img src="imgs/tables/table3.png" width="600">
</p>

<p align="center">
    <img src="imgs/tables/table4.png" width="600">
</p>


#### COCO dataset

<p align="center">
    <img src="imgs/tables/Table5.png" width="600">
</p>
<p align="center">
    <img src="imgs/tables/Table6.png" width="600">
</p>


## Qualitative results

![image](imgs/000005_000047.jpg)
![image](imgs/000007_000083.jpg)
![image](imgs/000009_000050.jpg)
![image](imgs/000016_000023.jpg)
![image](imgs/000019_000077.jpg)
![image](imgs/000024_000042.jpg)
![image](imgs/000046_000064.jpg)
