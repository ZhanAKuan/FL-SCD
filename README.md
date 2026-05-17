# FL-SCD

## Getting Started

To train on other datasets or splits, please modify
``dataset`` and ``split`` in [train.sh](https://github.com/LiheYoung/UniMatch/blob/main/more-scenarios/remote-sensing/scripts/train.sh).

Before starting training, you need to generate your own splits. The format of the splits is as follows:
```
├── [Your splits Path]
    ├── 5\
        └── ├──labeled.txt
            ├──unlabeled.txt
    ├── 10\
        └── ├──labeled.txt
            ├──unlabeled.txt
    ├── 20\
        └── ├──labeled.txt
            ├──unlabeled.txt
    ├── 40\
        └── ├──labeled.txt
            ├──unlabeled.txt
    ├──val.txt
    ├──test.txt
```

### Dataset

The LEVIR Building CD Dataset is openly available at https://justchenhao.github.io/LEVIR/.

The WHU Building Dataset is openly available at http://gpcv.whu.edu.cn/data/building_dataset.html.

The Google-GZ dataset is openly available at https://github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery.

The authors acknowledge the authors who provided the publicly available change detection datasets
