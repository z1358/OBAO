# Official Implement of ECCV 2024 paper "Reshaping the Online Data Buffering and Organizing Mechanism for Continual Test-Time Adaptation".
## Preparation

Please create and activate the following conda envrionment.

```bash
# It may take several minutes for conda to solve the environment
conda update conda
conda env create -f environment.yml
conda activate ctta
```
We recommend referring to [this repository](https://github.com/mariodoebler/test-time-adaptation/tree/main?tab=readme-ov-file#classification) to obtain the required datasets and source domain models. After downloading, please modify `_C.DATA_DIR` in `conf.py` accordingly.

## Experiment Execution

### Classification Experiments

You can use the provided configuration files to run experiments. Simply execute the following Python script with the corresponding configuration file:

```
# Tested on RTX4090
CUDA_VISIBLE_DEVICES=0 python test_time.py --cfg cfgs/[cifar10_c/cifar100_c/imagenet_c]/[source/norm_test/norm_alpha/tent/cotta/rotta/santa/obao].yaml
```
Alternatively, you can execute the provided shell script:
```
bash run.sh
```

### Segmentation Experiments

We heavily rely on the Cotta code in segmentation experiments. Please follow the instructions provided in [cotta](https://github.com/qinenergy/cotta/blob/main/README.md#segmentation-experiments) to download the segmentation code and set up the environment.

Next, replace `./tools/our.py` and `./mmseg/apis/test.py` with the our.py and test.py files from the ./seg folder of this repository. Then, you can run the following command to perform the segmentation experiment:

```bash
CUDA_VISIBLE_DEVICES=0 python ./tools/our.py ./local_configs/segformer/B5/segformer.b5.1024x1024.acdc.160k.py segformer.b5.1024x1024.city.160k.pth --rounds 10
```

## Citation
Please cite our work if you find it useful.
```bibtex
@inproceedings{zhu2024reshaping,
  title={Reshaping the Online Data Buffering and Organizing Mechanism for Continual Test-Time Adaptation},
  author={Zhu, Zhilin and Hong, Xiaopeng and Ma, Zhiheng and Zhuang, Weijun and Ma, Yaohui and Dai, Yong and Wang, Yaowei},
  booktitle={European Conference on Computer Vision},
  pages={415--433},
  year={2024}
}
```

## Acknowledgement 
+ Online Test-time Adaptation code is heavily used. [official](https://github.com/mariodoebler/test-time-adaptation/tree/main) 
+ CoTTA [official](https://github.com/qinenergy/cotta)
+ Robustbench [official](https://github.com/RobustBench/robustbench) 
