# Text-to-feature diffusion for audio-visual few-shot learning
## [Paper](https://arxiv.org/abs/2309.03869) | [Project Page](https://www.eml-unitue.de/publication/audio-visual-fsl)


This repository is the official implementation of [Text-to-feature diffusion for audio-visual few-shot
learning](https://arxiv.org/abs/2309.03869).

<img src="/img/AVDiff.png" width="700" height="700">

## Requirements
Install all required dependencies into a new virtual environment via conda.
```shell
conda env create -f AVDiff.yml
```
## Dataset

We base our datasets on the [TCAF repository](https://github.com/ExplainableML/TCAF-GZSL/). The features stored in the desired format can be found at the links below:

You can download our features for all three datasets here:
* [VGGSound-GFSL](https://mlcloud.uni-tuebingen.de:7443/akata0/omercea19/av-diff/VGGSound.zip) 
* [UCF-GFSL](https://mlcloud.uni-tuebingen.de:7443/akata0/omercea19/av-diff/UCF.zip)
* [ActivityNet-GFSL](https://mlcloud.uni-tuebingen.de:7443/akata0/omercea19/av-diff/ActivityNet.zip)

The features should be placed inside the ```avgzsl_benchmark_non_averaged_datasets/w2v_embedding``` folder which should be located in the current project:
```shell
unzip [DATASET].zip -d avgzsl_benchmark_non_averaged_datasets/w2v_embedding
```


# Training and Evaluation
In order to train the model run the following command:
```python3 main.py --cfg CFG_FILE  --root_dir ROOT_DIR --log_dir LOG_DIR --dataset_name DATASET_NAME --run all```

```
arguments:
--cfg CFG_FILE is the file containing all the hyperparameters for the experiments. These can be found in ```config/best/X-shot/best_Y.yaml``` where Y indicates the dataset and X indicate the number of shots.
--root_dir ROOT_DIR indicates the location where the dataset is stored.
--dataset_name {VGGSound, UCF, ActivityNet} indicate the name of the dataset.
--log_dir LOG_DIR indicates where to save the experiments.
--run {'all', 'stage-1', 'stage-2'}. 'all' indicates to run both training stages + evaluation, whereas 'stage-1', 'stage-2' indicates to run only those particular training stages
```

This script will train the model and evaluate it at the end.




# Model weights
The trained models can be downloaded from [here](https://mlcloud.uni-tuebingen.de:7443/akata0/omercea19/av-diff/models.zip).

# Results

### GFSL performance on VGGSound-GFSL, UCF-GFSL, ActivityNet-GFSL

| Method           | VGGSound-GFSL 1/5/10-shots    | UCF-GFSL 1/5/10-shots         | ActivityNet-GFSL 1/5/10-shots |
|------------------|-------------------------------|-------------------------------|-------------------------------|
| Attention fusion | 15.46/28.22/30.73             | 37.39/51.68/57.91             | 4.35/6.17/10.67               |
| Perceiver        | 17.97/29.92/33.65             | 44.12/48.60/55.33             | 17.34/25.75/29.88             |
| MBT              | 14.70/27.26/30.12             | 39.65/46.55/50.04             | 14.26/23.26/26.86             |
| TCAF             | 19.54/26.09/28.95             | 44.61/46.29/54.19             | 16.50/22.79/24.78             |
| ProtoGan         | 10.74/25.17/29.85             | 37.95/42.42/51.01             | 2.77/2.67/4.05                |
| SLDG             | 16.83/20.79/24.11             | 39.92/36.47/34.31             | 13.57/22.29/27.81             |
| TSL              | 18.73/19.49/21.93             | 44.51/51.08/60.93             | 9.53/10.97/10.39              |
| HiP              | 19.27/26.82/29.25             | 21.79/36.44/50.69             | 13.80/18.10/19.37             |
| Zorro            | 18.88/29.56/32.06             | 44.35/51.86/58.89             | 14.56/23.14/27.35             |
| AVCA             | 6.29/15.98/18.08              | 43.61/49.19/50.53             | 12.83/20.09/26.02             |
| **AV-DIFF**      | **20.31**/**31.19**/**33.99** | **51.50**/**59.96**/**64.18** | **18.47**/**26.96**/**30.86** |


### FSL performance on VGGSound-GFSL, UCF-GFSL, ActivityNet-GFSL

| Method             | VGGSound-GFSL 1/5/10-shots    | UCF-GFSL 1/5/10-shots         | ActivityNet-GFSL 1/5/10-shots |
|--------------------|-------------------------------|-------------------------------|-------------------------------|
| Attention fusion | 16.37/31.57/39.02             | 36.88/47.18/52.19             | 5.82/8.13/10.78               |
| Perceiver        | 18.51/33.58/40.73             | 33.73/40.47/47.86             | 12.53/21.50/26.46             |
| MBT              | 21.96/34.95/38.93             | 27.99/34.53/39.73             | 12.63/22.38/26.03             |
| TCAF             | 20.01/32.22/36.43             | 35.90/37.39/47.61             | 13.01/21.81/23.33             |
| ProtoGan         | 14.08/28.87/34.80             | 28.08/33.63/40.68             | 4.40/7.81/8.81                |
| SLDG             | 17.57/25.17/29.48             | 28.91/28.56/26.96             | 10.30/19.16/25.35             |
| TSL              | 22.44/29.50/31.29             | 35.17/42.42/55.63             | 10.77/12.77/12.18             |
| HiP              | 18.64/30.67/35.13             | 34.88/42.23/43.29             | 10.31/16.25/17.06             |
| Zorro            | 21.79/35.17/40.66             | 34.52/42.59/49.06             | 11.94/21.94/26.33             |
| AVCA             | 10.29/20.50/28.27             | 31.24/36.70/39.17             | 12.22/21.65/26.76             |
| **AV-DIFF**      | **22.95**/**36.56**/**41.39** | **39.89**/**51.45**/**57.39** | **13.80**/**23.00**/**27.81** |

# Project structure
```src``` - Contains the code used throughout the project for dataloaders/models/training/testing.   
```config``` - Contains the configuration files used for training/testing.
# References

If you find this code useful, please consider citing:
```
@inproceedings{mercea2023avdiff,
  author    = {Mercea, Otniel-Bogdan and Hummel, Thomas and Koepke, A. Sophia and Akata, Zeynep},
  title     = {Text-to-feature diffusion for audio-visual few-shot learning},
  booktitle = {DAGM GCPR},
  year      = {2023}
}
```
```
@inproceedings{mercea2022tcaf,
  author    = {Mercea, Otniel-Bogdan and Hummel, Thomas and Koepke, A. Sophia and Akata, Zeynep},
  title     = {Temporal and cross-modal attention for audio-visual zero-shot learning},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022}
}
```
```
@inproceedings{mercea2022avca,
  author    = {Mercea, Otniel-Bogdan and Riesch, Lukas and Koepke, A. Sophia and Akata, Zeynep},
  title     = {Audio-visual Generalised Zero-shot Learning with Cross-modal Attention and Language},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
```

