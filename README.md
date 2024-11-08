# BrainRAM

Official code implementation of [BrainRAM: Cross-Modality Retrieval-Augmented Image Reconstruction from Human Brain Activity](https://dl.acm.org/doi/abs/10.1145/3664647.3681296) (ACM MM 2024).

### Installation

pip:

    pip install -r requirements.txt

conda:

    conda env create -f environment.yml

### Dataset preparation

1. Download NSD dataset at https://naturalscenesdataset.org/.

2. Put `nsddata`, `nsddata_betas`, and `nsddata_stimuli` from NSD and place them under the `nsd` directory.

3. Specify your `NSD_PATH` at `prepare_nsd/prepare_nsddata.py`. 

4. Run following commands to preprocess NSD data:

```
cd prepare_nsd/
python prepare_nsddata.py --sub [1,2,5,7]
python prepare_coco_embeds.py --sub [1,2,5,7]
```



### Training on stage 1

```
python Train_BrainRAM --model_name image_prior --prior_modality image
python Train_BrainRAM --model_name text_prior --prior_modality text
```



### Reconstruction

```
python Reconstructions.py --model_name output \
                          --image_model train_logs/image_prior/last.pth \
                          --text_model train_logs/text_prior/last.pth \
                          --recons_per_sample 4
```

