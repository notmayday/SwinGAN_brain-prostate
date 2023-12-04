# SwinGAN

Official PyTorch implementation of SwinGAN described in the paper "Exploring SwinGAN for Accelerated MRI Reconstruction: Remarkable Noise Suppression".

<div align="center">
<img src="./asserts/framework.png" width="800px">
</div>

## Dependencies

```
python>=3.6.9
torch>=1.7.1
torchvision>=0.8.2
cuda=>10.1
Pillow == 6.2.1
PyYAML == 5.3.1
h5py == 2.10.0
ipython == 6.0.0
matplotlib == 3.1.1
numpy == 1.17.3
scipy == 0.19.1
tqdm == 4.37.0
nibabel == 2.2.1
ninja
```

## Installation
- Clone this repo:
```bash
git clone https://github.com/learnerzx/SwinGAN
cd SwinGAN
```

## Train

<br />

```
python3 train.py 

```


## Test

<br />

python3 difference_poisson.py 

<br />
<br />


# Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.


<br />

# Acknowledgements

This code uses libraries from [KIGAN], [SwinTransformer],[PatchGAN],[SwinGAN] repositories.
