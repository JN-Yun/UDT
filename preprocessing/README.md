<!-- <h1 align="center"> Preprocessing Guide </h1> -->

### This guide outlines the preprocessing pipeline, which is based on the code from [edm2](https://github.com/NVlabs/edm2) and a modified version of [REPA](https://github.com/sihyun-yu/REPA/tree/main/preprocessing). 

### 1. Dataset Download
You can use the following commands to download the dataset directly. Please replace `[DATA_PATH]` with your desired directory path.

```bash
mkdir [DATA_PATH]
cd [DATA_PATH]
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
```

```bash
# Run imagenet_tar.py to extract the tar archive using multiprocessing. 
# This will reduce the extraction time.
python imagenet_tar.py tar_path="[DATA_PATH]/ILSVRC2012_img_train.tar" extract_root="[DATA_PATH]/train_raw"
```

### 2. Resize to Desired Resolution (256x256 or 512x512)
After downloading and extracting ImageNet, run the following script (adapted from EDM2 and REPA) to crop and resize the images. By default, this will save the output in folders.

```bash
RESOLUTION=256x256     # 512x512 
python dataset_tools.py convert --source=[DATA_PATH]/train_raw \
    --dest=[DATA_PATH]/images --resolution=$RESOLUTION --transform=center-crop-dhariwal
```

### 3. Convert Pixels to VAE Latents
Extract the VAE latents from the resized images using the following command:

```bash
python dataset_tools.py encode --source=[DATA_PATH]/images \
    --dest=[DATA_PATH]/vae-sd
```

> **Note:** For your convenience, we provide a pre-processed `vae-sd.zip` (256x256 resolution). You can download it **[here](https://www.dropbox.com/scl/fo/83mky599uo7joa7kc9wvl/AH35N-yt7fi6jLKAWToor2E?rlkey=tzastx7fs0u9ok789d6j4gj6j&dl=0)**.  
> If you want to load the latents directly from this zip file without extracting it, please replace your existing dataset loader with the `CustomDataset_zip` class found in the `dataset.py` file included at the link above.

---

## Acknowledgement

This code is largely built upon the [edm2](https://github.com/NVlabs/edm2) and [REPA](https://github.com/sihyun-yu/REPA/tree/main/preprocessing) repositories.