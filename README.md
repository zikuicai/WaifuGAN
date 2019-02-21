# AnimeGAN

This project is influenced by 

https://github.com/jayleicn/animeGAN

https://github.com/carpedm20/DCGAN-tensorflow


## Dataset:

We can use get_dataset script to download the following 4 datasets, 
- Anime-faces dataset 
- Celeb-A dataset
- LSUN dataset
- MNIST dataset

To download anime-faces dataset, run the command
```bash
 $ python get_dataset.py anime-faces
```

About [anime-faces dataset](https://drive.google.com/file/d/0B4wZXrs0DHMHMEl1ODVpMjRTWEk/view?usp=sharing):


It consists of 48548 pictures of anime faces where each picture has size 96x96.


If get more dataset
https://www.adidas.com/us/shoes?sort=top-sellers&start=0



Learning to Synthesize and Manipulate Natural Images
https://youtu.be/MkluiD2lYCc?t=1h16m58s

Images: from Unsplash get galaxy images with width 1080

randomly crop 400x400 patched in the original image

## GAN

### loss functions
cross entropy

$−(ylog(p)+(1−y)log(1−p))$

### optimizers

adam

rsmp?


![DCGAN](doc/DCGAN.png)
### discriminator


layers:

what's convolution and deconvolution



