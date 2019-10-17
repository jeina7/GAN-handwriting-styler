[한글 버전](https://github.com/jeina7/Handwriting_styler/blob/master/readme.md)

# My Handwriting Styler, with GAN & Unet


　    




<p align="center"><img src="pngs/results_rigid_fonts.png" width="700"></p>
<p align="center">Left : Generated Image / Right : Ground Truth</p>




　    




## \# Introduction
This is a project that trains GAN-based model with the handwriting of human, and generates the character images that reflect those styles. Before learning human handwriting, it should be pre-trained on a large amount of about 75,000 digital font character images, and then transfer learning is done by small amounts of human handwritten data.


All details about this project can be seen in the [blog post](https://jeinalog.tistory.com/15). (in Korean)







　     


## \# Model Architecture
<p align="center"><img src="pngs/model.png" width="600"></p>

The basic model architecture is GAN, which consist of Generator and Discriminator.
- Generator gets Gothic type image for input, and do the style transfer with it. It has Encoder and Decoder, which is the different point with Vanilla GAN. Generator improves its result by evaluating its generated image by Discriminator.
- Discrinimator gets Real or Fake images, and calculate the probability of them to be the real image. It predicts the category of the font type as well.







　     




<p align="center"><img src="pngs/Unet_3d.png" width="600"></p>

It is 3D image of Encoder and Decoder. After the Encoder extracts features of image, the font category vector is concatenated at the end of the feature vector. Also, the middle-step vectors extracted by Encoder goes to the pair vectors which is decoded by Decoder. This architecture is [Unet](https://arxiv.org/abs/1505.04597).





　    




## \# Pre-Training
> Pre-Training processes are inspired and helped by [zi2zi](https://github.com/kaonashi-tyc/zi2zi) project of [kaonashi-tyc](https://github.com/kaonashi-tyc).



　    




<p align="center"><img src="gifz/old_training.gif" width="500"></p>
<p align="center">[Pre-Training] Data : 75,000 images / 150 epoch</p>







　     



Model trains 150epoch from the scratch.

- 1~30epoch : `L1_penalty=100`, `Lconst_penalty=15`
- 31~150epochh : `L1_penalty=500`, `Lconst_penalty=1000`


Until 30epoch, where is early stage yet, we give more weight to L1 loss to let the model learn overall shape first. After that, constant loss will be more weighted to make model learn more details to be sharp. Constant loss is introduced in [DTN](https://arxiv.org/abs/1611.02200).









　    


## \# Transfer Learning: Handwriting
<p align="center"><img src="gifz/jeina_training_imporv_1.gif" width="500"></p>
<p align="center">[Transfer Learning] Data : 210 images / 550 epoch</p>







　     



150epoch Pre-trained model now learns human handwriting. GIF shows the process of learning from 151epoch to 550epoch. It is lot more epochs, but it takes much shorter because of little amount of data.







　    




<p align="center"><img src="pngs/handwriting_ground_truth.png" width="500"></p>







　     



<p align="center"><img src="pngs/handwriting_generated.png" width="500"></p>







　     



The upper image is Ground Truth written by human, and the lower image is generated fake image.   
All 13 Korean characters written in image are not contained in 210-image training data set. It represents that model can generate unseen characters even if it has trained with only part of all Korean character set.




　    





## \# Interpolation
<p align="center"><img src="gifz/font_to_font_interpolation_short.gif" width="500"></p>


　    






Interpolation is the experiment to explore the latent space which model learned. It has introduced in [DCGAN](https://arxiv.org/abs/1511.06434). The GIF shows that there are middle-font between one type of font and another. It is the evidence that model trained the category vector space properly, not just 'memorizing' characters.








　    


## \# Codes
```
common
├── dataset.py    # load dataset, data pre-processing
├── function.py   # deep learning functions : conv2d, relu etc.
├── models.py     # Generator(Encoder, Decoder), Discriminator
├── train.py      # model Trainer
└── utils.py      # util functions

get_data
├── font2img.py   # font.ttf -> image
└── package.py    # .png -> .pkl
