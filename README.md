# Anime-DCGAN
### A DCGAN to generate anime faces 
### The model was trained on 2000 images for 5000 epochs
### The model was trained to generate images of size 64 * 64 with input noise of size 5 
![Example 1](examples/10000.png)<!-- --><br/>
![Example 2](examples/9236.png)<!-- --><br/>
![Example 3](examples/9435.png)<!-- --><br/>
### To train your DCGAN use the gan.ipynb 
### There are 500 images in images folder you can changes these images with any images of your choice with size 64 * 64 
### You can resize the images using rr.py
### Use commmand 
```python
python -d "directory of images to resize" -s "width height"
python -d D:/data/Anime-Gan/images/ -s 64 64
```
### you can change the size of input noise from 5 to any of your choice
```python
inputshape=5
```
### save the images in "images" folder
### Adjust the batch size according to you Gpu memory
### Generated images will be saved in folder "gen1" 
### Training checkpoints will be saved every 200 Epochs
### If in between the training your notebook crashes you can continue the training by restoring latest checkpoint
