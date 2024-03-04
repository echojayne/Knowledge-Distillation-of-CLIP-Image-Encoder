# Knowledge-Distillation-of-CLIP-Image-Encoder

Distillate the CLIP Image Encoder to a ViT Network

# System Architecture

![System architecture](./models/Arch.png "System architecture")

The overall architecture is shown in the figure above. The image encoder of CLIP is distilled into a small ViT architecture. Here, the features before the classification layer are distilled, using the MSE loss. The teacher and student models can each train their own classification networks.

# Usage

```bash
python3 main.py --batch-size 3000,1024 --epochs 18,200 --lr 0.01,0.001 --device cuda
```

The first parameter of batch_size, epoch, and lr is used for training the classifier, while the second parameter is used for distilling the student model.

If you want to change the architecture of student model, you can modify the vit_parameters under the models folder, the default is :

```python
vit_parameters = {
    'image_resolution': 224,
    'vision_patch_size': 32,
    'vision_width': 384,
    'vision_layers': 9,
    'vision_heads': 384//16,
    'embed_dim': 512
}
```
