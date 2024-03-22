# ComfyUI - Native DynamiCrafter
DynamiCrafter that works natively with ComfyUI's nodes, optimizations, and more.

![image](https://github.com/ExponentialML/ComfyUI_Native_DynamiCrafter/assets/59846140/fd1008ed-7660-454a-8253-1e032c9d054f)

|   |  |
| ------------- | ------------- |
| ![DynamiCrafter_00298](https://github.com/ExponentialML/ComfyUI_Native_DynamiCrafter/assets/59846140/e66a2559-b973-4a63-bc97-1a0701ab7dd3)  | ![DynamiCrafter_00327](https://github.com/ExponentialML/ComfyUI_Native_DynamiCrafter/assets/59846140/81b2b681-ef44-4966-8cb3-fa04692710a8)  |



> [!NOTE]  
> While this is still considered WIP (or beta), everything should be fully functional and adaptable to various workflows.

# Getting Started

Go to your `custom_nodes` directory in ComfyUI, and install by:

`git clone https://github.com/ExponentialML/ComfyUI_Native_DynamiCrafter.git`

# Installation

The pruned UNet checkpoints have been uploaded to HuggingFace. Each variant is working and fully functional.

https://huggingface.co/ExponentialML/DynamiCrafterUNet

## Instructions
You will also need a VAE, The CLIP model used with Stable Diffusion 2.1, and the Open CLIP Vision Model. All of the necessary model downloads are at that link.

If you aready have the base SD models, you do not need to download them (just use the CheckpointSimpleLoader without the model part).

Place the **DynamiCrafter** models inside `ComfyUI_Path/models/dynamicrafter_models`

If you are downloading the CLIP and VAE models separately, place them under their respective paths in the `ComfyUI_Path/models/` directory.

# Usage

- **model**: The loaded DynamiCrafter model.
  
- **clip_vision**: The CLIP Vision Checkpoint.
  
- **vae**: A Stable Diffusion VAE. If it works with < SD 2.1, it will work with this.
  
- **image_proj_model**: The Image Projection Model that is in the DynamiCrafter model file.
  
- **images**: The input images necessary for inference. If you are doing interpolation, you can simply batch two images together, check the toggle (see below), and everything will be handled automatically.
  
- **use_interpolation**: Use the interpolation mode with the interpolation model variant. You can interpolate any two frames (images), or predict the rest using one input.
  
- **fps**: Controls the speed of the video. If you're using a 256 based model, the highest plausible value is **4**
  
- **frames**: The amount of frames to use. If you're doing interpolation, the max is **16**. This is strictly enforced as it doesn't work properly (blurry results) if set higher.
  
- **model (output)**: The output into the a Sampler.
  
- **empty_latent**: An empty latent with the same size and frames as the processed ones.
  
- **latent_img**: If you're doing Img2Img based workflows, this is the necessary one to use.
  
> [!TIP]
> You don't have to use the latent outputs. As long as you use the same frame length (as your batch size) and same height and with as your image inputs, you can use your own latents.
> This means that you can experiment with inpainting and so on.

# TODO
- [x] Add various workflows.
- [ ] Ensure attention optimizations are working properly.
- [ ] Add autoregressive nodes (this may be a separate repository)
- [x] Add examples. (For more, [check here](https://github.com/Doubiiu/DynamiCrafter?tab=readme-ov-file#11-showcases-576x1024)).

# Credits

Thanks to @Doubiiu for for open sourcing [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter)https://github.com/Doubiiu/DynamiCrafter! Please support their work, and please follow any license terms they may uphold.
