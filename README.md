<div align="center">
  <img src="images/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_pulid</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_pulid">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_pulid">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_pulid/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_pulid.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Run PuLID, a tuning-free ID customization approach. It's is an ip-adapter alike method to restore facial identity. It uses insightface embedding, CLIP embedding and SDXL-Lightning for inferences in 4 steps. 


![output_1](https://raw.githubusercontent.com/Ikomia-hub/infer_pulid/main/images/output_1.jpg)

![illutration](https://private-user-images.githubusercontent.com/11482921/327072329-65610b0d-ba4f-4dc3-a74d-bd60f8f5ce37.jpeg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTU3NjIwODksIm5iZiI6MTcxNTc2MTc4OSwicGF0aCI6Ii8xMTQ4MjkyMS8zMjcwNzIzMjktNjU2MTBiMGQtYmE0Zi00ZGMzLWE3NGQtYmQ2MGY4ZjVjZTM3LmpwZWc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNTE1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDUxNVQwODI5NDlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0zZTAxZTNhMDJmNTJmZTUwMGVmNTNmNjliMzFhY2E5YWNhMjJjMmZiYWZkZTIyM2Q0MGI0YjAyYzgxNDMzYTBlJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.RlBFdYi8-iGmIhw6EF177HUMruadk7lznewRmEKhOcE)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_pulid", auto_connect=True)

# Run on your image  
wf.run_on(url="https://images.pexels.com/photos/4927360/pexels-photo-4927360.jpeg?cs=srgb&dl=pexels-anntarazevich-4927360.jpg&fm=jpg&w=1280&h=1920")

# Inpect your result
display(algo.get_output(0).get_image())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.
- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).
- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters
- **prompt** (str): Text prompt to guide the image generation.
- **negative_prompt** (str, *optional*) - default 'flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality,' \
            'artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands, ' \
            'low resolution, partially rendered objects,  deformed or partially rendered eyes, ' \
            'deformed, deformed eyeballs, cross-eyed, blurry'.
The prompt not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
- **num_inference_steps** (int) - default '4': Number of denoising steps. 
- **guidance_scale** (float) - default '1.2': Stable diffusion Scale for classifier-free guidance. Recommended between [1, 1.5]. 1 will be faster.
- **guidance_scale_id** (float) - default '0.8': ID guidance scale. Recommended between [0, 5].
- **seed** (int) - default '-1': Seed value. '-1' generates a random number between 0 and 1919655350.
- **num_images_per_prompt** (int) - default '1': Number of generated images. 
- **mode** (str) - default 'fidelity': Mode of the image generation 'fidelity' or 'extremely style'. We don't see much of a difference between the two.  
- **width** (int) - default '1024': Output width. If not divisible by 8 it will be automatically modified to a multiple of 8.
- **height** (int) - default '1024': Output height. If not divisible by 8 it will be automatically modified to a multiple of 8.
- **id_mix** (bool) - default 'False': If you want to mix two ID image, please turn this on, otherwise, turn this off.


```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Set main image
wf.set_image_input(
    url="https://images.pexels.com/photos/4927360/pexels-photo-4927360.jpeg?cs=srgb&dl=pexels-anntarazevich-4927360.jpg&fm=jpg&w=1280&h=1920",
    index=0
)

# Set additional image(s) [optional]
wf.set_image_input(
    url="https://images.pexels.com/photos/4927361/pexels-photo-4927361.jpeg?cs=srgb&dl=pexels-anntarazevich-4927361.jpg&fm=jpg&w=1280&h=1920",
    index=1
)

wf.set_image_input(
    url="https://images.pexels.com/photos/4927359/pexels-photo-4927359.jpeg?cs=srgb&dl=pexels-anntarazevich-4927359.jpg&fm=jpg&w=1280&h=1920",
    index=2
)

wf.set_image_input(
    url="https://images.pexels.com/photos/4927358/pexels-photo-4927358.jpeg?cs=srgb&dl=pexels-anntarazevich-4927358.jpg&fm=jpg&w=1280&h=1920",
    index=3
)

# Add algorithm
algo = wf.add_task(name="infer_pulid", auto_connect=False)

# Connect inputs
wf.connect_tasks(wf.root(), algo, [(0,0), (1,1), (2,2), (3,3)])

# Set parameters
algo.set_parameters({
    'prompt': 'portrait, color, cinematic, in garden, soft light, detailed face, wonderwoman costum, golden boomerang tiara, short hair',
    'guidance_scale': '1.2',
    'guidance_scale_id': '0.8',
    'num_inference_steps': '4',
    'seed': '-1',
    'width': '1024',
    'height': '1024',
    'mode': 'fidelity',
    'num_images_per_prompt':'2',
    'mix_id' : 'False'
    })

# Run your workflow
wf.run()

# Inpect your result
display(algo.get_output(0).get_image())
display(algo.get_output(1).get_image())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_pulid", auto_connect=True)

# Run on your image  
wf.run_on(url="https://images.pexels.com/photos/4927360/pexels-photo-4927360.jpeg?cs=srgb&dl=pexels-anntarazevich-4927360.jpg&fm=jpg&w=1280&h=1920")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

## :fast_forward: Troubleshooting 

If you encounter issues while installing Insightface on Windows, please follow this [guide](https://github.com/Gourieff/sd-webui-reactor#insightfacebuild).