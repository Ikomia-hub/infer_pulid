import copy
import numpy as np
import torch
from ikomia import core, dataprocess, utils
import random

from infer_pulid.PuLID.pulid import attention_processor as attention
from infer_pulid.PuLID.pulid.pipeline import PuLIDPipeline
from infer_pulid.PuLID.pulid.utils import resize_numpy_image_long, seed_everything

# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferPulidParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.prompt = "portrait, color, cinematic, in garden, soft light, detailed face"
        self.guidance_scale = 1.2
        self.guidance_scale_id = 0.8
        self.negative_prompt = 'flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality,' \
            'artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands, ' \
            'low resolution, partially rendered objects,  deformed or partially rendered eyes, ' \
            'deformed, deformed eyeballs, cross-eyed, blurry'
        self.num_inference_steps = 4
        self.num_images_per_prompt = 1
        self.seed = -1
        self.width = 1024
        self.height = 1024
        self.mode = 'fidelity'
        self.id_mix = False
        self.update = False

    def set_values(self, params):
        self.prompt = params["prompt"]
        self.guidance_scale = float(params["guidance_scale"])
        self.guidance_scale_id = float(params["guidance_scale_id"])
        self.negative_prompt = params["negative_prompt"]
        self.seed = int(params["seed"])
        self.num_inference_steps = int(params["num_inference_steps"])
        self.num_images_per_prompt = int(params["num_images_per_prompt"])
        self.width = int(params["width"])
        self.height = int(params["height"])
        self.mode = str(params['mode'])
        self.id_mix = utils.strtobool(params['id_mix'])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia Studio or API
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["prompt"] = str(self.prompt)
        param_map["guidance_scale"] = str(self.guidance_scale)
        param_map["guidance_scale_id"] = str(self.guidance_scale_id)
        param_map["negative_prompt"] = str(self.negative_prompt)
        param_map["num_inference_steps"] = str(self.num_inference_steps)
        param_map["num_images_per_prompt"] = str(self.num_images_per_prompt)
        param_map["seed"] = str(self.seed)
        param_map["width"] = str(self.width)
        param_map["height"] = str(self.height)
        param_map["mode"] = str(self.mode)
        param_map["id_mix"] = str(self.id_mix)
        return param_map


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferPulid(core.CWorkflowTask):

    def __init__(self, name, param):
        core.CWorkflowTask.__init__(self, name)
        # Main image
        self.add_input(dataprocess.CImageIO())
        # Optional additional images
        self.add_input(dataprocess.CImageIO())
        self.add_input(dataprocess.CImageIO())
        self.add_input(dataprocess.CImageIO())

        # Create parameters object
        if param is None:
            self.set_param_object(InferPulidParam())
        else:
            self.set_param_object(copy.deepcopy(param))
        
        self.pipeline = None
        self.max_seed_value = 1919655350
        self.resize_long_edge = 1024

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def inference(self, img, supp_images):
        # Get parameters
        param = self.get_param_object()
        
        # Adjust attention settings based on mode
        if param.mode == 'fidelity':
            attention.NUM_ZERO = 8
            attention.ORTHO = False
            attention.ORTHO_v2 = True
        elif param.mode == 'extremely style':
            attention.NUM_ZERO = 16
            attention.ORTHO = True
            attention.ORTHO_v2 = False
        else:
            raise ValueError("Unsupported mode")

        # Process the ID images and resize them
        if img is not None:
            id_image = resize_numpy_image_long(img, 1024)
            id_embeddings = self.pipeline.get_id_embedding(id_image)
            if supp_images is not None:
                for supp_image in supp_images:
                    # Process supplementary images
                    supp_image = resize_numpy_image_long(supp_image, self.resize_long_edge)
                    supp_id_embeddings = self.pipeline.get_id_embedding(supp_image)
                    # Conditionally mix embeddings
                    id_embeddings = torch.cat(
                        (id_embeddings, supp_id_embeddings if param.id_mix else supp_id_embeddings[:, :5]), dim=1
                    )      
        else:
            id_embeddings = None

        # Set the random seed
        if param.seed == -1:
            seed = random.randint(0, self.max_seed_value)
        else:
            seed = param.seed
        seed_everything(seed)

        # Generate images
        ims = []
        for _ in range(param.num_images_per_prompt):
            img = self.pipeline.inference(
                param.prompt,
                (1, param.height, param.width),
                param.negative_prompt,
                id_embeddings,
                param.guidance_scale_id,
                param.guidance_scale,
                param.num_inference_steps
            )[0]

            ims.append(np.array(img))

        print(f"Prompt:\t{param.prompt}\nSeed:\t{seed}")
        return ims

    def run(self):
        # Main function of your algorithm
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get param
        param = self.get_param_object()  

       # Load pipelines
        if self.pipeline is None:
            self.pipeline = PuLIDPipeline()

        torch.set_grad_enabled(False)
   
        # Get the main input image
        image_in = self.get_input(0)
        image = image_in.get_image() if image_in.is_data_available() else None

        # Collect supplementary images from additional inputs
        supp_images = [self.get_input(i).get_image() for i in range(1, 4) if self.get_input(i).is_data_available()]
        
        # Edit output size
        if param.width % 8 != 0:
            param.width = param.width // 8 * 8
            print("Updating width to {} to be a multiple of 8".format(param.width))
        if param.height % 8 != 0:
            param.height = param.height // 8 * 8
            print("Updating height to {} to be a multiple of 8".format(param.height))

        # Execute inference with the main and supplementary images
        results = self.inference(image, supp_images if supp_images else None)            

        # Set image output
        for _ in self.get_outputs():
            self.remove_output(0)
        # Set image output  
        for i, image in enumerate(results):
                self.add_output(dataprocess.CImageIO())
                img = np.array(image)
                output = self.get_output(i)
                output.set_image(img)

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()
# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferPulidFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_pulid"
        self.info.short_description = "Pure and Lightning ID customization (PuLID) is a novel " \
                                        "tuning-free ID customization method for text-to-image generation."
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Diffusion"
        self.info.version = "1.0.0"
        # self.info.icon_path = "your path to a specific icon"
        self.info.authors = "Guo, Zinan and Wu, Yanze and Chen, Zhuowei and Chen, Lang and He, Qian"
        self.info.article = "PuLID: Pure and Lightning ID Customization via Contrastive Alignment"
        self.info.journal = "arXiv preprint"
        self.info.year = 2024
        self.info.license = "Apache License 2.0"
        # URL of documentation
        self.info.documentation_link = "https://arxiv.org/abs/2404.16022"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_pulid"
        self.info.original_repository = "https://github.com/ToTheBeginning/PuLID"
        # Python version
        self.info.min_python_version = "3.10.0"
        # Keywords used for search
        self.info.keywords = "Stable Diffusion, Hugging Face, text-to-image, Generative, ID Customization"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "IMAGE_GENERATION"

    def create(self, param=None):
        # Create algorithm object
        return InferPulid(self.info.name, param)
