"""
Model worker for Hunyuan3D API server.
"""
import os
import time
import uuid
import base64
import trimesh
from io import BytesIO
from PIL import Image
import torch

# Apply torchvision compatibility fix before other imports
import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.utils import logger


def load_image_from_base64(image):
    """
    Load an image from base64 encoded string.
    
    Args:
        image (str): Base64 encoded image string
        
    Returns:
        PIL.Image: Loaded image
    """
    return Image.open(BytesIO(base64.b64decode(image)))


class ModelWorker:
    """
    Worker class for handling 3D model generation tasks.
    """
    
    def __init__(self,
                 model_path='tencent/Hunyuan3D-2.1',
                 subfolder='hunyuan3d-dit-v2-1',
                 device='cuda',
                 low_vram_mode=False,
                 worker_id=None,
                 model_semaphore=None,
                 save_dir='gradio_cache'):
        """
        Initialize the model worker.
        
        Args:
            model_path (str): Path to the shape generation model
            subfolder (str): Subfolder containing the model files
            device (str): Device to run the model on ('cuda' or 'cpu')
            low_vram_mode (bool): Whether to use low VRAM mode
            worker_id (str): Unique identifier for this worker
            model_semaphore: Semaphore for controlling model concurrency
            save_dir (str): Directory to save generated files
        """
        self.model_path = model_path
        self.worker_id = worker_id or str(uuid.uuid4())[:6]
        self.device = device
        self.low_vram_mode = low_vram_mode
        self.model_semaphore = model_semaphore
        self.save_dir = save_dir
        
        logger.info(f"Loading the model {model_path} on worker {self.worker_id} ...")

        # Initialize background remover
        self.rembg = BackgroundRemover()
        
        # Initialize shape generation pipeline (matching demo.py)
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
            
    def get_queue_length(self):
        """
        Get the current queue length for model processing.
        
        Returns:
            int: Number of tasks in the queue
        """
        if self.model_semaphore is None:
            return 0
        else:
            return (self.model_semaphore._value if hasattr(self.model_semaphore, '_value') else 0) + \
                   (len(self.model_semaphore._waiters) if hasattr(self.model_semaphore, '_waiters') and self.model_semaphore._waiters is not None else 0)

    def get_status(self):
        """
        Get the current status of the worker.
        
        Returns:
            dict: Status information including speed and queue length
        """
        return {
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate(self, uid, params):
        """
        Generate a 3D model from the given parameters.
        
        Args:
            uid: Unique identifier for this generation task
            params (dict): Generation parameters including image and options
            
        Returns:
            tuple: (file_path, uid) - Path to generated file and task ID
        """
        start_time = time.time()
        
        # Handle input image
        if 'image' in params:
            image = params["image"]
            image = load_image_from_base64(image)
        else:
            raise ValueError("No input image provided")

        # Convert to RGBA and remove background if needed (matching demo.py)
        image = image.convert("RGBA")
        if image.mode == "RGB":
            image = self.rembg(image)

        # Generate mesh using the same simple approach as demo.py
        try:
            mesh = self.pipeline(image=image)[0]
            logger.info("---Shape generation takes %s seconds ---" % (time.time() - start_time))
        except Exception as e:
            logger.error(f"Shape generation failed: {e}")
            raise ValueError(f"Failed to generate 3D mesh: {str(e)}")

        # Export final mesh without texture
        file_type = params.get('type', 'glb')
        save_path = os.path.join(self.save_dir, f'{str(uid)}.{file_type}')
        mesh.export(save_path)

        if self.low_vram_mode:
            torch.cuda.empty_cache()
            
        logger.info("---Total generation takes %s seconds ---" % (time.time() - start_time))
        return save_path, uid 