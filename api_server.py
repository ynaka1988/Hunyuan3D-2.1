# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

"""
A model worker executes the model.
"""
import argparse
import asyncio
import base64
import logging
import logging.handlers
import os
import sys
import tempfile
import threading
import traceback
import uuid
import time
from io import BytesIO

# Apply torchvision compatibility fix before other imports
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

import torch
import trimesh
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse

# Updated imports to match gradio_app.py
from hy3dshape import FaceReducer, FloaterRemover, DegenerateFaceRemover, MeshSimplifier, \
    Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.pipelines import export_to_trimesh
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.utils import logger

# Texture generation imports
try:
    from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
    from hy3dpaint.convert_utils import create_glb_with_pbr_materials
    HAS_TEXTUREGEN = True
except ImportError:
    print("Warning: Texture generation not available")
    HAS_TEXTUREGEN = False

LOGDIR = '.'

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


SAVE_DIR = 'gradio_cache'
os.makedirs(SAVE_DIR, exist_ok=True)

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("controller", f"{SAVE_DIR}/controller.log")


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def export_mesh(mesh, save_folder, textured=False, type='glb'):
    """
    Export a mesh to a file in the specified folder, optionally including textures.

    Args:
        mesh (trimesh.Trimesh): The mesh object to export.
        save_folder (str): Directory path where the mesh file will be saved.
        textured (bool, optional): Whether to include textures/normals in the export. Defaults to False.
        type (str, optional): File format to export ('glb' or 'obj' supported). Defaults to 'glb'.

    Returns:
        str: The full path to the exported mesh file.
    """
    if textured:
        path = os.path.join(save_folder, f'textured_mesh.{type}')
    else:
        path = os.path.join(save_folder, f'white_mesh.{type}')
    if type not in ['glb', 'obj']:
        mesh.export(path)
    else:
        mesh.export(path, include_normals=textured)
    return path


class ModelWorker:
    def __init__(self,
                 model_path='tencent/Hunyuan3D-2.1',
                 tex_model_path='tencent/Hunyuan3D-2.1',
                 subfolder='hunyuan3d-dit-v2-1',
                 device='cuda',
                 enable_tex=False,
                 low_vram_mode=False):
        self.model_path = model_path
        self.worker_id = worker_id
        self.device = device
        self.low_vram_mode = low_vram_mode
        logger.info(f"Loading the model {model_path} on worker {worker_id} ...")

        # Initialize background remover
        self.rembg = BackgroundRemover()
        
        # Initialize shape generation pipeline
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            use_safetensors=False,
            device=device,
        )
            
        # Initialize texture generation pipeline if enabled
        if enable_tex and HAS_TEXTUREGEN:
            try:
                conf = Hunyuan3DPaintConfig(max_num_view=8, resolution=768)
                conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
                conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
                conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
                self.pipeline_tex = Hunyuan3DPaintPipeline(conf)
            except Exception as e:
                logger.error(f"Failed to initialize texture pipeline: {e}")
                self.pipeline_tex = None
        else:
            self.pipeline_tex = None
            
        # Initialize mesh processing workers
        self.floater_remove_worker = FloaterRemover()
        self.degenerate_face_remove_worker = DegenerateFaceRemover()
        self.face_reduce_worker = FaceReducer()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate(self, uid, params):
        start_time = time.time()
        
        # Handle input image
        if 'image' in params:
            image = params["image"]
            image = load_image_from_base64(image)
        else:
            raise ValueError("No input image provided")

        # Remove background if needed
        if params.get('remove_background', True) or image.mode == "RGB":
            image = self.rembg(image.convert('RGB'))

        # Handle existing mesh or generate new one
        if 'mesh' in params:
            mesh = trimesh.load(BytesIO(base64.b64decode(params["mesh"])), file_type='glb')
        else:
            # Generate new mesh
            seed = params.get("seed", 1234)
            generator = torch.Generator(self.device).manual_seed(seed)
            octree_resolution = params.get("octree_resolution", 256)
            num_inference_steps = params.get("num_inference_steps", 5)
            guidance_scale = params.get('guidance_scale', 5.0)
            num_chunks = params.get('num_chunks', 8000)
            
            outputs = self.pipeline(
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                octree_resolution=octree_resolution,
                num_chunks=num_chunks,
                output_type='mesh'
            )
            
            mesh = export_to_trimesh(outputs)[0]
            logger.info("---Shape generation takes %s seconds ---" % (time.time() - start_time))

        # Apply texture if requested
        if params.get('texture', False) and self.pipeline_tex is not None:
            # Post-process mesh for texture generation
            mesh = self.floater_remove_worker(mesh)
            mesh = self.degenerate_face_remove_worker(mesh)
            mesh = self.face_reduce_worker(mesh, max_facenum=params.get('face_count', 40000))
            
            # Generate texture
            tex_start_time = time.time()
            temp_obj_path = os.path.join(SAVE_DIR, f'{str(uid)}_temp.obj')
            mesh.export(temp_obj_path)
            
            text_path = os.path.join(SAVE_DIR, f'{str(uid)}_textured.obj')
            self.pipeline_tex(mesh_path=temp_obj_path, 
                             image_path=image, 
                             output_mesh_path=text_path, 
                             save_glb=False)
            logger.info("---Texture generation takes %s seconds ---" % (time.time() - tex_start_time))
            
            # Convert to GLB with PBR materials if requested
            file_type = params.get('type', 'glb')
            if file_type == 'glb':
                glb_path = os.path.join(SAVE_DIR, f'{str(uid)}.glb')
                # Create texture paths (these would be generated by the texture pipeline)
                textures = {
                    'albedo': text_path.replace('.obj', '_albedo.png'),
                    'metallic': text_path.replace('.obj', '_metallic.png'),
                    'roughness': text_path.replace('.obj', '_roughness.jpg')
                }
                try:
                    create_glb_with_pbr_materials(text_path, textures, glb_path)
                    save_path = glb_path
                except Exception as e:
                    logger.warning(f"Failed to create PBR GLB, using regular export: {e}")
                    mesh = trimesh.load(text_path)
                    mesh.export(save_path)
            else:
                # Load textured mesh for other formats
                mesh = trimesh.load(text_path)
                mesh.export(save_path)
        else:
            # Export final mesh without texture
            file_type = params.get('type', 'glb')
            save_path = os.path.join(SAVE_DIR, f'{str(uid)}.{file_type}')
            mesh.export(save_path)

        if self.low_vram_mode:
            torch.cuda.empty_cache()
            
        logger.info("---Total generation takes %s seconds ---" % (time.time() - start_time))
        return save_path, uid


app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 你可以指定允许的来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)


@app.post("/generate")
async def generate(request: Request):
    logger.info("Worker generating...")
    try:
        params = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse JSON request: {e}")
        return JSONResponse({"error": "Invalid JSON request"}, status_code=400)
    
    # Validate required parameters
    if not params.get('image'):
        return JSONResponse({"error": "Image parameter is required"}, status_code=400)
    
    uid = uuid.uuid4()
    try:
        file_path, uid = worker.generate(uid, params)
        return FileResponse(file_path)
    except ValueError as e:
        traceback.print_exc()
        logger.error(f"Caught ValueError: {e}")
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)
    except torch.cuda.CudaError as e:
        logger.error(f"Caught torch.cuda.CudaError: {e}")
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)
    except Exception as e:
        logger.error(f"Caught Unknown Error: {e}")
        traceback.print_exc()
        ret = {
            "text": server_error_msg,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)


@app.post("/send")
async def generate(request: Request):
    logger.info("Worker send...")
    try:
        params = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse JSON request: {e}")
        return JSONResponse({"error": "Invalid JSON request"}, status_code=400)
    
    # Validate required parameters
    if not params.get('image'):
        return JSONResponse({"error": "Image parameter is required"}, status_code=400)
    
    uid = uuid.uuid4()
    try:
        threading.Thread(target=worker.generate, args=(uid, params,)).start()
        ret = {"uid": str(uid)}
        return JSONResponse(ret, status_code=200)
    except Exception as e:
        logger.error(f"Failed to start generation thread: {e}")
        ret = {"error": "Failed to start generation"}
        return JSONResponse(ret, status_code=500)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({"status": "healthy", "worker_id": worker_id}, status_code=200)


@app.get("/status/{uid}")
async def status(uid: str):
    save_file_path = os.path.join(SAVE_DIR, f'{uid}.glb')
    print(save_file_path, os.path.exists(save_file_path))
    if not os.path.exists(save_file_path):
        response = {'status': 'processing'}
        return JSONResponse(response, status_code=200)
    else:
        try:
            base64_str = base64.b64encode(open(save_file_path, 'rb').read()).decode()
            response = {'status': 'completed', 'model_base64': base64_str}
            return JSONResponse(response, status_code=200)
        except Exception as e:
            logger.error(f"Error reading file {save_file_path}: {e}")
            response = {'status': 'error', 'message': 'Failed to read generated file'}
            return JSONResponse(response, status_code=500)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2.1')
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-1')
    parser.add_argument("--tex_model_path", type=str, default='tencent/Hunyuan3D-2.1')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument('--enable_tex', action='store_true')
    parser.add_argument('--low_vram_mode', action='store_true')
    parser.add_argument('--cache-path', type=str, default='./gradio_cache')
    parser.add_argument('--mc_algo', type=str, default='mc')
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    logger.info(f"args: {args}")

    # Update SAVE_DIR based on cache-path argument
    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)

    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)

    worker = ModelWorker(
        model_path=args.model_path, 
        tex_model_path=args.tex_model_path,
        subfolder=args.subfolder,
        device=args.device, 
        enable_tex=args.enable_tex,
        low_vram_mode=args.low_vram_mode
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
