from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import os
import uuid

from nst import neural_style_transfer, neural_style_transfer_with_segmentation, neural_style_transfer_mixed

IMG_HEIGHT = 400

# paths
default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
content_images_dir = os.path.join(default_resource_dir, 'content-images')
style_images_dir = os.path.join(default_resource_dir, 'style-images')
output_img_dir = os.path.join(default_resource_dir, 'output-images')
web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'web')

# create app
app = FastAPI()

# add cors
origins = [
    "http://localhost:3000",  # React development server
    "http://localhost:5000",  # Other potential local ports
    "http://fokakefir.go.ro",  # Your production domain
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve the web UI
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join(web_dir, 'index.html'), 'r', encoding='utf-8') as f:
        return f.read()

# serve static files (CSS, JS)
@app.get("/{file_name}")
async def serve_static(file_name: str):
    if file_name in ['style.css', 'app.js']:
        file_path = os.path.join(web_dir, file_name)
        return FileResponse(file_path)

# content image uploader
@app.post('/content/upload/')
async def upload_content_image(file: UploadFile = File(...)):

    file.filename = f'{uuid.uuid4()}.jpg'
    contents = await file.read()

    # save the file
    image_path = os.path.join(content_images_dir, file.filename)
    with open(image_path, 'wb') as fout:
        fout.write(contents)

    return {'image_name': file.filename}  

# style image uploader
@app.post('/style/upload/')
async def upload_style_image(file: UploadFile = File(...)):

    file.filename = f'{uuid.uuid4()}.jpg'
    contents = await file.read()

    # save the file
    image_path = os.path.join(style_images_dir, file.filename)
    with open(image_path, 'wb') as fout:
        fout.write(contents)

    return {'image_name': file.filename}  

# list all available style images
@app.get('/styles/list')
async def list_style_images():
    try:
        files = os.listdir(style_images_dir)
        # Filter only image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        style_images = [f for f in files if os.path.splitext(f.lower())[1] in image_extensions]
        return {'styles': style_images}
    except Exception as e:
        return {'styles': [], 'error': str(e)}

# list all available content images
@app.get('/content/list')
async def list_content_images():
    try:
        files = os.listdir(content_images_dir)
        # Filter only image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        content_images = [f for f in files if os.path.splitext(f.lower())[1] in image_extensions]
        return {'content': content_images}
    except Exception as e:
        return {'content': [], 'error': str(e)}

# returns a style image
@app.get('/image/style/{image_name}')
async def get_style_image(image_name: str):
    image_path = os.path.join(style_images_dir, image_name)
    return FileResponse(image_path)

# returns a content image
@app.get('/image/content/{image_name}')
async def get_content_image(image_name: str):
    image_path = os.path.join(content_images_dir, image_name)
    return FileResponse(image_path)

# return a generated image
@app.get('/image/generated/{image_name}')
async def get_generated_image(image_name: str):
    image_path = os.path.join(output_img_dir, image_name)
    return FileResponse(
        image_path,
        headers={
            "Content-Disposition": f"attachment; filename={image_name}",
            "Access-Control-Expose-Headers": "Content-Disposition",
            "Access-Control-Allow-Origin": "*" 
        }
    )


@app.post('/generate')
async def generate(
    content_img: str,
    style_img: str,
    init_method: str = "content",
    style_weight: int = 30000,
    tv_weight: int = 1,
    iterations: int = 1000,
    use_original_size: bool = False
):
    # Set height based on use_original_size flag
    height = None if use_original_size else IMG_HEIGHT
    
    config = {
        'content_img_name': content_img,
        'style_img_name': style_img,
        'init_method': init_method,
        'content_weight': 1e5,
        'style_weight': style_weight,
        'tv_weight': tv_weight,
        'iterations': iterations,
        'model': 'vgg19',
        'content_images_dir': content_images_dir,
        'style_images_dir': style_images_dir,
        'output_img_dir': output_img_dir,
        'img_format': (4, '.jpg'),
        'height': height,
        'saving_freq': -1
    }
    img_name = neural_style_transfer(config)
    return {'image': img_name}


@app.post('/generate_seg')
async def generate_seg(
    content_img: str,
    style_person_img: str = "", 
    style_background_img: str = "",  
    init_method: str = "content",
    style_person_weight: float = None,
    style_background_weight: float = None,
    tv_weight: float = 1.0,
    iterations: int = 1000,
    use_original_size: bool = False
):
    person_img = style_person_img or None
    background_img = style_background_img or None
    
    # Set height based on use_original_size flag
    height = None if use_original_size else IMG_HEIGHT

    if person_img is None and background_img is None:
        raise HTTPException(
            status_code=400,
            detail="At least one of style_person_img or style_background_img must be provided"
        )

    if person_img is not None and style_person_weight is None:
        raise HTTPException(
            status_code=400,
            detail="style_person_weight is required when style_person_img is provided"
        )
    if background_img is not None and style_background_weight is None:
        raise HTTPException(
            status_code=400,
            detail="style_background_weight is required when style_background_img is provided"
        )

    config = {
        'content_img_name':            content_img,
        'style_person_img_name':       person_img,
        'style_background_img_name':   background_img,
        'init_method':                 init_method,
        'content_weight':              1e5,
        'style_person_weight':         style_person_weight or 0.0,
        'style_background_weight':     style_background_weight or 0.0,
        'tv_weight':                   tv_weight,
        'iterations':                  iterations,
        'model':                       'vgg19',
        'content_images_dir':          content_images_dir,
        'style_images_dir':            style_images_dir,
        'output_img_dir':              output_img_dir,
        'img_format':                  (4, '.jpg'),
        'height':                      height,
        'saving_freq':                 -1
    }

    img_name = neural_style_transfer_with_segmentation(config)

    return {'image': img_name}


@app.post('/generate_mixed')
async def generate_mixed(
    content_img: str,
    style_img_1: str,
    style_img_2: str,
    init_method: str = "content",
    style_weight: float = None,
    alpha: float = 0.5,
    tv_weight: float = 1.0,
    iterations: int = 1000,
    use_original_size: bool = False
):
    # Set height based on use_original_size flag
    height = None if use_original_size else IMG_HEIGHT
    
    # 1) Kötelező mezők
    if not style_img_1 or not style_img_2:
        raise HTTPException(
            status_code=400,
            detail="Both style_img_1 and style_img_2 must be provided for mixed-style generation"
        )
    if style_weight is None:
        raise HTTPException(
            status_code=400,
            detail="style_weight is required for mixed-style generation"
        )
    if alpha is None:
        raise HTTPException(
            status_code=400,
            detail="alpha is required for mixed-style generation"
        )

    config = {
        'content_img_name':     content_img,
        'style_img_name_1':     style_img_1,
        'style_img_name_2':     style_img_2,
        'init_method':          init_method,
        'content_weight':       1e5,
        'style_weight':         style_weight,
        'alpha':                alpha,
        'tv_weight':            tv_weight,
        'iterations':           iterations,
        'model':                'vgg19',
        'content_images_dir':   content_images_dir,
        'style_images_dir':     style_images_dir,
        'output_img_dir':       output_img_dir,
        'img_format':           (4, '.jpg'),
        'height':               height,
        'saving_freq':          -1
    }

    img_name = neural_style_transfer_mixed(config)

    return {'image': img_name}