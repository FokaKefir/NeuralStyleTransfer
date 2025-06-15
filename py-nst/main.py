from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import os
import uuid

from nst import neural_style_transfer, neural_style_transfer_with_segmentation, neural_style_transfer_mixed
from utils.db_utils import update_gen_document
from utils.api_const import BASE_URL

IMG_HEIGHT = 400

# paths
default_resource_dir = os.path.join(os.path.dirname(__file__), 'data')
content_images_dir = os.path.join(default_resource_dir, 'content-images')
style_images_dir = os.path.join(default_resource_dir, 'style-images')
output_img_dir = os.path.join(default_resource_dir, 'output-images')

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


#@app.get('/user/{user_id}/generate')
@app.post('/generate')
async def generate(
    doc_id: str,
    content_img: str,
    style_img: str,
    init_method: str,
    style_weight: int,
    tv_weight: int,
    iterations: int
):
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
        'height': IMG_HEIGHT,
        'saving_freq': -1
    }
    img_name = neural_style_transfer(config)
    update_gen_document(doc_id, BASE_URL + "image/generated/" + img_name)
    return {'image': img_name}


@app.post('/generate_seg')
async def generate_seg(
    doc_id: str,
    content_img: str,
    style_person_img: str = "", 
    style_background_img: str = "",  
    init_method: str = "content",
    style_person_weight: float = None,
    style_background_weight: float = None,
    tv_weight: float = 1.0,
    iterations: int = 1000
):
    person_img = style_person_img or None
    background_img = style_background_img or None

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
        'height':                      IMG_HEIGHT,
        'saving_freq':                 -1
    }

    img_name = neural_style_transfer_with_segmentation(config)

    update_gen_document(doc_id,BASE_URL + "image/generated/" + img_name)

    return {'image': img_name}


@app.post('/generate_mixed')
async def generate_mixed(
    doc_id: str,
    content_img: str,
    style_img_1: str,
    style_img_2: str,
    init_method: str = "content",
    style_weight: float = None,
    alpha: float = 0.5,
    tv_weight: float = 1.0,
    iterations: int = 1000
):
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
        'height':               IMG_HEIGHT,
        'saving_freq':          -1
    }

    img_name = neural_style_transfer_mixed(config)

    update_gen_document(doc_id, BASE_URL + "image/generated/" + img_name)

    return {'image': img_name}