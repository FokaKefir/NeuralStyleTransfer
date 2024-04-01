from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import os
import uuid

from nst import neural_style_transfer
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
    return FileResponse(image_path)


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