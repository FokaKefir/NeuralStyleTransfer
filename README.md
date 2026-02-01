# AI the Artist – Neural Style Transfer Backend

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

> 🏆 **Award-Winning Project**: Bachelor thesis grade 10/10 | 1st Place at 2024 Scientific Student Conference | Accenture Special Award | Presented at 2025 National Scientific Student Conference

**AI the Artist** (StyleApp) is a high-performance Neural Style Transfer (NST) backend that powers a cross-platform creative image stylization application. Transform everyday photos into stunning artwork by applying the style of famous paintings or custom artistic styles.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [API Endpoints](#api-endpoints)
  - [CLI Usage](#cli-usage)
- [Advanced Features](#advanced-features)
- [Technical Details](#technical-details)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## ✨ Features

- **🎨 Classic Neural Style Transfer**: Transform images using Gatys et al.'s optimization-based approach
- **👤 Segmentation-Based Stylization**: Apply different styles to foreground (person) and background separately
- **🎭 Mixed Style Transfer**: Blend two artistic styles into a single output with adjustable weights
- **🚀 RESTful API**: Production-ready FastAPI backend with CORS support
- **⚡ GPU Acceleration**: CUDA support for fast processing
- **🔧 Flexible Configuration**: Multiple initialization methods, customizable loss weights, and iteration counts
- **📊 Metrics & Monitoring**: Built-in quality metrics (SSIM, FID, style loss) and Weights & Biases integration
- **🎯 Pre-trained Models**: VGG16 and VGG19 architectures for feature extraction

## 🏗️ Architecture

The system implements Neural Style Transfer using the following approach:

1. **Feature Extraction**: Pre-trained VGG networks extract content and style features
2. **Loss Computation**: 
   - **Content Loss**: MSE between content feature maps
   - **Style Loss**: MSE between Gram matrices of style features
   - **Total Variation Loss**: Regularization for spatial smoothness
3. **Optimization**: Adam optimizer iteratively updates pixel values to minimize combined loss
4. **Segmentation** (optional): DeepLabV3 for person detection and separate stylization

### Three Operating Modes

1. **Standard NST**: Single content image + single style image
2. **Segmented NST**: Different styles for person vs. background (using semantic segmentation)
3. **Mixed NST**: Blend two different artistic styles with adjustable alpha parameter

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/neural-style-transfer.git
cd neural-style-transfer/py-nst

# Install dependencies
pip install torch torchvision
pip install fastapi uvicorn
pip install opencv-python numpy
pip install piqa  # for metrics
pip install wandb  # optional, for experiment tracking

# Create data directories
mkdir -p data/content-images data/style-images data/output-images
```

## 🚀 Usage

### API Server

Start the FastAPI server:

```bash
# Using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000

# Or using the provided script
bash start_api.sh
```

The API will be available at `http://localhost:8000`. View interactive API docs at `http://localhost:8000/docs`.

### API Endpoints

#### 1. **Upload Images**

```bash
# Upload content image
curl -X POST "http://localhost:8000/content/upload/" \
  -F "file=@your_photo.jpg"

# Upload style image
curl -X POST "http://localhost:8000/style/upload/" \
  -F "file=@vangogh_starry_night.jpg"
```

#### 2. **Generate Stylized Image**

**Standard Style Transfer:**
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "unique_id",
    "content_img": "content_filename.jpg",
    "style_img": "style_filename.jpg",
    "init_method": "content",
    "style_weight": 30000,
    "tv_weight": 1.0,
    "iterations": 1000
  }'
```

**Segmented Style Transfer:**
```bash
curl -X POST "http://localhost:8000/generate_seg" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "unique_id",
    "content_img": "portrait.jpg",
    "style_person_img": "picasso.jpg",
    "style_background_img": "monet.jpg",
    "style_person_weight": 25000,
    "style_background_weight": 30000,
    "iterations": 1000
  }'
```

**Mixed Style Transfer:**
```bash
curl -X POST "http://localhost:8000/generate_mixed" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "unique_id",
    "content_img": "content.jpg",
    "style_img_1": "style1.jpg",
    "style_img_2": "style2.jpg",
    "style_weight": 30000,
    "alpha": 0.5,
    "iterations": 1000
  }'
```

#### 3. **Retrieve Generated Image**

```bash
curl "http://localhost:8000/image/generated/{image_name}" -o output.jpg
```

### CLI Usage

For standalone processing without the API:

```python
from nst import neural_style_transfer

config = {
    'content_img_name': 'photo.jpg',
    'style_img_name': 'style.jpg',
    'init_method': 'content',  # 'random', 'content', or 'style'
    'content_weight': 1e5,
    'style_weight': 3e4,
    'tv_weight': 1e0,
    'iterations': 1000,
    'model': 'vgg19',  # or 'vgg16'
    'content_images_dir': 'data/content-images',
    'style_images_dir': 'data/style-images',
    'output_img_dir': 'data/output-images',
    'img_format': (4, '.jpg'),
    'height': 400,
    'saving_freq': -1  # -1 saves only final result
}

neural_style_transfer(config)
```

## 🔬 Advanced Features

### Initialization Methods

- **`content`**: Start optimization from content image (recommended)
- **`style`**: Start from resized style image
- **`random`**: Start from Gaussian noise

### Hyperparameter Tuning

- **`content_weight`**: Controls content preservation (default: 1e5)
- **`style_weight`**: Controls style strength (default: 3e4)
- **`tv_weight`**: Total variation regularization (default: 1.0)
- **`iterations`**: Optimization steps (500-3000, depending on quality needs)
- **`height`**: Output image height in pixels (width auto-scaled)

### Quality Metrics

Evaluate generated images using [metrics.py](py-nst/metrics.py):

```python
# Computes SSIM (structural similarity with content)
# and FID (Fréchet Inception Distance for style quality)
python metrics.py
```

### Weights & Biases Integration

Track experiments and compare results:

```python
# In wandb_nst.py - logs losses and generated images to W&B dashboard
wandb.init(project="neural-style-transfer")
# Run NST with logging enabled
```

## 🔧 Technical Details

### Models

- **VGG16**: 4 layers (`relu1_2`, `relu2_2`, `relu3_3`, `relu4_3`)
- **VGG19**: 6 layers (`relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`, `conv4_2`, `relu5_1`)

Content is typically extracted from `relu2_2` (VGG16) or `conv4_2` (VGG19), while style is extracted from multiple layers.

### Loss Function

$$
\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{content} + \beta \cdot \mathcal{L}_{style} + \gamma \cdot \mathcal{L}_{tv}
$$

Where:
- $\mathcal{L}_{content}$ is the MSE between content feature maps
- $\mathcal{L}_{style}$ is the MSE between Gram matrices
- $\mathcal{L}_{tv}$ penalizes spatial variations

### Gram Matrix

Style representation uses Gram matrices to capture texture/color correlations:

$$
G_{ij}^l = \sum_k F_{ik}^l F_{jk}^l
$$

### Segmentation

Person segmentation uses DeepLabV3 (ResNet-101 backbone) with post-processing:
- Morphological opening to remove noise
- Connected component analysis to isolate largest person region

## 📁 Project Structure

```
py-nst/
├── main.py                           # FastAPI server & API endpoints
├── nst.py                           # Core NST implementation (3 modes)
├── neural_style_transfer.py         # Original NST implementation
├── segmentation.py                  # Person segmentation with DeepLabV3
├── metrics.py                       # Quality metrics (SSIM, FID)
├── wandb_nst.py                     # W&B experiment tracking
├── models/
│   └── definitions/
│       ├── vgg_nets.py              # VGG16/VGG19 implementations
│       └── __init__.py
├── utils/
│   ├── utils.py                     # Image processing & model prep
│   ├── video_utils.py               # Video generation from frames
│   ├── db_utils.py                  # Database utilities
│   └── __init__.py
└── data/
    ├── content-images/              # Input photos
    ├── style-images/                # Artistic style references
    └── output-images/               # Generated results
```

## 📖 API Reference

### POST `/content/upload/`
Upload a content image.
- **Input**: Multipart form data with image file
- **Output**: `{"image_name": "uuid.jpg"}`

### POST `/style/upload/`
Upload a style image.
- **Input**: Multipart form data with image file
- **Output**: `{"image_name": "uuid.jpg"}`

### POST `/generate`
Standard Neural Style Transfer.
- **Parameters**:
  - `doc_id`: Unique document identifier
  - `content_img`: Content image filename
  - `style_img`: Style image filename
  - `init_method`: `"content"`, `"style"`, or `"random"`
  - `style_weight`: Style loss weight (10000-50000)
  - `tv_weight`: Total variation weight (0.1-10)
  - `iterations`: Number of optimization steps (500-3000)

### POST `/generate_seg`
Segmented style transfer (different styles for person vs. background).
- **Additional Parameters**:
  - `style_person_img`: Style for person region (optional)
  - `style_background_img`: Style for background (optional)
  - `style_person_weight`: Style weight for person
  - `style_background_weight`: Style weight for background

### POST `/generate_mixed`
Mixed style transfer (blend two styles).
- **Additional Parameters**:
  - `style_img_1`: First style image
  - `style_img_2`: Second style image
  - `alpha`: Blending factor (0.0-1.0, controls style_img_2 influence)

### GET `/image/generated/{image_name}`
Download generated image.

## ⚡ Performance

- **Processing Time**: 30-60 seconds per image (GPU) / 5-15 minutes (CPU)
- **Image Size**: 400px height (default), auto-scaled width
- **Memory**: ~2-4GB GPU memory for standard images
- **Iterations**: 1000 iterations provide good quality; 2000+ for high quality

**Optimization Tips**:
- Use GPU acceleration for 10-20x speedup
- Lower `height` parameter for faster processing
- Reduce `iterations` for quick previews
- Use `init_method='content'` for faster convergence

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 Citation

If you use this project in your research or application, please cite:

```bibtex
@thesis{styleapp2024,
  title={AI the Artist: Creative Image Stylization with Neural Style Transfer},
  author={Your Name},
  year={2024},
  school={Your University},
  note={1st Place, Scientific Student Conference 2024; Accenture Special Award}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original Neural Style Transfer paper: [Gatys et al., 2015](https://arxiv.org/abs/1508.06576)
- VGG networks: [Simonyan & Zisserman, 2014](https://arxiv.org/abs/1409.1556)
- DeepLabV3: [Chen et al., 2017](https://arxiv.org/abs/1706.05587)
- PyTorch team for excellent deep learning framework

## 📬 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or contact [your-email@example.com].

---

**Note**: This is the backend component of the StyleApp project. For the complete cross-platform application (React web + Android), please visit the main repository.
