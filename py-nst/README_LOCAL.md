# Neural Style Transfer - Local Setup

## Project Structure

```
NeuralStyleTransfer/
├── py-nst/          # Backend FastAPI application
│   ├── main.py      # API server
│   ├── nst.py       # Neural style transfer logic
│   ├── models/      # VGG network definitions
│   ├── utils/       # Utility functions
│   └── data/        # Image storage
└── web/             # Frontend web interface
    ├── index.html   # Main UI
    ├── style.css    # Styling
    └── app.js       # JavaScript
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install fastapi uvicorn python-multipart torch torchvision
   ```

2. **Start the server:**
   ```bash
   cd py-nst
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Open your browser:**
   Navigate to `http://localhost:8000`

## Usage

1. **Upload Content Image**: Select and upload the image you want to stylize
2. **Upload Style Image**: Select and upload the artistic style image
3. **Adjust Parameters**:
   - Style Weight: Controls how strong the style is applied (1000-100000)
   - TV Weight: Total variation weight for smoothness (0.1-10)
   - Iterations: Number of optimization steps (100-3000)
   - Init Method: Starting point (content/random/style)
4. **Generate**: Click to create your styled image
5. **Download**: Save the generated result

## Features

- ✅ Basic neural style transfer
- ✅ Segmentation-based transfer (separate person/background)
- ✅ Mixed-style transfer (blend two styles)
- ✅ No database required - fully local
- ✅ Simple web UI

## API Endpoints

- `GET /` - Web UI
- `POST /content/upload/` - Upload content image
- `POST /style/upload/` - Upload style image
- `POST /generate` - Generate styled image
- `POST /generate_seg` - Generate with segmentation
- `POST /generate_mixed` - Generate with mixed styles
- `GET /image/generated/{name}` - Download generated image

## Notes

- Processing time depends on iterations and your hardware
- GPU recommended for faster processing
- Images are stored locally in `data/` directory
