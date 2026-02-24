# Neural Style Transfer - Web UI

This folder contains the frontend web interface for the Neural Style Transfer application.

## Files

- **index.html** - Main web interface
- **style.css** - Styling and layout
- **app.js** - JavaScript functionality (image upload, API calls, UI updates)

## Features

- Upload content and style images
- Adjust generation parameters with interactive sliders
- Real-time image previews
- Progress indicators during generation
- Download generated images
- Responsive design for mobile and desktop

## API Integration

The UI communicates with the FastAPI backend running on `http://localhost:8000`:

- `/content/upload/` - Upload content images
- `/style/upload/` - Upload style images
- `/generate` - Generate styled images
- `/image/generated/{name}` - Download generated images

## Usage

The web UI is automatically served by the FastAPI backend. Just start the server and navigate to `http://localhost:8000` in your browser.

No separate web server needed!
