const API_URL = 'http://localhost:8000';

// State
let contentImageFile = null;
let styleImageFile = null;
let contentImageName = null;
let styleImageName = null;

// DOM Elements
const contentInput = document.getElementById('contentInput');
const styleInput = document.getElementById('styleInput');
const contentPreview = document.getElementById('contentPreview');
const stylePreview = document.getElementById('stylePreview');
const uploadContentBtn = document.getElementById('uploadContentBtn');
const uploadStyleBtn = document.getElementById('uploadStyleBtn');
const contentStatus = document.getElementById('contentStatus');
const styleStatus = document.getElementById('styleStatus');
const generateBtn = document.getElementById('generateBtn');
const styleWeightSlider = document.getElementById('styleWeight');
const tvWeightSlider = document.getElementById('tvWeight');
const iterationsSlider = document.getElementById('iterations');
const styleWeightValue = document.getElementById('styleWeightValue');
const tvWeightValue = document.getElementById('tvWeightValue');
const iterationsValue = document.getElementById('iterationsValue');
const initMethod = document.getElementById('initMethod');
const progress = document.getElementById('progress');
const resultPanel = document.getElementById('resultPanel');
const resultImage = document.getElementById('resultImage');
const downloadBtn = document.getElementById('downloadBtn');
const newGenerationBtn = document.getElementById('newGenerationBtn');
const styleGallery = document.getElementById('styleGallery');
const contentGallery = document.getElementById('contentGallery');
const tabBtns = document.querySelectorAll('.tab-btn');
const styleLibrary = document.getElementById('styleLibrary');
const styleUpload = document.getElementById('styleUpload');
const contentLibrary = document.getElementById('contentLibrary');
const contentUpload = document.getElementById('contentUpload');
const useOriginalSize = document.getElementById('useOriginalSize');

// Event Listeners
contentInput.addEventListener('change', (e) => handleImageSelect(e, 'content'));
styleInput.addEventListener('change', (e) => handleImageSelect(e, 'style'));
uploadContentBtn.addEventListener('click', () => uploadImage('content'));
uploadStyleBtn.addEventListener('click', () => uploadImage('style'));
generateBtn.addEventListener('click', generateImage);
downloadBtn.addEventListener('click', downloadImage);
newGenerationBtn.addEventListener('click', resetApp);

// Tab switching
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        
        // Update button states
        const parentTabs = btn.parentElement;
        parentTabs.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Update content visibility
        if (tab === 'library') {
            styleLibrary.classList.add('active');
            styleUpload.classList.remove('active');
        } else if (tab === 'upload') {
            styleLibrary.classList.remove('active');
            styleUpload.classList.add('active');
        } else if (tab === 'content-library') {
            contentLibrary.classList.add('active');
            contentUpload.classList.remove('active');
        } else if (tab === 'content-upload') {
            contentLibrary.classList.remove('active');
            contentUpload.classList.add('active');
        }
    });
});

// Load content library
async function loadContentLibrary() {
    try {
        const response = await fetch(`${API_URL}/content/list`);
        const data = await response.json();
        
        if (data.content && data.content.length > 0) {
            contentGallery.innerHTML = '';
            data.content.forEach(contentName => {
                const item = document.createElement('div');
                item.className = 'content-item';
                item.innerHTML = `<img src="${API_URL}/image/content/${contentName}" alt="${contentName}">`;
                item.onclick = () => selectContentFromLibrary(contentName, item);
                contentGallery.appendChild(item);
            });
        } else {
            contentGallery.innerHTML = '<p class="loading">No content images available. Upload some!</p>';
        }
    } catch (error) {
        contentGallery.innerHTML = '<p class="loading">Error loading content images</p>';
        console.error('Failed to load content library:', error);
    }
}

// Select content from library
function selectContentFromLibrary(contentName, itemElement) {
    // Remove previous selection
    document.querySelectorAll('.content-item').forEach(item => {
        item.classList.remove('selected');
    });
    
    // Mark as selected
    itemElement.classList.add('selected');
    
    // Update preview
    contentPreview.classList.remove('empty');
    contentPreview.innerHTML = `<img src="${API_URL}/image/content/${contentName}" alt="${contentName}">`;
    
    // Set content name
    contentImageName = contentName;
    contentImageFile = null; // Clear file as we're using library
    
    // Update status
    contentStatus.textContent = '✓ Content selected from library';
    contentStatus.className = 'status success';
    
    // Update generate button
    updateGenerateButton();
}

// Load style library
async function loadStyleLibrary() {
    try {
        const response = await fetch(`${API_URL}/styles/list`);
        const data = await response.json();
        
        if (data.styles && data.styles.length > 0) {
            styleGallery.innerHTML = '';
            data.styles.forEach(styleName => {
                const item = document.createElement('div');
                item.className = 'style-item';
                item.innerHTML = `<img src="${API_URL}/image/style/${styleName}" alt="${styleName}">`;
                item.onclick = () => selectStyleFromLibrary(styleName, item);
                styleGallery.appendChild(item);
            });
        } else {
            styleGallery.innerHTML = '<p class="loading">No styles available. Upload some!</p>';
        }
    } catch (error) {
        styleGallery.innerHTML = '<p class="loading">Error loading styles</p>';
        console.error('Failed to load style library:', error);
    }
}

// Select style from library
function selectStyleFromLibrary(styleName, itemElement) {
    // Remove previous selection
    document.querySelectorAll('.style-item').forEach(item => {
        item.classList.remove('selected');
    });
    
    // Mark as selected
    itemElement.classList.add('selected');
    
    // Update preview
    stylePreview.classList.remove('empty');
    stylePreview.innerHTML = `<img src="${API_URL}/image/style/${styleName}" alt="${styleName}">`;
    
    // Set style name
    styleImageName = styleName;
    styleImageFile = null; // Clear file as we're using library
    
    // Update status
    styleStatus.textContent = '✓ Style selected from library';
    styleStatus.className = 'status success';
    
    // Update generate button
    updateGenerateButton();
}

// Event Listeners
contentInput.addEventListener('change', (e) => handleImageSelect(e, 'content'));
styleInput.addEventListener('change', (e) => handleImageSelect(e, 'style'));
uploadContentBtn.addEventListener('click', () => uploadImage('content'));
uploadStyleBtn.addEventListener('click', () => uploadImage('style'));
generateBtn.addEventListener('click', generateImage);
downloadBtn.addEventListener('click', downloadImage);
newGenerationBtn.addEventListener('click', resetApp);

// Conversion functions
function getStyleWeight(sliderValue) {
    // Pattern: whole numbers -> 10^value, x.5 -> 3 * 10^floor(value)
    const val = parseFloat(sliderValue);
    const floorVal = Math.floor(val);
    const isHalfStep = (val % 1) === 0.5;
    
    if (isHalfStep) {
        return 3 * Math.pow(10, floorVal);
    } else {
        return Math.pow(10, val);
    }
}

function getTvWeight(sliderValue) {
    // 10^value
    return Math.pow(10, parseFloat(sliderValue));
}

// Update slider values
styleWeightSlider.addEventListener('input', (e) => {
    const actualWeight = getStyleWeight(e.target.value);
    styleWeightValue.textContent = actualWeight.toLocaleString();
});

tvWeightSlider.addEventListener('input', (e) => {
    const actualWeight = getTvWeight(e.target.value);
    tvWeightValue.textContent = actualWeight;
});

iterationsSlider.addEventListener('input', (e) => {
    iterationsValue.textContent = e.target.value;
});

// Initialize display values
styleWeightValue.textContent = getStyleWeight(styleWeightSlider.value).toLocaleString();
tvWeightValue.textContent = getTvWeight(tvWeightSlider.value);

// Handle image selection
function handleImageSelect(event, type) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = type === 'content' ? contentPreview : stylePreview;
        preview.classList.remove('empty');
        preview.innerHTML = `<img src="${e.target.result}" alt="${type} preview">`;
        
        if (type === 'content') {
            contentImageFile = file;
            uploadContentBtn.disabled = false;
            contentImageName = null;
            contentStatus.textContent = '';
        } else {
            styleImageFile = file;
            uploadStyleBtn.disabled = false;
            styleImageName = null;
            styleStatus.textContent = '';
        }
        
        updateGenerateButton();
    };
    reader.readAsDataURL(file);
}

// Upload image to server
async function uploadImage(type) {
    const file = type === 'content' ? contentImageFile : styleImageFile;
    const statusElement = type === 'content' ? contentStatus : styleStatus;
    const uploadBtn = type === 'content' ? uploadContentBtn : uploadStyleBtn;
    
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    uploadBtn.disabled = true;
    statusElement.textContent = 'Uploading...';
    statusElement.className = 'status';
    
    try {
        const endpoint = type === 'content' ? '/content/upload/' : '/style/upload/';
        const response = await fetch(API_URL + endpoint, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('Upload failed');
        
        const data = await response.json();
        
        if (type === 'content') {
            contentImageName = data.image_name;
            statusElement.textContent = '✓ Content uploaded successfully';
            // Reload content library to include the new content
            loadContentLibrary();
        } else {
            styleImageName = data.image_name;
            statusElement.textContent = '✓ Style uploaded successfully';
            // Reload style library to include the new style
            loadStyleLibrary();
        }
        
        statusElement.className = 'status success';
        updateGenerateButton();
        
    } catch (error) {
        statusElement.textContent = `✗ Upload failed: ${error.message}`;
        statusElement.className = 'status error';
        uploadBtn.disabled = false;
    }
}

// Update generate button state
function updateGenerateButton() {
    generateBtn.disabled = !(contentImageName && styleImageName);
}

// Generate styled image
async function generateImage() {
    generateBtn.disabled = true;
    progress.style.display = 'block';
    resultPanel.style.display = 'none';
    
    const params = new URLSearchParams({
        content_img: contentImageName,
        style_img: styleImageName,
        init_method: initMethod.value,
        style_weight: getStyleWeight(styleWeightSlider.value),
        tv_weight: getTvWeight(tvWeightSlider.value),
        iterations: iterationsSlider.value,
        use_original_size: useOriginalSize.checked
    });
    
    try {
        const response = await fetch(API_URL + '/generate?' + params.toString(), {
            method: 'POST'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Generation failed');
        }
        
        const data = await response.json();
        
        // Display result
        resultImage.src = `${API_URL}/image/generated/${data.image}`;
        resultImage.dataset.imageName = data.image;
        resultPanel.style.display = 'block';
        
        // Re-enable generate button for new generation with different parameters
        generateBtn.disabled = false;
        
        // Scroll to result
        resultPanel.scrollIntoView({ behavior: 'smooth' });
        
    } catch (error) {
        alert(`Generation failed: ${error.message}`);
        generateBtn.disabled = false;
    } finally {
        progress.style.display = 'none';
    }
}

// Download generated image
function downloadImage() {
    const imageName = resultImage.dataset.imageName;
    if (!imageName) return;
    
    const link = document.createElement('a');
    link.href = `${API_URL}/image/generated/${imageName}`;
    link.download = imageName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Reset app for new generation
function resetApp() {
    contentImageFile = null;
    styleImageFile = null;
    contentImageName = null;
    styleImageName = null;
    
    contentInput.value = '';
    styleInput.value = '';
    contentPreview.innerHTML = '';
    stylePreview.innerHTML = '';
    contentPreview.classList.add('empty');
    stylePreview.classList.add('empty');
    contentStatus.textContent = '';
    styleStatus.textContent = '';
    
    // Clear style gallery selection
    document.querySelectorAll('.style-item').forEach(item => {
        item.classList.remove('selected');
    });
    
    // Clear content gallery selection
    document.querySelectorAll('.content-item').forEach(item => {
        item.classList.remove('selected');
    });
    
    uploadContentBtn.disabled = true;
    uploadStyleBtn.disabled = true;
    generateBtn.disabled = true;
    
    resultPanel.style.display = 'none';
    
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Initialize empty previews
contentPreview.classList.add('empty');
stylePreview.classList.add('empty');

// Load both libraries on page load
loadContentLibrary();
loadStyleLibrary();
