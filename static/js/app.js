/**
 * Plant Disease Classifier - Main JavaScript
 * Version: 1.1.0 (38 Classes)
 * Author: Plant AI Team
 */

// ============================================
// 1. CONFIGURATION & CONSTANTS
// ============================================
const CONFIG = {
    API_BASE_URL: window.location.origin + '/api',
    MAX_FILE_SIZE: 16 * 1024 * 1024,
    ALLOWED_FILE_TYPES: ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
};

// Plant icons mapping
const PLANT_ICONS = {
    'Apple': 'apple-alt',
    'Cherry': 'cherries',
    'Corn': 'corn',
    'Grape': 'grapes',
    'Peach': 'peach',
    'Pepper': 'pepper-hot',
    'Potato': 'carrot',
    'Strawberry': 'strawberry',
    'Tomato': 'tomato',
    'Blueberry': 'seedling',
    'Orange': 'lemon',
    'Raspberry': 'seedling',
    'Soybean': 'seedling',
    'Squash': 'seedling'
};

// ============================================
// 2. STATE MANAGEMENT
// ============================================
let AppState = {
    currentFile: null,
    isProcessing: false,
    systemInfo: {
        modelLoaded: false,
        totalClasses: 38,
        totalPlants: 14,
        modelAccuracy: '95%'
    }
};

// ============================================
// 3. DOM ELEMENTS
// ============================================
function getElement(id) {
    const element = document.getElementById(id);
    if (!element) {
        console.warn(`Element with id "${id}" not found`);
    }
    return element;
}

const Elements = {
    // Loading
    get loadingScreen() { return getElement('loadingScreen'); },
    
    // Stats
    get statClasses() { return getElement('statClasses'); },
    get statPlants() { return getElement('statPlants'); },
    get modelLayers() { return getElement('modelLayers'); },
    get modelAccuracy() { return getElement('modelAccuracy'); },
    
    // Upload
    get uploadArea() { return getElement('uploadArea'); },
    get singleUploadBtn() { return getElement('singleUploadBtn'); },
    get fileInput() { return getElement('fileInput'); },
    get filePreview() { return getElement('filePreview'); },
    get predictBtn() { return getElement('predictBtn'); },
    get clearBtn() { return getElement('clearBtn'); },
    
    // Loading State
    get loadingState() { return getElement('loadingState'); },
    
    // Results
    get resultsCard() { return getElement('resultsCard'); },
    get resultsContent() { return getElement('resultsContent'); },
    get closeResults() { return getElement('closeResults'); },
    
    // Plants
    get plantsGrid() { return getElement('plantsGrid'); },
    
    // Model Info
    get modelInfo() { return getElement('modelInfo'); },
    
    // Modals
    get plantModal() { return getElement('plantModal'); },
    get helpModal() { return getElement('helpModal'); },
    get modelStatusModal() { return getElement('modelStatusModal'); },
    get modalTitle() { return getElement('modalTitle'); },
    get modalBody() { return getElement('modalBody'); },
    get statusContent() { return getElement('statusContent'); },
    
    // Toast
    get toastContainer() { return getElement('toastContainer'); }
};

// ============================================
// 4. UTILITY FUNCTIONS (Declare First)
// ============================================
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`;
}

function getConfidenceLevel(confidence) {
    if (confidence >= 90) return 'Very High';
    if (confidence >= 75) return 'High';
    if (confidence >= 60) return 'Moderate';
    if (confidence >= 40) return 'Low';
    return 'Very Low';
}

function showToast(title, message, type = 'info') {
    try {
        if (!Elements.toastContainer) return;
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        let icon = 'info-circle';
        if (type === 'success') icon = 'check-circle';
        if (type === 'error') icon = 'exclamation-circle';
        if (type === 'warning') icon = 'exclamation-triangle';
        
        toast.innerHTML = `
            <i class="fas fa-${icon}"></i>
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                <div class="toast-message">${message}</div>
            </div>
            <button class="btn-icon close-toast">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        Elements.toastContainer.appendChild(toast);
        
        toast.querySelector('.close-toast').addEventListener('click', () => {
            toast.remove();
        });
        
        setTimeout(() => {
            if (toast.parentNode) {
                toast.remove();
            }
        }, 5000);
        
    } catch (error) {
        console.error('Error showing toast:', error);
    }
}

function closeModal() {
    try {
        const modals = document.querySelectorAll('.modal.active');
        modals.forEach(modal => modal.classList.remove('active'));
    } catch (error) {
        console.error('Error closing modal:', error);
    }
}

function showPlantDetails(plantType, icon) {
    try {
        console.log(`Showing details for ${plantType}`);
        // Simple alert for now - you can expand this later
        alert(`Plant: ${plantType}\nIcon: ${icon}\n\nDetailed information would appear here in the full version.`);
    } catch (error) {
        console.error('Error showing plant details:', error);
    }
}

function showModelStatusModal() {
    try {
        alert('Model Status:\n\n• Architecture: EfficientNet B4\n• Classes: 38\n• Plant Types: 14\n• Status: Online\n\nDetailed status would appear here in the full version.');
    } catch (error) {
        console.error('Error showing model status:', error);
    }
}

function showHelpModal() {
    try {
        alert('Help Information:\n\n1. Upload a plant leaf image\n2. Click "Analyze Image"\n3. View results\n\nMaximum file size: 16MB\nSupported formats: JPG, PNG, WebP');
    } catch (error) {
        console.error('Error showing help modal:', error);
    }
}

function handleKeyboardShortcuts(e) {
    if (e.key === 'Escape') {
        hideResults();
        closeModal();
    }
}

// ============================================
// 5. FILE HANDLING FUNCTIONS
// ============================================
function validateFile(file) {
    if (!file) return false;
    
    if (!CONFIG.ALLOWED_FILE_TYPES.includes(file.type)) {
        showToast('Error', 'Please upload a JPG, JPEG, or PNG image', 'error');
        return false;
    }
    
    if (file.size > CONFIG.MAX_FILE_SIZE) {
        showToast('Error', `File too large. Maximum size is ${CONFIG.MAX_FILE_SIZE / (1024*1024)}MB`, 'error');
        return false;
    }
    
    return true;
}

function showFilePreview(file) {
    if (!Elements.filePreview) return;
    
    const reader = new FileReader();
    
    reader.onload = function(e) {
        Elements.filePreview.innerHTML = `
            <img src="${e.target.result}" alt="Preview">
            <div class="file-info">
                <div>
                    <div class="file-name">${file.name}</div>
                    <div class="file-size">${formatFileSize(file.size)}</div>
                </div>
                <button class="btn-icon" onclick="removeFile()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        Elements.filePreview.classList.add('active');
    };
    
    reader.onerror = function() {
        showToast('Error', 'Failed to read image file', 'error');
    };
    
    reader.readAsDataURL(file);
}

function removeFile() {
    try {
        AppState.currentFile = null;
        if (Elements.fileInput) Elements.fileInput.value = '';
        if (Elements.filePreview) Elements.filePreview.classList.remove('active');
        if (Elements.predictBtn) Elements.predictBtn.disabled = true;
        if (Elements.clearBtn) Elements.clearBtn.disabled = true;
    } catch (error) {
        console.error('Error removing file:', error);
    }
}

function processFile(file) {
    if (!validateFile(file)) return;
    
    try {
        AppState.currentFile = file;
        showFilePreview(file);
        
        if (Elements.predictBtn) Elements.predictBtn.disabled = false;
        if (Elements.clearBtn) Elements.clearBtn.disabled = false;
        
        showToast('Success', 'Image uploaded successfully', 'success');
    } catch (error) {
        console.error('Error processing file:', error);
        showToast('Error', 'Failed to process image', 'error');
    }
}

function handleFileSelect(event) {
    try {
        const file = event.target.files[0];
        if (file) {
            processFile(file);
        }
    } catch (error) {
        console.error('Error handling file select:', error);
        showToast('Error', 'Failed to process file', 'error');
    }
}

function handleDrop(e) {
    try {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        if (file) {
            processFile(file);
        }
    } catch (error) {
        console.error('Error handling file drop:', error);
        showToast('Error', 'Failed to process dropped file', 'error');
    }
}

function handleClear() {
    removeFile();
    hideResults();
    showToast('Cleared', 'Upload cleared successfully', 'success');
}

// ============================================
// 6. RESULTS FUNCTIONS
// ============================================
function displayResults(result) {
    if (!Elements.resultsCard || !Elements.resultsContent) return;
    
    try {
        const icon = PLANT_ICONS[result.plant_type] || 'leaf';
        
        let html = `
            <div class="prediction-header">
                <div class="predicted-plant">
                    <i class="fas fa-${icon}"></i>
                    <div class="plant-name">
                        <div class="predicted-class">${result.formatted_class}</div>
                        <div class="plant-type">${result.plant_type}</div>
                    </div>
                </div>
                
                <div class="confidence-display">${result.confidence}%</div>
                
                <div class="confidence-meter">
                    <div class="confidence-fill" style="width: ${result.confidence}%">
                        ${result.confidence}%
                    </div>
                </div>
                
                <p>Confidence level: ${getConfidenceLevel(result.confidence)}</p>
            </div>
            
            <div class="card-actions">
                <button class="btn btn-primary" onclick="handlePredict()">
                    <i class="fas fa-redo"></i> Analyze Again
                </button>
            </div>
        `;
        
        Elements.resultsContent.innerHTML = html;
        showResults();
        
        showToast('Analysis Complete', `Identified as ${result.formatted_class}`, 'success');
    } catch (error) {
        console.error('Error displaying results:', error);
        showToast('Error', 'Failed to display results', 'error');
    }
}

function showResults() {
    if (!Elements.resultsCard) return;
    Elements.resultsCard.classList.add('active');
}

function hideResults() {
    if (!Elements.resultsCard) return;
    Elements.resultsCard.classList.remove('active');
}

// ============================================
// 7. PREDICTION HANDLING
// ============================================
async function handlePredict() {
    if (!AppState.currentFile || AppState.isProcessing) return;
    
    try {
        AppState.isProcessing = true;
        if (Elements.loadingState) Elements.loadingState.classList.add('active');
        if (Elements.predictBtn) Elements.predictBtn.disabled = true;
        
        const formData = new FormData();
        formData.append('file', AppState.currentFile);
        
        const response = await fetch(`${CONFIG.API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            displayResults(result);
        } else {
            throw new Error(result.error || 'Prediction failed');
        }
        
    } catch (error) {
        console.error('Prediction error:', error);
        showToast('Prediction Error', error.message, 'error');
    } finally {
        AppState.isProcessing = false;
        if (Elements.loadingState) Elements.loadingState.classList.remove('active');
        if (Elements.predictBtn) Elements.predictBtn.disabled = false;
    }
}

// ============================================
// 8. DATA LOADING FUNCTIONS
// ============================================
function getDefaultPlantsData() {
    return [
        { plant_type: 'Apple', diseases: 3, healthy: 1, total_classes: 4 },
        { plant_type: 'Cherry', diseases: 1, healthy: 1, total_classes: 2 },
        { plant_type: 'Corn', diseases: 3, healthy: 1, total_classes: 4 },
        { plant_type: 'Grape', diseases: 3, healthy: 1, total_classes: 4 },
        { plant_type: 'Peach', diseases: 1, healthy: 1, total_classes: 2 },
        { plant_type: 'Pepper', diseases: 1, healthy: 1, total_classes: 2 },
        { plant_type: 'Potato', diseases: 2, healthy: 1, total_classes: 3 },
        { plant_type: 'Strawberry', diseases: 1, healthy: 1, total_classes: 2 },
        { plant_type: 'Tomato', diseases: 9, healthy: 1, total_classes: 10 },
        { plant_type: 'Blueberry', diseases: 0, healthy: 1, total_classes: 1 },
        { plant_type: 'Orange', diseases: 1, healthy: 0, total_classes: 1 },
        { plant_type: 'Raspberry', diseases: 0, healthy: 1, total_classes: 1 },
        { plant_type: 'Soybean', diseases: 0, healthy: 1, total_classes: 1 },
        { plant_type: 'Squash', diseases: 1, healthy: 0, total_classes: 1 }
    ];
}

function getDefaultModelData() {
    return {
        architecture: 'EfficientNet B4',
        input_size: '380×380',
        total_parameters: '80M',
        trainable_parameters: '1.2M',
        frozen: true,
        dropout: '0.3'
    };
}

async function loadInitialData() {
    try {
        // Test API connection first
        const healthResponse = await fetch(`${CONFIG.API_BASE_URL}/health`, {
            timeout: 5000
        }).catch(() => null);
        
        if (!healthResponse || !healthResponse.ok) {
            console.warn('API not responding, using default data');
            AppState.plantsData = getDefaultPlantsData();
            AppState.modelData = getDefaultModelData();
            return;
        }
        
        const healthData = await healthResponse.json();
        
        // Load plants data
        const plantsResponse = await fetch(`${CONFIG.API_BASE_URL}/plants`);
        const plantsData = plantsResponse.ok ? await plantsResponse.json() : null;
        
        // Load model info
        const modelResponse = await fetch(`${CONFIG.API_BASE_URL}/model/info`);
        const modelData = modelResponse.ok ? await modelResponse.json() : null;
        
        // Update state
        AppState.systemInfo = {
            modelLoaded: healthData.model?.loaded || false,
            totalClasses: plantsData?.total_classes || 38,
            totalPlants: plantsData?.total_plants || 14,
            modelAccuracy: '95%'
        };
        
        AppState.plantsData = plantsData?.plants || getDefaultPlantsData();
        AppState.modelData = modelData?.model || getDefaultModelData();
        
    } catch (error) {
        console.warn('Error loading initial data, using defaults:', error);
        AppState.plantsData = getDefaultPlantsData();
        AppState.modelData = getDefaultModelData();
    }
}

// ============================================
// 9. UI UPDATE FUNCTIONS
// ============================================
function loadPlantsGrid() {
    if (!Elements.plantsGrid) return;
    
    try {
        const plants = AppState.plantsData || getDefaultPlantsData();
        let html = '';
        
        plants.forEach(plant => {
            const icon = PLANT_ICONS[plant.plant_type] || plant.icon || 'leaf';
            
            html += `
                <div class="plant-card" onclick="showPlantDetails('${plant.plant_type}', '${icon}')">
                    <i class="fas fa-${icon}"></i>
                    <div class="plant-name">${plant.plant_type}</div>
                    <div class="plant-stats">
                        <span class="plant-stat diseases">
                            <i class="fas fa-bug"></i> ${plant.diseases || 0}
                        </span>
                        <span class="plant-stat healthy">
                            <i class="fas fa-heart"></i> ${plant.healthy || 0}
                        </span>
                    </div>
                </div>
            `;
        });
        
        Elements.plantsGrid.innerHTML = html;
    } catch (error) {
        console.error('Error loading plants grid:', error);
        Elements.plantsGrid.innerHTML = '<p>Error loading plant information</p>';
    }
}

function loadModelInfo() {
    if (!Elements.modelInfo) return;
    
    try {
        const modelData = AppState.modelData || getDefaultModelData();
        
        let html = `
            <div class="model-info-item">
                <span class="model-info-label">Architecture</span>
                <span class="model-info-value">${modelData.architecture || 'EfficientNet B4'}</span>
            </div>
            <div class="model-info-item">
                <span class="model-info-label">Input Size</span>
                <span class="model-info-value">${modelData.input_size || '380×380'}</span>
            </div>
            <div class="model-info-item">
                <span class="model-info-label">Parameters</span>
                <span class="model-info-value">${modelData.total_parameters || '80M'}</span>
            </div>
            <div class="model-info-item">
                <span class="model-info-label">Trainable</span>
                <span class="model-info-value">${modelData.trainable_parameters || '1.2M'}</span>
            </div>
            <div class="model-info-item">
                <span class="model-info-label">Frozen Layers</span>
                <span class="model-info-value">${modelData.frozen ? 'Yes' : 'No'}</span>
            </div>
        `;
        
        Elements.modelInfo.innerHTML = html;
    } catch (error) {
        console.error('Error loading model info:', error);
        Elements.modelInfo.innerHTML = '<p>Error loading model information</p>';
    }
}

function updateUI() {
    try {
        // Update stats
        if (Elements.statClasses) {
            Elements.statClasses.textContent = AppState.systemInfo.totalClasses;
        }
        
        if (Elements.statPlants) {
            Elements.statPlants.textContent = AppState.systemInfo.totalPlants;
        }
        
        if (Elements.modelAccuracy) {
            Elements.modelAccuracy.textContent = AppState.systemInfo.modelAccuracy;
        }
        
        if (Elements.modelLayers && AppState.modelData.total_parameters) {
            Elements.modelLayers.textContent = AppState.modelData.total_parameters;
        }
        
        // Load plants grid
        loadPlantsGrid();
        
        // Load model info
        loadModelInfo();
        
    } catch (error) {
        console.error('Error updating UI:', error);
    }
}

// ============================================
// 10. DRAG & DROP FUNCTIONS
// ============================================
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function initializeDragAndDrop() {
    if (!Elements.uploadArea) return;
    
    try {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            Elements.uploadArea.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });
        
        // Highlight drop area
        ['dragenter', 'dragover'].forEach(eventName => {
            Elements.uploadArea.addEventListener(eventName, () => {
                Elements.uploadArea.classList.add('drag-over');
            }, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            Elements.uploadArea.addEventListener(eventName, () => {
                Elements.uploadArea.classList.remove('drag-over');
            }, false);
        });
        
        // Handle dropped files
        Elements.uploadArea.addEventListener('drop', handleDrop, false);
        
    } catch (error) {
        console.error('Error initializing drag and drop:', error);
    }
}

// ============================================
// 11. EVENT LISTENERS
// ============================================
function initializeEventListeners() {
    try {
        // Single upload button
        if (Elements.singleUploadBtn) {
            Elements.singleUploadBtn.addEventListener('click', () => {
                if (Elements.fileInput) Elements.fileInput.click();
            });
        }
        
        // File input
        if (Elements.fileInput) {
            Elements.fileInput.addEventListener('change', handleFileSelect);
        }
        
        // Upload area click
        if (Elements.uploadArea) {
            Elements.uploadArea.addEventListener('click', (e) => {
                if (e.target === Elements.uploadArea && Elements.fileInput) {
                    Elements.fileInput.click();
                }
            });
        }
        
        // Buttons with null checks
        if (Elements.predictBtn) {
            Elements.predictBtn.addEventListener('click', handlePredict);
        }
        
        if (Elements.clearBtn) {
            Elements.clearBtn.addEventListener('click', handleClear);
        }
        
        if (Elements.closeResults) {
            Elements.closeResults.addEventListener('click', hideResults);
        }
        
        // Modal close buttons - Use the actual function reference
        document.querySelectorAll('.close-modal').forEach(btn => {
            btn.addEventListener('click', closeModal);
        });
        
        // Close modal on overlay click
        document.querySelectorAll('.modal').forEach(modal => {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    closeModal();
                }
            });
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', handleKeyboardShortcuts);
        
    } catch (error) {
        console.error('Error initializing event listeners:', error);
    }
}

// ============================================
// 12. INITIALIZATION
// ============================================
async function initializeApp() {
    try {
        // Check if required elements exist
        if (!Elements.loadingScreen) {
            console.warn('Loading screen element not found');
        }
        
        // Initialize components one by one with error handling
        initializeEventListeners();
        initializeDragAndDrop();
        
        // Load initial data
        await loadInitialData();
        
        // Update UI
        updateUI();
        
        // Hide loading screen after a delay
        setTimeout(() => {
            if (Elements.loadingScreen) {
                Elements.loadingScreen.style.opacity = '0';
                setTimeout(() => {
                    if (Elements.loadingScreen) {
                        Elements.loadingScreen.style.display = 'none';
                    }
                }, 300);
            }
        }, 1000);
        
        console.log('✅ Application initialized successfully (38 Classes)');
        
    } catch (error) {
        console.error('Initialization error:', error);
        showToast('Initialization Error', error.message, 'error');
        
        // Show app anyway even if some features fail
        if (Elements.loadingScreen) {
            Elements.loadingScreen.style.display = 'none';
        }
    }
}

// ============================================
// 13. GLOBAL EXPORTS
// ============================================
// Export all functions that are called from HTML onclick attributes
window.removeFile = removeFile;
window.handlePredict = handlePredict;
window.showPlantDetails = showPlantDetails;
window.showModelStatusModal = showModelStatusModal;
window.showHelpModal = showHelpModal;
window.closeModal = closeModal;

// ============================================
// 14. START APPLICATION
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    console.log('🌱 Plant Disease Classifier Initializing... (38 Classes)');
    
    // Add error boundary
    try {
        initializeApp();
    } catch (error) {
        console.error('Fatal initialization error:', error);
        // Show error message on page
        if (Elements.loadingScreen) {
            Elements.loadingScreen.innerHTML = `
                <div style="text-align: center; padding: 40px; color: #721c24;">
                    <i class="fas fa-exclamation-triangle" style="font-size: 48px; margin-bottom: 20px;"></i>
                    <h2>Application Error</h2>
                    <p>${error.message}</p>
                    <button onclick="location.reload()" style="
                        padding: 10px 20px;
                        background: #721c24;
                        color: white;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                        margin-top: 20px;
                    ">Reload Application</button>
                </div>
            `;
        }
    }
});