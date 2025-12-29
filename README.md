🌱 Plant Disease Analyzer
AI-powered plant disease detection system that instantly identifies 33 diseases across 9 plant types using EfficientNet B4 deep learning model.

🚀 Features
AI-Powered Detection: 33 plant disease classes across 9 plant types

Real-Time Analysis: Fast image processing with confidence scores

User-Friendly Interface: Drag & drop, single-click upload

Detailed Results: Top predictions with visual confidence indicators

Plant Database: Complete information about supported plants and diseases

Responsive Design: Works on desktop and mobile devices

📸 Supported Plants
🍎 Apple (4 classes)

🍒 Cherry (2 classes)

🌽 Corn (4 classes)

🍇 Grape (4 classes)

🍑 Peach (2 classes)

🌶️ Pepper (2 classes)

🥔 Potato (3 classes)

🍓 Strawberry (2 classes)

🍅 Tomato (10 classes)

🛠️ Technology Stack
Backend: Flask, PyTorch, EfficientNet B4

Frontend: HTML5, CSS3, JavaScript

ML Framework: TorchVision, PIL

Deployment: Flask development server (production-ready with Gunicorn)

📋 Prerequisites
Python 3.8+

pip (Python package manager)

2GB+ RAM

Web browser with JavaScript enabled

⚡ Quick Start
1. Clone & Setup
bash
git clone <repository-url>
cd plant-disease-analyzer

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
2. Place Model File
Place the trained model file (model_imbalanced.pth) in the project root directory.

3. Run the Application
bash
python app.py
4. Access the Application
Open your browser and navigate to:

text
http://localhost:5000
📁 Project Structure
text
plant-disease-analyzer/
├── app.py                 # Flask backend application
├── model_imbalanced.pth   # Trained model weights
├── requirements.txt       # Python dependencies
├── static/               # Static assets (CSS, JS, images)
│   ├── css/
│   │   └── style.css    # Stylesheet
│   └── js/
│       └── app.js       # Frontend JavaScript
├── templates/
│   └── index.html       # Main web interface
├── uploads/             # User uploaded images
└── logs/               # Application logs
🔧 API Endpoints
GET / - Web interface

GET /api/health - System health check

GET /api/model/info - Model information

GET /api/plants - List all supported plants

POST /api/predict - Analyze single image

GET /api/plant/<type> - Get specific plant details

📝 Usage Instructions
Upload Image: Click "Upload Image" button or drag & drop

Select File: Choose a clear image of a plant leaf

Analyze: Click "Analyze Image" to process

View Results: See disease prediction with confidence score

Image Requirements:
Formats: JPG, JPEG, PNG

Maximum size: 16MB

Clear, focused images of plant leaves

Good lighting conditions

🎯 Model Information
Architecture: EfficientNet B4

Input Size: 380×380 pixels

Parameters: ~17 million

Accuracy: 95% (on test dataset)

Classes: 38 plant diseases/conditions

🔒 Security Notes
No personal data is collected or stored

All processing happens locally on the server

CORS enabled for development

🤝 Contributing
Fork the repository

Create a feature branch

Commit your changes

Push to the branch

Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Plant Village dataset for training images

PyTorch and TorchVision teams

EfficientNet research team

FontAwesome for icons

