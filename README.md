# Handwritten Equation Solver using CNN

A Streamlit-based web application that recognizes handwritten mathematical equations and provides real-time solutions using a Convolutional Neural Network (CNN). Users can draw equations on a canvas, and the application will identify the digits and operators, then calculate the result.

## 🚀 Features

- **Interactive Drawing Canvas**: Draw mathematical equations directly on the web interface
- **Real-time Recognition**: CNN-based recognition of handwritten digits (0-9) and operators (+, -, *, ÷)
- **Automatic Equation Solving**: Automatically evaluates recognized equations and displays results
- **Visual Feedback**: Shows bounding boxes around detected symbols with predicted labels
- **Responsive Design**: Clean and user-friendly Streamlit interface with custom styling
- **Symbol Detection**: Processes multiple symbols in sequence from left to right

## 🛠️ Technology Stack

- **Frontend**: Streamlit with streamlit-drawable-canvas
- **Deep Learning**: TensorFlow/Keras CNN Model
- **Image Processing**: OpenCV (cv2)
- **UI Components**: PIL (Pillow), NumPy
- **Model**: Pre-trained CNN model (`cnn_model.keras`)

## 📋 Prerequisites

Before running the application, ensure you have:

1. **Python 3.8+** installed
2. **Required Python packages** (see installation section)
3. **Pre-trained CNN model** (`cnn_model.keras` file)
4. **Logo image** (`logo.png` file)

## 🔧 Installation

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd handwritten-equation-solver
   ```

2. **Install required dependencies**:
```
pip install streamlit streamlit-drawable-canvas tensorflow opencv-python pillow numpy
```

3. **Ensure model and assets are in place**:
- Place `cnn_model.keras` in the root directory
- Place `logo.png` in the root directory
- Create `temp/` directory (auto-created by application)

## 🚀 Usage

1. **Start the Streamlit application**:
```
streamlit run app.py
```

2. **Access the application** in your browser (typically `http://localhost:8501`)

3. **Draw your equation**:
- Use the drawing canvas to write digits and mathematical operators
- Supported symbols: 0-9, +, -, *, ÷
- Adjust stroke width and colors as needed

4. **Get results**:
- Click the "Predict" button
- View the recognized equation and calculated result
- See visual feedback with bounding boxes around detected symbols

## 💡 How It Works

### Image Processing Pipeline
<img width="1109" height="722" alt="Screenshot 2025-08-02 015457" src="https://github.com/user-attachments/assets/d65feb86-9711-4ca8-9e51-20d7f8ef599b" />

1. **Canvas Capture**: User drawing is captured from the Streamlit canvas
2. **Image Preprocessing**: 
- Convert to grayscale
- Apply binary thresholding (THRESH_BINARY_INV)
- Detect contours for symbol segmentation

### Symbol Recognition
1. **Contour Detection**: Find individual symbols using OpenCV contours
2. **Bounding Box Extraction**: Get coordinates for each detected symbol
3. **Sorting**: Arrange symbols from left to right for proper equation order
4. **ROI Extraction**: Extract regions of interest with padding
5. **Preprocessing**: Resize to 32x32 pixels and normalize

### CNN Prediction
1. **Model Input**: Processed 32x32 grayscale images
2. **Prediction**: CNN model predicts symbol class (0-13)
3. **Label Mapping**: Convert predictions to readable symbols
4. **Equation Assembly**: Combine symbols to form complete equation

### Result Calculation
1. **String Processing**: Convert symbols to evaluable mathematical expression
2. **Safety Conversion**: Replace ÷ with / for Python evaluation
3. **Evaluation**: Calculate final result using Python's eval function

<img width="1011" height="744" alt="Screenshot 2025-08-02 015546" src="https://github.com/user-attachments/assets/3b369efc-4e7d-473a-a3ff-fd4d3ff09364" />

## ⚙️ Model Details
- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 32x32 grayscale images
- **Classes**: 14 classes (0-9 digits, +, -, *, ÷ operators)
- **Training Data**: Handwritten math symbols dataset
- **Model File**: `cnn_model.keras`

### Label Mapping
```
{0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
8: '8', 9: '9', 10: '+', 11: '÷', 12: '*', 13: '-'}
```

## 📁 Project Structure
```
handwritten-equation-solver/
├── app.py # Main Streamlit application
├── predict.py # Prediction and image processing logic
├── cnn_model.keras # Pre-trained CNN model
├── logo.png # Application logo
├── temp/ # Temporary image storage (auto-created)
├── notebook79e8678d82.ipynb # Training notebook (if included)
└── README.md # This file
```

## 🔍 Key Functions

### app.py
- **Main Interface**: Streamlit web application setup
- **Canvas Integration**: Drawing interface with customizable options
- **Result Display**: Shows equation and calculated result with styling

### predict.py
- `predict(image_path)`: Main prediction function
- **Image Processing**: Contour detection and ROI extraction
- **Model Inference**: CNN prediction and label mapping
- **Visualization**: Draws bounding boxes and labels on detected symbols

## 🎨 UI Features

- **Custom Styling**: CSS styling for improved visual appeal
- **Drawing Controls**: Adjustable stroke width and color options
- **Background Options**: Customizable canvas background
- **Result Formatting**: Dynamic sizing based on result length
- **Social Links**: GitHub and LinkedIn integration in header

## 🔧 Configuration

### Canvas Settings
- **Default stroke width**: 3-5 pixels
- **Canvas size**: 700x300 pixels
- **Background**: Customizable color
- **Drawing modes**: Freedraw, line, rectangle, etc.

### Image Processing Parameters
- **Threshold**: 128 (binary threshold)
- **Minimum contour area**: 100 pixels
- **Padding**: 25 pixels around detected symbols
- **Resize target**: 32x32 pixels

## 🛡️ Error Handling

- **File Management**: Automatic temp directory creation
- **Image Validation**: Contour area filtering to remove noise
- **Model Loading**: Graceful handling of model file issues
- **Evaluation Safety**: Safe mathematical expression evaluation

## 🔧 Troubleshooting

**Common Issues:**

1. **Model not found**: Ensure `cnn_model.keras` is in the root directory
2. **Poor recognition**: Try drawing symbols larger and clearer
3. **Canvas not responding**: Refresh the browser page
4. **Import errors**: Verify all dependencies are installed correctly
5. **Temp directory issues**: Check write permissions in project folder

## 📊 Performance Tips

- **Clear drawings**: Draw symbols with good contrast and spacing
- **Proper sizing**: Make symbols reasonably large on the canvas
- **Sequential writing**: Write equations from left to right
- **Clean strokes**: Avoid overlapping or broken lines

## 🤝 Contributing

Feel free to contribute to this project by:
- Improving the CNN model accuracy
- Adding support for more mathematical operators
- Enhancing the UI/UX design
- Adding data augmentation for better recognition
- Implementing equation history features

## 🙏 Acknowledgments

- Dataset: Handwritten Math Symbols Dataset
- Framework: Streamlit for rapid web app development
- Deep Learning: TensorFlow/Keras for CNN implementation
- Image Processing: OpenCV for computer vision tasks

**Note**: Ensure all dependencies are properly installed and the pre-trained model file is available before running the application.

