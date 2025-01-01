# Fruit-Quality-Detection

Here's a well-structured and detailed README.md for your "Fruit Quality Detection" project. You can copy and paste it directly to your GitHub repository. If you want to customize it further, let me know!

🍎 Fruit Quality Detection using CNN
A deep learning project to classify fruit quality as fresh or rotten using a Convolutional Neural Network (CNN). This project processes images of apples and bananas to determine their quality, enabling automated quality detection through a trained model.

🚀 Project Overview
This project is designed to classify fruit quality as either fresh or rotten using a CNN-based model. The dataset consists of labeled images of apples and bananas, categorized into fresh and rotten. The model is trained to recognize patterns in the images and predict the quality of new fruit images.

The solution uses:

Convolutional Neural Networks (CNN) for feature extraction and classification.
A user-friendly approach to input test images and evaluate predictions.

<br>
Fruit-Quality-Detection/
├── dataset/                # Dataset folder
│   ├── train/              # Training data
│   │   ├── fresh/          # Fresh fruit images
│   │   ├── rotten/         # Rotten fruit images
│   ├── test/               # Testing data
│       ├── fresh/          # Fresh fruit images
│       ├── rotten/         # Rotten fruit images
├── fruit_quality_model.h5  # Saved CNN model
├── fruit_quality_detection.ipynb # Jupyter notebook with code
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation <br>


✨ Features
Dataset Preparation: Preprocessed images of apples and bananas (150x150 resolution) into train and test folders.

CNN Architecture:
Convolutional Layers for feature extraction.
Pooling Layers to reduce dimensionality.
Dense Layers for classification.
Training and Validation:
Tracks accuracy and loss over epochs.
Visualizes training performance.

Image Prediction:
Allows users to test images through a simple file path input.
Outputs the classification as either "Fresh" or "Rotten."
User Interaction:
Drag-and-drop interface replaced with manual file path input for simplicity.


📊 Dataset
The dataset used in this project contains:

Fresh Apples: Images of fresh apples.
Rotten Apples: Images of rotten apples.
Fresh Bananas: Images of fresh bananas.
Rotten Bananas: Images of rotten bananas.
The dataset is organized into training and testing folders. Example:

dataset/
├── train/
│   ├── fresh/
│   ├── rotten/
├── test/
│   ├── fresh/
│   ├── rotten/

🛠️ Tools and Technologies
Python 3.9+
TensorFlow 2.x
Keras
OpenCV
NumPy
Matplotlib

⚙️ Installation and Setup
Clone the repository:
git clone https://github.com/yourusername/fruit-quality-detection.git
cd fruit-quality-detection

pip install -r requirements.txt

Dataset Preparation:

Place the dataset in the dataset/ folder as described in the structure.
Ensure images are organized into train/ and test/ folders.
Run the Jupyter Notebook: Open the notebook fruit_quality_detection.ipynb in Jupyter Notebook or Google Colab and run the cells step-by-step.

🧠 Model Training
The model is trained using the following parameters:

Epochs: 10
Batch Size: 32
Loss Function: Binary Crossentropy
Optimizer: Adam
Metrics: Accuracy
To retrain the model, run the training cells in the notebook.

🎯 How to Use
Train the Model:

Use the fruit_quality_detection.ipynb to train the model on your dataset.
Save the Model:

The trained model is saved as fruit_quality_model.h5.
Test Predictions:

Use the classify_image() function to test a single image.
Provide the path to the image (e.g., C:/path/to/image.jpg).
The model will output either Fresh or Rotten.
Example:
image_path = "C:/path/to/image.jpg"
result = classify_image(image_path)
print(f"The fruit is: {result}")

Here's a well-structured and detailed README.md for your "Fruit Quality Detection" project. You can copy and paste it directly to your GitHub repository. If you want to customize it further, let me know!

🍎 Fruit Quality Detection using CNN
A deep learning project to classify fruit quality as fresh or rotten using a Convolutional Neural Network (CNN). This project processes images of apples and bananas to determine their quality, enabling automated quality detection through a trained model.

🚀 Project Overview
This project is designed to classify fruit quality as either fresh or rotten using a CNN-based model. The dataset consists of labeled images of apples and bananas, categorized into fresh and rotten. The model is trained to recognize patterns in the images and predict the quality of new fruit images.

The solution uses:
Convolutional Neural Networks (CNN) for feature extraction and classification.
A user-friendly approach to input test images and evaluate predictions.

📂 Project Structure
bash
Copy code
Fruit-Quality-Detection/
├── dataset/                # Dataset folder
│   ├── train/              # Training data
│   │   ├── fresh/          # Fresh fruit images
│   │   ├── rotten/         # Rotten fruit images
│   ├── test/               # Testing data
│       ├── fresh/          # Fresh fruit images
│       ├── rotten/         # Rotten fruit images
├── fruit_quality_model.h5  # Saved CNN model
├── fruit_quality_detection.ipynb # Jupyter notebook with code
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

✨ Features
1. Dataset Preparation:
2. Preprocessed images of apples and bananas (150x150 resolution) into train and test folders.
3. CNN Architecture:
Convolutional Layers for feature extraction.
Pooling Layers to reduce dimensionality.
Dense Layers for classification.
4. Training and Validation:
Tracks accuracy and loss over epochs.
Visualizes training performance.
5. Image Prediction:
Allows users to test images through a simple file path input.
Outputs the classification as either "Fresh" or "Rotten."
6. User Interaction:
Drag-and-drop interface replaced with manual file path input for simplicity.

📊 Dataset
The dataset used in this project contains:
Fresh Apples: Images of fresh apples.
Rotten Apples: Images of rotten apples.
Fresh Bananas: Images of fresh bananas.
Rotten Bananas: Images of rotten bananas.
The dataset is organized into training and testing folders.
Example:

dataset/
├── train/
│   ├── fresh/
│   ├── rotten/
├── test/
│   ├── fresh/
│   ├── rotten/

🛠️ Tools and Technologies
Python 3.9+
TensorFlow 2.x
Keras
OpenCV
NumPy
Matplotlib

⚙️ Installation and Setup
Clone the repository:
git clone https://github.com/yourusername/fruit-quality-detection.git ------------> [bash]
cd fruit-quality-detection
Install dependencies:

pip install -r requirements.txt ------------> [bash]
Dataset Preparation:

Place the dataset in the dataset/ folder as described in the structure.
Ensure images are organized into train/ and test/ folders.
Run the Jupyter Notebook: Open the notebook fruit_quality_detection.ipynb in Jupyter Notebook or Google Colab and run the cells step-by-step.

🧠 Model Training
The model is trained using the following parameters:
Epochs: 10
Batch Size: 32
Loss Function: Binary Crossentropy
Optimizer: Adam
Metrics: Accuracy
To retrain the model, run the training cells in the notebook.

🎯 How to Use
Train the Model:
Use the fruit_quality_detection.ipynb to train the model on your dataset.
Save the Model:
The trained model is saved as fruit_quality_model.h5.
Test Predictions:
Use the classify_image() function to test a single image.
Provide the path to the image (e.g., C:/path/to/image.jpg).
The model will output either Fresh or Rotten.

Example:
image_path = "C:/path/to/image.jpg"
result = classify_image(image_path)
print(f"The fruit is: {result}")

📊 Results
Achieved training accuracy of ~98% and validation accuracy of ~95% after 10 epochs.
The model generalizes well for unseen test images.

📈 Performance Visualization
The model's performance during training is visualized using Matplotlib:
Training and Validation Accuracy:
Training and Validation Loss:

🔑 Key Learnings
How to preprocess image datasets for deep learning tasks.
Implementing a Convolutional Neural Network (CNN) for classification.
Visualizing and interpreting model performance metrics.
Deploying a deep learning model for practical use cases.

🖼️ Examples
Fresh Apple:
Rotten Banana:

🤝 Contributions
Feel free to fork this repository, create issues, or submit pull requests if you want to contribute or improve the project.
