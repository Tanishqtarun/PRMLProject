Handwritten Digit Recognition with MNIST

📁 Project Structure
PRMLProject/
├── Data/                            # Dataset files
├── Dataset/                         # Backup or preprocessed data
├── src/                             # Source code
│   └── utils.py                     # Utility functions
├── Bayesian_Classifier_model.ipynb
├── Decision_tree_model.ipynb
├── KNN_model.ipynb
├── Linear_Regression_model.ipynb
├── Support_Vector_Machine_model.ipynb
├── Handwritten Digit Recognition_Midreport.pdf
└── README.md

1. Project Overview

This project implements and compares multiple machine learning algorithms to classify handwritten digits from the MNIST dataset. It aims to explore both linear and nonlinear classifiers, highlight their performance, and evaluate their real-world applicability, especially in real-time digit recognition systems like postal code readers or bank check digitizers.

2.Motivation
Handwriting recognition is a classic machine learning task with practical significance. Despite being well-studied, it offers valuable insights into:
Feature selection and dimensionality reduction
Model training and evaluation
Trade-offs between accuracy and computational complexity
Generalization and overfitting behavior


📂 Dataset
Name: Modified National Institute of Standards and Technology (MNIST)
Source: Yann LeCun’s MNIST Database
Structure:
60,000 training images
10,000 testing images
Each image: 28x28 pixels (grayscale)
Labels: Digits from 0 to 9

3. Problem Statement

Train and evaluate models to accurately classify handwritten digits based on pixel values. The final system should support real-time digit recognition.

4. Progress So Far

Data Collection and Preprocessing
- Collected and converted images to array format.
- Pixel values binarized:
  - 0 for pixel values between 0–100
  - 1 for pixel values between 100–255
- Saved dataset in CSV format.
- Data shuffled for unbiased training.

5.Model Training

Implemented and evaluated the following machine learning models:

- Linear Regression 
  - Used as a baseline classifier.
  - Performed poorly on classification due to its continuous output and inability to model complex patterns.

- K-Nearest Neighbors (KNN) 
  - Achieved significantly better accuracy.
  - Sensitive to the choice of distance metric and number of neighbors.
  - Issues: high memory usage, slow prediction time.

- Decision Tree 
  - Capable of learning non-linear boundaries.
  - Tends to overfit without pruning or regularization.

- Bayesian Classifier (Naive Bayes)  
  - Fast and memory-efficient.
  - Assumes feature independence, which may not hold in image data.
  - Still provided decent accuracy with minimal resource usage.

- Support Vector Machine (SVM)  
  - Implemented with both linear and RBF kernels.
  - Provided high accuracy and good generalization.
  - Computationally intensive for large datasets.

Implemented Models

Model                        | Description                                                            | Notebook
___________________________________________________________________________________________________________________________________________
K-Nearest Neighbors (KNN)    | Non-parametric model that classifies based on distance from neighbors. | KNN_model.ipynb                    |
Support Vector Machine (SVM) | A powerful linear classifier with kernel support for non-linear cases. | Support_Vector_Machine_model.ipynb |
Decision Tree                | A simple yet interpretable tree-based classification model.            | Decision_tree_model.ipynb          |
Bayesian Classifier          | Probabilistic model based on Bayes’ theorem.                           | Bayesian_Classifier_model.ipynb    |
Linear Regression            | Adapted for classification using thresholding.                         | Linear_Regression_model.ipynb      |  

Evaluation Metrics:

Each model is evaluated using the following metrics:
Accuracy
Precision, Recall & F1-Score
Confusion Matrix
Inference Time

Visualizations:

Learning curves
Confusion matrices
PCA-based 2D visualizations of high-dimensional data
Sample misclassifications

Live Prediction
- Developed a real-time digit recognition interface allowing users to draw digits for prediction.

6. Challenges Encountered

- Low Accuracy with Linear Regression: Unable to model complex patterns.
- Data Noise: Needs improved preprocessing and noise filtering.
- Overfitting in Decision Trees: Requires pruning and regularization.
- KNN Issues:
  - High computational complexity
  - High memory usage
  - Slow prediction time
  - Choice of distance metric impacts accuracy
  - Curse of dimensionality affects performance
    
7. Technologies Used

- Python
- NumPy, Pandas
- Scikit-learn
- OpenCV (for live digit capture)
- Tkinter / PyQt / Streamlit (for GUI)

8. How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   
2.Install dependencies:
pip install -r requirements.txt

3.Run the training script:
python train_model.py

4.Launch the real-time prediction UI:
python live_prediction.py

9.Acknowledgments

MNIST dataset by Yann LeCun et al.
References from PRML lectures and external scikit-learn documentation
Visualization libraries: matplotlib, seaborn, plotly

10.Future Work

Hyperparameter tuning using Grid Search or Bayesian Optimization
Integration with a GUI using Tkinter or Streamlit for digit input
Real-time recognition via webcam and OpenCV
Neural Network / Deep Learning models (CNNs)



