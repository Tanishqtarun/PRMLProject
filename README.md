Handwritten Digit Recognition with MNIST

1. Project Overview

This project aims to build a machine learning model capable of recognizing handwritten digits (0–9) using the MNIST dataset. It explores multiple classification algorithms and evaluates their performance, culminating in a real-time digit recognition system.

2. Dataset

- Source: [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- Size: 70,000 grayscale images (60,000 for training, 10,000 for testing)
- Format: 28x28 pixel grayscale images of handwritten digits

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

Model Training

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

Live Prediction
- Developed a real-time digit recognition interface allowing users to draw digits for prediction.

5. Challenges Encountered

- Low Accuracy with Linear Regression: Unable to model complex patterns.
- Data Noise: Needs improved preprocessing and noise filtering.
- Overfitting in Decision Trees: Requires pruning and regularization.
- KNN Issues:
  - High computational complexity
  - High memory usage
  - Slow prediction time
  - Choice of distance metric impacts accuracy
  - Curse of dimensionality affects performance

6. Remaining Tasks

- [ ] Optimize KNN, Decision Tree, Bayesian Classifier, and SVM models
- [ ] Improve preprocessing using noise reduction and data augmentation
- [ ] Tune hyperparameters for all models
- [ ] Evaluate models using accuracy, precision, recall, and F1-score
- [ ] Build a user interface for real-time digit prediction
- [ ] Finalize deployment

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

9.Acknowledgements

MNIST dataset by Yann LeCun
Scikit-learn and OpenCV documentation
