# Breast Cancer Classification Project

## Overview
This project focuses on building machine learning models to classify breast cancer as malignant or benign. The goal is to compare the performance of various machine learning algorithms to identify the most effective model for this classification task.

## Key Features
- **Dataset**: The project uses a breast cancer dataset from https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset/data
- **Algorithms Implemented**:
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - Support Vector Machine (SVM)
  - Naive Bayes
- **Metrics Used**:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC

## Dataset
The dataset includes features such as radius, texture, perimeter, area, and smoothness of the cell nuclei. The target variable indicates whether the tumor is **malignant** or **benign**.

### Dataset Structure
- **Features**: Numerical attributes derived from image processing.
- **Target**: Binary label (0 for benign, 1 for malignant).

## Methodology
1. **Data Preprocessing**:
   - Handled missing values and outliers.
   - Normalized numerical features for algorithms sensitive to scaling.
   - Split data into training and testing sets.

2. **Model Training**:
   - Implemented five machine learning algorithms.
   - Performed hyperparameter tuning using grid search or cross-validation.

3. **Evaluation**:
   - Compared model performance using the defined metrics.
   - Visualized results with confusion matrices and ROC curves.

4. **Model Selection**:
   - Selected the best-performing model based on evaluation metrics.

## Results
- Each algorithm’s performance is summarized in a comparison table and corresponding visualizations.
- Insights into the effectiveness of each algorithm for breast cancer classification are discussed.

## Prerequisites
- Python (3.8 or higher)
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd breast-cancer-classification
   ```
3. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main script to train and evaluate models:
```bash
python main.py
```

## File Structure
- **data/**: Contains the dataset.
- **notebooks/**: Jupyter notebooks for data exploration and model development.
- **scripts/**: Python scripts for preprocessing, training, and evaluation.
- **results/**: Outputs such as performance metrics and visualizations.

## Future Work
- Experiment with additional machine learning algorithms such as Gradient Boosting and Neural Networks.
- Implement feature engineering techniques to enhance model performance.
- Integrate the model into a web application for real-time predictions.

## Conclusion
This project demonstrates the application of machine learning to a critical healthcare problem. The comparative analysis provides insights into the strengths and limitations of different algorithms for breast cancer classification.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Dataset courtesy of [(https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset/data)].
- Inspiration and guidance from the machine learning community.

