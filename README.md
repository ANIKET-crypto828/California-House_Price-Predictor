California Housing Price Prediction

This repository contains a simple Linear Regression model using the California Housing dataset from sklearn.datasets. The model predicts housing prices based on a single feature (average number of rooms per dwelling).

Features

Uses Scikit-learn's California Housing dataset

Linear Regression Model for price prediction

Performance Evaluation using Mean Squared Error (MSE) and R-squared (R²) Score

Visualization of Predictions

Installation

Clone the repository:

git clone https://github.com/ANIKET-crypto828
/california-housing-prediction.git
cd california-housing-prediction

Install dependencies:

pip install -r requirements.txt

Dependencies

Python 3.x

NumPy

Pandas

Matplotlib

Scikit-learn

Install dependencies using:

pip install numpy pandas matplotlib scikit-learn

Usage

Run the script using:

python main.py

Code Overview

Load Dataset: Fetches the California Housing dataset from Scikit-learn.

Preprocessing: Extracts the third feature (MedInc - Median Income) and splits data into training/testing sets.

Model Training: Implements Linear Regression.

Evaluation:

Computes Mean Squared Error (MSE)

Computes R² Score

Visualization:

Plots actual vs. predicted prices

Model Evaluation

The model is evaluated using:

Mean Squared Error (MSE): Measures prediction error.

R² Score: Determines how well the model explains variance.

Example Output

Mean Squared Error: 0.5
R-squared (R²) Score: 0.72
Weights: [0.42]
Intercept: 1.23

Results

If R² is close to 1, the model is performing well.

If R² is near 0 or negative, the model needs improvement (try using more features).

Contributions

Feel free to fork this repository and submit pull requests for improvements!

License

This project is licensed under the MIT License.

Author: Aniket Santra
GitHub: ANIKET-crypto828


