# 6740-Group-Project---Alpha-Group

MGT6769-Final-Project
Bond Risk Premia with Machine Learning
This project aims to replicate the findings of the paper "Bond Risk Premia with Machine Learning" by Daniele Bianchi, Matthias Büchner, and Andrea Tamoni (2019). The paper explores the use of various machine learning methods for predicting bond excess returns and measuring bond risk premia.

Overview
The main objectives of this project are:

Implement and compare different machine learning methods for predicting bond excess returns, including:

Principal Component Regression (PCR)
Partial Least Squares (PLS)
Penalized Linear Regressions (Ridge, Lasso, Elastic Net)
Regression Trees and Random Forests
Neural Networks (Shallow and Deep)
Evaluate the out-of-sample predictive performance of these methods using mean squared prediction error (MSPE) and out-of-sample R-squared (R2_oos).

Investigate the economic significance of the predictability by calculating Sharpe ratios based on the predicted bond excess returns.

Analyze the relative importance of macroeconomic variables in predicting bond excess returns using neural networks.

Data
The project uses the following datasets:

Yield curve data: U.S. Treasury bond yields for different maturities (1-year, 2-year, 3-year, 5-year).
Macroeconomic variables: A large panel of macroeconomic and financial variables.
Code Structure
The code for this project is organized as follows:

main.ipynb: Jupyter notebook containing the main code for data preprocessing, model training, evaluation, and analysis.
utils.py: Python module containing utility functions and classes used in the main notebook.
data/: Directory containing the raw and processed datasets.
models/: Directory to store the trained machine learning models.
results/: Directory to store the evaluation results and figures.
Dependencies
The project requires the following dependencies:

Python 3.x
NumPy
Pandas
Scikit-learn
PyTorch
Matplotlib
Seaborn
Usage
Install the required dependencies.
Place the raw datasets in the data/ directory.
Run the main.ipynb notebook to execute the code and reproduce the results.
Results
The main findings of this project are:

Machine learning methods, particularly neural networks, can capture a significant amount of time-series variation in expected bond excess returns and outperform traditional benchmarks like PCR.
Macroeconomic information has substantial out-of-sample forecasting power for bond excess returns, especially when combined with non-linear methods like deep neural networks.
The composition of the best predictors varies across the term structure, with financial variables being more important for short-term bonds and macroeconomic variables being more relevant for long-term bonds.
For detailed results and analysis, please refer to the main.ipynb notebook and the results/ directory.

References
Bianchi, D., Büchner, M., & Tamoni, A. (2019). Bond Risk Premia with Machine Learning. Journal of Financial Economics, forthcoming.
