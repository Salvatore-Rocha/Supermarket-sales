# Urban Sales Metrics of Myanmar HUB

Welcome to this Supermarket Sales Dashboard! 
This project is built using the Python library of Dash and uses the dataset of the historical sales from a supermarket company. This dash app analyzes the records from three branches over three months and provides predictions and insights to help users understand store performance better.

The data set can be found at: 
[Supermarket sales](https://www.kaggle.com/datasets/aungpyaeap/supermarket-sales)

Here is a preview of the prediction tab of the app and the sales overview on the rest of them:

![Prediction Tab](https://github.com/Salvatore-Rocha/Supermarket-sales/blob/3e39c4f5880c260cf61c87e0c0a732c44d8a2191/Imgs/Sales_Dashboard_Ex1.jpg)
![Sales OVerview](https://github.com/Salvatore-Rocha/Supermarket-sales/blob/3e39c4f5880c260cf61c87e0c0a732c44d8a2191/Imgs/Sales_Dashboard_Ex2.jpg)

Features
1. Store Rating and Sales Predictions
Our dashboard predicts store rating (on a scale from 1 to 10) and total sales using a trained Random Forest Regressor model. Users can input various categorical variables, and the model will generate predictions based on these inputs.

2. Feature Importance Evaluation
The app evaluates and displays the importance of different features, helping users understand which variables most impact store ratings and sales.

3. Training and Validation Curves
To ensure the model's effectiveness, we provide both training and validation curves:

Training Curve: Shows the model's performance on the training dataset over iterations, indicating how well the model learns from the data.
Validation Curve: Illustrates the model's performance on a separate validation dataset, crucial for evaluating the model's ability to generalize to new data.
The comparison between these curves helps identify potential issues like overfitting (good performance on training data but poor on validation data) or underfitting (poor performance on both).
4. Data Segmentation and Visualization
The app includes four interactive tabs, each performing data transformation and constructing visualizations to provide insights into different aspects of the sales data:

Sales Segmentation: Analyze how sales vary across different stores and overall.
Cost of Goods Sold (COGS): Understand the distribution and behavior of COGS in each store and collectively.
Customer Behavior: Visualize customer patterns and behaviors in individual stores and across all branches.
Combined Analysis: Get a comprehensive view of sales, COGS, and customer data for all branches.
This notebook presents plots derived from historical sales data collected from three distinct supermarkets. The original dataset is accessible at the following URL:
 https://www.kaggle.com/datasets/aungpyaeap/supermarket-sales

The plots presented here will be used as baseline to construct an interactive dash app.
A brief description of what this project does and who it's for.

 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

Instructions for setting up the project locally. For example:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
npm install
