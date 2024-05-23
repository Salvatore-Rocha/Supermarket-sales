#  Supermarket Sales Dashboard: Metrics of Myanmar HUB

Welcome to this Supermarket Sales Dashboard! 
This project is built using the Python library of Dash and uses the dataset of the historical sales from a supermarket company. This dash app analyzes the records from three branches over three months and provides predictions and insights to help users understand store performance better.

Link to the online app on render. It uses the free tier of render, so it might take up to 2 minutes to load all the components; please load the main code in the "scr" folder and run locally for faster results.

[Urban Sales Metrics of Myanmar HUB](https://urban-insights-predict-and-visualize.onrender.com/)

The data set can be found at: 

[Supermarket sales](https://www.kaggle.com/datasets/aungpyaeap/supermarket-sales)

Here is a Jupyter notebook containing draft ideas and a demonstration of the logic behind data processing and function declaration.

[Sales Jupyter Notebook](https://colab.research.google.com/drive/1UX7Bah8Sn1WaajXQQzInp-4RprBnz2qD?usp=sharing)

Here is a preview of the prediction tab of the app and the sales overview on the rest of them:

![Prediction Tab](https://github.com/Salvatore-Rocha/Supermarket-sales/blob/3e39c4f5880c260cf61c87e0c0a732c44d8a2191/Imgs/Sales_Dashboard_Ex1.jpg)
![Sales OVerview](https://github.com/Salvatore-Rocha/Supermarket-sales/blob/3e39c4f5880c260cf61c87e0c0a732c44d8a2191/Imgs/Sales_Dashboard_Ex2.jpg)

Features
1. Store Rating and Sales Predictions
This dashboard predicts store rating (on a scale from 1 to 10) and total sales using a trained "Random Forest Regressor" model. Users can decide which categorical variables to include, and the model will generate predictions based on these inputs.

2. Feature Importance Evaluation
The app also evaluates and displays the importance of the input features, helping users understand which variables most impact store ratings and sales.

3. Training and Validation Curves
To visualizse the model's effectiveness, the app provide both training and validation curves:

Training Curve: Shows the model's performance on the training dataset over iterations, indicating how well the model learns from the data.
Validation Curve: Illustrates the model's performance on a separate validation dataset, crucial for evaluating the model's ability to generalize to new data.

The comparison between these curves helps identify potential issues like overfitting (good performance on training data but poor on validation data) or underfitting (poor performance on both).

4. Data Segmentation and Visualization
The app includes four interactive tabs, each performing data transformation and constructing visualizations to provide insights into different aspects of the sales data:

Sales Segmentation: Analyze how sales vary across different stores and overall.
Cost of Goods Sold (COGS): Understand the distribution and behavior of COGS in each store and collectively.
Customer Segmentatio: Visualize customer segmentation and behaviors in individual stores and across all branches.
Combined Analysis: Get a comprehensive view of sales, COGS, and customer data for all branches
