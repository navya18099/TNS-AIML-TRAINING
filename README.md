# Retail Sales Forecasting

Forecasting daily retail sales using Holt-Winters, Random Forest and XGBoost.

## Dataset
Retail sales data from Kaggle (link here). Not included because of file size.

## Approach
- Daily total sales from 4.5M records (2019-2023)
- Time-based 80/20 train/test split
- Features: weekday, month, day of year, promo, price, time index, 364-day lag

## Results
| Model | MAE | RMSE |
|---|---|---|
| Naive | 27,612 | 31,492 |
| Random Forest | 506 | 820 |
| XGBoost | 438 | 715 |

## How to run
pip install -r requirements.txt
Download the dataset, place retail_sales.csv in this folder, run the notebook.