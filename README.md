 # Data Science Midterm Project

## Project/Goals
This project aims to predict the selling price of real estate properties using multiple machine learning models. The dataset contains features such as yar built, building ratio, square footage, city encoding, garage availability, number of stores and bedrooms.  The models implmented inclue 
Linear Regression

XGBoost

Random Forest

K-Nearest Neighbors (KNN)

Support Vector Regression (SVR)

## Process
### Extracting JSON data
- Looped over all JSON data, filtering records with no listings, loaded and saved to Pandas Dataframe for cleaning 
- Loaded the data into a Pandas DataFrame for further cleaning and preprocessing.
### Data Preparation  
#### Handling Missing Data: 
- The 4 columns with the most (>70%) missing data were removed, as well as irrelevant columns and rows whre the target is null.
- Null values for features such as `stories`, `garage`, `beds` etc were imputed with 0
- Null values for the `year_built` column was imputed based on the `type` column:
  - there were many entries of `single_family` type, entries were grouped by city, then imputed with the most frequently occuring year for the listing
  - `land` and `condo` types only had one entry each in the whole dataframe, so null values where imputed according to the single entry
  - there were only 2 entries of `other` type, and they were removed upon manual inspection of being incomplete
  - The remaining few entries were imputed by manual search, or removed if manual search yielded no results
#### Encoding Categorical Columns:
- Encode_tags function was applied with minimum occurance of 100
- `City` and `state` columns were encoded using `TargetEncoder` from the `category_encoder` module to account for both the group and global mean 
#### Feature Engineering: 
- `Total sqft`: sum of `sqft` and `lot_sqft`
- `building_ratio`: ratio of house sqft to lot sqft, higher ratio = more building area, 0 = all lot no building
#### Scaling: using sk-learn StandardScaler() on select columns 
#### Train/Test Split: Divided the dataset using K-Fold 
#### Scaling & Normalization: Applied scaling to numeric features to improve model performance.


## Results

| Model               | Mean Squared Error | RMSE ($)     | R² Score | Mean Absolute Error ($) | Adjusted R² Score |
|---------------------|------------------  |--------------|----------|------------------       |----------------|
| Linear Regression   | 55,286,464,174.43  | N/A          | 0.43     | 151,410.45              | 0.38           |
| XGBoost             | N/A                | 188,629.67   | 0.61     | 95,876.97               | N/A            |
| Random Forest       | N/A                | 196,497.12   | 0.57     | 97,171.21               | N/A            |
| K-Nearest Neighbors | N/A                | 259,243.99   | 0.26     | 145,165.42              | N/A            |
| Support Vector Reg. | N/A                | 305,003.43   | -0.03    | 190,197.95              | N/A            |



## Challenges and Limitations 
### Data Processing 
It took some thought to determine the best way to impute missing values for the `year_built` column. Ultimately it was done using the grouped city/state mode or the value of the single entry in the dataframe. This sort of imputation will result in the model having less variability. 

## Future Goals
A next step for this project would be to test the model and predict on a new set of data. Implementing external datasets such as city/statte climate or economy information can also be used to construct a more robust model. 
