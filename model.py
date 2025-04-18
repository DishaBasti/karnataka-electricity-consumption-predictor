import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
#from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
 
def linear_reg(year, district, house_size, household_size):
    data = pd.read_csv(r"C:\Users\msacc\Downloads\ML\ML\templates\energyKarnataka.csv")

    # Define the feature and target variables
    X = data[['Year', 'District', 'House Size (sqft)', 'Household Size']]
    y_energy = data['Energy Consumption (kWh)']
    y_price = data['Price (in Rupees)']

    # One-hot encode the District column
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = pd.DataFrame(encoder.fit_transform(X[['District']]).toarray(), columns=encoder.get_feature_names_out(['District']))
    X = X.drop('District', axis=1)
    X = pd.concat([X, X_encoded], axis=1)

    # Split the data into training and testing sets
    X_train, X_test, y_energy_train, y_energy_test = train_test_split(X, y_energy, test_size=0.2, random_state=42)
    X_train, X_test, y_price_train, y_price_test = train_test_split(X, y_price, test_size=0.2, random_state=42)

    # Create and train a linear regression model for Energy Consumption
    lr_energy = LinearRegression()
    lr_energy.fit(X_train, y_energy_train)
    lr_energy_pred = lr_energy.predict(X_test)
    mse_energy = mean_squared_error(lr_energy_pred, y_energy_test)
    r2_energy = r2_score(lr_energy_pred, y_energy_test)
    # Create and train a linear regression model for Price
    lr_price = LinearRegression()
    lr_price.fit(X_train, y_price_train)
    # Prepare input data for prediction
    user_input = {'Year': [year], 'District': [district], 'House Size (sqft)': [house_size], 'Household Size': [household_size]}
    user_df = pd.DataFrame(user_input)

    # Encode categorical variables like 'District'
    encoded_districts_user = encoder.transform(user_df[['District']]).toarray()
    encoded_districts_user_df = pd.DataFrame(encoded_districts_user, columns=encoder.get_feature_names_out(['District']))
    user_df = user_df.drop('District', axis=1)
    user_df = pd.concat([user_df, encoded_districts_user_df], axis=1)

    # Predict using the trained models
    energy_prediction = lr_energy.predict(user_df)
    price_prediction = lr_price.predict(user_df)

    price_prediction[0] = round(price_prediction[0],2)
    energy_prediction[0] = round(energy_prediction[0],2)
    mse_energy = round(mse_energy,2)
    r2_energy = round(r2_energy,2)

    return energy_prediction[0], price_prediction[0], mse_energy, r2_energy
    

def knn_reg(year, district, house_size, household_size):
    data = pd.read_csv(r"C:\Users\msacc\Downloads\ML\ML\templates\energyKarnataka.csv")

    # Define the feature and target variables
    X = data[['Year', 'District', 'House Size (sqft)', 'Household Size']]
    y_energy = data['Energy Consumption (kWh)']
    y_price = data['Price (in Rupees)']

    # One-hot encode the District column
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = pd.DataFrame(encoder.fit_transform(X[['District']]).toarray(), columns=encoder.get_feature_names_out(['District']))
    X = X.drop('District', axis=1)
    X = pd.concat([X, X_encoded], axis=1)

    # Split the data into training and testing sets
    X_train, X_test, y_energy_train, y_energy_test = train_test_split(X, y_energy, test_size=0.2, random_state=42)
    X_train, X_test, y_price_train, y_price_test = train_test_split(X, y_price, test_size=0.2, random_state=42)

    # Create and train a KNN regressor model for Energy Consumption
    knn_energy = KNeighborsRegressor()
    knn_energy.fit(X_train, y_energy_train)

    # Create and train a KNN regressor model for Price
    knn_price = KNeighborsRegressor()
    knn_price.fit(X_train, y_price_train)
    # Prepare input data for prediction
    user_input = {'Year': [year], 'District': [district], 'House Size (sqft)': [house_size], 'Household Size': [household_size]}
    user_df = pd.DataFrame(user_input)

    # Encode categorical variables like 'District'
    encoded_districts_user = encoder.transform(user_df[['District']]).toarray()
    encoded_districts_user_df = pd.DataFrame(encoded_districts_user, columns=encoder.get_feature_names_out(['District']))
    user_df = user_df.drop('District', axis=1)
    user_df = pd.concat([user_df, encoded_districts_user_df], axis=1)

    # Predict using the trained models
    energy_prediction = knn_energy.predict(user_df)
    price_prediction = knn_price.predict(user_df)

    # Evaluate model performance on test set
    energy_pred_test = knn_energy.predict(X_test)
    price_pred_test = knn_price.predict(X_test)
    mse_energy = mean_squared_error(y_energy_test, energy_pred_test)
    r2_energy = r2_score(y_energy_test, energy_pred_test)

    price_prediction[0] = round(price_prediction[0],2)
    energy_prediction[0] = round(energy_prediction[0],2)
    mse_energy = round(mse_energy,2)
    r2_energy = round(r2_energy,2)

    return energy_prediction[0], price_prediction[0], mse_energy, r2_energy

def rf_reg(year, district, house_size, household_size):

    data = pd.read_csv(r"C:\Users\msacc\Downloads\ML\ML\templates\energyKarnataka.csv")

    # Define the feature and target variables
    X = data[['Year', 'District', 'House Size (sqft)', 'Household Size']]
    y_energy = data['Energy Consumption (kWh)']
    y_price = data['Price (in Rupees)']

    # One-hot encode the District column
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = pd.DataFrame(encoder.fit_transform(X[['District']]).toarray(), columns=encoder.get_feature_names_out(['District']))
    X = X.drop('District', axis=1)
    X = pd.concat([X, X_encoded], axis=1)

    # Split the data into training and testing sets
    X_train, X_test, y_energy_train, y_energy_test = train_test_split(X, y_energy, test_size=0.2, random_state=42)
    X_train, X_test, y_price_train, y_price_test = train_test_split(X, y_price, test_size=0.2, random_state=42)

    # Create and train a Random Forest regressor model for Energy Consumption
    rf_energy = RandomForestRegressor(random_state=42)
    rf_energy.fit(X_train, y_energy_train)

    # Create and train a Random Forest regressor model for Price
    rf_price = RandomForestRegressor(random_state=42)
    rf_price.fit(X_train, y_price_train)
    # Prepare input data for prediction
    user_input = {'Year': [year], 'District': [district], 'House Size (sqft)': [house_size], 'Household Size': [household_size]}
    user_df = pd.DataFrame(user_input)

    # Encode categorical variables like 'District'
    encoded_districts_user = encoder.transform(user_df[['District']]).toarray()
    encoded_districts_user_df = pd.DataFrame(encoded_districts_user, columns=encoder.get_feature_names_out(['District']))
    user_df = user_df.drop('District', axis=1)
    user_df = pd.concat([user_df, encoded_districts_user_df], axis=1)

    # Predict using the trained models
    energy_prediction = rf_energy.predict(user_df)
    price_prediction = rf_price.predict(user_df)

    # Evaluate model performance on test set
    energy_pred_test = rf_energy.predict(X_test)
    mse_energy = mean_squared_error(y_energy_test, energy_pred_test)
    r2_energy = r2_score(y_energy_test, energy_pred_test)

    price_prediction[0] = round(price_prediction[0],2)
    energy_prediction[0] = round(energy_prediction[0],2)
    mse_energy = round(mse_energy,2)
    r2_energy = round(r2_energy,2)

    return energy_prediction[0], price_prediction[0], mse_energy, r2_energy

def ada_reg(year, district, house_size, household_size):
    data = pd.read_csv(r"D:\Shreya Files\Projects\ML\templates\energyKarnataka.csv")

    # Define the feature and target variables
    X = data[['Year', 'District', 'House Size (sqft)', 'Household Size']]
    y_energy = data['Energy Consumption (kWh)']
    y_price = data['Price (in Rupees)']

    # One-hot encode the District column
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = pd.DataFrame(encoder.fit_transform(X[['District']]).toarray(), columns=encoder.get_feature_names_out(['District']))
    X = X.drop('District', axis=1)
    X = pd.concat([X, X_encoded], axis=1)

    # Split the data into training and testing sets
    X_train, X_test, y_energy_train, y_energy_test = train_test_split(X, y_energy, test_size=0.2, random_state=42)
    X_train, X_test, y_price_train, y_price_test = train_test_split(X, y_price, test_size=0.2, random_state=42)

    # Create and train an AdaBoost regressor model for Energy Consumption
    ada_energy = AdaBoostRegressor(random_state=42)
    ada_energy.fit(X_train, y_energy_train)

    # Create and train an AdaBoost regressor model for Price
    ada_price = AdaBoostRegressor(random_state=42)
    ada_price.fit(X_train, y_price_train)

    # Prepare input data for prediction
    user_input = {'Year': [year], 'District': [district], 'House Size (sqft)': [house_size], 'Household Size': [household_size]}
    user_df = pd.DataFrame(user_input)

    # Encode categorical variables like 'District'
    encoded_districts_user = encoder.transform(user_df[['District']]).toarray()
    encoded_districts_user_df = pd.DataFrame(encoded_districts_user, columns=encoder.get_feature_names_out(['District']))
    user_df = user_df.drop('District', axis=1)
    user_df = pd.concat([user_df, encoded_districts_user_df], axis=1)

    # Predict using the trained models
    energy_prediction = ada_energy.predict(user_df)
    price_prediction = ada_price.predict(user_df)

    # Evaluate model performance on test set
    energy_pred_test = ada_energy.predict(X_test)
    mse_energy = mean_squared_error(y_energy_test, energy_pred_test)
    r2_energy = r2_score(y_energy_test, energy_pred_test)

    price_prediction[0] = round(price_prediction[0],2)
    energy_prediction[0] = round(energy_prediction[0],2)
    mse_energy = round(mse_energy,2)
    r2_energy = round(r2_energy,2)

    return energy_prediction[0], price_prediction[0], mse_energy, r2_energy

"""def xgb_reg(year, district, house_size, household_size):
    data = pd.read_csv("F:\\Python\\ML\\templates\\energyKarnataka.csv")

    # Define the feature and target variables
    X = data[['Year', 'District', 'House Size (sqft)', 'Household Size']]
    y_energy = data['Energy Consumption (kWh)']
    y_price = data['Price (in Rupees)']

    # One-hot encode the District column
    encoder = OneHotEncoder(handle_unknown='ignore')
    X_encoded = pd.DataFrame(encoder.fit_transform(X[['District']]).toarray(), columns=encoder.get_feature_names_out(['District']))
    X = X.drop('District', axis=1)
    X = pd.concat([X, X_encoded], axis=1)

    # Split the data into training and testing sets
    X_train, X_test, y_energy_train, y_energy_test = train_test_split(X, y_energy, test_size=0.2, random_state=42)
    X_train, X_test, y_price_train, y_price_test = train_test_split(X, y_price, test_size=0.2, random_state=42)

    # Create and train an XGBoost regressor model for Energy Consumption
   # xgb_energy = XGBRegressor(random_state=42)
    #xgb_energy.fit(X_train, y_energy_train)

    # Create and train an XGBoost regressor model for Price
   # xgb_price = XGBRegressor(random_state=42)
    #xgb_price.fit(X_train, y_price_train)
    # Prepare input data for prediction
    #user_input = {'Year': [year], 'District': [district], 'House Size (sqft)': [house_size], 'Household Size': [household_size]}
    #user_df = pd.DataFrame(user_input)

    # Encode categorical variables like 'District'
    encoded_districts_user = encoder.transform(user_df[['District']]).toarray()
    encoded_districts_user_df = pd.DataFrame(encoded_districts_user, columns=encoder.get_feature_names_out(['District']))
    user_df = user_df.drop('District', axis=1)
    user_df = pd.concat([user_df, encoded_districts_user_df], axis=1)

    # Predict using the trained models
    energy_prediction = xgb_energy.predict(user_df)
    price_prediction = xgb_price.predict(user_df)

    # Evaluate model performance on test set
    energy_pred_test = xgb_energy.predict(X_test)
    mse_energy = mean_squared_error(y_energy_test, energy_pred_test)
    r2_energy = r2_score(y_energy_test, energy_pred_test)

    price_prediction[0] = round(price_prediction[0],2)
    energy_prediction[0] = round(energy_prediction[0],2)
    mse_energy = round(mse_energy,2)
    r2_energy = round(r2_energy,2)
    
    return energy_prediction[0], price_prediction[0], mse_energy, r2_energy"""