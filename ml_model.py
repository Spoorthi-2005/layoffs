import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_excel(r"C:\Users\spoor\Downloads\layoffs\tech_layoffs.xlsx")

# Data Preprocessing
# Remove currency symbols and commas, then convert to numeric
df["Money_Raised_in_$_mil"] = df["Money_Raised_in_$_mil"].replace(r'[\$,]', '', regex=True)
df["Money_Raised_in_$_mil"] = pd.to_numeric(df["Money_Raised_in_$_mil"], errors='coerce')

# Handle missing values by filling with the median
num_cols = ["Laid_Off", "Percentage", "Company_Size_before_Layoffs", "Company_Size_after_layoffs", "Money_Raised_in_$_mil"]
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Encode Company Names into numerical values
if "Company" in df.columns:
    label_encoder = LabelEncoder()
    df["Company_Encoded"] = label_encoder.fit_transform(df["Company"].astype(str))
else:
    raise ValueError("Column 'Company' not found in dataset")

# Define features (X) and target variable (y)
required_columns = ['Year', 'Company_Encoded', 'Company_Size_before_Layoffs', 'Company_Size_after_layoffs', 'Percentage']
if all(col in df.columns for col in required_columns):
    X = df[required_columns]
    y = df['Laid_Off']
else:
    missing_cols = [col for col in required_columns if col not in df.columns]
    raise ValueError(f"Missing required columns: {missing_cols}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model's performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model Training Completed. Mean Squared Error: {mse}")

# Save the trained model and the label encoder for future use
with open("layoff_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("company_encoder.pkl", "wb") as encoder_file:
    pickle.dump(label_encoder, encoder_file)
