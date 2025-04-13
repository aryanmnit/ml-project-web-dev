import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
df=pd.read_csv("ppd2.csv")
df.columns = range(df.shape[1])
df2 = df.drop(columns=[0, 20,21])
for col in df2.select_dtypes(include=['object']).columns:
    df2[col] = df2[col].astype(str).str.extract(r'(\d+)')

# Convert extracted values to numeric type
df3 = df2.apply(pd.to_numeric, errors='coerce')
x = df3
y = df[20]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#Decision tree
dt_model = RandomForestClassifier(n_estimators=100, random_state=42)
dt_model.fit(x_train, y_train)
#making pickle file for our model
pickle.dump(dt_model,open("ml_model.pkl","wb"))
