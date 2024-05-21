import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder 

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
car_evaluation = fetch_ucirepo(id=19) 
  
# data (as pandas dataframes) 
X = car_evaluation.data.features 
y = car_evaluation.data.targets 

# Split dataset into train and test sets
X_en = pd.DataFrame()

label_encoders = {}
for column in X.columns:
    label_encoders[column] = LabelEncoder()
    X_en[column] = label_encoders[column].fit_transform(X[column])

y_en = pd.DataFrame()
label_encoder = LabelEncoder()
y_en['class'] = label_encoder.fit_transform(y['class'])

#Split data
X_train, X_test, y_train, y_test = train_test_split(X_en, y_en, test_size=0.2, random_state=42)

# Standardize features
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

# Train KNN classifier
dct_classifier = DecisionTreeClassifier()
dct_classifier.fit(X_train, y_train)

# Function to predict flower species
def predict_species():
    try:
        # Get input values
        buying = float(entry_buying.get())
        maint = float(entry_maint.get())
        doors = float(entry_doors.get())
        persons = float(entry_persons.get())
        lug_boot = float(entry_lug_boot.get())
        safety = float(entry_safety.get()) 
        
        # Standardize input
        input_data = [[buying, maint, doors, persons, lug_boot, safety]]
        
        # Predict
        prediction = dct_classifier.predict(input_data)
        hasil = ""
        if prediction[0] == 0:
            hasil ="diterima"
        elif prediction[0] == 1:
            hasil ="bagus"
        elif prediction[0] == 2:
            hasil ="tidak diterima"
        elif prediction[0] == 3:
            hasil ="sangat bagus"            
        
        # Map prediction to flower species
        #identification = car_evaluation.target_names[prediction[0]]
        
        # Show prediction
        messagebox.showinfo("Prediction", f"The predicted type is: {hasil}")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values for all input fields.")

# Create GUI
root = tk.Tk()
root.title("car identification")

# Labels
label_buying = tk.Label(root, text="buying :")
label_buying.grid(row=0, column=0)
label_maint = tk.Label(root, text="maint :")
label_maint.grid(row=1, column=0)
label_doors = tk.Label(root, text="doors :")
label_doors.grid(row=2, column=0)
label_persons = tk.Label(root, text="persons :")
label_persons.grid(row=3, column=0)
label_lug_boot = tk.Label(root, text="lug boot :")
label_lug_boot.grid(row=4, column=0)
label_safety = tk.Label(root, text="safety :")
label_safety.grid(row=5, column=0)


# Entry fields
entry_buying = tk.Entry(root)
entry_buying.grid(row=0, column=1)
entry_maint = tk.Entry(root)
entry_maint.grid(row=1, column=1)
entry_doors = tk.Entry(root)
entry_doors.grid(row=2, column=1)
entry_persons = tk.Entry(root)
entry_persons.grid(row=3, column=1)
entry_lug_boot = tk.Entry(root)
entry_lug_boot.grid(row=4, column=1)
entry_safety = tk.Entry(root)
entry_safety.grid(row=5, column=1)

# Button
button_predict = tk.Button(root, text="prediction", command=predict_species)
button_predict.grid(row=6, column=0, columnspan=2)

root.mainloop()
