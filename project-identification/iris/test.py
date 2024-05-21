import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN classifier
k = 3  # K value
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)

# Function to predict flower species
def predict_species():
    try:
        # Get input values
        sepal_length = float(entry_sepal_length.get())
        sepal_width = float(entry_sepal_width.get())
        petal_length = float(entry_petal_length.get())
        petal_width = float(entry_petal_width.get())
        
        # Standardize input
        input_data = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Predict
        prediction = knn_classifier.predict(input_data)
        
        # Map prediction to flower species
        species = iris.target_names[prediction[0]]
        
        # Show prediction
        messagebox.showinfo("Prediction", f"The predicted species is: {species}")
    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values for all input fields.")

# Create GUI
root = tk.Tk()
root.title("Flower Species Prediction")

# Labels
label_sepal_length = tk.Label(root, text="Sepal Length:")
label_sepal_length.grid(row=0, column=0)
label_sepal_width = tk.Label(root, text="Sepal Width:")
label_sepal_width.grid(row=1, column=0)
label_petal_length = tk.Label(root, text="Petal Length:")
label_petal_length.grid(row=2, column=0)
label_petal_width = tk.Label(root, text="Petal Width:")
label_petal_width.grid(row=3, column=0)

# Entry fields
entry_sepal_length = tk.Entry(root)
entry_sepal_length.grid(row=0, column=1)
entry_sepal_width = tk.Entry(root)
entry_sepal_width.grid(row=1, column=1)
entry_petal_length = tk.Entry(root)
entry_petal_length.grid(row=2, column=1)
entry_petal_width = tk.Entry(root)
entry_petal_width.grid(row=3, column=1)

# Button
button_predict = tk.Button(root, text="Predict", command=predict_species)
button_predict.grid(row=4, column=0, columnspan=2)

root.mainloop()
