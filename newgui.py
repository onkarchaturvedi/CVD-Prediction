import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the trained model
model = joblib.load('model1167.pkl')

# Load dataset to fit the scaler
df = pd.read_csv("heart.csv")
selected_features = ['age', 'cp', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'thal']
a = df[selected_features]

scaler = StandardScaler()
scaler.fit(a)

# Create the main window
root = tk.Tk()
root.title("Cardiovascular Disease Prediction")
root.geometry("500x400")
root.resizable(False, False)

title_label = tk.Label(root, text="Cardiovascular Disease Prediction", font=("Helvetica", 16, 'bold'))
title_label.pack(pady=20)

feature_names = ['Age', 'Chest Pain Type', 'Resting Blood Pressure', 'Cholesterol', 'Max Heart Rate', 'Oldpeak', 'CA (Major Vessels)', 'Thal']

entries = {}
entry_vars = {}  # To track the entry widget variables (for value checking)

for feature in feature_names:
    frame = tk.Frame(root)
    frame.pack(pady=5)

    label = tk.Label(frame, text=feature, width=30, anchor="w", font=("Helvetica", 10))
    label.pack(side="left")

    entry_var = tk.StringVar()
    entry = tk.Entry(frame, width=20, font=("Helvetica", 10), textvariable=entry_var)
    entry.pack(side="right")
    entries[feature] = entry
    entry_vars[feature] = entry_var

def highlight_field(entry, is_valid):
    if is_valid:
        entry.config(bg="lightgreen")  
    else:
        entry.config(bg="lightcoral")  

def predict():
    features = []
    is_valid = True
    invalid_fields = []  

    # Collecting the input features and checking if they are valid
    for feature in feature_names:
        value = entry_vars[feature].get()
        try:
            # Convert the input value to float
            feature_value = float(value)
            features.append(feature_value)
            highlight_field(entries[feature], True)  
        except ValueError:
            highlight_field(entries[feature], False)
            is_valid = False
            invalid_fields.append(feature)

    if not is_valid:
        # Show a message indicating which fields are invalid
        messagebox.showerror("Input Error", f"Please enter valid numerical values for: {', '.join(invalid_fields)}.")
        return

    try:
        # Apply feature scaling (StandardScaler)
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Make prediction using the ensemble model
        prediction = model.predict(features_scaled)

        # Display the result in a popup window
        if prediction == 0:
            message = "No Cardiovascular Disease Detected"
        else:
            message = "Risk of Cardiovascular Disease Detected"
        
        messagebox.showinfo("Prediction Result", message)

    except Exception as e:
        # Catch any unexpected error and display the specific error message
        messagebox.showerror("Unexpected Error", f"An error occurred during prediction: {str(e)}")

# Create the 'Predict' button
predict_button = tk.Button(root, text="Predict", command=predict, width=20, height=2, font=("Helvetica", 12, 'bold'), bg="#4CAF50", fg="white")
predict_button.pack(pady=20)

# Run the Tkinter event loop
root.mainloop()
