#importing necesseray modules 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import openai
from tensorflow.keras.callbacks import EarlyStopping

openai.api_key = "API-KEY"



climateFile = "climate_policy_database_policies_export.csv"
dataset = pd.read_csv(climateFile)
relevantColumns = ["policy_description", "high_impact"]
dataset = dataset[relevantColumns].dropna()
label_encoder = LabelEncoder()
dataset["high_impact"] = label_encoder.fit_transform(dataset["high_impact"])
x_train, x_test, y_train, y_test = train_test_split(
    dataset["policy_description"], dataset["high_impact"], test_size=0.2, random_state=42
)
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
tokenizer.fit_on_texts(dataset["policy_description"])
x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)
x_train_pad = tf.keras.preprocessing.sequence.pad_sequences(x_train_seq, maxlen=100)
x_test_pad = tf.keras.preprocessing.sequence.pad_sequences(x_test_seq, maxlen=100)

avgLength = np.mean([len(seq) for seq in x_train_seq])
print(f"Average sequence length: {avgLength}")

model = Sequential([
    Embedding(input_dim=10000, output_dim=128), 
    Bidirectional(LSTM(128, return_sequences=True)),  
    LSTM(64, return_sequences=True), 
    Dropout(0.32), 
    LSTM(32),  
    Dense(16, activation='relu'),  
    Dense(1, activation='sigmoid')  
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
earlyStopping = EarlyStopping(
    monitor='val_loss', 
    patience=2,         # Number of epochs with no improvement
    restore_best_weights=True
    )

model.fit(x_train_pad, y_train, epochs=3, validation_data=(x_test_pad, y_test), callbacks=[earlyStopping])

train_loss, train_accuracy = model.evaluate(x_train_pad, y_train)
test_loss, test_accuracy = model.evaluate(x_test_pad, y_test)
print(f"Training Accuracy: {train_accuracy:.4f}, Training Loss: {train_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

for i in range(14):
    print(" ")



def generate_policy(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Generate 1 effective very specific governmental (local or national) climate policy to tackle the given issue in less than 100 words. Preferably around 70-77 words"}, 
        {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]




# Function to evaluate generated policies
def evaluatePolicy(policy_text):
    policy_seq = tokenizer.texts_to_sequences([policy_text])
    policy_pad = tf.keras.preprocessing.sequence.pad_sequences(policy_seq, maxlen=200)
    prediction = model.predict(policy_pad)
    return prediction

userQuery = input("What specific climate problem needs to be tackled ")
policyOne = generate_policy(userQuery)
policyTwo = generate_policy("Your job is to answer this query: " + userQuery + "; And it has to be different than " + policyOne)
policyThree = generate_policy("Your job is to answer this query: " + userQuery + "; And it has to be different than " + policyOne + " and " + policyTwo)

policyScores = [evaluatePolicy(policyOne), evaluatePolicy(policyTwo), evaluatePolicy(policyThree)]

desiredPolicy = ""
if(policyScores[0] > policyScores[1] and policyScores[0] > policyScores[2]):
    desiredPolicy = policyOne
elif(policyScores[1] > policyScores[0] and policyScores[1] > policyScores[2]):
    desiredPolicy = policyTwo
elif(policyScores[2] > policyScores[0] and policyScores[2] > policyScores[1]):
    desiredPolicy = policyThree


print("One of the best policy ideas that could tackle your issue is to: " + desiredPolicy)
