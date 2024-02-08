import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("transaction_anomalies_dataset.csv")
print(data)
relevant_features = ['Transaction_Amount', 'Average_Transaction_Amount', 'Frequency_of_Transactions']

st.title("Transaction Anomaly Detection ")
st.header("Enter Transaction Details:")

user_inputs = {}
for feature in relevant_features:
    user_input = st.number_input(f"{feature}: ", min_value=0.0)
    user_inputs[feature] = user_input

k = 2

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[relevant_features])    
kmeans = KMeans(n_clusters=k, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

if st.button("Predict Anomaly"):
    user_df = pd.DataFrame([user_inputs])
    user_inputs_scaled = scaler.transform(user_df)
    user_cluster = kmeans.predict(user_inputs_scaled)
    

    st.write(f"Predicted Cluster: {user_cluster[0]}")
    
    if user_cluster[0] == 1:
        st.warning("Anomaly detected: This transaction is flagged as an anomaly.")
    else:
        st.success("No anomaly detected: This transaction is normal.")

fig, ax = plt.subplots()
for cluster_label in range(k):
    cluster_data = data[data['Cluster'] == cluster_label]
    ax.scatter(cluster_data['Transaction_Amount'], cluster_data['Average_Transaction_Amount'], label=f'Cluster {cluster_label}')


centroids = scaler.inverse_transform(kmeans.cluster_centers_)
ax.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='red', s=200, label='Centroids')

ax.set_title('K-Means Clustering')
ax.set_xlabel('Transaction Amount')
ax.set_ylabel('Average Transaction Amount')
ax.legend()

st.pyplot(fig)
print(data)
