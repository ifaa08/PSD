import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Memuat objek StandardScaler dan model MLPClassifier dari file pickle
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('mlp_model.pkl', 'rb') as model_file:
    mlp = pickle.load(model_file)

# Fungsi untuk melakukan prediksi
def predict_verification_result(data):
    # Mengubah data masukan pengguna menjadi DataFrame
    input_data = pd.DataFrame([data], columns=['process.b1.capacity', 'process.b2.capacity', 'process.b3.capacity', 'process.b4.capacity', 'property.price', 'property.product', 'property.winner'])
    
    # Normalisasi data menggunakan objek StandardScaler yang telah di-load
    input_data = scaler.transform(input_data)
    
    # Melakukan prediksi dengan model MLPClassifier
    prediction = mlp.predict(input_data)
    
    return prediction[0]

# Judul aplikasi Streamlit
st.title('Klasifikasi Hasil Verifikasi')

# Input data dari pengguna
st.write('Masukkan data berikut untuk mengklasifikasikan hasil verifikasi:')
process_b1_capacity = st.number_input('process.b1.capacity', value=0.0)
process_b2_capacity = st.number_input('process.b2.capacity', value=0.0)
process_b3_capacity = st.number_input('process.b3.capacity', value=0.0)
process_b4_capacity = st.number_input('process.b4.capacity', value=0.0)
property_price = st.number_input('property.price', value=0.0)
property_product = st.number_input('property.product', value=0.0)
property_winner = st.number_input('property.winner', value=0.0)

# Tombol untuk melakukan prediksi
if st.button('Prediksi Hasil Verifikasi'):
    # Membuat data masukan dari input pengguna
    user_input = {
        'process.b1.capacity': process_b1_capacity,
        'process.b2.capacity': process_b2_capacity,
        'process.b3.capacity': process_b3_capacity,
        'process.b4.capacity': process_b4_capacity,
        'property.price': property_price,
        'property.product': property_product,
        'property.winner': property_winner
    }
    
    # Melakukan prediksi
    prediction = predict_verification_result(user_input)
    
    # Menampilkan hasil prediksi
    st.write(f'Hasil Verifikasi: {prediction}')
