import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD

# CSV 파일에서 데이터 읽기
def load_data(file_type):
    if file_type == "Full Factorial":
        return pd.read_csv("./data/EC_result_full_factrorial.csv")
    elif file_type == "Optimal Space Filling":
        return pd.read_csv("./data/EC_result_Optimal_fiiling_space.csv")
    else:
        raise ValueError("Invalid file type selected.")

# Streamlit UI 설정
st.title("메타모델을 활용한 PKG 및 표면온도 예측 도구")

# 데이터 초기화
data_type = st.selectbox("Choose DOE Data Set:", ["Full Factorial", "Optimal Space Filling"])
data = load_data(data_type)

# 전역 변수 초기화
scaler_X = MinMaxScaler()
scaler_y1 = MinMaxScaler()
scaler_y2 = MinMaxScaler()
X = data[["length", "width", "fin_count"]].values
y1 = data["Simulation_temperature"].values
y2 = data["Surface_temperature"].values
X_scaled = scaler_X.fit_transform(X)
y1_scaled = scaler_y1.fit_transform(y1.reshape(-1, 1)).flatten()
y2_scaled = scaler_y2.fit_transform(y2.reshape(-1, 1)).flatten()

# SVD 설정
svd = TruncatedSVD(n_components=3, random_state=42)
X_svd = svd.fit_transform(X_scaled)

# 모델 학습 함수
def train_models():
    global rf_model1, gb_model1, linear_model1, rf_model2, gb_model2, linear_model2
    rf_model1 = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model1.fit(X_scaled, y1_scaled)

    gb_model1 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_model1.fit(X_scaled, y1_scaled)

    linear_model1 = LinearRegression()
    linear_model1.fit(X_scaled, y1_scaled)

    rf_model2 = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model2.fit(X_scaled, y2_scaled)

    gb_model2 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_model2.fit(X_scaled, y2_scaled)

    linear_model2 = LinearRegression()
    linear_model2.fit(X_scaled, y2_scaled)

# 초기 모델 학습
train_models()

# 입력 값
st.subheader("HeatSink Input Parameters")
col1, col2, col3 = st.columns(3)
with col1:
    length = st.number_input("Length:", min_value=30, max_value=130, value=50, step=1)
with col2:
    width = st.number_input("Width:", min_value=30, max_value=85, value=50, step=1)
with col3:
    fin_count = st.number_input("Fin Count:", min_value=2, max_value=25, value=20, step=1)

model_type = st.selectbox("Choose Model:", ["RandomForest", "GradientBoosting", "LinearRegression", "ROM"])

# 예측 함수
def predict_simulation(length, width, fin_count, model_type="RandomForest"):
    new_input = scaler_X.transform([[length, width, fin_count]])
    new_input_svd = svd.transform(new_input)

    if model_type == "RandomForest":
        scaled_prediction1 = rf_model1.predict(new_input)
        scaled_prediction2 = rf_model2.predict(new_input)
    elif model_type == "GradientBoosting":
        scaled_prediction1 = gb_model1.predict(new_input)
        scaled_prediction2 = gb_model2.predict(new_input)
    elif model_type == "LinearRegression":
        scaled_prediction1 = linear_model1.predict(new_input)
        scaled_prediction2 = linear_model2.predict(new_input)
    elif model_type == "ROM":
        rom_model1 = LinearRegression()
        rom_model1.fit(X_svd, y1_scaled)
        scaled_prediction1 = rom_model1.predict(new_input_svd)

        rom_model2 = LinearRegression()
        rom_model2.fit(X_svd, y2_scaled)
        scaled_prediction2 = rom_model2.predict(new_input_svd)
    else:
        raise ValueError("Invalid model type. Choose RandomForest, GradientBoosting, LinearRegression, or ROM.")

    prediction1 = scaler_y1.inverse_transform(scaled_prediction1.reshape(-1, 1))[0][0]
    prediction2 = scaler_y2.inverse_transform(scaled_prediction2.reshape(-1, 1))[0][0]
    return max(0, prediction1), max(0, prediction2)

# 결과 예측 및 출력
if st.button("Predict"):
    result1, result2 = predict_simulation(length, width, fin_count, model_type)
    st.subheader("Prediction Results")
    st.write(f"Pkg Temperature: {result1:.2f}")
    st.write(f"Surface Temperature: {result2:.2f}")
