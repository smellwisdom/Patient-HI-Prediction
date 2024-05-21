import streamlit as st
import shap
import pandas as pd
import joblib
import numpy as np
import streamlit.components.v1 as components
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV
import os


# 仅保留训练数据中的特征列
feature_names = ['Age', 'BMI','Admission_mRS_Score', 'Thrombolytic_Drug', 'PreTreat_ASPECT_Score',
                 'Onset_to_Puncture_Time', 'Antiplatelet_Therapy', 'Anticoagulation_Therapy']

# 对训练数据进行校准

# 保存校准后的模型
calibrated_model_path = 'calibrated_model.pkl'
model_path = 'best_model(5-20).pkl'

# Streamlit 应用程序接口
st.title("Patient HI Prediction")

st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
.force-plot-container {
    display: flex;
    justify-content: flex-start; /* Align items to the left */
    margin-left: -100px; /* Adjust this value based on your needs */
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Input Patient Details:</p>', unsafe_allow_html=True)

# 创建输入表单并添加合理的限制
input_data = {}
input_data['Age'] = st.number_input('Age', min_value=18.0, max_value=100.0, value=18.0, step=1.0)
input_data['BMI'] = st.number_input('BMI', min_value=14.0, max_value=50.0, value=14.0, step=0.01)
input_data['Admission_mRS_Score'] = st.number_input('Admission mRS Score', min_value=0.0, max_value=5.0, value=0.0,
                                                    step=1.0)
input_data['Thrombolytic_Drug'] = st.selectbox('Thrombolytic Drug', options=['rt-pa', 'Urokinase', 'Other'], index=0)
input_data['PreTreat_ASPECT_Score'] = st.number_input('PreTreat ASPECT Score', min_value=1.0, max_value=2.0, value=1.0,
                                                      step=1.0)
input_data['Onset_to_Puncture_Time'] = st.number_input('Onset to Puncture Time (min)', min_value=1.0, max_value=3660.0,
                                                       value=20.0, step=1.0)
input_data['Antiplatelet_Therapy'] = st.selectbox('Antiplatelet Therapy', options=['No', 'Yes'], index=1)
input_data['Anticoagulation_Therapy'] = st.selectbox('Anticoagulation Therapy', options=['No', 'Yes'], index=1)

# 将输入数据转换为 DataFrame
input_df = pd.DataFrame([input_data])

# 将 Thrombolytic_Drug 转换为数值
input_df['Thrombolytic_Drug'] = input_df['Thrombolytic_Drug'].map({'rt-pa': 0.0, 'Urokinase': 1.0, 'Other': 2.0})
input_df['Antiplatelet_Therapy'] = input_df['Antiplatelet_Therapy'].map({'No': 0.0, 'Yes': 1.0})
input_df['Anticoagulation_Therapy'] = input_df['Anticoagulation_Therapy'].map({'No': 0.0, 'Yes': 1.0})

# 确保特征顺序一致
input_df = input_df[feature_names]

# 加载校准后的模型
calibrated_model = joblib.load(calibrated_model_path)
loaded_model = joblib.load(model_path)
# 进行预测
if st.button('Predict'):
    # 模型预测
    prediction_prob = calibrated_model.predict_proba(input_df)[0, 1]
    st.markdown('<p class="large-font">Prediction Result:</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="big-font">Based on feature values, predicted probability of HI is {prediction_prob * 100:.2f}%</p>',
        unsafe_allow_html=True)

    # 创建 SHAP 解释器，使用训练数据进行初始化
    explainer = shap.Explainer(loaded_model, feature_names=feature_names)
    shap_values = explainer(input_df)

    # 输出基准值和SHAP值以供调试
    base_value = explainer.expected_value
    # st.write(f"SHAP base value: {base_value}")
    # st.write(f"Sum of SHAP values: {shap_values.values[0].sum()}")

    # 转换SHAP值为概率值
    prob_value = expit(base_value + shap_values.values[0].sum())
    # st.markdown(f'<p class="large-font">SHAP calculated probability of HI is {float(prob_value) * 100:.2f}%</p>',
    #             unsafe_allow_html=True)

    # 创建 Force Plot 并保存为HTML文件
    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value, shap_values.values[0], input_df)
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"

    # 显示力图
    st.markdown('<div class="force-plot-container">', unsafe_allow_html=True)
    components.html(shap_html, height=2300, width=1100)
    st.markdown('</div>', unsafe_allow_html=True)
