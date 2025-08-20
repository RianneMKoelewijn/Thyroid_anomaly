import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
from scipy.special import expit


st.set_page_config(layout="wide")
st.markdown("""<style>.block-container {padding-left: 1rem;padding-right: 1rem;}</style>""", unsafe_allow_html=True)

## loading in the models, data, scaler, and shap explainer
@st.cache_resource
def load_models():
    models = {
        "3d": {
            "scaler": joblib.load("tsh_ft3_ft4_scaler.sav"),
            "model": joblib.load("tsh_ft3_ft4_model.sav"),
            "data": pd.read_excel('data/tsh_ft3_ft4_trainingdata.xlsx'),
            "explainer": joblib.load("3d_explainer.sav")
        },
        "2d": {
            "scaler": joblib.load("tsh_ft4_scaler.sav"),
            "model": joblib.load("tsh_ft4_2D_model.sav"),
            "data": pd.read_excel('data/tsh_ft4_trainingdata.xlsx'),
            "explainer": joblib.load("2d_explainer.sav")
        }
    }
    for model in models.values():
        model["data"]['anomaly'] = pd.Categorical(model["data"]['anomaly'])
        # model["data"]["anomaly"] = model["data"]["anomaly"].astype(int)
    for col in ["tsh", "ft3", "ft4", "age"]:
        if col in models["3d"]["data"].columns:
            models["3d"]["data"][col] = pd.to_numeric(models["3d"]["data"][col], errors="coerce")
        if col in models["2d"]["data"].columns:
            models["2d"]["data"][col] = pd.to_numeric(models["2d"]["data"][col], errors="coerce")

    st.dataframe(models["3d"]["data"])
    st.dataframe(models["2d"]["data"])

    return models


models = load_models()

st.write(models["3d"]["data"].head())
st.write(models["3d"]["data"].dtypes)
st.write(models["3d"]["data"].isna().sum())


## setting session_state
if "show_3d_graph" not in st.session_state:
    st.session_state["show_3d_graph"] = True
if "points" not in st.session_state:
    st.session_state["points"] = []

def toggle_mode():
    st.session_state["show_3d_graph"] = not st.session_state["show_3d_graph"]
    st.session_state["points"]=[]

## Sidebar input
st.sidebar.header("Add your own datapoint")
st.sidebar.caption("Entered datapoint will be deleted when switching models or adding a new one.")
st.sidebar.button("Without FT3" if st.session_state['show_3d_graph'] else "With FT3", on_click=toggle_mode)

with st.sidebar.form("add_data"):
    tsh_val = st.number_input("TSH (mU/L)", value=0.007, min_value=0.007, max_value=151.0, step=0.001, format="%.3f")
    ft3_val = ft4_val = None
    if st.session_state['show_3d_graph']:
        st.session_state.points = []
        ft3_val = st.number_input("FT3 (pmol/l)", value=0.30, min_value=0.30, max_value=31.0, step=0.001, format="%.2f")
    ft4_val = st.number_input("FT4 (pmol/l)", value=1.5, min_value=1.5, max_value=130.0, step=0.001, format="%.2f")
    age_val = st.number_input("Age (years)", value=0.0, min_value=0.0, max_value=120.0, step=0.001, format="%.2f")
    gender_val = st.selectbox("Gender", ("Female", "Male"))
    submitted = st.form_submit_button("Classify and show")
st.sidebar.caption("SHAP explanation will be shown below the plot")

## helper functions
def compute_ref_val(model_key, tsh, ft4, ft3=None, age=None):
    tsh_con = int(0.30 <= tsh <= 4.78)
    ft4_con = int(11.5 <= ft4 <= 22.7)
    if model_key == "3d" and ft3 is not None and age is not None:
        age_cond = [
            ((age < 2) & (5.1 <= ft3 <= 8.0)),
            ((2 <= age <= 12) & (5.1 <= ft3 <= 7.4)),
            ((13 <= age <= 20) & (4.7 <= ft3 <= 7.2)),
            ((age > 20) & (3.5 <= ft3 <= 6.5))
        ]
        ft3_con = np.select(age_cond, [1,1,1,1], default=0)
        return 1 if (tsh_con + ft3_con + ft4_con) < 3 else 0
    else:
        return 1 if (tsh_con + ft4_con) < 2 else 0

def make_prediction(model_key, inputs, gender_val, ref_val):
    model = models[model_key]
    log_tsh = np.log(inputs["tsh"])
    if model_key == "3d":
        scaler_input = np.array([inputs["age"], inputs["tsh"], inputs["ft3"], inputs["ft4"], log_tsh])
        scaled = model["scaler"].transform([scaler_input])
        model_input = np.array([gender_val, ref_val, scaled[0][0], scaled[0][2], scaled[0][3], scaled[0][4]]).reshape(1,-1)
        feature_names = ["gender", "ref", "s_age", "s_ft3", "s_ft4", "sl_tsh"]
    else:
        scaler_input = np.array([inputs["age"], inputs["tsh"], inputs["ft4"], inputs["ft4"], log_tsh])
        scaled = model["scaler"].transform([scaler_input])
        model_input = np.array([gender_val, ref_val, scaled[0][0], scaled[0][2], scaled[0][4]]).reshape(1,-1)
        feature_names = ["gender", "ref", "s_age", "s_ft4", "sl_tsh"]
    prediction = model["model"].predict(model_input)[0]
    shap_values = model["explainer"](pd.DataFrame(model_input, columns=feature_names))
    return prediction, shap_values, model_input, feature_names

def plot_3d(model_key):
    model = models[model_key]
    data = model["data"]
    fig = px.scatter_3d(data, x='ft3', y='ft4', z='tsh', color='anomaly', 
                        color_discrete_map={0:'lightgreen',1:'tomato'},
                        opacity=0.5, hover_data=['age','gender'], log_z=True)
    added_labels=set()
    for pt in st.session_state.points:
        if "ft3" not in pt: continue
        color = "green" if pt["prediction"]==0 else "red"
        label = "Inlier" if pt["prediction"] == 0 else "Outlier"

        if label in added_labels:
            show_legend=False
        else: 
            show_legend =True
            added_labels.add(label)
            
        fig.add_trace(go.Scatter3d(x=[pt['ft3']], y=[pt['ft4']], z=[pt['tsh']],
                                   mode='markers', name=label, showlegend=show_legend,
                                   marker=dict(size=10, color=color, symbol='diamond', line=dict(width=2, color='black'))))

        fig.update_layout(
            scene=dict(
                xaxis_title='FT3',
                yaxis_title='FT4',
                zaxis_title='TSH'
            ),
            legend=dict(itemsizing='constant')
        )
    return fig

def plot_2d(model_key, points=None):
    model = models[model_key]
    data = model["data"]
    fig = px.scatter(data, x='ft4', y='tsh', color='anomaly', 
                     color_discrete_map={0:'lightgreen',1:'tomato'},
                     opacity=0.5, hover_data=['age','gender'], log_y=True)
    fig.show()
    if points:
        for pt in points:
            color = "green" if pt["prediction"]==0 else "red"
            label = "Inlier" if pt["prediction"] == 0 else "Outlier"
            fig.add_trace(go.Scatter(x=[pt['ft4']], y=[pt['tsh']],
                                     mode='markers', name=label,
                                     marker=dict(size=10, color=color, symbol='diamond', line=dict(width=2, color='black'))))
    return fig


def show_shap(shap_values, model_key):
    st.subheader("SHAP explanation of your datapoint")
    
    st.write("Shows us how each feature contributes to the model's decision.\nThe threshold for considering whether a new datapoint is an outlier is:", round(models[model_key]["model"].threshold_,2))
    st.write("Everything below the threshold is considered an inlier and everything above an outlier and thus an anomaly.")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

## main logic
st.title("Thyroid dataset")

st.caption(
    """
    **Disclaimer:**  
    This app is intended for educational and research purposes only. It provides model-based classifications and visual explanations of 
    thyroid function test results, but these outputs should not be interpreted as medical advice. 
    Clinical decisions must always be based on a qualified healthcare professional’s judgment,
    following established medical protocols and guidelines. The creators and contributors are not responsible for any 
    consequences resulting from the use of this app. The app is provided without warranty, and no guarantees are made regarding its content or availability.
    """
)

model_key = "3d" if st.session_state["show_3d_graph"] else "2d"
if model_key=="3d":
    st.subheader(f"{'Model 1 (TSH, FT3, FT4)'}")
col1, col2 = st.columns(2)

if submitted:
    # st.session_state.points = []
    gender_bin = 1 if gender_val=="Female" else 0
    inputs = {"age": age_val, "tsh": tsh_val, "ft4": ft4_val, "ft3": ft3_val}
    ref_val = compute_ref_val(model_key, tsh_val, ft4_val, ft3_val, age_val)
    prediction, shap_values, model_input, _ = make_prediction(model_key, inputs, gender_bin, ref_val)
    new_point = {"tsh": tsh_val, "ft4": ft4_val, "prediction": prediction}
    if model_key=="3d": new_point["ft3"]=ft3_val
    st.session_state.points.append(new_point)

    
if model_key=="3d":
    
    with col1:
        st.text("3D plot of model 1\nTotal of 3182 datapoints with 160 anomalies")
        fig3d = plot_3d(model_key)
        st.plotly_chart(fig3d)
        if submitted:
            show_shap(shap_values, model_key)
    with col2:
        st.text("2D plot of model 1\nTotal of 3182 datapoints with 160 anomalies")
        fig2d = plot_2d("3d", st.session_state.points)
        st.plotly_chart(fig2d)
else:
    with col1:
        st.subheader(f"{'Model 1 (TSH, FT4)'}")
        st.text("2D plot of model 1\nTotal of 3182 datapoints with 160 anomalies")
        
        fig2d = plot_2d("2d", st.session_state.points)
        st.plotly_chart(fig2d)
    with col2:
        if submitted:
            show_shap(shap_values, model_key)    



