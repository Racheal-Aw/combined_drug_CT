import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import ast
import pickle
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="KG Genie Prototype", layout="wide")
st.title("🧠  KG Genie – Activity Predictor + Network Context")

BEST_THRESH = 0.50          # ← replace with your Youden threshold
MODEL_PATH  = "New_RF_pipeline.sav"  # or "trained_pipe_knn.sav" if you saved one
SCALER_PATH = None          # or "scaler.pkl" if you saved one
FEATURE_CSV = "triplet_df_cleaned_good.csv"

import re

# 🔑 fixed lengths
FINGERPRINT_LEN = 2048   # adjust if you used a different size
EMBED_LEN       = 64

# ---- robust parser that pads / truncates ----
def parse_vector(cell, target_len):
    if isinstance(cell, (list, np.ndarray)):
        arr = np.asarray(cell, dtype=float)
    else:
        tokens = re.findall(r"[-+]?\d*\.\d+|\d+", str(cell))
        arr = np.asarray([float(t) for t in tokens], dtype=float)

    if arr.size < target_len:                       # pad with zeros
        arr = np.hstack([arr, np.zeros(target_len - arr.size)])
    elif arr.size > target_len:                     # truncate
        arr = arr[:target_len]
    return arr


# ------------------------------------------------------------------
# 🚀  Load artefacts *once* (cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts(feature_csv, model_pkl, scaler_pkl=None):
    # features
    df_feat = pd.read_csv(feature_csv)
    # parse & fix lengths
    df_feat["morgan_fp_array"]   = df_feat["morgan_fp_array"].apply(lambda x: parse_vector(x, FINGERPRINT_LEN))
    df_feat["graph_embed"] = df_feat["graph_embed"].apply(lambda x: parse_vector(x, EMBED_LEN))

    # sanity‑check
    assert all(v.size == FINGERPRINT_LEN for v in df_feat["morgan_fp_array"])
    assert all(v.size == EMBED_LEN       for v in df_feat["graph_embed"])

    # model
    with open(model_pkl, "rb") as f:
        model = pickle.load(f)

    # scaler (if you saved one).  Otherwise create fresh—less ideal but works.
    if scaler_pkl:
        with open(scaler_pkl, "rb") as f:
            scaler = pickle.load(f)
    else:
        scaler = StandardScaler().fit(
            np.vstack(
                df_feat.apply(lambda r: np.hstack((r["morgan_fp_array"], r["graph_embed"])), axis=1)
            )
        )

    return df_feat, model, scaler


df_feat, model, scaler = load_artifacts(
    feature_csv="triplet_df_cleaned_good.csv",
    model_pkl="New_RF_pipeline.sav",
)


## ❹  If the choice is a drug row → run prediction
drug_choices = df_feat["source"] if "source" in df_feat else df_feat["smiles"]
choice = st.selectbox("Select drug", drug_choices)

# Fetch the row
row = df_feat.loc[df_feat["source"] == choice] if "source" in df_feat else \
      df_feat.loc[df_feat["smiles"] == choice]

if row.empty:
    st.error("Drug not found!")
else:
    row = row.iloc[0]  # convert to Series
    morgan_fp = row["morgan_fp_array"]
    graph_emb = row["graph_embed"]

    # Combine and scale
    features = np.hstack((morgan_fp, graph_emb)).reshape(1, -1)
    features_scaled = scaler.transform(features)

    if st.button("🔮 Predict"):
        pred = int(model.predict(features_scaled)[0])
        prob = model.predict_proba(features_scaled)[0][pred]
        st.success(f"**Prediction:** {pred}  (prob = {prob:.2f})")
