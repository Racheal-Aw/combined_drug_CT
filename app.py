# app.py

import streamlit as st
import pandas as pd
from pyvis.network import Network
from streamlit.components.v1 import html
import os

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="TNBC KG", layout="wide")
DATA_PATH = "data/tnbc_kg_triplets.csv"

# ----------------------------
# Load KG
# ----------------------------
@st.cache_data
def load_kg():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        st.error("❌ Data file not found.")
        return pd.DataFrame()

trip = load_kg()

# ----------------------------
# Page Title
# ----------------------------
st.markdown("<h1>🔬 TNBC Clinical Trial Knowledge Graph</h1><hr>", unsafe_allow_html=True)




import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import ast
import pickle
from sklearn.preprocessing import StandardScaler

#st.set_page_config(page_title="KG Genie Prototype", layout="wide")
#st.title("🧠  KG Genie – Activity Predictor + Network Context")

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
        
# ----------------------------
# Sidebar Filters
# ----------------------------
st.sidebar.header("🎯 Filters")
q = st.sidebar.text_input("🔍 Search")
rels = st.sidebar.multiselect("Relations", ["All"] + sorted(trip["relation"].unique()), default=["All"])

df = trip if "All" in rels else trip[trip["relation"].isin(rels)]

if q:
    df = df[df["head"].str.contains(q, case=False) | df["tail"].str.contains(q, case=False)]

# ----------------------------
# Show Table
# ----------------------------
st.subheader("📄 Triplets")
st.dataframe(df.head(2), use_container_width=True)

# ----------------------------
# Node Coloring Function
# ----------------------------
def infer_type(node):
    node = str(node).strip()
    if node.startswith("NCT"):
        return "TRIAL"
    elif node.upper() in ["PD-1", "PD-L1", "BRCA1", "BRCA2", "VEGF", "HER2", "EGFR", "TP53", "AKT1", "PIK3CA"]:
        return "GENE"
    elif "university" in node.lower() or "center" in node.lower() or "institute" in node.lower():
        return "SPONSOR"
    elif any(k in node.lower() for k in ["umab", "limab", "tinib", "drug", "ol", "inib", "umab"]):
        return "DRUG"
    elif len(node) < 12 and node.isupper():
        return "BIOMARKER"
    else:
        return "OTHER"

color_map = {
    "GENE": "#FF4C4C",      # brighter red
    "DRUG": "#4FC3F7",      # brighter blue
    "TRIAL": "#A3E635",     # brighter lime green
    "SPONSOR": "#FFEA00",   # brighter yellow
    "BIOMARKER": "#FF8C69", # brighter salmon
    "OTHER": "#E0E0E0"      # slightly brighter light gray
}


# ----------------------------
# Draw Network Function
# ----------------------------
def draw_network(df, limit=100):
    df = df.dropna(subset=["head", "tail", "relation"]).copy()
    df = df.head(limit)

    net = Network(height="600px", width="100%", directed=True)

    nodes = set(df["head"]).union(set(df["tail"]))

    for node in nodes:
        ntype = infer_type(node)
        color = color_map.get(ntype, "#D3D3D3")
        tooltip = f"{ntype} Node: {node}"
        net.add_node(str(node), label=str(node), color=color, title=tooltip)

    for _, r in df.iterrows():
        relation_label = str(r["relation"])
        net.add_edge(str(r["head"]), str(r["tail"]), label=relation_label, title=f"{r['head']} → {r['tail']}: {relation_label}")

    return net

# ----------------------------
# Show Graph
# ----------------------------
st.subheader("🧠 Graph View")

if df.empty:
    st.warning("No data to show.")
else:
    net = draw_network(df, limit=150)
    net.save_graph("kg.html")
    with open("kg.html", "r", encoding="utf-8") as f:
        html(f.read(), height=600, scrolling=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("🚀 Built with ❤️ for TNBC Research • Streamlit • PyVis • ClinicalTrials.gov")


