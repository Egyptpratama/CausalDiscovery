import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from causallearn.search.FCMBased import lingam
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from streamlit_option_menu import option_menu
from sklearn.metrics import confusion_matrix

# Judul Aplikasi
st.title("Aplikasi Visualisasi Graf kausal temporal")

# Sidebar menu 
with st.sidebar:
    selected = option_menu("Menu", 
                           ["Dashboard", "Dataset", "Causal Discovery","Evaluation"], 
                           icons=["house", "cloud-upload", "bar-chart-line"], 
                           menu_icon="cast", default_index=0)

# Inisialisasi variabel data di luar menu untuk digunakan bersama
if 'data' not in st.session_state:
    st.session_state['data'] = None

# Menu pertama: Dashboard
if selected == "Dashboard":
    st.header("Dashboard")

    st.write("Selamat Datang")
    
    
# Menu kedua: Dataset
elif selected == "Dataset":
    st.header("Dataset")

    # Upload file CSV
    uploaded_file = st.file_uploader("Unggah file CSV", type="csv")
    if uploaded_file is not None:
        # Baca data CSV
        st.session_state['data'] = pd.read_csv(uploaded_file)
        
        # Tampilkan data dalam bentuk tabel
        st.write("Data CSV yang diunggah:")
        st.dataframe(st.session_state['data'])

# Menu ketiga: Causal Discovery
elif selected == "Causal Discovery":
    st.header("Causal Discovery")

    data = st.session_state.get('data')
    
    if data is not None:
        # Dataset
        st.write("Dataset:")
        st.dataframe(data)

        # Jalankan causal discovery
        if st.button("Run Causal Discovery"):
            st.write("Running Causal Discovery... This may take a while.")
        
            # Varlingam
            model = lingam.VARLiNGAM()
            model.fit(data)

            true_graph = model.adjacency_matrices_[0]
            estimated_graph = model.adjacency_matrices_[1]

            st.session_state['true_graph'] = true_graph
            st.session_state['estimated_graph'] = estimated_graph

            G = nx.from_numpy_array(true_graph, create_using=nx.DiGraph)

            plt.figure(figsize=(8, 6))
            pos = nx.spring_layout(G)  
            nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', arrows=True)
            plt.title("Causal Graph from Adjacency Matrix")

            st.pyplot(plt)  

            st.write("Causal Discovery Complete!")
        else:
            st.write("Please upload a dataset in the 'Dataset' menu.")

# Menu keempat: Prediction
elif selected == "Predict":
    st.header("Prediction")

# Menu kelima: Evaluation         
elif selected == "Evaluation":
    st.header("Model Evaluation")

    if 'true_graph' in st.session_state and 'estimated_graph' in st.session_state:
        true_graph = st.session_state['true_graph']
        estimated_graph = st.session_state['estimated_graph']
        
        # Function to compute SHD
        def compute_shd(true_graph, estimated_graph):
            if true_graph.shape != estimated_graph.shape:
                raise ValueError("The shapes of true_graph and estimated_graph must be the same.")
                
            added_edges = np.sum((estimated_graph - true_graph) == 1)
            removed_edges = np.sum((estimated_graph - true_graph) == -1)
            return added_edges + removed_edges

        # Function to compute SID
        def compute_sid(true_graph, estimated_graph):
            if true_graph.shape != estimated_graph.shape:
                raise ValueError("The shapes of true_graph and estimated_graph must be the same.")
                
            true_edges = (true_graph != 0).astype(int)
            estimated_edges = (estimated_graph != 0).astype(int)
            return np.sum(np.abs(true_edges - estimated_edges))

        # Function to compute FDR
        def compute_fdr(true_graph, estimated_graph):
            if true_graph.shape != estimated_graph.shape:
                raise ValueError("The shapes of true_graph and estimated_graph must be the same.")
                
            tp = np.sum((true_graph == 1) & (estimated_graph == 1))  # True Positives
            fp = np.sum((true_graph == 0) & (estimated_graph == 1))  # False Positives
            return fp / (tp + fp) if (tp + fp) > 0 else 0

        # Function to compute MCC
        def compute_mcc(true_graph, estimated_graph):
            if true_graph.shape != estimated_graph.shape:
                raise ValueError("The shapes of true_graph and estimated_graph must be the same.")
                
            # Flatten the matrices for binary classification
            y_true = (true_graph != 0).flatten()
            y_pred = (estimated_graph != 0).flatten()
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0
            return mcc


        shd = compute_shd(true_graph, estimated_graph)
        sid = compute_sid(true_graph, estimated_graph)
        fdr = compute_fdr(true_graph, estimated_graph)
        mcc = compute_mcc(true_graph, estimated_graph)

        
        st.write("SHD:",shd)
        st.write("SHD:",sid)
        st.write("SHD:",fdr)
        st.write("SHD:",mcc)

    else:
        st.write("")
