import streamlit as st
import pandas as pd
import yfinance as yf
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from streamlit_option_menu import option_menu
from causallearn.graph.ArrowConfusion import ArrowConfusion
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion
from causallearn.graph.SHD import SHD

# Set page configuration
st.set_page_config(page_title="Stock Market Causal Discovery", layout="wide")


def calculate_mcc(arrowsTp, arrowsTn, arrowsFp, arrowsFn):
    numerator = (arrowsTp * arrowsTn) - (arrowsFp * arrowsFn)
    denominator = ((arrowsTp + arrowsFp) * (arrowsTp + arrowsFn) * (arrowsTn + arrowsFp) * (arrowsTn + arrowsFn)) ** 0.5
    if denominator == 0:
        return 0
    return numerator / denominator
    
def calculate_fdr(arrowsTp, arrowsFp):
    """Menghitung False Discovery Rate (FDR)."""
    if (arrowsTp + arrowsFp) == 0:
        return 0  # Untuk menghindari pembagian dengan nol
    return arrowsFp / (arrowsTp + arrowsFp)

def calculate_sid(arrowsTp, arrowsTn, arrowsFp, arrowsFn):
    """Menghitung Structural Independence Distance (SID)."""
    return arrowsFp + arrowsFn

# Sidebar menu
with st.sidebar:
    selected = option_menu("Menu",
                           ["Dataset", "Causal Discovery","Evaluation"],
                           icons=["cloud-upload", "bar-chart-line", "bar-chart"],
                           menu_icon="cast", default_index=0)

# Dataset Section
if selected == "Dataset":
    st.title("Dataset Upload & Stock Data Fetching")
    st.write("Pilih apakah Anda ingin mengupload dataset Anda atau mengambil data saham dari Yahoo Finance.")

    # Menu pilihan untuk upload atau ambil dari Yahoo Finance
    option = st.radio("Pilih opsi data:", ["Upload Dataset", "Ambil Data dari Yahoo Finance"])

    # Upload dataset
    if option == "Upload Dataset":
        st.subheader("Upload Dataset")
        uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx"])

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file, engine='python', on_bad_lines='skip')
                elif uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(uploaded_file)

                st.session_state['data'] = data
                st.success("Dataset Berhasil Diupload!")
                st.dataframe(data)  # Menampilkan seluruh data yang diupload

                # Check for missing values
                st.write("Missing values in the dataset:")
                st.write(data.isnull().sum())

                # Option to clean missing values
                if st.checkbox("Clean missing values (drop rows with missing data)?"):
                    data = data.dropna()
                    st.success("Missing values removed.")
                    st.write(data.head())

                # Feature selection
                st.subheader("Pilih Fitur untuk Analisis")
                selected_features = st.multiselect("Pilih kolom untuk analisis:", options=data.columns.tolist(), default=data.columns.tolist())
                filtered_data = data[selected_features]
                st.session_state['filtered_data'] = filtered_data
                st.write("Data Terpilih:")
                st.dataframe(filtered_data)  # Menampilkan data yang dipilih untuk analisis

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memuat file: {e}")

    # Ambil data dari Yahoo Finance
    elif option == "Ambil Data dari Yahoo Finance":
        st.subheader("Ambil Data Saham dari Yahoo Finance")
        tickers = st.text_input("Masukkan Ticker Saham (dipisahkan koma)", "AAPL, MSFT, GOOG")
        start_date = st.date_input("Tanggal Mulai", value=pd.to_datetime("2015-01-01"))
        end_date = st.date_input("Tanggal Selesai", value=pd.to_datetime("2024-01-01"))

        if st.button("Ambil Data"):
            try:
                tickers_list = [ticker.strip() for ticker in tickers.split(",")]
                stock_data = yf.download(tickers_list, start=start_date, end=end_date)['Adj Close']

                st.session_state['data'] = stock_data
                st.success("Data Saham Berhasil Diambil!")
                st.dataframe(stock_data)  # Menampilkan seluruh data saham yang diambil

                # Skip feature selection for Yahoo Finance data
                st.session_state['filtered_data'] = stock_data

            except Exception as e:
                st.error(f"Terjadi kesalahan saat mengambil data saham: {e}")


# Causal Discovery Section
elif selected == "Causal Discovery":
    st.title("Causal Discovery dengan FCI")
    st.write("Lakukan Causal Discovery menggunakan algoritma Fast Causal Inference (FCI) dari pustaka causal-learn.")

    if 'filtered_data' in st.session_state:
        data = st.session_state['filtered_data']

        # Ensure data is numeric and calculate returns if necessary
        data = data.apply(pd.to_numeric, errors='coerce').dropna()
        returns = data.pct_change().dropna()

        # Validate data dimensions
        if returns.shape[0] <= returns.shape[1]:
            st.error("Jumlah sampel (baris) harus lebih besar daripada jumlah fitur (kolom). Kurangi jumlah fitur yang dipilih atau tambahkan lebih banyak data.")
        else:
            st.write("Dataset for Causal Discovery:")
            st.dataframe(returns.head(5))
            column_labels = returns.columns.tolist()  # Menampilkan seluruh dataset untuk analisis Causal Discovery

            if st.button("Run Causal Discovery"):
                st.write("Running FCI Algorithm...")

                # Import FCI model from causal-learn
                from causallearn.search.ConstraintBased.FCI import fci
                from causallearn.utils.cit import fisherz
                from causallearn.utils.GraphUtils import GraphUtils
                from sklearn.model_selection import train_test_split

                data_array = returns.values
                train, test = train_test_split(data_array, test_size=0.2, random_state=42)

                fci_train, edges_train = fci(train, fisherz, alpha=0.05)
                fci_result, edges = fci(test, fisherz, alpha=0.05) 
                
                edges = fci_result.get_graph_edges()
                nodes = fci_result.get_nodes()

                truth_cpdag = fci_train
                est = fci_result


                st.session_state['truth_cpdag'] = truth_cpdag
                st.session_state['est'] = est

                # Convert graph to pydot and display
                
                pyd = GraphUtils.to_pydot(fci_result)
                pyd.write_png('pag.png')

                from PIL import Image
                img = Image.open('pag.png')
                st.image(img)

                st.write("Edges detected by FCI:")
                symbol_to_description = {
                "-->": "A is a cause of B",
                "o-o": "No set d-separates A and B.",
                "o->": "B is not an ancestor of A",
                "<->": "There is a latent common cause of A and B."
                }
                for edge in edges:
                	edge_str = str(edge)
                	if "o-o" in edge_str:
                		description = symbol_to_description["o-o"]
                	elif "o->" in edge_str:
                		description = symbol_to_description["o->"]
                	elif "-->" in edge_str:
                		description = symbol_to_description["-->"]
                	elif "<->" in edge_str:
                		description = symbol_to_description["<->"]
                	else:
                		description = "Tidak ada relasi."

                	st.write(f"Edge: {edge_str}, Description: {description}")
            else:
            	st.error("No data available. Please upload or fetch data in the Dataset section.")
    with st.expander('Penjelasan Edges'):
    	st.image('C:/Users/ACER/Documents/causal/stock/edge/1.png', width=250)
    	st.write('Jika A menyebabkan B, yang menunjukkan hubungan sebab-akibat langsung dari A ke B.')
    	st.image('C:/Users/ACER/Documents/causal/stock/edge/2.png', width=250)
    	st.write('Jika terdapat variabel laten (tidak teramati) yang menjadi penyebab bersama untuk A dan B. Artinya, hubungan sebab-akibat ini dipengaruhi oleh variabel tersembunyi.')
    	st.image('C:/Users/ACER/Documents/causal/stock/edge/3.png', width=250)
    	st.write('Jika B tidak menjadi nenek moyang A, menunjukkan arah hubungan, namun bukan hubungan sebab-akibat langsung.')
    	st.image('C:/Users/ACER/Documents/causal/stock/edge/4.png', width=250)
    	st.write('D-separation adalah konsep dalam teori graf berbasis probabilitas untuk menentukan independensi. Jika tidak ada set yang bisa memisahkan A dan B, maka hubungan kausal tidak dapat ditentukan dengan jelas')
        

elif selected == "Evaluation":
    st.header("Model Evaluation")

    if 'truth_cpdag' in st.session_state and 'est' in st.session_state:
        truth_cpdag = st.session_state['truth_cpdag']
        est = st.session_state['est']

        arrow = ArrowConfusion(truth_cpdag, est)

        arrowsTp = arrow.get_arrows_tp()
        arrowsFp = arrow.get_arrows_fp()
        arrowsFn = arrow.get_arrows_fn()
        arrowsTn = arrow.get_arrows_tn()

        arrowPrec = arrow.get_arrows_precision()
        arrowRec = arrow.get_arrows_recall()

        # For adjacency matrices

        adj = AdjacencyConfusion(truth_cpdag, est)
        adjTp = adj.get_adj_tp()
        adjFp = adj.get_adj_fp()
        adjFn = adj.get_adj_fn()
        adjTn = adj.get_adj_tn()
        adjPrec = adj.get_adj_precision()
        adjRec = adj.get_adj_recall()

        # Calculate evaluation metrics
        shd = SHD(truth_cpdag, est).get_shd()
        sid = calculate_sid(arrowsTp, arrowsTn, arrowsFp, arrowsFn)
        fdr = calculate_fdr(arrowsTp, arrowsFp)
        mcc = calculate_mcc(arrowsTp, arrowsTn, arrowsFp, arrowsFn)

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("SHD", shd)
        b2.metric("SID", sid)
        b3.metric("FDR", fdr)
        b4.metric("MCC", mcc)

    else:
        st.write("Jalankan 'Causal Discovery' untuk menghasilkan dan menyimpan struktur grafik.")
    
    with st.expander('Penjelasan Matrik Evaluasi'):
        st.write('**SHD (Structural Hamming Distance)**: Metrik ini menghitung jumlah penambahan, penghapusan, dan pembalikan arah edge yang diperlukan untuk menyelaraskan graf estimasi dengan graf ground-truth. jika menghasilkan nilai yang lebih rendah maka menunjukkan akurasi struktur graf yang lebih tinggi, dengan SHD = 0 menandakan graf estimasi sepenuhnya sesuai dengan ground-truth')
        st.write('**SID (Structural Intervention Distance)**: Metrik ini menilai dampak kesalahan struktur graf pada interpretasi kausalitas, yang sangat relevan untuk analisis kausal berbasis intervensi.')
        st.write('**FDR (False Discovery Rate)**: FDR mengukur proporsi kesalahan positif dalam hubungan kausal yang ditemukan, yaitu jumlah hubungan kausal yang salah teridentifikasi dibandingkan dengan semua hubungan yang ditemukan. Semakin rendah nilai FDR, semakin baik kemampuan model dalam menghindari kesalahan')
        st.write('**MCC (Mattews correlation coefficient)**: Metrik ini mengevaluasi akurasi keseluruhan dengan mempertimbangkan semua elemen dalam matriks kebingungan, termasuk true positives, false positives, true negatives, dan false negatives. MCC memberikan nilai antara -1 hingga 1, di mana 1 menunjukkan kesesuaian sempurna dan 0 menunjukkan tebakan acak')



