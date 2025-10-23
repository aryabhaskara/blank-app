import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict
from streamlit_option_menu import option_menu
import pandas as pd
import pydeck as pdk
from txt2csv import txt2csv
from fft import compute_fft, plot_fft, normalize, extract_features, ordered_columns
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
#from gsheet import init_gspread, save_to_gsheet

hide_fork_me = """
    <style>
    .stApp > header {
        visibility: hidden;
    }
    </style>
"""
st.markdown(hide_fork_me, unsafe_allow_html=True)
BRIN = "images/brin.png"
Axis = "images/"
with st.sidebar:
    st.image(BRIN, caption=None, use_container_width=False)
    selected = option_menu(
        menu_title="Menu Utama",
        options=["Beranda","Framework","Teori","Eksperimen","Prediksi (Ganda)","Prediksi (Tunggal)","Obrolan","Kontak","Lokasi"],
        icons=["house","window","book","pencil","bullseye","check2-circle","chat","envelope","pin"],
        menu_icon="cast",
        default_index=0,
    )
if selected == "Beranda":
    st.title("Prediksi Kondisi Mass Imbalance Pada Rotating Equipment Menggunakan Machine Learning")

    st.markdown("""
    Selamat datang pada situs web ini. Situs web ini dibuat untuk melakukan prediksi anomali mesin kondisi massa tidak seimbang (mass imbalance) menggunakan machine learning.  

    Dataset yang digunakan untuk membangun model ML ini diperoleh dari hasil eksperimen tim RP-OREM mass imbalance. Mesin yang digunakan adalah simulator rotating equipment dengan kecepatan 600 - 2400 rpm.  

    Data (input) yang dibutuhkan untuk deteksi mass imbalance adalah data dari tiga akselerometer yang dipasang pada rotating equipment:  
    - Horizontal (sumbu x)  
    - Vertikal (sumbu y)  
    - Aksial (sumbu z)  
                
    Data tersebut akan diproses dengan fitur Fast Fourier Transformation (FFT) dan menggunakan rekayasa fitur (feature engineering) terkait pemrosesan sinyal untuk ketiga sumbu tersebut, yaitu :
    - Mean 
    - Standard Deviation
    - Shape Factor
    - Root Mean Square
    - Impulse Factor
    - Peak to Peak
    - Kurtosis
    - Crest Factor
    - Skewness
                
    27 fitur ini (9 x 3 axis) akan digunakan sebagai input model ML. Algoritma yang digunakan adalah **Extreme Gradient Boost (XGB)** dengan optimasi Bayesian dengan Expected Improvement acquisition function.  

    Output yang dihasilkan oleh machine learning model adalah kondisi mesin (rotating equipment) baik normal maupun tidak seimbang (imbalance).

    Program ini diselenggarakan dalam kegiatan Rumah Program Manufaktur - Organisasi Riset Energi dan Manufaktur 2025 dan bekerjasama dengan PT. Daun Biru Engineering.
    
    """)
if selected == "Framework":
    FW = "images/framework.png"
    st.image(FW, caption="Framework Machine Learning", use_container_width=True)
if selected == "Eksperimen":
    sensor = "images/sensor.jpeg"
    daq = "images/daq.jpeg"
    sim_atas = "images/simulator_atas.jpeg"
    sim_depan = "images/simulator.jpeg"
    st.title("Pengambilan Data")
    st.markdown("""
<div style='text-align: justify; text-justify: inter-word;'>
Pengambilan data dilakukan di PT. Daun Biru Engineering, Depok Jawa Barat selama periode April - Desember 2025. 
Pengambilan data kecepatan mesin berputar dengan kondisi massa yang tidak seimbang (imbalance) dilakukan dengan kecepatan 600 – 2400 rpm. 
Pengambilan data dilakukan dengan kenaikan frekuensi ±0.4 Hertz (±25 rpm) untuk mendapatkan resolusi yang memadai. 
Sebuah bandul dengan berat 9.5 gram dipasang pada impeller nomor 2 (tengah) untuk menciptakan kondisi imbalance di sudut 0°.
Akselerometer ditempatkan di bagian horizontal, vertikal, dan aksial dari mesin (x, y, dan z). 
Gambar 2 menunjukkan proses pemasangan alat untuk melaksanakan eksperimen, sedangkan Gambar 3 menunjukkan perangkat sensor dan akuisisi data yang digunakan. 
Data yang diambil akan dijadikan bahan pelatihan model machine learning dan pengujian untuk menghasilkan model yang valid. 
Data diambil dengan sampling rate yang tinggi (18 kHz) untuk memastikan data berkualitas tinggi selama 11 detik dengan averaging window 3 dan parameter “FFT size 65536”.
</div>,<br>         
""",unsafe_allow_html=True)
    st.image(sim_depan,caption="Simulator Rotating Equipment",use_container_width=True)
    st.title("Spesifikasi Alat dan Data")
    st.markdown("Tabel 1. Spesifikasi Akselerometer (AC-102)")
    spek_sensor = {
        "Specifications": [
            "Part Number",
            "Sensitivity (±10%)",
            "Frequency Response (±3dB)",
            "Dynamic Range",
        ],
        "Standard": [
            "AC102",
            "100 mV/g",
            "30 – 810,000 CPM",
            "±80 g, peak, Vsource ≥ 22V, 12V Bias",  # empty cell for last row
        ]
    }
    spek_sensor = pd.DataFrame(spek_sensor)
    st.table(spek_sensor)
    st.image(sensor, caption="Akselerometer (AC-102)", use_container_width=True)
    st.markdown("Tabel 2. Spesifikasi Sistem Akusisi Data (DT9837)")
    spek_daq = {
        "Part No.": [
            "Channel",
            "Resolution",
            "High-pass filter",
            "Max. sampling rate",
            "Current source",
        ],
        "DT9837": [
            "4-channel",
            "24-bit",
            "0.5 Hertz",
            "<52.7 kHz",
            "4 mA",  # empty cell for last row
        ]
    }
    spek_daq = pd.DataFrame(spek_daq)
    st.table(spek_daq)
    st.image(daq, caption="Sistem Akuisisi Data (DT9837)", use_container_width=True)
if selected == "Teori":
    st.title("Teori Mass Imbalance")
    st.markdown("""
    Mass imbalance pada rotating equipment (peralatan berputar, misalnya rotor, kipas, turbin, atau pompa) adalah kondisi ketika massa tidak terdistribusi merata di sekitar sumbu putar. Akibatnya, saat peralatan berputar, timbul gaya sentrifugal yang tidak seimbang sehingga menghasilkan getaran berlebih.

Penyebab umum imbalance antara lain:

- Adanya kotoran atau material menempel pada rotor.
- Keausan atau kerusakan pada bagian tertentu.
- Cacat manufaktur sehingga bentuknya tidak simetris.
- Pemasangan komponen yang tidak presisi.

Dampak mass imbalance:
- Meningkatkan getaran dan kebisingan.
- Mempercepat keausan bantalan (bearing) dan poros.
- Menurunkan efisiensi dan umur pakai mesin.
- Untuk mengatasinya biasanya dilakukan balancing, yaitu menambahkan atau mengurangi massa pada bagian tertentu agar distribusinya kembali seimbang.
                
    """)
    st.title("Teori Bayesian Optimization")
    st.markdown("""
    Bayesian Optimization (BO) adalah metode optimasi yang digunakan untuk mencari kombinasi hyperparameter terbaik pada model machine learning (misalnya XGB atau ANN) agar performanya maksimal. Berbeda dengan grid search atau random search yang mencoba secara menyeluruh atau acak, BO memilih percobaan berikutnya berdasarkan informasi dari percobaan sebelumnya sehingga lebih efisien. Pada penelitian ini digunakan Upper Confidence Bound (UCB) sebagai fungsi akuisisi untuk menyeimbangkan eksploitasi (memilih nilai yang sudah menjanjikan) dan eksplorasi (mencoba nilai baru), dengan dukungan Gaussian Process (GP) yang memperkirakan rata-rata (μ) dan ketidakpastian (σ). Batasan nilai setiap hyperparameter dijelaskan pada Tabel 1.
    """)
    BO = "images/bo.png"
    st.image(BO, caption="Diagram Alir Bayesian Optimisation", width='content')
    st.title("Teori Machine Learning - Extreme Gradient Boost")
    st.markdown("""

    Extreme Gradient Boost (XGB) adalah salah satu algoritma machine learning berbasis tree (pohon keputusan) yang sangat populer karena cepat, akurat, dan efisien. Prinsip kerjanya adalah membuat banyak pohon keputusan secara bertahap (boosting), di mana setiap pohon baru berusaha memperbaiki kesalahan dari pohon sebelumnya. Hasil akhirnya adalah gabungan dari semua pohon tersebut sehingga prediksinya menjadi lebih kuat dan akurat.
    XGB banyak dipakai karena:
    - Bisa menangani data besar dengan cepat.
    - Memberikan akurasi tinggi dibanding banyak metode lain.
    - Dapat digunakan untuk berbagai tugas, seperti klasifikasi (misalnya menentukan ya/tidak) maupun regresi (memperkirakan angka).
    """)
    XGB = "images/xgb.png"
    st.image(XGB, caption="Diagram XGB", use_container_width=True)
if selected == "Prediksi (Tunggal)":
    st.title("Prediksi Kondisi Mesin (1 file)")
    uploaded_file = st.file_uploader("Unggah file .csv", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(
                uploaded_file,
                skiprows=9,          # skip header lines (data starts at line 10)
                header=None,         # no column names in data section
                engine='python',     # tolerate inconsistent commas
                sep=r',\s*',         # handle comma + optional space
                comment='#'          # ignore comment lines if any
            )

            # Drop empty trailing column caused by extra comma at line end
            df = df.dropna(axis=1, how='all')

            # Assign consistent column names
            df.columns = ['Time', 'X', 'Y', 'Z']

            # Ensure all numeric
            df = df.apply(pd.to_numeric, errors='coerce').dropna()

        except Exception as e:
            st.error(f"⚠️ Error reading file: {e}")
            st.stop()  # safely stop Streamlit execution
        st.write("📊 Raw Data:", df.head())
    # Convert to numeric and drop NaN
        # Extract signals (assuming col1=x, col2=y, col3=z)
        t, x, y, z = df.iloc[:,0],df.iloc[:, 1].values, df.iloc[:, 2].values, df.iloc[:, 3].values
        axis_data = {
            'x': normalize(compute_fft(x)),
            'y': normalize(compute_fft(y)),
            'z': normalize(compute_fft(z))
            }
        features_all = {}
        for f in ["mean", "std", "shape_factor", "rms", "impulse_factor","peak_to_peak","kurtosis","crest_factor","skewness"]:
                for axis in ['x','y','z']:
                    features_all[f'{f}_{axis}'] = extract_features(axis_data[axis])[f]
    # === Show FFT plots ===
    st.subheader("🔊 Spektrum FFT")
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    colors = ["red", "green", "blue"]  # one color per axis
    for idx, (signal, label) in enumerate([(x, "X"), (y, "Y"), (z, "Z")]):
        freqs, fft_vals, fs = plot_fft(signal, t)
        ax[idx].plot(freqs, fft_vals, color=colors[idx])
        ax[idx].set_title(f"FFT of {label}-axis")
        ax[idx].set_xlabel("Frequency (Hz)")
        ax[idx].set_ylabel("Amplitude")
        ax[idx].grid(True)
    plt.tight_layout()
    st.pyplot(fig)
    # === Show STFT plots ===
    st.subheader("📡 Spektrogram STFT")
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    for idx, (signal, label) in enumerate([(x,"X"), (y,"Y"), (z,"Z")]):
        f, t, Sxx = spectrogram(signal, fs, nperseg=256)
        ax[idx].pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud')
        ax[idx].set_title(f"STFT of {label}-axis")
        ax[idx].set_xlabel("Time [s]")
        ax[idx].set_ylabel("Frequency [Hz]")
    plt.tight_layout()
    st.pyplot(fig)
    feature_df = pd.DataFrame([[features_all[col] for col in ordered_columns]], columns=ordered_columns)
    st.write("🔍 Extracted Features:", feature_df)
    # Prediction
    st.write("🤖 Hasil Prediksi Machine Learning :")
    prediction = predict(feature_df)
    if prediction[0]==1:
        st.success(f"❌ Kondisi Mesin Imbalance")
    else:
        st.success(f"✅ Kondisi Mesin Tidak Imbalance")        
if selected == "Prediksi (Ganda)":
    st.title("Prediksi Kondisi Mesin")
    st.markdown("Masukan data dalam bentuk tabular (.csv/.txt)")
    uploaded = st.file_uploader("Pilih file CSV atau TXT", type=["csv", "txt"])
    if uploaded is not None:
        if uploaded.name.endswith(".txt"):
            df, csv_filename = txt2csv(uploaded)
        else:
            df = pd.read_csv(uploaded)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    x, y, z = df.iloc[:, 1].values, df.iloc[:, 2].values, df.iloc[:, 3].values

    axis_data = {
            'x': normalize(compute_fft(x)),
            'y': normalize(compute_fft(y)),
            'z': normalize(compute_fft(z))}
    features_all = {}
    for f in ["mean", "std", "shape_factor", "rms", "impulse_factor","peak_to_peak","kurtosis","crest_factor","skewness"]:
        for axis in ['x','y','z']:
            features_all[f'{f}_{axis}'] = extract_features(axis_data[axis])[f]
    #new_rows.append([features_all[col] for col in ordered_columns])
    if df is not None:
       st.write("📊 Data yang diupload:")
       st.dataframe(df)
       csv_data = df.to_csv(index=False).encode("utf-8")
       st.download_button(
            label="💾 Download sebagai CSV",
            data=csv_data,
            file_name=csv_filename,
            mime="text/csv"
        )
#sh, ws_features, ws_labels, df_existing, df_labels = init_gspread()
    st.title("📊 FFT Feature Extraction Tool")
    uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
    if uploaded_files:
        new_rows, new_labels = [], []
        for uploaded in uploaded_files:
            try:
                df = pd.read_csv(uploaded, skiprows=[0, 1], header=None)
                df = df.apply(pd.to_numeric, errors='coerce').dropna()

                x, y, z = df.iloc[:, 1].values, df.iloc[:, 2].values, df.iloc[:, 3].values

                axis_data = {
                    'x': normalize(compute_fft(x)),
                    'y': normalize(compute_fft(y)),
                    'z': normalize(compute_fft(z))
                }

                features_all = {}
                for f in ["mean", "std", "shape_factor", "rms", "impulse_factor","peak_to_peak","kurtosis","crest_factor","skewness"]:
                    for axis in ['x','y','z']:
                        features_all[f'{f}_{axis}'] = extract_features(axis_data[axis])[f]

                new_rows.append([features_all[col] for col in ordered_columns])

                # Label from filename
                filename = uploaded.name
                if 'Norm' in filename:
                    y_class = 0
                elif 'Bandul' in filename:
                    y_class = 1
                else:
                    y_class = 'Unknown'

                new_labels.append([filename, y_class])

            except Exception as e:
                st.error(f"⚠️ Error in {uploaded.name}: {e}")

        # Save to Google Sheets
        #save_to_gsheet(ws_features, ws_labels, df_existing, df_labels, new_rows, new_labels, ordered_columns)
        st.success("✅ Data saved to Google Sheets")
        #st.markdown(f"🔗 [Open Google Sheet](https://docs.google.com/spreadsheets/d/{sh.id})")
if selected == "Kontak":
    st.title("Tim Riset Mass Imbalance - Rumah Program Manufaktur - Organisasi Riset dan Manufaktur")
    st.write("Kelompok Riset Bioenergi dan Energi Alternatif - Pusat Riset Konversi dan Konservasi Energi")
    st.markdown("- Frendy Rian Saputro")
    st.markdown("- Trisno Anggoro")
    st.markdown("- Wargiantoro Prabowo")
    st.markdown("- Erlan Rosyadi")
    st.markdown("- Dhani Avianto Sugeng")
    st.markdown("- Ade Syafrinaldy")
    st.markdown("- Bambang Muharto")
    st.markdown("- Arya Bhaskara Adiprabowo")
    st.write("Kelompok Riset Sarana Transportasi Air - Pusat Riset Teknologi Transportasi")
    st.markdown("- Nanda Yustina")
    st.write("PT. Daun Biru Engineering")
    st.markdown("- Herry Susanto")
    st.markdown("- Zona Amrullah")
    st.markdown("- Suwarjono")
if selected == "Obrolan":
    with st.chat_message("ai",avatar=":material/robot:"):
        st.write("Halo! Ada yang bisa dibantu? 👋")
        prompt = st.chat_input("Say something") 
        if prompt:
            st.write(f"User has sent the following prompt: {prompt}")
if selected == "Lokasi":
    col3, col4 = st.columns(2)
    with col3:
        st.title("Kontak Kami :")
        st.write("Kelompok Riset Bioenergi dan Energi Alternatif - Pusat Riset Konversi dan Konservasi Energi")
        st.write("Gedung Energi 625, Kawasan Puspiptek Setu Serpong, Muncul, Kec. Setu, Kota Tangerang Selatan, Banten 15314")
        st.write("https://brin.go.id/orem/pusat-riset-konversi-dan-konservasi-energi/page/kontak-pusat-riset-konversi-dan-konservasi-energi")
        PTSEIK = "images/ptseik.jpeg"
        st.image(PTSEIK, caption="Laboratorium PRKKE-BRIN", use_container_width=True)
    with col4:
        puspiptek = pd.DataFrame({
            'lat': [-6.35864],
            'lon': [106.66618]
                })
        view_state = pdk.ViewState(
            latitude=puspiptek['lat'][0],
            longitude=puspiptek['lon'][0],
            zoom=17,  
            pitch=0
            )
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=puspiptek,
            get_position='[lon, lat]',
            get_radius=5,
            get_color=[255, 0, 0],
            pickable=True
            )
        map_deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state
            )
        st.pydeck_chart(map_deck)
