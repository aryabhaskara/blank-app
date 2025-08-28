import streamlit as st
import pandas as pd
import numpy as np
#from prediction import predict
from streamlit_option_menu import option_menu
import pandas as pd
import pydeck as pdk

hide_fork_me = """
    <style>
    .stApp > header {
        visibility: hidden;
    }
    </style>
"""
st.markdown(hide_fork_me, unsafe_allow_html=True)
BRIN = "images/brin.png"
st.image(BRIN, caption=None, use_container_width=True)
Axis = "images/"
selected = option_menu(
    menu_title=None,
    options = ["Beranda","Framework","Teori","Prediksi","Obrolan", "Kontak", "Lokasi"],
    icons = ["house","window","book","gear","chat","envelope","pin"],
    menu_icon = "cast",
    default_index = 0,
    orientation = "horizontal",
)
if selected == "Beranda":
    st.title("Prediksi Kondisi Mass Imbalance Pada Rotating Equipment Menggunakan Machine Learning")

    st.markdown("""
    Selamat datang pada situs web ini. Situs web ini dibuat untuk melakukan prediksi anomali mesin kondisi massa tidak seimbang (mass imbalance) menggunakan machine learning).  

    Dataset yang digunakan untuk membangun model ML ini diperoleh dari [1]. Mesin yang digunakan adalah pompa dengan merk Panasonic GP–129JXK dengan kecepatan 3000 rpm.  

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

    **Referensi:**  
    [1] H. Ihsannur, B. T. Atmaja, Suyanto and D. Arifianto, “VBL-VA001: Lab-scale vibration analysis dataset”, Journal of Vibration Engineering & Technologies, no. 123456789. Zenodo, Surabaya, Agu 18, 2022. doi: 10.5281/zenodo.7006575.
    [2] Atmaja, B.T., Ihsannur, H., Suyanto et al. Lab-Scale Vibration Analysis Dataset and Baseline Methods for Machinery Fault Diagnosis with Machine Learning. J. Vib. Eng. Technol. 12, 1991–2001 (2024). https://doi.org/10.1007/s42417-023-00959-9
    """)
if selected == "Framework":
    FW = "images/framework.png"
    st.image(FW, caption="Framework Machine Learning", use_container_width=True)
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
    st.image(BO, caption="Diagram Alir Bayesian Optimisation", use_container_width=True)
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
if selected == "Prediksi":
    st.title("Prediksi Pembelajaran Mesin")
    st.markdown("Masukan data dalam bentuk tabular (.csv)")
    st.header("Input Prediksi")
    col1, col2 = st.columns(2)
    with col1:
        speed = st.number_input("Masukan kecepatan mesin dalam rpm",min_value=800, max_value=1200)
        st.write("Kecepatan mesin saat ini adalah", speed, "rpm (800 - 1200 rpm)") 
        load = st.number_input("Masukan beban dalam Watt",min_value=1000, max_value=4000)
        st.write("Beban saat ini adalah", load, "Watt (1000 - 4000 Watt)")      
    with col2:
        bio_d = st.number_input("Masukan persentase biodiesel dalam %",min_value=0, max_value=50)
        st.write("Persentase biodiesel adalah", bio_d, "% (0 - 50%)")
        bio_bt = st.number_input("Masukan temperatur campuran biodiesel dalam derajat Celcius",min_value=26, max_value=60)
        st.write("Temperatur campuran biodiesel saat ini adalah", bio_bt, "degC (26 - 60 degC)")  
    if st.button("Prediksi!"):
        result = predict(np.array([[speed, load, bio_d, bio_bt]]))
        torque = round(result[0].item(), 4)
        sfc = round(result[1].item(), 4)
        thermal_efficiency = round(result[2].item(), 4)
        st.text(f"Torsi mesin anda adalah: {torque} Nm")
        st.text(f"Specific Fuel Consumption (SFC) mesin anda adalah: {sfc} g/kWh")
        st.text(f"Efisiensi Termal (Thermal Efficiency) mesin anda adalah: {thermal_efficiency} %")
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
    st.write("Kelompok Riset Sarana Transportasi Air- Pusat Riset Teknologi Transportasi")
    st.markdown("- Nanda Yustina")
    st.write("PT. Daun Biru Engineering")
    st.markdown("- Herry Susanto")
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
        st.image(PTSEIK, caption=None, use_container_width=True)
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
