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
selected = option_menu(
    menu_title=None,
    options = ["Beranda","Prediksi","Obrolan", "Kontak", "Lokasi"],
    icons = ["house","gear","chat","envelope","pin"],
    menu_icon = "cast",
    default_index = 0,
    orientation = "horizontal",
)
if selected == "Beranda":
    st.title("Prediksi & Optimasi Performa Mesin Diesel Menggunakan Algoritma Pembelajaran Mesin (Machine Learning)")
    st.write(
    "Selamat datang pada situs web ini. Situs web ini dibuat untuk melakukan prediksi performa mesin diesel menggunakan pembelajaran mesin (machine learning).")
    st.write(
    "Dataset yang digunakan untuk membangun model pembelajaran mesin ini diperoleh dari [1]. Mesin yang digunakan adalah mesin diesel 4 langkah dengan merk Yanmar dengan kapasitas 7.5 kW dan bahan bakar diesel/biodiesel. Mesin tersebut digunakan sebagai sumber energi listrik (genset) dan diberikan beban listrik")
    st.write(
    "Masukan data (input) yang digunakan oleh model pembelajaran mesin ini adalah kecepatan (0 - 1200 rpm), beban (0 - 4000 Watt), persentase biodiesel (0 - 50%), dan suhu campuran biodiesel (26 - 60 oC).")
    st.write(
    "Algoritma ML yang digunakan dalam laman web ini adalah Extreme Gradient Boost (XGB) dengan optimasi")
    st.write(
    "Luaran data (output) yang dihasilkan oleh model pembelajaran mesin ini adalah torsi (Nm), specific fuel consumption (g/kWh), dan efisiensi termal (%).")
    st.write("[1] Suardi, S., Setiawan, W., Nugraha, A. M., Alamsyah, A., & Ikhwani, R. J. (2023). Evaluation of Diesel Engine Performance Using Biodiesel from Cooking Oil Waste (WCO). Jurnal Riset Teknologi Pencegahan Pencemaran Industri, 14(1), 29–39. https://doi.org/10.21771/jrtppi.2023.v14.no1.p29-39")
if selected == "Prediksi":
    st.title("Prediksi Pembelajaran Mesin")
    st.markdown("Masukan nilai yang digunakan untuk memprediksi performa mesin diesel")
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
    st.title("Tim Riset Inovasi Indonesia Maju - Lembaga Pengelola Dana Pendidikan (RIIM-LPDP) ")
    st.write("Kelompok Riset Pemodelan Sarana Transportasi Berkelanjutan - Pusat Riset Teknologi Transportasi")
    st.markdown("- Nilam Sari Octaviani")
    st.markdown("- Rizqon Fajar")
    st.markdown("- Kurnia Fajar Adhi Sukra")
    st.markdown("- Sigit Tri Atmaja")
    st.markdown("- Fitra Hidiyanto")
    st.markdown("- Raditya Hendra Pratama")
    st.markdown("- Dhani Avianto Sugeng")
    st.markdown("- Ardani Cesario Zuhri")
    st.write("Kelompok Riset Bioenergi dan Energi Alternatif - Pusat Riset Konversi dan Konservasi Energi")
    st.markdown("- Arya Bhaskara Adiprabowo")
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
        st.write("Kelompok Riset Pemodelan Sarana Transportasi Berkelanjutan - Pusat Riset Teknologi Transportasi")
        st.write("Gedung 230, Kawasan Puspiptek Setu Serpong, Muncul, Kec. Setu, Kota Tangerang Selatan, Banten 15314")
        st.write("http://elsa.brin.go.id/")
    with col4:
        puspiptek = pd.DataFrame({
            'lat': [-6.3473723],
            'lon': [106.663]
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
