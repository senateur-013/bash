import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nuees_dynamiques import NueesDynamiques

st.title("Méthode des Nuées Dynamiques (Diday, 1971)")

uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])
K = st.slider("Nombre de classes", 2, 10, 3)
ni = st.slider("Nombre d'étalons par classe (ni)", 2, 20, 5)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    X = df.values

    nd = NueesDynamiques(K=K, ni=ni)
    classes = nd.fit(X)

    st.success("Clustering terminé !")

    fig, ax = plt.subplots()
    colors = ["red", "blue", "green", "purple", "orange", "brown", "pink"]

    for i, Ci in enumerate(classes):
        Ci = np.array(Ci)
        if Ci.size > 0:
            ax.scatter(Ci[:,0], Ci[:,1], color=colors[i % len(colors)], label=f"Classe {i+1}")

        Ei = np.array(nd.E[i])
        ax.scatter(Ei[:,0], Ei[:,1], marker="x", s=100, color="black")

    ax.legend()
    st.pyplot(fig)

    st.write("Nombre d’éléments par classe :")
    for i, Ci in enumerate(classes):
        st.write(f"Classe {i+1} : {len(Ci)} éléments")

