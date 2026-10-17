import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Configuração da Página do Streamlit
st.set_page_config(
    page_title="Previsão de Consumo Industrial",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Painel de Previsão de Consumo de Energia Industrial")
st.markdown("Este painel utiliza aprendizado de máquina (Regressão Linear) para estimar o consumo de energia com base nas variáveis operacionais da fábrica.")

# 1. Geração do Dataset Sintético
@st.cache_data
def gerar_dados():
    np.random.seed(42)
    n_registros = 50
    
    velocidade = np.random.uniform(20.0, 100.0, n_registros) # m/min
    maquinas = np.random.randint(2, 25, n_registros)           # unidades
    temp_ambiente = np.random.uniform(18.0, 35.0, n_registros) # °C
    
    # Criando o consumo com uma relação linear simulada mais algum ruído aleatório
    consumo = (45.5 * velocidade) + (110.2 * maquinas) + (12.5 * temp_ambiente) + np.random.normal(0, 150, n_registros)
    
    df = pd.DataFrame({
        'Velocidade_Esteira': velocidade,
        'Maquinas_Ativas': maquinas,
        'Temp_Ambiente': temp_ambiente,
        'Consumo_kWh': consumo
    })
    return df

df_industrial = gerar_dados()

# 2. Treinamento do Modelo de Regressão
X = df_industrial[['Velocidade_Esteira', 'Maquinas_Ativas', 'Temp_Ambiente']]
y = df_industrial['Consumo_kWh']

modelo = LinearRegression()
modelo.fit(X, y)

# 3. Controles na Barra Lateral (Sidebar)
st.sidebar.header("⚙️ Parâmetros Operacionais")

vel_input = st.sidebar.slider(
    "Velocidade da Esteira (m/min)",
    float(X['Velocidade_Esteira'].min()),
    float(X['Velocidade_Esteira'].max()),
    float(X['Velocidade_Esteira'].mean())
)

maq_input = st.sidebar.slider(
    "Máquinas Ativas (un)",
    int(X['Maquinas_Ativas'].min()),
    int(X['Maquinas_Ativas'].max()),
    int(X['Maquinas_Ativas'].mean())
)

temp_input = st.sidebar.slider(
    "Temperatura Ambiente (°C)",
    float(X['Temp_Ambiente'].min()),
    float(X['Temp_Ambiente'].max()),
    float(X['Temp_Ambiente'].mean())
)

# 4. Previsão com base na entrada do usuário
entrada_usuario = pd.DataFrame([[vel_input, maq_input, temp_input]], columns=['Velocidade_Esteira', 'Maquinas_Ativas', 'Temp_Ambiente'])
consumo_previsto = modelo.predict(entrada_usuario)[0]

# 5. Exibição na Tela Principal
st.markdown("---")
st.subheader("📊 Resultado da Previsão")

st.metric(
    label="Consumo Estimado de Energia",
    value=f"{consumo_previsto:.2f} kWh"
)

# 6. Gráfico de Contribuição das Variáveis (Coeficientes do Modelo)
st.markdown("---")
st.subheader("📈 Contribuição das Variáveis no Modelo")
st.markdown("O gráfico abaixo exibe o peso (coeficiente) de cada parâmetro na preensão final do consumo:")

fig, ax = plt.subplots(figsize=(8, 4))
variaveis = ['Velocidade Esteira', 'Máquinas Ativas', 'Temp. Ambiente']
coeficientes = modelo.coef_

barras = ax.bar(variaveis, coeficientes, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_ylabel("Coeficiente de Impacto")
ax.set_title("Importância Relativa dos Parâmetros")
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adicionando os valores em cima das barras
for barra in barras:
    altura = barra.get_height()
    ax.annotate(f'{altura:.1f}',
                xy=(barra.get_x() + barra.get_width() / 2, altura),
                xytext=(0, 3),  # 3 pontos de deslocamento vertical
                textcoords="offset points",
                ha='center', va='bottom')

st.pyplot(fig)

# Expansor opcional para visualizar o dataset gerado
with st.expander("🔍 Visualizar Dataset de Treinamento"):
    st.dataframe(df_industrial.head(10))
