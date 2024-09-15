# Imports
import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Carregar o modelo e o scaler salvos
model_file = 'modelos/modelo_v1.pkl'
scaler_file = 'padronizadores/scaler_v1.pkl'
modelo = joblib.load(model_file)
scaler = joblib.load(scaler_file)

# Função para fazer a recomendação de manutenção
def recomenda_manutencao(novo_dado):
    # Definir os nomes das colunas conforme o scaler foi ajustado
    colunas = ['vibracao', 'temperatura', 'pressao', 'umidade', 'horas_trabalho']
    
    # Converter o novo dado para DataFrame com os nomes de colunas corretos
    novo_dado_df = pd.DataFrame([novo_dado], columns=colunas)
    
    # Aplicar o scaler ao novo dado
    novo_dado_scaled = scaler.transform(novo_dado_df)
    
    # Fazer a previsão
    predicao = modelo.predict(novo_dado_scaled)
    predicao_proba = modelo.predict_proba(novo_dado_scaled)[:, 1]

    # Retornar a classe e a probabilidade
    return predicao[0], predicao_proba[0]

# Configura a página da aplicação no Streamlit
st.set_page_config(page_title="Manutenção IoT", page_icon=":100:", layout="centered")

# Define o título da aplicação
st.title("Sistema de Recomendação de Manutenção Preventiva IoT 🌐")

# Define uma legenda explicativa da aplicação
st.caption("Recomendações baseadas em Machine Learning")

st.header("Insira os valores dos Sensores:")
vibracao = st.number_input("Vibração", value=0.0)
temperatura = st.number_input("Temperatura (°C)", value=0.0)
pressao = st.number_input("Pressão (PSI)", value=0.0)
umidade = st.number_input("Umidade (%)", value=0.0)
horas_trabalho = st.number_input("Horas de Trabalho", value=0)

# Inicializa o histórico no estado da sessão, se ainda não estiver definido
if "history" not in st.session_state:
    st.session_state["history"] = []

# Botão para realizar a previsão
if st.button("Verificar necessidade de manutenção"):
    novo_dado = [vibracao, temperatura, pressao, umidade, horas_trabalho]
    classe, probabilidade = recomenda_manutencao(novo_dado)
    
    # Adicionar os dados ao histórico
    st.session_state["history"].append(novo_dado + [classe, probabilidade])

    # Mostrar os resultados
    st.write(f"Classe da Previsão: {'Realizar manutenção' if classe == 1 else 'Nenhuma manutenção necessária'}")
    st.write(f"Probabilidade de manutenção: {probabilidade:.2%}")

    # Exibe o histórico de análises
    st.subheader("Histórico de Análises")
    history_df = pd.DataFrame(st.session_state["history"], columns=["Vibração", "Temperatura (ºC)", "Pressão (PSI)", "Umidade (%)", "Horas Oper.", "Classe", "Probabilidade"])
    st.write(history_df)

st.caption("By Marcelo Medeiros | Cientista de Dados")
