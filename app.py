"""
Gradient Descent Visualizer - Interface Streamlit.

Aplicação interativa para visualizar e explorar o algoritmo de gradiente descendente
com suporte a momentum. Permite ao usuário ajustar hiperparâmetros em tempo real e
ver como o algoritmo navega pela topografia da função em busca do mínimo.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

import config
import visualizador
from tools import executar_gradiente

# Configuração da página
st.set_page_config(page_title=config.TITULO_PAGINA, layout=config.LAYOUT)
st.title(config.TITULO_APP)

# --- BARRA LATERAL ---
st.sidebar.header("Parâmetros")
funcao_texto = st.sidebar.text_input("Função f(x):", value=config.FUNCAO_PADRAO)

# Hiperparâmetros
lr = st.sidebar.slider(
    "Learning Rate",
    config.LR_MIN,
    config.LR_MAX,
    config.LR_DEFAULT,
    format=config.LR_FORMAT
)
momentum = st.sidebar.slider(
    "Momentum (Inércia)",
    config.MOMENTUM_MIN,
    config.MOMENTUM_MAX,
    config.MOMENTUM_DEFAULT,
    step=config.MOMENTUM_STEP,
    help="0.0 = Bolinha de Isopor (Para fácil)\n0.9 = Bola de Chumbo (Passa por tudo)"
)
iteracoes = st.sidebar.slider(
    "Iterações",
    config.ITERACOES_MIN,
    config.ITERACOES_MAX,
    config.ITERACOES_DEFAULT
)
x_ini = st.sidebar.slider(
    "Início (x)",
    config.X_INI_MIN,
    config.X_INI_MAX,
    config.X_INI_DEFAULT
)

# Validação da função de entrada
try:
    func_usuario = lambda x: eval(funcao_texto, {"x": x, "np": np})
    func_usuario(0)  # Teste rápido
except Exception as e:
    st.error(f"❌ Erro na função: {e}")
    st.stop()

# --- EXECUÇÃO DO ALGORITMO ---
resultado = executar_gradiente(func_usuario, x_ini, lr, iteracoes, momentum)

# Extrai dados para uso local
historico_x = resultado.x
historico_y = resultado.y
status = resultado.status

# --- VISUALIZAÇÃO ---
col1, col2 = st.columns(config.LAYOUT_COLUNAS)

with col1:
    # Tenta criar visualização
    try:
        fig = visualizador.criar_visualizacao(func_usuario, resultado, momentum)
        st.pyplot(fig)
    except ValueError as e:
        st.warning(f"⚠️ {e}")

with col2:
    st.subheader("Diagnóstico")
    
    # Exibe status com cor e ícone apropriados
    if status in config.MENSAGENS_STATUS:
        mensagem, tipo = config.MENSAGENS_STATUS[status]
        getattr(st, tipo)(mensagem)
    else:
        st.info(f"ℹ️ {status.upper()}")
    
    # Exibe mensagem detalhada
    st.write(resultado.msg)
    st.write("---")
    
    # Métricas finais (se houver dados)
    if resultado.tem_dados():
        st.metric("Posição Final", f"{resultado.x[-1]:.4f}")
        st.metric("Inclinação Final", f"{resultado.incl_final:.4f}")
    
    # Avisos condicionais
    if momentum > config.MOMENTUM_ALTO_THRESHOLD:
        st.caption("💡 Com Momentum alto, é normal a IA 'balançar' no fundo.")