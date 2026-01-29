import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tools import derivada_numerica

st.set_page_config(page_title="Laboratório de Gradiente Descendente", layout="wide")

st.title("Explorador de Otimização")
st.markdown("""
Este painel simula o comportamento de um algoritmo de otimização tentando encontrar o mínimo de uma função.
**Ajuste os parâmetros na barra lateral e veja o gráfico reagir!**
""")

# --- BARRA LATERAL (CONTROLES) ---
st.sidebar.header("Parâmetros do Experimento")

# 1. Input da Função Matemática
funcao_texto = st.sidebar.text_input(
    "Digite a função f(x):", 
    value="x**4 - 2*x**2 + 1",
    help="Use sintaxe Python/Numpy. Ex: x**2 ou np.sin(x)"
)

# 2. Sliders de Controle
learning_rate = st.sidebar.slider("Learning Rate (Passo)", 0.001, 1.5, 0.05, step=0.001, format="%.3f")
iteracoes = st.sidebar.slider("Número de Iterações", 1, 100, 30)
x_inicial = st.sidebar.slider("Posição Inicial (x)", -5.0, 5.0, 2.0, step=0.1)

# --- PROCESSAMENTO LÓGICO ---
# Transformando o texto do usuário em uma função Python real
try:
    # O 'eval' é perigoso em produção, mas ótimo para estudos locais.
    # Ele permite que o Python leia a string "x**2" e execute como matemática.
    func_usuario = lambda x: eval(funcao_texto, {"x": x, "np": np})
    
    # Teste rápido para ver se a função é válida
    func_usuario(0)
except Exception as e:
    st.error(f"Erro na função digitada: {e}")
    st.stop()

# --- MOTOR DE CÁLCULO (O ALGORITMO) ---
# Aqui rodamos o Gradiente Descendente inteiro de uma vez para pegar o histórico
historico_x = [x_inicial]
historico_y = [func_usuario(x_inicial)]
x_atual = x_inicial

for i in range(iteracoes):
    inclinacao = derivada_numerica(func_usuario, x_atual)
    
    # Atualização (Fórmula do Gradiente)
    x_atual = x_atual - (learning_rate * inclinacao)
    
    # Salvar para plotar
    historico_x.append(x_atual)
    historico_y.append(func_usuario(x_atual))

# --- VISUALIZAÇÃO ---
col1, col2 = st.columns([3, 1])

with col1:
    # Criando o gráfico com Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Desenhar o "Chão" (Função de Fundo)
    # Criamos um intervalo dinâmico baseado onde a bolinha andou
    limite_visual = max(abs(min(historico_x)), abs(max(historico_x))) + 2
    x_fundo = np.linspace(-limite_visual, limite_visual, 200)
    y_fundo = [func_usuario(val) for val in x_fundo] # List comprehension para segurança
    
    ax.plot(x_fundo, y_fundo, label="Terreno f(x)", color='navy', alpha=0.3)
    
    # 2. Desenhar a Trajetória (O Caminho do Alpinista)
    # Linhas pontilhadas mostrando o caminho
    ax.plot(historico_x, historico_y, color='red', linestyle='--', alpha=0.5)
    # Pontos de cada passo (o mais escuro é o último)
    ax.scatter(historico_x, historico_y, c=range(len(historico_x)), cmap='Reds', s=80, edgecolors='black', label="Passos")
    
    # Ponto Inicial (Verde) e Final (Estrela Dourada)
    ax.scatter(historico_x[0], historico_y[0], c='green', s=150, label='Início', zorder=5)
    ax.scatter(historico_x[-1], historico_y[-1], c='gold', marker='*', s=300, edgecolors='black', label='Final', zorder=5)

    ax.set_title(f"Trajetória da Otimização (LR: {learning_rate})")
    ax.set_xlabel("x")
    ax.set_ylabel("Custo")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Passar o gráfico para o Streamlit
    st.pyplot(fig)

with col2:
    st.subheader("Resultados")
    st.metric(label="Posição Final (x)", value=f"{historico_x[-1]:.4f}")
    st.metric(label="Custo Final (y)", value=f"{historico_y[-1]:.4f}")
    
    derivada_final = derivada_numerica(func_usuario, historico_x[-1])
    st.metric(label="Inclinação Final", value=f"{derivada_final:.4f}")
    
    if abs(derivada_final) < 0.01:
        st.success("Convergiu para um mínimo! (Plano)")
    else:
        st.warning("Ainda descendo ou oscilando.")

    st.write("---")
    st.write("**Histórico dos últimos 5 passos:**")
    st.write(historico_x[-5:])