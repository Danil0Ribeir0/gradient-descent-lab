import streamlit as st
import config
import visualizador
from tools import executar_gradiente
from validators import ValidadorParametros

@st.cache_data(show_spinner=False)
def obter_resultado_gradiente(funcao_texto, x_ini, lr, iteracoes, momentum, otimizador):
    _, _, func_usuario = ValidadorParametros.validar_funcao(funcao_texto)
    return executar_gradiente(func_usuario, x_ini, lr, iteracoes, momentum, otimizador)

st.set_page_config(page_title=config.TITULO_PAGINA, layout=config.LAYOUT)
st.title(config.TITULO_APP)

# --- SIDEBAR: MODELO A ---
st.sidebar.header("Parâmetros - Modelo A")
funcao_texto = st.sidebar.text_input("Função f(x):", value=config.FUNCAO_PADRAO)

otimizador_a = st.sidebar.selectbox("Otimizador", ["SGD", "AdaGrad", "RMSProp", "Adam"], index=0, key="opt_a")

lr_a = st.sidebar.slider("Learning Rate", config.LR_MIN, config.LR_MAX, config.LR_DEFAULT, format=config.LR_FORMAT)
momentum_a = st.sidebar.slider("Momentum / Beta1", config.MOMENTUM_MIN, config.MOMENTUM_MAX, config.MOMENTUM_DEFAULT, step=config.MOMENTUM_STEP)
iteracoes = st.sidebar.slider("Iterações", config.ITERACOES_MIN, config.ITERACOES_MAX, config.ITERACOES_DEFAULT)
x_ini = st.sidebar.slider("Início (x)", config.X_INI_MIN, config.X_INI_MAX, config.X_INI_DEFAULT)

valido_func, msg_func, func_usuario = ValidadorParametros.validar_funcao(funcao_texto)
if not valido_func:
    st.error(f"❌ {msg_func}")
    st.stop()


st.sidebar.divider()
comparar = st.sidebar.toggle("Comparar com Modelo B")

if comparar:
    st.sidebar.header("Parâmetros - Modelo B")
    otimizador_b = st.sidebar.selectbox("Otimizador B", ["SGD", "AdaGrad", "RMSProp", "Adam"], index=3, key="opt_b")
    
    lr_b = st.sidebar.slider("Learning Rate B", config.LR_MIN, config.LR_MAX, config.LR_DEFAULT, format=config.LR_FORMAT, key="lr_b")
    momentum_b = st.sidebar.slider("Momentum / Beta1 B", config.MOMENTUM_MIN, config.MOMENTUM_MAX, 0.9, step=config.MOMENTUM_STEP, key="mom_b")
else:
    lr_b, momentum_b, otimizador_b = None, None, None

resultado_a = obter_resultado_gradiente(funcao_texto, x_ini, lr_a, iteracoes, momentum_a, otimizador_a)
resultado_b = obter_resultado_gradiente(funcao_texto, x_ini, lr_b, iteracoes, momentum_b, otimizador_b) if comparar else None

col1, col2 = st.columns(config.LAYOUT_COLUNAS)

with col1:
    try:
        fig = visualizador.criar_visualizacao(
            func_usuario, 
            resultado_a, momentum_a, 
            resultado_b, momentum_b if comparar else 0.0,
            otimizador_a, otimizador_b if comparar else "SGD"
        )
        st.plotly_chart(fig, width='stretch')
    except ValueError as e:
        st.warning(f"⚠️ {e}")

with col2:
    st.subheader("Diagnóstico")
    
    def renderizar_diagnostico(resultado, momentum):
        if resultado.status in config.MENSAGENS_STATUS:
            mensagem, tipo = config.MENSAGENS_STATUS[resultado.status]
            match tipo:
                case "success": st.success(mensagem)
                case "warning": st.warning(mensagem)
                case "error": st.error(mensagem)
                case "info": st.info(mensagem)
        else:
            st.info(f"ℹ️ {resultado.status.upper()}")
        
        st.write(resultado.msg)
        st.write("---")
        if resultado.tem_dados():
            st.metric("Posição Final", f"{resultado.x[-1]:.4f}")
            st.metric("Inclinação Final", f"{resultado.incl_final:.4f}")
            st.metric("Iterações Executadas", len(resultado.x) - 1)
        
        if momentum > config.MOMENTUM_ALTO_THRESHOLD:
            st.caption("💡 Com Momentum alto, a IA pode 'balançar' no fundo.")

    if comparar:
        aba_a, aba_b = st.tabs(["Modelo A", "Modelo B"])
        with aba_a:
            renderizar_diagnostico(resultado_a, momentum_a)
        with aba_b:
            renderizar_diagnostico(resultado_b, momentum_b)
    else:
        renderizar_diagnostico(resultado_a, momentum_a)