import streamlit as st
import config
import visualizador
from tools import executar_gradiente
from validators import ValidadorParametros

@st.cache_data(show_spinner=False)
def obter_resultado_gradiente(funcao_texto, x_ini, lr, iteracoes, momentum):
    _, _, func_usuario = ValidadorParametros.validar_funcao(funcao_texto)
    return executar_gradiente(func_usuario, x_ini, lr, iteracoes, momentum)

st.set_page_config(page_title=config.TITULO_PAGINA, layout=config.LAYOUT)

st.set_page_config(page_title=config.TITULO_PAGINA, layout=config.LAYOUT)
st.title(config.TITULO_APP)

st.sidebar.header("Parâmetros")
funcao_texto = st.sidebar.text_input("Função f(x):", value=config.FUNCAO_PADRAO)

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
    help="0.0 = Sem inércia (Pode estagnar em mínimos locais)\n0.9 = Alta inércia (Acumula velocidade para superar platôs e mínimos locais)"
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

valido_func, msg_func, func_usuario = ValidadorParametros.validar_funcao(funcao_texto)

if not valido_func:
    st.error(f"❌ {msg_func}")
    st.stop()

valido_todos, erros = ValidadorParametros.validar_todos_parametros(
    funcao_texto, lr, momentum, iteracoes, x_ini
)

if not valido_todos:
    st.sidebar.error("❌ Erros nos parâmetros:")
    for erro in erros:
        st.sidebar.error(f"  • {erro}")
    st.stop()

resultado = obter_resultado_gradiente(funcao_texto, x_ini, lr, iteracoes, momentum)

historico_x = resultado.x
historico_y = resultado.y
status = resultado.status

col1, col2 = st.columns(config.LAYOUT_COLUNAS)

with col1:
    try:
        fig = visualizador.criar_visualizacao(func_usuario, resultado, momentum)
        st.pyplot(fig)
    except ValueError as e:
        st.warning(f"⚠️ {e}")

with col2:
    st.subheader("Diagnóstico")
    
    if status in config.MENSAGENS_STATUS:
        mensagem, tipo = config.MENSAGENS_STATUS[status]
        match tipo:
            case "success":
                st.success(mensagem)
            case "warning":
                st.warning(mensagem)
            case "error":
                st.error(mensagem)
            case "info":
                st.info(mensagem)
    else:
        st.info(f"ℹ️ {status.upper()}")
    
    st.write(resultado.msg)
    st.write("---")
    
    if resultado.tem_dados():
        st.metric("Posição Final", f"{resultado.x[-1]:.4f}")
        st.metric("Inclinação Final", f"{resultado.incl_final:.4f}")
        st.metric("Iterações Executadas", len(resultado.x) - 1)
    
    if momentum > config.MOMENTUM_ALTO_THRESHOLD:
        st.caption("💡 Com Momentum alto, é normal a IA 'balançar' no fundo.")