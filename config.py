TITULO_PAGINA = "Gradiente com Momentum"
TITULO_APP = "Gradiente Descendente"
LAYOUT = "wide"

# Learning Rate
LR_MIN = 0.001
LR_MAX = 1.0
LR_DEFAULT = 0.05
LR_FORMAT = "%.3f"

# Momentum
MOMENTUM_MIN = 0.0
MOMENTUM_MAX = 0.99
MOMENTUM_DEFAULT = 0.0
MOMENTUM_STEP = 0.01

# Iterações
ITERACOES_MIN = 1
ITERACOES_MAX = 200
ITERACOES_DEFAULT = 100

# Posição inicial (X)
X_INI_MIN = -4.0
X_INI_MAX = 4.0
X_INI_DEFAULT = -2.0

# Função padrão
FUNCAO_PADRAO = "x**4 - 2*x**3 + 1"

# Derivada numérica
DERIVADA_H = 0.0001

# Tolerâncias de convergência
TOLERANCIA_INCLINACAO = 0.01
TOLERANCIA_VELOCIDADE = 0.01

# Limite de segurança para explosão
LIMITE_X_EXPLOSAO = 1e10

# Dimensões do gráfico
FIGSIZE_X = 10
FIGSIZE_Y = 6
LAYOUT_COLUNAS = [3, 1]

# Eixo X (zoom)
AMPLITUDE_MINIMA_EIXO_X = 6.0
ZOOM_FACTOR = 1.5

# Eixo Y (auto-scaling)
MARGEM_Y_PERCENTUAL = 0.1
AMPLITUDE_Y_MINIMA = 0.1
AMPLITUDE_Y_PADRAO = 2.0
LIMITE_MAXIMO_EIXO_Y = 1000
LIMITE_MINIMO_EIXO_Y_ABSOLUTO = 100

# Cores e estilos
COR_TERRENO = 'navy'
ALPHA_TERRENO = 0.3
LARGURA_TERRENO = 2

COR_TRAJETORIA = 'red'
ALPHA_TRAJETORIA = 0.5
ESTILO_TRAJETORIA = '--'

CMAP_PONTOS = 'Reds'
TAMANHO_PONTOS = 40

TAMANHO_INICIO = 120
COR_INICIO = 'lime'

TAMANHO_FIM = 250
MARCADOR_FIM = '*'
COR_FIM = 'gold'

ESTILO_GRID = '--'
ALPHA_GRID = 0.5

LOCALIZACAO_LEGENDA = 'upper right'

# Pontos na grade de visualização
PONTOS_GRID_X = 300

MENSAGENS_STATUS = {
    "otimo": ("MÍNIMO ENCONTRADO", "success"),
    "descendo": ("EM MOVIMENTO", "warning"),
    "explosao": ("EXPLOSÃO", "error"),
    "erro": ("ERRO", "error"),
    "sucesso": ("SUCESSO", "info"),
    "proximo": ("PRÓXIMO AO MÍNIMO", "info"),
}

MOMENTUM_ALTO_THRESHOLD = 0.5