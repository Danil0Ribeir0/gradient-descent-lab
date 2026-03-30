import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
import config
from modelos import ResultadoGradiente


def calcular_limites_eixo_x(
    historico_x: List[float],
) -> Tuple[float, float]:
    """
    Calcula os limites do eixo X com zoom inteligente.
    
    Garante uma amplitude mínima visível e centra o gráfico na trajetória.
    
    Args:
        historico_x: Lista de valores x visitados.
        
    Returns:
        Tupla (lim_x_min, lim_x_max) para uso em np.linspace.
    """
    if not historico_x:
        return (-config.AMPLITUDE_MINIMA_EIXO_X / 2, config.AMPLITUDE_MINIMA_EIXO_X / 2)
    
    min_p = min(historico_x)
    max_p = max(historico_x)
    
    # Calcula a largura visual com zoom factor
    largura_visual = max(
        (max_p - min_p) * config.ZOOM_FACTOR,
        config.AMPLITUDE_MINIMA_EIXO_X
    )
    
    # Centra no meio da trajetória
    centro = (max_p + min_p) / 2
    
    lim_x_min = centro - largura_visual / 2
    lim_x_max = centro + largura_visual / 2
    
    return lim_x_min, lim_x_max


def calcular_limites_eixo_y(
    y_grid: np.ndarray,
    historico_y: List[float],
) -> Tuple[float, float]:
    """
    Calcula os limites do eixo Y com auto-scaling inteligente.
    
    Baseia-se na amplitude visível do terreno e trajetória, com proteção
    contra funções explosivas.
    
    Args:
        y_grid: Valores de y calculados para o grid de visualização.
        historico_y: Valores de y da trajetória.
        
    Returns:
        Tupla (chao, teto) para uso em ax.set_ylim.
    """
    # Encontra min/max visível
    y_min_visivel = min(np.min(y_grid), min(historico_y))
    y_max_visivel = max(np.max(y_grid), max(historico_y))
    
    # Calcula amplitude
    amplitude_y = y_max_visivel - y_min_visivel
    
    # Se amplitude for muito pequena, usa valor padrão
    if amplitude_y < config.AMPLITUDE_Y_MINIMA:
        amplitude_y = config.AMPLITUDE_Y_PADRAO
    
    # Adiciona margens (10% em cima e embaixo)
    chao = y_min_visivel - (amplitude_y * config.MARGEM_Y_PERCENTUAL)
    teto = y_max_visivel + (amplitude_y * config.MARGEM_Y_PERCENTUAL)
    
    # Trava de segurança para funções explosivas
    # Se o teto for muito alto e o chão próximo de zero, limita o teto
    if teto > config.LIMITE_MAXIMO_EIXO_Y and abs(chao) < config.LIMITE_MINIMO_EIXO_Y_ABSOLUTO:
        teto = config.LIMITE_MAXIMO_EIXO_Y
    
    return chao, teto


def plotar_terreno(
    ax: plt.Axes,
    funcao: Callable[[float], float],
    lim_x_min: float,
    lim_x_max: float,
) -> np.ndarray:
    """
    Plota o "terreno" (gráfico da função).
    
    Args:
        ax: Eixo matplotlib para plotagem.
        funcao: Função f(x) a ser plotada.
        lim_x_min: Limite mínimo do eixo X.
        lim_x_max: Limite máximo do eixo X.
        
    Returns:
        Array y_grid com valores calculados.
        
    Raises:
        ValueError: Se a função não for avaliável no domínio.
    """
    x_grid = np.linspace(lim_x_min, lim_x_max, config.PONTOS_GRID_X)
    
    try:
        y_grid = funcao(x_grid)
        ax.plot(
            x_grid, y_grid,
            color=config.COR_TERRENO,
            alpha=config.ALPHA_TERRENO,
            linewidth=config.LARGURA_TERRENO,
            label='Terreno f(x)'
        )
        return y_grid
    except Exception as e:
        raise ValueError(f"Erro ao plotar terreno: {e}")


def plotar_trajetoria(
    ax: plt.Axes,
    historico_x: List[float],
    historico_y: List[float],
) -> None:
    """
    Plota a trajetória do algoritmo no gráfico.
    
    Inclui a linha de trajetória, pontos de progresso, ponto inicial e final.
    
    Args:
        ax: Eixo matplotlib para plotagem.
        historico_x: Valores de x visitados.
        historico_y: Valores de y correspondentes.
    """
    # Linha de trajetória
    ax.plot(
        historico_x, historico_y,
        color=config.COR_TRAJETORIA,
        alpha=config.ALPHA_TRAJETORIA,
        linestyle=config.ESTILO_TRAJETORIA,
        label='Trajetória'
    )
    
    # Pontos de progresso (gradualmente mais vermelhos)
    ax.scatter(
        historico_x, historico_y,
        c=range(len(historico_x)),
        cmap=config.CMAP_PONTOS,
        s=config.TAMANHO_PONTOS,
        edgecolors='k',
        zorder=3
    )
    
    # Ponto inicial
    ax.scatter(
        historico_x[0], historico_y[0],
        c=config.COR_INICIO,
        s=config.TAMANHO_INICIO,
        zorder=5,
        edgecolors='k',
        label='Início'
    )
    
    # Ponto final
    ax.scatter(
        historico_x[-1], historico_y[-1],
        c=config.COR_FIM,
        marker=config.MARCADOR_FIM,
        s=config.TAMANHO_FIM,
        zorder=5,
        edgecolors='k',
        label='Fim'
    )


def gerar_titulo_grafico(resultado: ResultadoGradiente, momentum: float) -> str:
    """
    Gera o título do gráfico baseado no resultado e momentum.
    
    Args:
        resultado: Resultado da otimização.
        momentum: Valor do coeficiente de momentum.
        
    Returns:
        String com o título formatado.
    """
    titulo = f"Status: {resultado.status.upper()}"
    if momentum > 0:
        titulo += f" | Momentum: {momentum:.2f}"
    return titulo


def criar_visualizacao(
    funcao: Callable[[float], float],
    resultado: ResultadoGradiente,
    momentum: float,
) -> plt.Figure:
    """
    Cria a visualização completa do resultado da otimização.
    
    Integra terreno, trajetória e configurações visuais em um único gráfico.
    
    Args:
        funcao: Função f(x) original.
        resultado: Resultado da execução do algoritmo.
        momentum: Valor do momentum usado.
        
    Returns:
        Figura matplotlib com a visualização.
        
    Raises:
        ValueError: Se houver erro ao plotar terreno ou se não há dados.
    """
    if not resultado.tem_dados():
        raise ValueError("Nenhum dado disponível para visualizar.")
    
    fig, ax = plt.subplots(figsize=(config.FIGSIZE_X, config.FIGSIZE_Y))
    
    # Calcula limites dos eixos
    lim_x_min, lim_x_max = calcular_limites_eixo_x(resultado.x)
    
    # Plota terreno
    try:
        y_grid = plotar_terreno(ax, funcao, lim_x_min, lim_x_max)
    except ValueError as e:
        # Mesmo com erro, tenta plotar trajetória
        ax.text(0.5, 0.5, f"Erro: {e}", ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Calcula e aplica limites do eixo Y
    chao, teto = calcular_limites_eixo_y(y_grid, resultado.y)
    ax.set_ylim(chao, teto)
    
    # Plota trajetória
    plotar_trajetoria(ax, resultado.x, resultado.y)
    
    # Configurações finais
    titulo = gerar_titulo_grafico(resultado, momentum)
    ax.set_title(titulo)
    ax.legend(loc=config.LOCALIZACAO_LEGENDA)
    ax.grid(True, linestyle=config.ESTILO_GRID, alpha=config.ALPHA_GRID)
    
    return fig
