import numpy as np
import plotly.graph_objects as go
from typing import Callable, Tuple, List
import config
from modelos import ResultadoGradiente

def calcular_limites_eixo_x(historico_x: List[float]) -> Tuple[float, float]:
    if not historico_x:
        return (-config.AMPLITUDE_MINIMA_EIXO_X / 2, config.AMPLITUDE_MINIMA_EIXO_X / 2)
    
    min_p, max_p = min(historico_x), max(historico_x)
    largura_visual = max((max_p - min_p) * config.ZOOM_FACTOR, config.AMPLITUDE_MINIMA_EIXO_X)
    centro = (max_p + min_p) / 2
    
    return centro - largura_visual / 2, centro + largura_visual / 2

def criar_visualizacao(
    funcao: Callable[[float], float],
    resultado: ResultadoGradiente,
    momentum: float,
) -> go.Figure:
    if not resultado.tem_dados():
        raise ValueError("Nenhum dado disponível para visualizar.")
    
    lim_x_min, lim_x_max = calcular_limites_eixo_x(resultado.x)
    
    x_grid = np.linspace(lim_x_min, lim_x_max, config.PONTOS_GRID_X)
    try:
        y_grid = [funcao(x) for x in x_grid]
    except Exception as e:
        raise ValueError(f"Erro ao plotar terreno: {e}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_grid, y=y_grid,
        mode='lines',
        name='Terreno f(x)',
        line=dict(color=config.COR_TERRENO, width=config.LARGURA_TERRENO),
        opacity=0.6
    ))

    fig.add_trace(go.Scatter(
        x=resultado.x, y=resultado.y,
        mode='lines',
        name='Caminho',
        line=dict(color=config.COR_TRAJETORIA, dash='dash', width=2),
        opacity=0.5
    ))

    fig.add_trace(go.Scatter(
        x=resultado.x, y=resultado.y,
        mode='markers',
        name='Passos',
        marker=dict(
            size=8,
            color=list(range(len(resultado.x))),
            colorscale='Reds',
            showscale=False,
            line=dict(width=1, color='black')
        ),
        hovertemplate="x: %{x:.4f}<br>f(x): %{y:.4f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=[resultado.x[0]], y=[resultado.y[0]], mode='markers', name='Início',
        marker=dict(size=14, color=config.COR_INICIO, line=dict(width=2, color='black'))
    ))
    fig.add_trace(go.Scatter(
        x=[resultado.x[-1]], y=[resultado.y[-1]], mode='markers', name='Fim',
        marker=dict(size=18, color=config.COR_FIM, symbol='star', line=dict(width=2, color='black'))
    ))

    titulo = f"Status: {resultado.status.upper()}" + (f" | Momentum: {momentum:.2f}" if momentum > 0 else "")
    fig.update_layout(
        title=titulo,
        xaxis_title="Eixo X",
        yaxis_title="f(x)",
        template="plotly_white",
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig