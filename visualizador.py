import numpy as np
import plotly.graph_objects as go
from typing import Callable, Tuple, List, Optional
import config
from modelos import ResultadoGradiente

def calcular_limites_eixo_x(historico_a: List[float], historico_b: Optional[List[float]] = None) -> Tuple[float, float]:
    todos_x = list(historico_a)
    if historico_b:
        todos_x.extend(historico_b)
        
    if not todos_x:
        return (-config.AMPLITUDE_MINIMA_EIXO_X / 2, config.AMPLITUDE_MINIMA_EIXO_X / 2)
    
    min_p, max_p = min(todos_x), max(todos_x)
    largura_visual = max((max_p - min_p) * config.ZOOM_FACTOR, config.AMPLITUDE_MINIMA_EIXO_X)
    centro = (max_p + min_p) / 2
    
    return centro - largura_visual / 2, centro + largura_visual / 2

def criar_visualizacao(
    funcao: Callable[[float], float],
    resultado_a: ResultadoGradiente,
    momentum_a: float,
    resultado_b: Optional[ResultadoGradiente] = None,
    momentum_b: float = 0.0,
    otimizador_a: str = "SGD",
    otimizador_b: str = "SGD"
) -> go.Figure:
    
    if not resultado_a.tem_dados():
        raise ValueError("Nenhum dado disponível no Modelo A para visualizar.")
    
    lim_x_min, lim_x_max = calcular_limites_eixo_x(
        resultado_a.x, 
        resultado_b.x if resultado_b and resultado_b.tem_dados() else None
    )
    
    x_grid = np.linspace(lim_x_min, lim_x_max, config.PONTOS_GRID_X)
    try:
        y_grid = [funcao(x) for x in x_grid]
    except Exception as e:
        raise ValueError(f"Erro ao plotar terreno: {e}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_grid, y=y_grid, mode='lines', name='Terreno f(x)',
        line=dict(color=config.COR_TERRENO, width=config.LARGURA_TERRENO), opacity=0.6
    ))

    fig.add_trace(go.Scatter(
        x=resultado_a.x, y=resultado_a.y, mode='lines', name='Caminho A',
        line=dict(color=config.COR_TRAJETORIA, dash='dash', width=2), opacity=0.5
    ))
    fig.add_trace(go.Scatter(
        x=resultado_a.x, y=resultado_a.y, mode='markers', name='Passos A',
        marker=dict(size=8, color=list(range(len(resultado_a.x))), colorscale='Reds', line=dict(width=1, color='black')),
        hovertemplate="Modelo A<br>x: %{x:.4f}<br>f(x): %{y:.4f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=[resultado_a.x[-1]], y=[resultado_a.y[-1]], mode='markers', name='Fim A',
        marker=dict(size=18, color=config.COR_FIM, symbol='star', line=dict(width=2, color='black'))
    ))

    if resultado_b and resultado_b.tem_dados():
        fig.add_trace(go.Scatter(
            x=resultado_b.x, y=resultado_b.y, mode='lines', name='Caminho B',
            line=dict(color='blue', width=2), opacity=0.6
        ))
        fig.add_trace(go.Scatter(
            x=resultado_b.x, y=resultado_b.y, mode='markers', name='Passos B',
            marker=dict(size=8, color=list(range(len(resultado_b.x))), colorscale='Blues', line=dict(width=1, color='black')),
            hovertemplate="Modelo B<br>x: %{x:.4f}<br>f(x): %{y:.4f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=[resultado_b.x[-1]], y=[resultado_b.y[-1]], mode='markers', name='Fim B',
            marker=dict(size=16, color='cyan', symbol='star-diamond', line=dict(width=2, color='black'))
        ))

    fig.add_trace(go.Scatter(
        x=[resultado_a.x[0]], y=[resultado_a.y[0]], mode='markers', name='Início',
        marker=dict(size=14, color=config.COR_INICIO, line=dict(width=2, color='black'))
    ))

    titulo = f"Modelo A ({otimizador_a}): {resultado_a.status.upper()} (Mom/Beta: {momentum_a:.2f})"
    if resultado_b:
        titulo += f" | Modelo B ({otimizador_b}): {resultado_b.status.upper()} (Mom/Beta: {momentum_b:.2f})"

    fig.update_layout(
        title=titulo,
        xaxis_title="Eixo X", yaxis_title="f(x)",
        template="plotly_white", hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig