"""
Módulo de cálculo matemático para o Gradient Descent Visualizer.

Implementa o algoritmo de gradiente descendente com derivação numérica
e suporte para momentum (inércia).
"""

from typing import Callable, Dict, Any
import numpy as np
import config
from modelos import ResultadoGradiente


def derivada_numerica(
    funcao: Callable[[float], float],
    x: float,
    h: float = config.DERIVADA_H,
) -> float:
    """
    Calcula a derivada de uma função em um ponto usando aproximação numérica.
    
    Usa a definição de limite: f'(x) ≈ (f(x+h) - f(x)) / h
    
    Args:
        funcao: Função para a qual calcular a derivada.
        x: Ponto em que calcular a derivada.
        h: Pequeno deslocamento para aproximação (padrão: 0.0001).
        
    Returns:
        Valor aproximado da derivada em x.
        
    Raises:
        ValueError: Se a função não for avaliável em x ou x+h.
    """
    try:
        y_atual = funcao(x)
        y_frente = funcao(x + h)
        
        if np.isnan(y_atual) or np.isnan(y_frente):
            raise ValueError(f"Função retornou NaN em x={x}")
        
        return (y_frente - y_atual) / h
    except Exception as e:
        raise ValueError(f"Erro ao calcular derivada: {e}")


def executar_gradiente(
    funcao: Callable[[float], float],
    x_inicial: float,
    learning_rate: float,
    iteracoes: int,
    momentum: float = 0.0,
) -> ResultadoGradiente:
    """
    Executa o algoritmo de gradiente descendente com suporte a momentum.
    
    O algoritmo iterativamente:
    1. Calcula a derivada (inclinação) no ponto atual
    2. Atualiza velocidade (momentum): v = momentum*v + lr*derivada
    3. Move-se contra a derivada: x_novo = x_atual - v
    
    Args:
        funcao: Função f(x) a ser minimizada.
        x_inicial: Ponto de partida.
        learning_rate: Taxa de aprendizado (tamanho do passo).
        iteracoes: Número máximo de iterações.
        momentum: Coeficiente de momentum em [0.0, 0.99] (padrão: 0.0).
        
    Returns:
        ResultadoGradiente com histórico, status e diagnóstico.
    """
    # Inicializa histórico
    historico_x = [x_inicial]
    historico_y = []
    
    # Validação inicial
    try:
        y_ini = funcao(x_inicial)
        if np.isnan(y_ini) or np.isinf(y_ini):
            return ResultadoGradiente(
                x=[],
                y=[],
                status="erro",
                msg="Ponto inicial retorna NaN ou infinito.",
                incl_final=0.0,
            )
        historico_y.append(y_ini)
    except Exception as e:
        return ResultadoGradiente(
            x=[],
            y=[],
            status="erro",
            msg=f"Ponto inicial inválido: {e}",
            incl_final=0.0,
        )
    
    # Estado do algoritmo
    x_atual = x_inicial
    velocidade = 0.0  # Para momentum
    status = "sucesso"
    mensagem = "Convergência realizada."
    
    # Loop de otimização
    for i in range(iteracoes):
        try:
            # Calcula derivada
            inclinacao = derivada_numerica(funcao, x_atual)
            
            # Atualiza velocidade com momentum
            velocidade = (momentum * velocidade) + (learning_rate * inclinacao)
            
            # Atualiza posição
            x_novo = x_atual - velocidade
            
            # Verifica explosão
            if abs(x_novo) > config.LIMITE_X_EXPLOSAO or np.isnan(x_novo):
                status = "explosao"
                mensagem = f"Explodiu na iteração {i+1}!"
                break
            
            x_atual = x_novo
            historico_x.append(x_atual)
            
            # Calcula novo y
            y_novo = funcao(x_atual)
            if np.isnan(y_novo) or np.isinf(y_novo):
                status = "erro"
                mensagem = f"Função retornou NaN/inf na iteração {i+1}."
                break
            
            historico_y.append(y_novo)
            
        except Exception as e:
            status = "erro"
            mensagem = f"Erro na iteração {i+1}: {str(e)}"
            break
    
    # Diagnóstico final
    incl_final = 0.0
    if historico_x:
        try:
            incl_final = derivada_numerica(funcao, historico_x[-1])
        except Exception:
            pass  # Se não conseguir calcular, deixa 0.0
    
    # Determina status final
    if status == "sucesso":
        if abs(incl_final) < config.TOLERANCIA_INCLINACAO:
            status = "otimo"
            mensagem = "Mínimo encontrado."
        elif abs(velocidade) > config.TOLERANCIA_VELOCIDADE:
            status = "descendo"
            mensagem = "Passando pelo plano com inércia..."
        else:
            status = "proximo"
            mensagem = "Perto do mínimo."
    
    return ResultadoGradiente(
        x=historico_x,
        y=historico_y,
        status=status,
        msg=mensagem,
        incl_final=incl_final,
    )