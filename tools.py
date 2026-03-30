from typing import Callable
import numpy as np
import config
from modelos import ResultadoGradiente


def derivada_numerica(
    funcao: Callable[[float], float],
    x: float,
    h: float = config.DERIVADA_H,
) -> float:
    try:
        y_atual = funcao(x)
        y_frente = funcao(x + h)
        
        if np.isnan(y_atual) or np.isnan(y_frente):
            raise ValueError(f"Função retornou NaN em x={x}")
        
        derivada = (y_frente - y_atual) / h
        return derivada
        
    except Exception as e:
        raise ValueError(f"Erro ao calcular derivada: {e}")


def executar_gradiente(
    funcao: Callable[[float], float],
    x_inicial: float,
    learning_rate: float,
    iteracoes: int,
    momentum: float = 0.0,
) -> ResultadoGradiente:
    historico_x = [x_inicial]
    historico_y = []
    
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
    
    x_atual = x_inicial
    velocidade = 0.0
    status = "sucesso"
    mensagem = "Convergência realizada."
    
    for i in range(iteracoes):
        try:
            inclinacao = derivada_numerica(funcao, x_atual)
            velocidade = (momentum * velocidade) + (learning_rate * inclinacao)
            x_novo = x_atual - velocidade
            
            if abs(x_novo) > config.LIMITE_X_EXPLOSAO:
                status = "explosao"
                mensagem = f"Explodiu na iteração {i+1}!"
                break
            
            if np.isnan(x_novo):
                status = "explosao"
                mensagem = f"Explodiu na iteração {i+1}!"
                break
            
            x_atual = x_novo
            historico_x.append(x_atual)
            
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
    
    incl_final = 0.0
    if historico_x:
        try:
            incl_final = derivada_numerica(funcao, historico_x[-1])
        except Exception as e:
            pass
    
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