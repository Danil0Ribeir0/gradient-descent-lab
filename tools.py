from typing import Callable
import numpy as np
import config
from modelos import ResultadoGradiente


class OtimizadorBase:
    def __init__(self, learning_rate: float, momentum: float):
        self.lr = learning_rate
        self.momentum = momentum
        self.ultimo_passo = 0.0

    def calcular_passo(self, inclinacao: float, t: int) -> float:
        raise NotImplementedError("Deve ser implementado pela classe filha")


class SGD(OtimizadorBase):
    def __init__(self, learning_rate: float, momentum: float):
        super().__init__(learning_rate, momentum)
        self.velocidade = 0.0
        
    def calcular_passo(self, inclinacao: float, t: int) -> float:
        self.velocidade = (self.momentum * self.velocidade) + (self.lr * inclinacao)
        self.ultimo_passo = self.velocidade
        return self.ultimo_passo


class AdaGrad(OtimizadorBase):
    def __init__(self, learning_rate: float, momentum: float):
        super().__init__(learning_rate, momentum)
        self.G = 0.0
        self.epsilon = 1e-8
        
    def calcular_passo(self, inclinacao: float, t: int) -> float:
        self.G += inclinacao ** 2
        self.ultimo_passo = (self.lr / (np.sqrt(self.G) + self.epsilon)) * inclinacao
        return self.ultimo_passo


class RMSProp(OtimizadorBase):
    def __init__(self, learning_rate: float, momentum: float):
        super().__init__(learning_rate, momentum)
        self.G = 0.0
        self.epsilon = 1e-8
        self.beta = momentum if momentum > 0 else 0.9
        
    def calcular_passo(self, inclinacao: float, t: int) -> float:
        self.G = self.beta * self.G + (1 - self.beta) * (inclinacao ** 2)
        self.ultimo_passo = (self.lr / (np.sqrt(self.G) + self.epsilon)) * inclinacao
        return self.ultimo_passo


class Adam(OtimizadorBase):
    def __init__(self, learning_rate: float, momentum: float):
        super().__init__(learning_rate, momentum)
        self.M = 0.0
        self.G = 0.0
        self.epsilon = 1e-8
        self.beta1 = momentum
        self.beta2 = 0.999
        
    def calcular_passo(self, inclinacao: float, t: int) -> float:
        self.M = self.beta1 * self.M + (1 - self.beta1) * inclinacao
        self.G = self.beta2 * self.G + (1 - self.beta2) * (inclinacao ** 2)
        
        M_hat = self.M / (1 - self.beta1 ** t)
        G_hat = self.G / (1 - self.beta2 ** t)
        
        self.ultimo_passo = (self.lr / (np.sqrt(G_hat) + self.epsilon)) * M_hat
        return self.ultimo_passo

def derivada_numerica(funcao: Callable[[float], float], x: float, h: float = config.DERIVADA_H) -> float:
    try:
        y_frente = funcao(x + h)
        y_tras = funcao(x - h)
        
        if np.isnan(y_frente) or np.isnan(y_tras):
            raise ValueError(f"Função retornou NaN ao redor de x={x}")
        
        return (y_frente - y_tras) / (2 * h)
    except Exception as e:
        raise ValueError(f"Erro ao calcular derivada: {e}")


def executar_gradiente(
    funcao: Callable[[float], float],
    x_inicial: float,
    learning_rate: float,
    iteracoes: int,
    momentum: float = 0.0,
    otimizador_nome: str = "SGD"
) -> ResultadoGradiente:
    
    try:
        y_ini = funcao(x_inicial)
        if np.isnan(y_ini) or np.isinf(y_ini):
            return ResultadoGradiente(x=[], y=[], status="erro", msg="Ponto inicial retorna NaN/infinito.", incl_final=0.0)
    except Exception as e:
        return ResultadoGradiente(x=[], y=[], status="erro", msg=f"Ponto inicial inválido: {e}", incl_final=0.0)
    
    mapa_otimizadores = {
        "SGD": SGD,
        "AdaGrad": AdaGrad,
        "RMSProp": RMSProp,
        "Adam": Adam
    }
    
    if otimizador_nome not in mapa_otimizadores:
         return ResultadoGradiente(x=[], y=[], status="erro", msg=f"Otimizador desconhecido: {otimizador_nome}", incl_final=0.0)
         
    motor = mapa_otimizadores[otimizador_nome](learning_rate, momentum)
    
    historico_x = [x_inicial]
    historico_y = [y_ini]
    x_atual = x_inicial
    status = "sucesso"
    mensagem = "Convergência realizada."

    incl_final = 0.0
    
    for t in range(1, iteracoes + 1):
        try:
            inclinacao = derivada_numerica(funcao, x_atual)
            passo = motor.calcular_passo(inclinacao, t)

            if abs(inclinacao) < config.TOLERANCIA_INCLINACAO and abs(passo) <= config.TOLERANCIA_VELOCIDADE:
                incl_final = inclinacao
                status, mensagem = "otimo", f"Mínimo encontrado na iteração {t}."
                break
            
            x_novo = x_atual - passo
            
            if abs(x_novo) > config.LIMITE_X_EXPLOSAO or np.isnan(x_novo):
                status, mensagem = "explosao", f"Explodiu na iteração {t}!"
                break
            
            y_novo = funcao(x_novo)
            if np.isnan(y_novo) or np.isinf(y_novo):
                status, mensagem = "erro", f"Função retornou NaN/inf na iteração {t}."
                break
            
            x_atual = x_novo
            historico_x.append(x_atual)
            historico_y.append(y_novo)
            
        except Exception as e:
            status, mensagem = "erro", f"Erro na iteração {t}: {str(e)}"
            break
            
    if status == "sucesso" and historico_x:
        try:
            incl_final = derivada_numerica(funcao, historico_x[-1])
            
            if abs(incl_final) < config.TOLERANCIA_INCLINACAO:
                status, mensagem = "otimo", "Mínimo encontrado no limite de iterações."
            elif abs(motor.ultimo_passo) > config.TOLERANCIA_VELOCIDADE:
                status, mensagem = "descendo", "Passando pelo plano com inércia..."
            else:
                status, mensagem = "proximo", "Perto do mínimo ou velocidade decaiu."
                
        except Exception as e:
            status, mensagem = "erro", f"Erro final (possível overflow): {str(e)}"
            
    return ResultadoGradiente(x=historico_x, y=historico_y, status=status, msg=mensagem, incl_final=incl_final)