from dataclasses import dataclass
from typing import List


@dataclass
class ResultadoGradiente:
    """
    Encapsula o resultado de uma execução do algoritmo de gradiente descendente.
    
    Attributes:
        x (List[float]): Historicamente de valores de x visitados.
        y (List[float]): Historicamente de valores f(x) correspondentes.
        status (str): Status da otimização ('otimo', 'descendo', 'explosao', etc).
        msg (str): Mensagem descritiva sobre o resultado.
        incl_final (float): Inclinação (derivada) no ponto final.
    """
    x: List[float]
    y: List[float]
    status: str
    msg: str
    incl_final: float
    
    def tem_dados(self) -> bool:
        """Verifica se há dados válidos para plotagem."""
        return len(self.x) > 0 and len(self.y) > 0
    
    def convergiu(self) -> bool:
        """Verifica se o algoritmo convergiu para um mínimo."""
        return self.status == "otimo"
    
    def explodiu(self) -> bool:
        """Verifica se o algoritmo divergiu."""
        return self.status == "explosao"
    
    def obteve_erro(self) -> bool:
        """Verifica se ocorreu um erro durante a execução."""
        return self.status == "erro"


@dataclass
class ParametrosOtimizacao:
    """
    Encapsula os parâmetros de entrada do algoritmo de otimização.
    
    Attributes:
        learning_rate (float): Taxa de aprendizado (tamanho do passo).
        momentum (float): Coeficiente de momentum (inércia).
        iteracoes (int): Número máximo de iterações.
        x_inicial (float): Ponto de partida.
    """
    learning_rate: float
    momentum: float
    iteracoes: int
    x_inicial: float
    
    def __post_init__(self):
        """Valida os parâmetros após inicialização."""
        if not 0.001 <= self.learning_rate <= 1.0:
            raise ValueError(f"Learning rate deve estar entre 0.001 e 1.0, recebeu {self.learning_rate}")
        if not 0.0 <= self.momentum <= 0.99:
            raise ValueError(f"Momentum deve estar entre 0.0 e 0.99, recebeu {self.momentum}")
        if self.iteracoes < 1 or self.iteracoes > 200:
            raise ValueError(f"Iterações deve estar entre 1 e 200, recebeu {self.iteracoes}")


@dataclass
class DadosVisualizacao:
    """
    Encapsula dados necessários para visualizar a otimização.
    
    Attributes:
        historico_x (List[float]): Valores de x.
        historico_y (List[float]): Valores de y = f(x).
        resultado (ResultadoGradiente): Resultado da otimização.
        titulo (str): Título do gráfico.
    """
    historico_x: List[float]
    historico_y: List[float]
    resultado: ResultadoGradiente
    titulo: str
