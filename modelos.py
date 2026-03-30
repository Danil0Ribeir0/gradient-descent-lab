from dataclasses import dataclass
from typing import List


@dataclass
class ResultadoGradiente:
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
