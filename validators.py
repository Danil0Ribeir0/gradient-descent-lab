import numpy as np
from typing import Callable, Tuple
import config


class ValidadorParametros:
    
    @staticmethod
    def validar_funcao(funcao_texto: str) -> Tuple[bool, str, Callable]:
        try:
            func_usuario = lambda x: eval(funcao_texto, {"x": x, "np": np})
            test_result = func_usuario(0)
            
            if np.isnan(test_result) or np.isinf(test_result):
                return False, "Função retorna NaN ou infinito", None
            
            return True, "Função válida", func_usuario
        
        except SyntaxError as e:
            return False, f"Erro de sintaxe: {e}", None
        except NameError as e:
            return False, f"Variável não definida: {e}", None
        except Exception as e:
            return False, f"Erro ao validar função: {e}", None
    
    @staticmethod
    def validar_learning_rate(lr: float) -> Tuple[bool, str]:
        if not isinstance(lr, (int, float)):
            return False, "Learning Rate deve ser um número"
        if lr < config.LR_MIN or lr > config.LR_MAX:
            return False, f"Learning Rate deve estar entre {config.LR_MIN} e {config.LR_MAX}"
        return True, "OK"
    
    @staticmethod
    def validar_momentum(momentum: float) -> Tuple[bool, str]:
        if not isinstance(momentum, (int, float)):
            return False, "Momentum deve ser um número"
        if momentum < config.MOMENTUM_MIN or momentum > config.MOMENTUM_MAX:
            return False, f"Momentum deve estar entre {config.MOMENTUM_MIN} e {config.MOMENTUM_MAX}"
        return True, "OK"
    
    @staticmethod
    def validar_iteracoes(iteracoes: int) -> Tuple[bool, str]:
        if not isinstance(iteracoes, int):
            return False, "Iterações deve ser um inteiro"
        if iteracoes < config.ITERACOES_MIN or iteracoes > config.ITERACOES_MAX:
            return False, f"Iterações deve estar entre {config.ITERACOES_MIN} e {config.ITERACOES_MAX}"
        return True, "OK"
    
    @staticmethod
    def validar_x_inicial(x_ini: float) -> Tuple[bool, str]:
        if not isinstance(x_ini, (int, float)):
            return False, "X inicial deve ser um número"
        if np.isnan(x_ini) or np.isinf(x_ini):
            return False, "X inicial não pode ser NaN ou infinito"
        return True, "OK"
    
    @staticmethod
    def validar_todos_parametros(
        funcao_texto: str,
        lr: float,
        momentum: float,
        iteracoes: int,
        x_ini: float
    ) -> Tuple[bool, list]:
        erros = []
        
        valido, msg, _ = ValidadorParametros.validar_funcao(funcao_texto)
        if not valido:
            erros.append(f"Função: {msg}")
        
        valido, msg = ValidadorParametros.validar_learning_rate(lr)
        if not valido:
            erros.append(f"Learning Rate: {msg}")
        
        valido, msg = ValidadorParametros.validar_momentum(momentum)
        if not valido:
            erros.append(f"Momentum: {msg}")
        
        valido, msg = ValidadorParametros.validar_iteracoes(iteracoes)
        if not valido:
            erros.append(f"Iterações: {msg}")
        
        valido, msg = ValidadorParametros.validar_x_inicial(x_ini)
        if not valido:
            erros.append(f"X inicial: {msg}")
        
        return len(erros) == 0, erros
