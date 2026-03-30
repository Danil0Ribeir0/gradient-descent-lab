import numpy as np
import sympy as sp
from typing import Callable, Tuple
import config

class ValidadorParametros:
    
    @staticmethod
    def validar_funcao(funcao_texto: str) -> Tuple[bool, str, Callable]:
        try:
            x_sym = sp.Symbol('x')
            expr = sp.sympify(funcao_texto)
            
            func_usuario = sp.lambdify(x_sym, expr, modules=['numpy'])
            
            test_result = func_usuario(np.array([0.0, 1.0]))
            
            if isinstance(test_result, (int, float)):
                func_usuario_original = func_usuario
                func_usuario = lambda x: np.full_like(x, func_usuario_original(x), dtype=float)
                test_result = func_usuario(np.array([0.0, 1.0]))

            if np.any(np.isnan(test_result)) or np.any(np.isinf(test_result)):
                return False, "Função retorna NaN ou infinito no ponto inicial", None
            
            return True, "Função válida", func_usuario
        
        except Exception as e:
            return False, f"Erro na expressão matemática. Verifique a sintaxe. (Detalhe: {e})", None