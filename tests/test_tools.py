import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import derivada_numerica, executar_gradiente
from modelos import ResultadoGradiente


class TestDerivadaNumerica:
    
    def test_derivada_x_ao_quadrado(self):
        f = lambda x: x**2
        resultado = derivada_numerica(f, 2.0)
        assert abs(resultado - 4.0) < 0.01
    
    def test_derivada_x_cubica(self):
        f = lambda x: x**3
        resultado = derivada_numerica(f, 1.0)
        assert abs(resultado - 3.0) < 0.1
    
    def test_derivada_exponencial(self):
        f = lambda x: np.exp(x)
        resultado = derivada_numerica(f, 0.0)
        assert abs(resultado - 1.0) < 0.01
    
    def test_derivada_seno(self):
        f = lambda x: np.sin(x)
        resultado = derivada_numerica(f, 0.0)
        assert abs(resultado - 1.0) < 0.01
    
    def test_derivada_funcao_nan(self):
        f = lambda x: np.nan if x > 0 else x
        with pytest.raises(ValueError):
            derivada_numerica(f, 1.0)
    
    def test_derivada_com_h_customizado(self):
        f = lambda x: x**2
        resultado = derivada_numerica(f, 3.0, h=0.001)
        assert abs(resultado - 6.0) < 0.1


class TestExecutarGradiente:
    
    def test_convergencia_simples(self):
        f = lambda x: (x - 1)**2
        resultado = executar_gradiente(f, 0.0, 0.1, 50, 0.0)
        
        assert isinstance(resultado, ResultadoGradiente)
        assert resultado.tem_dados()
        assert resultado.x[-1] > 0.5
    
    def test_ponto_inicial_invalido_nan(self):
        f = lambda x: x**2
        resultado = executar_gradiente(f, np.nan, 0.1, 50)
        
        assert resultado.status == "erro"
        assert not resultado.tem_dados()
    
    def test_ponto_inicial_invalido_inf(self):
        f = lambda x: x**2
        resultado = executar_gradiente(f, np.inf, 0.1, 50)
        
        assert resultado.status == "erro"
        assert not resultado.tem_dados()
    
    def test_funcao_invalida(self):
        f = lambda x: np.inf if x > 0 else x**2
        resultado = executar_gradiente(f, -1.0, 0.1, 50)
        
        assert resultado.status in ["erro", "explosao"]
    
    def test_com_momentum(self):
        f = lambda x: (x - 2)**2
        resultado_sem = executar_gradiente(f, 0.0, 0.1, 50, 0.0)
        resultado_com = executar_gradiente(f, 0.0, 0.1, 50, 0.5)
        
        assert len(resultado_sem.x) > 0
        assert len(resultado_com.x) > 0
    
    def test_learning_rate_muito_grande(self):
        f = lambda x: (x - 1)**2
        resultado = executar_gradiente(f, 0.0, 1.0, 10)
        
        assert resultado.status in ["explosao", "sucesso", "descendo"]
    
    def test_learning_rate_muito_pequeno(self):
        f = lambda x: (x - 1)**2
        resultado = executar_gradiente(f, 0.0, 0.00001, 5)
        
        assert resultado.tem_dados()
        assert abs(resultado.x[1] - resultado.x[0]) < 0.01
    
    def test_funcao_linear(self):
        f = lambda x: 2*x + 1
        resultado = executar_gradiente(f, 0.0, 0.01, 10)
        
        assert resultado.status == "descendo"
    
    def test_convergencia_minimo_local(self):
        f = lambda x: np.sin(x)
        resultado = executar_gradiente(f, 1.5, 0.1, 100, 0.0)
        
        assert resultado.tem_dados()
        assert resultado.status in ["otimo", "proximo", "descendo"]
    
    def test_explosao_funcao_explosiva(self):
        f = lambda x: x**4
        resultado = executar_gradiente(f, 5.0, 0.5, 50)
        
        if resultado.status == "explosao":
            assert resultado.obteve_erro() or resultado.explodiu()
    
    def test_iteracoes_zero(self):
        f = lambda x: x**2
        resultado = executar_gradiente(f, 1.0, 0.1, 0)
        
        assert len(resultado.x) == 1
        assert resultado.x[0] == 1.0
    
    def test_retorna_resultado_gradiente(self):
        f = lambda x: x**2
        resultado = executar_gradiente(f, 0.5, 0.1, 10)
        
        assert isinstance(resultado, ResultadoGradiente)
        assert hasattr(resultado, 'x')
        assert hasattr(resultado, 'y')
        assert hasattr(resultado, 'status')
        assert hasattr(resultado, 'msg')
        assert hasattr(resultado, 'incl_final')


class TestResultadoGradiente:
    
    def test_tem_dados_valido(self):
        resultado = ResultadoGradiente(
            x=[1.0, 2.0], 
            y=[1.0, 4.0], 
            status="sucesso",
            msg="Teste",
            incl_final=0.5
        )
        assert resultado.tem_dados()
    
    def test_tem_dados_vazio(self):
        resultado = ResultadoGradiente(
            x=[], 
            y=[], 
            status="erro",
            msg="Erro",
            incl_final=0.0
        )
        assert not resultado.tem_dados()
    
    def test_convergiu(self):
        resultado = ResultadoGradiente(
            x=[1.0], 
            y=[1.0], 
            status="otimo",
            msg="Convergiu",
            incl_final=0.001
        )
        assert resultado.convergiu()
    
    def test_explodiu(self):
        resultado = ResultadoGradiente(
            x=[1.0, 100.0], 
            y=[1.0, 10000.0], 
            status="explosao",
            msg="Explodiu",
            incl_final=1000.0
        )
        assert resultado.explodiu()
    
    def test_obteve_erro(self):
        resultado = ResultadoGradiente(
            x=[], 
            y=[], 
            status="erro",
            msg="Erro",
            incl_final=0.0
        )
        assert resultado.obteve_erro()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
