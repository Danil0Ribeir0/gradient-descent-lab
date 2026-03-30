import numpy as np
from tools import executar_gradiente
from validators import ValidadorParametros


def exemplo_1_quadratica_simples():
    print("\n" + "="*60)
    print("EXEMPLO 1: Minimizar Funcao Quadratica Simples")
    print("="*60)
    
    f = lambda x: (x - 2)**2
    
    print("Funcao: f(x) = (x - 2)^2")
    print("Minimo teorico: x = 2, f(x) = 0\n")
    
    resultado = executar_gradiente(f, 0.0, 0.1, 50)
    
    print(f"Posicao final: {resultado.x[-1]:.4f}")
    print(f"Valor f(x): {resultado.y[-1]:.6f}")
    print(f"Iteracoes: {len(resultado.x) - 1}")
    print(f"Status: {resultado.status}")
    print(f"Mensagem: {resultado.msg}")


def exemplo_2_polinomio_grau_4():
    print("\n" + "="*60)
    print("EXEMPLO 2: Minimizar Polinomio de Grau 4")
    print("="*60)
    
    f = lambda x: x**4 - 2*x**2 + 1
    
    print("Funcao: f(x) = x^4 - 2x^2 + 1")
    print("Minimos locais em: x ≈ ±1\n")
    
    r1 = executar_gradiente(f, 0.5, 0.1, 200, momentum=0.0)
    r2 = executar_gradiente(f, 0.5, 0.1, 200, momentum=0.8)
    
    print("Sem momentum:")
    print(f"  Posicao: {r1.x[-1]:.4f}")
    print(f"  Iteracoes: {len(r1.x) - 1}")
    print(f"  Status: {r1.status}\n")
    
    print("Com momentum (0.8):")
    print(f"  Posicao: {r2.x[-1]:.4f}")
    print(f"  Iteracoes: {len(r2.x) - 1}")
    print(f"  Status: {r2.status}")


def exemplo_3_learning_rate_impact():
    print("\n" + "="*60)
    print("EXEMPLO 3: Impacto do Learning Rate")
    print("="*60)
    
    f = lambda x: (x - 3)**2
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    
    print("Funcao: f(x) = (x - 3)^2\n")
    
    for lr in learning_rates:
        resultado = executar_gradiente(f, 0.0, lr, 200)
        print(f"LR = {lr:5.3f} | Posicao: {resultado.x[-1]:7.4f} | Iteracoes: {len(resultado.x)-1:3} | Status: {resultado.status}")


def exemplo_4_momentum_comparacao():
    print("\n" + "="*60)
    print("EXEMPLO 4: Comparacao de Momentums")
    print("="*60)
    
    f = lambda x: x**4 - 3*x**2 + 2
    momentums = [0.0, 0.3, 0.5, 0.7, 0.9]
    
    print("Funcao: f(x) = x^4 - 3x^2 + 2")
    print("Ponto inicial: x = 1.0\n")
    
    for m in momentums:
        resultado = executar_gradiente(f, 1.0, 0.05, 300, m)
        x_final = resultado.x[-1]
        y_final = resultado.y[-1]
        
        print(f"Momentum = {m:.1f} | x_final = {x_final:7.4f} | f(x) = {y_final:8.6f} | Iteracoes: {len(resultado.x)-1:3} | {resultado.status}")


def exemplo_5_validacao():
    print("\n" + "="*60)
    print("EXEMPLO 5: Validar Parametros")
    print("="*60)
    
    print("Testando validacao de parametros...\n")
    
    casos = [
        ("x**2", 0.1, 0.5, 100, 0.0, True),
        ("x**2", 2.0, 0.5, 100, 0.0, False),
        ("x**2", 0.1, 1.5, 100, 0.0, False),
        ("1/0", 0.1, 0.5, 100, 0.0, False),
    ]
    
    for func_texto, lr, momentum, iteracoes, x_ini, esperado in casos:
        valido, erros = ValidadorParametros.validar_todos_parametros(
            func_texto, lr, momentum, iteracoes, x_ini
        )
        
        status = "OK" if valido == esperado else "FALHOU"
        print(f"{status:6} | Funcao: {func_texto:10} | LR: {lr:5.2f} | Valido: {valido}")


def exemplo_6_multiplas_funcoes():
    print("\n" + "="*60)
    print("EXEMPLO 6: Otimizar Multiplas Funcoes")
    print("="*60)
    
    funcoes = {
        "Quadratica": lambda x: x**2,
        "Cubica": lambda x: (x - 1)**3,
        "Exponencial": lambda x: np.exp(x),
        "Seno": lambda x: np.sin(x),
        "Seno*x": lambda x: np.sin(x) * x,
    }
    
    print("Ponto inicial: 1.0, Learning Rate: 0.05, Iteracoes: 100\n")
    
    for nome, f in funcoes.items():
        try:
            resultado = executar_gradiente(f, 1.0, 0.05, 100, 0.3)
            print(f"{nome:15} | x_final: {resultado.x[-1]:7.4f} | Status: {resultado.status:10} | Iter: {len(resultado.x)-1:3}")
        except:
            print(f"{nome:15} | ERRO NA EXECUCAO")


def exemplo_7_analisando_convergencia():
    print("\n" + "="*60)
    print("EXEMPLO 7: Analisando Convergencia Passo a Passo")
    print("="*60)
    
    f = lambda x: (x - 2)**2
    resultado = executar_gradiente(f, 0.0, 0.2, 10)
    
    print("Funcao: f(x) = (x - 2)^2\n")
    print("| Iter |     x     |    f(x)    | Decrescimo |")
    print("|" + "-"*49 + "|")
    
    for i in range(len(resultado.x)):
        if i == 0:
            decrescimo = "---"
        else:
            decrescimo = f"{resultado.y[i-1] - resultado.y[i]:.6f}"
        
        print(f"| {i:4d} | {resultado.x[i]:9.6f} | {resultado.y[i]:10.6f} | {decrescimo:>10} |")
    
    print(f"\nStatus Final: {resultado.status}")
    print(f"Posicao Final: {resultado.x[-1]:.6f}")
    print(f"Valor Final: {resultado.y[-1]:.6f}")


def exemplo_8_divergencia():
    print("\n" + "="*60)
    print("EXEMPLO 8: Detectando Divergencia")
    print("="*60)
    
    f_explosiva = lambda x: x**4
    
    print("Funcao: f(x) = x^4\n")
    print("Testando com diferentes learning rates...\n")
    
    learning_rates = [0.01, 0.1, 0.5, 1.0]
    
    for lr in learning_rates:
        resultado = executar_gradiente(f_explosiva, 2.0, lr, 50)
        
        if resultado.explodiu():
            print(f"LR = {lr:.2f} | DIVERGENCIA | {resultado.msg}")
        elif resultado.obteve_erro():
            print(f"LR = {lr:.2f} | ERRO | {resultado.msg}")
        else:
            print(f"LR = {lr:.2f} | OK | x_final: {resultado.x[-1]:.4f}")


def exemplo_9_mínimo_local():
    print("\n" + "="*60)
    print("EXEMPLO 9: Minimos Locais vs Globais")
    print("="*60)
    
    f = lambda x: np.sin(x)
    
    print("Funcao: f(x) = sin(x)")
    print("Minimos locais em: x = -pi/2, -5pi/2, etc\n")
    
    pontos_iniciais = [-1.0, 0.0, 1.5, 3.0, 5.0]
    
    print("Ponto Inicial | Convergencia x | Status")
    print("-" * 45)
    
    for x_ini in pontos_iniciais:
        resultado = executar_gradiente(f, x_ini, 0.1, 100)
        print(f"    {x_ini:5.1f}     |   {resultado.x[-1]:8.4f}    | {resultado.status:10}")


def exemplo_10_reproducibilidade():
    print("\n" + "="*60)
    print("EXEMPLO 10: Reproducibilidade")
    print("="*60)
    
    f = lambda x: x**2 - 4*x + 3
    
    print("Executando o mesmo gradiente 3 vezes...\n")
    
    resultados = [
        executar_gradiente(f, 0.0, 0.1, 50, 0.3),
        executar_gradiente(f, 0.0, 0.1, 50, 0.3),
        executar_gradiente(f, 0.0, 0.1, 50, 0.3),
    ]
    
    iguais = (
        resultados[0].x == resultados[1].x and
        resultados[1].x == resultados[2].x
    )
    
    print(f"Todos os resultados igual? {iguais}")
    print(f"Posicao final (execucao 1): {resultados[0].x[-1]:.6f}")
    print(f"Posicao final (execucao 2): {resultados[1].x[-1]:.6f}")
    print(f"Posicao final (execucao 3): {resultados[2].x[-1]:.6f}")


if __name__ == "__main__":
    exemplo_1_quadratica_simples()
    exemplo_2_polinomio_grau_4()
    exemplo_3_learning_rate_impact()
    exemplo_4_momentum_comparacao()
    exemplo_5_validacao()
    exemplo_6_multiplas_funcoes()
    exemplo_7_analisando_convergencia()
    exemplo_8_divergencia()
    exemplo_9_mínimo_local()
    exemplo_10_reproducibilidade()
    
    print("\n" + "="*60)
    print("TODOS OS EXEMPLOS CONCLUÍDOS!")
    print("="*60)
