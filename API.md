# Documentação da API - Gradient Descent Visualizer

## Índice
- [Módulo `tools`](#módulo-tools)
- [Módulo `modelos`](#módulo-modelos)
- [Módulo `validators`](#módulo-validators)
- [Exemplos de Uso](#exemplos-de-uso)

---

## Módulo `tools`

### `derivada_numerica(funcao, x, h=0.0001) -> float`

Calcula a derivada de uma função em um ponto usando aproximação numérica.

**Parâmetros:**
- `funcao` (Callable): Função f(x) para calcular derivada
- `x` (float): Ponto onde calcular a derivada
- `h` (float): Tamanho do deslocamento (padrão: 0.0001)

**Retorna:**
- `float`: Valor aproximado da derivada em x

**Raises:**
- `ValueError`: Se a função retornar NaN ou infinito

**Exemplo:**
```python
from tools import derivada_numerica

f = lambda x: x**2
derivada_em_3 = derivada_numerica(f, 3.0)
print(derivada_em_3)
```

**Saída:** ~6.0 (porque d/dx(x²) = 2x = 2*3 = 6)

---

### `executar_gradiente(funcao, x_inicial, learning_rate, iteracoes, momentum=0.0) -> ResultadoGradiente`

Executa o algoritmo de gradiente descendente com suporte a momentum.

**Parâmetros:**
- `funcao` (Callable): Função f(x) a ser minimizada
- `x_inicial` (float): Ponto de partida
- `learning_rate` (float): Taxa de aprendizado (0.001 a 1.0)
- `iteracoes` (int): Número máximo de iterações (1 a 200)
- `momentum` (float): Coeficiente de inércia (0.0 a 0.99), padrão: 0.0

**Retorna:**
- `ResultadoGradiente`: Objeto contendo histórico, status e diagnóstico

**Raises:**
- Nenhuma (sempre retorna ResultadoGradiente, mesmo em erros)

**Status Possíveis:**
- `"otimo"`: Convergiu para mínimo (inclinação < 0.01)
- `"proximo"`: Próximo ao mínimo
- `"descendo"`: Ainda descendo (velocidade > 0.01)
- `"explosao"`: Divergiu (|x| > 1e10)
- `"erro"`: Erro durante execução

**Exemplo:**
```python
from tools import executar_gradiente

f = lambda x: (x - 2)**2
resultado = executar_gradiente(f, 0.0, 0.1, 100, momentum=0.5)

print(f"Status: {resultado.status}")
print(f"Posição final: {resultado.x[-1]:.4f}")
print(f"Valor f(x): {resultado.y[-1]:.4f}")
print(f"Iterações: {len(resultado.x) - 1}")
```

---

## Módulo `modelos`

### `ResultadoGradiente`

Dataclass que encapsula o resultado de uma execução do gradiente descendente.

**Atributos:**
```python
@dataclass
class ResultadoGradiente:
    x: List[float]          # Histórico de valores x
    y: List[float]          # Histórico de valores f(x)
    status: str             # Status da otimização
    msg: str                # Mensagem descritiva
    incl_final: float       # Inclinação no ponto final
```

**Métodos:**

#### `tem_dados() -> bool`
Verifica se há dados válidos para plotagem.
```python
if resultado.tem_dados():
    print(f"Pontos visitados: {len(resultado.x)}")
```

#### `convergiu() -> bool`
Verifica se convergiu para mínimo.
```python
if resultado.convergiu():
    print("Encontrou o mínimo!")
```

#### `explodiu() -> bool`
Verifica se divergiu.
```python
if resultado.explodiu():
    print("Algoritmo divergiu!")
```

#### `obteve_erro() -> bool`
Verifica se ocorreu erro.
```python
if resultado.obteve_erro():
    print(resultado.msg)
```

**Exemplo Completo:**
```python
from tools import executar_gradiente

f = lambda x: x**4 - 3*x**2 + 2
resultado = executar_gradiente(f, 1.0, 0.05, 200)

if resultado.tem_dados():
    print(f"Iterações: {len(resultado.x) - 1}")
    print(f"Status: {resultado.status}")
    print(f"Mensagem: {resultado.msg}")
    print(f"Posição: {resultado.x[-1]:.6f}")
    print(f"Valor: {resultado.y[-1]:.6f}")
    print(f"Inclinação: {resultado.incl_final:.6f}")
```

### `ParametrosOtimizacao`

Dataclass para validação de parâmetros.

```python
@dataclass
class ParametrosOtimizacao:
    learning_rate: float
    momentum: float
    iteracoes: int
    x_inicial: float
```

Valida automaticamente na inicialização:
```python
from modelos import ParametrosOtimizacao

try:
    params = ParametrosOtimizacao(
        learning_rate=0.1,
        momentum=0.5,
        iteracoes=100,
        x_inicial=0.0
    )
except ValueError as e:
    print(f"Erro: {e}")
```

---

## Módulo `validators`

### `ValidadorParametros`

Classe com métodos estáticos para validação.

#### `validar_funcao(funcao_texto: str) -> Tuple[bool, str, Callable]`

```python
from validators import ValidadorParametros

valido, msg, func = ValidadorParametros.validar_funcao("x**2 - 4*x + 1")

if valido:
    resultado = func(2.0)
    print(f"f(2) = {resultado}")
else:
    print(f"Erro: {msg}")
```

#### `validar_learning_rate(lr: float) -> Tuple[bool, str]`

```python
valido, msg = ValidadorParametros.validar_learning_rate(0.1)
if not valido:
    print(f"Erro: {msg}")
```

#### `validar_momentum(momentum: float) -> Tuple[bool, str]`

```python
valido, msg = ValidadorParametros.validar_momentum(0.9)
if not valido:
    print(f"Erro: {msg}")
```

#### `validar_iteracoes(iteracoes: int) -> Tuple[bool, str]`

```python
valido, msg = ValidadorParametros.validar_iteracoes(100)
if not valido:
    print(f"Erro: {msg}")
```

#### `validar_x_inicial(x_ini: float) -> Tuple[bool, str]`

```python
valido, msg = ValidadorParametros.validar_x_inicial(2.5)
if not valido:
    print(f"Erro: {msg}")
```

#### `validar_todos_parametros(...) -> Tuple[bool, list]`

Valida todos os parâmetros de uma vez.

```python
from validators import ValidadorParametros

valido, erros = ValidadorParametros.validar_todos_parametros(
    funcao_texto="x**2",
    lr=0.1,
    momentum=0.5,
    iteracoes=100,
    x_ini=0.0
)

if valido:
    print("Todos os parâmetros estão válidos!")
else:
    for erro in erros:
        print(f"- {erro}")
```

---

## Exemplos de Uso

### Exemplo 1: Minimizar Função Simples

```python
from tools import executar_gradiente

f = lambda x: (x - 3)**2 + 2

resultado = executar_gradiente(
    funcao=f,
    x_inicial=0.0,
    learning_rate=0.1,
    iteracoes=100,
    momentum=0.0
)

print(f"Mínimo encontrado em x ≈ {resultado.x[-1]:.4f}")
print(f"Valor da função: {resultado.y[-1]:.4f}")
print(f"Status: {resultado.status}")
```

### Exemplo 2: Comparar Momentum vs Sem Momentum

```python
from tools import executar_gradiente

f = lambda x: x**4 - 2*x**2 + 1

r_sem = executar_gradiente(f, 0.5, 0.05, 200, momentum=0.0)
r_com = executar_gradiente(f, 0.5, 0.05, 200, momentum=0.8)

print(f"Sem momentum: {len(r_sem.x)} iterações")
print(f"Com momentum: {len(r_com.x)} iterações")
```

### Exemplo 3: Validar e Executar

```python
from tools import executar_gradiente
from validators import ValidadorParametros

funcao_texto = "x**2 - 6*x + 5"
lr = 0.1
momentum = 0.5
iteracoes = 100
x_ini = 0.0

valido, erros = ValidadorParametros.validar_todos_parametros(
    funcao_texto, lr, momentum, iteracoes, x_ini
)

if valido:
    valido_f, _, func = ValidadorParametros.validar_funcao(funcao_texto)
    if valido_f:
        resultado = executar_gradiente(func, x_ini, lr, iteracoes, momentum)
        print(f"Convergiu: {resultado.convergiu()}")
else:
    for erro in erros:
        print(f"Erro: {erro}")
```

### Exemplo 4: Detectar Erro de Convergência

```python
from tools import executar_gradiente

f_explosiva = lambda x: x**10

resultado = executar_gradiente(f_explosiva, 2.0, 0.5, 50)

if resultado.explodiu():
    print("⚠️ Função explodiu!")
    print(resultado.msg)
elif resultado.obteve_erro():
    print("❌ Erro na execução!")
    print(resultado.msg)
elif resultado.convergiu():
    print("✅ Convergiu!")
else:
    print("⏳ Ainda descendo...")
```

---

## Configuração

Ver [config.py](config.py) para ajustar constantes globais:
- `DERIVADA_H`: Tamanho do passo para derivada numérica
- `TOLERANCIA_INCLINACAO`: Limite para considerar convergência
- `LIMITE_X_EXPLOSAO`: Limite máximo de divergência
