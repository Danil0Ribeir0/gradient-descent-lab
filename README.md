# Gradient Descent Visualizer

> Um laboratório interativo para explorar visualmente como algoritmos de otimização "aprendem" encontrando mínimos de funções.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red)
![Math](https://img.shields.io/badge/Math-Calculus-green)

## Sobre o Projeto

Este projeto é uma ferramenta visual construída para desmistificar o algoritmo **Gradient Descent** (Gradiente Descendente), a base matemática por trás do treinamento de Redes Neurais e Machine Learning moderno.

Em vez de usar equações prontas, o projeto implementa um **Motor de Derivada Numérica**, permitindo que o usuário inpute *qualquer* função matemática e veja, em tempo real, como o algoritmo navega pela topografia da função para encontrar o ponto de menor custo (mínimo global ou local).

## Funcionalidades

* **Matemática Genérica:** Aceita qualquer função $f(x)$ definida pelo usuário (ex: `x**4 - 2*x**2 + 1` ou `x * np.sin(x)`).
* **Ajuste de Hiperparâmetros:** Controle total sobre:
    * `Learning Rate` (Taxa de aprendizado): Veja o que acontece quando o passo é muito pequeno (lento) ou muito grande (divergência/explosão).
    * `Iterações`: Defina quanto tempo o algoritmo tem para treinar.
    * `Start Point`: Mude o ponto de partida para testar problemas de **Mínimos Locais**.
* **Visualização em Tempo Real:** Gráfico interativo que plota a trajetória passo-a-passo do otimizador.

## Como Funciona (O Código)

O núcleo do projeto não utiliza diferenciação automática pronta (como PyTorch/TensorFlow). A derivada é calculada "na raça" utilizando a definição fundamental de limite:

$$f'(x) \approx \frac{f(x+h) - f(x)}{h}$$

Isso permite que o algoritmo calcule a inclinação (gradiente) de funções complexas sem saber as regras analíticas de derivação, apenas sondando o terreno ao redor.

### Estrutura de Arquivos
* `app.py`: O dashboard interativo construído com **Streamlit**.
* `tools.py`: Módulo matemático contendo a lógica da derivada numérica.

## 🚀 Como Rodar Localmente

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/SEU-USUARIO/gradient-descent-lab.git](https://github.com/SEU-USUARIO/gradient-descent-lab.git)
    cd gradient-descent-lab
    ```

2.  **Crie um ambiente virtual (Recomendado):**
    ```bash
    python -m venv venv
    # Windows:
    source venv/Scripts/activate
    # Linux/Mac:
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute o dashboard:**
    ```bash
    streamlit run app.py
    ```

## 🎮 Exemplos para Testar

Tente estas funções no painel para ver comportamentos interessantes:

1.  **O "W" (Padrão):** `x**4 - 2*x**2 + 1`
    * *Desafio:* Tente cair nos dois buracos diferentes mudando apenas o ponto inicial.
2.  **O Caos (Senoide):** `x * np.sin(x)`
    * *Desafio:* Aumente o limite do gráfico e veja como o algoritmo fica preso em vales locais dependendo de onde começa.
3.  **A Parábola Simples:** `x**2`
    * *Desafio:* Aumente o Learning Rate para > 1.0 e veja o algoritmo oscilar ou explodir para o infinito.

---
*Desenvolvido por Danilo Ribeiro*