# Gradient Descent Visualizer

> Um laboratório interativo para explorar visualmente como algoritmos de otimização "aprendem" encontrando mínimos de funções.

[![Acesse ao Vivo](https://img.shields.io/badge/Acessar_App_Online-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://gradient-descent-lab.streamlit.app)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red)
![SymPy](https://img.shields.io/badge/SymPy-Math_Parsing-lightgrey)
![Plotly](https://img.shields.io/badge/Plotly-Data_Viz-indigo)

## Sobre o Projeto

Este projeto é uma ferramenta visual construída para desmistificar o algoritmo **Gradient Descent** (Gradiente Descendente) e suas variantes mais modernas. Ele é a base matemática por trás do treinamento de Redes Neurais e modelos de Machine Learning.

Em vez de usar equações prontas ou diferenciação automática de frameworks pesados, o projeto implementa um **Motor de Derivada Numérica** acoplado a um parser matemático seguro. Isso permite que você digite *qualquer* função e veja, em tempo real, como diferentes algoritmos navegam pela topografia para encontrar o ponto de menor custo (mínimo global ou local).

## Funcionalidades

* **Múltiplos Otimizadores:** Compare o comportamento de algoritmos clássicos e modernos do Deep Learning: `SGD`, `AdaGrad`, `RMSProp` e `Adam`.
* **Modo Comparação (A/B Test):** Coloque dois modelos rodando simultaneamente na mesma função para ver qual converge mais rápido ou qual escapa de mínimos locais.
* **Matemática Descomplicada & Segura:** Graças à integração com `SymPy`, você pode digitar funções matemáticas naturalmente (ex: `x**4 - 2*x**2 + 1` ou `x * sin(x)`) sem precisar de sintaxe de programação.
* **Ajuste Fino de Hiperparâmetros:** Controle em tempo real:
    * `Learning Rate` (Taxa de aprendizado)
    * `Momentum / Beta1` (Inércia do otimizador)
    * `Iterações` (Com sistema inteligente de *Early Stopping* para poupar processamento)
    * `Start Point` (Ponto de partida)
* **Diagnóstico Rico:** Acompanhe a inclinação final, posição e status de convergência (Sucesso, Explosão de Gradiente, etc.).

## Como Funciona (Under the Hood)

O núcleo do projeto calcula a derivada "na raça" utilizando a definição fundamental de limite por diferenças centrais:

$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$$

Isso permite sondar a inclinação do terreno dinamicamente. O motor gráfico usa vetorização via `NumPy` para renderizar o terreno instantaneamente, garantindo uma experiência interativa e sem engasgos.

## Como Rodar Localmente

Se quiser rodar na sua própria máquina para explorar o código:

1. **Clone o repositório:**
    ```bash
    git clone [https://github.com/SEU-USUARIO/gradient-descent-lab.git](https://github.com/SEU-USUARIO/gradient-descent-lab.git)
    cd gradient-descent-lab
    ```

2. **Crie um ambiente virtual (Recomendado):**
    ```bash
    python -m venv venv
    # Windows:
    source venv/Scripts/activate
    # Linux/Mac:
    source venv/bin/activate
    ```

3. **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Execute o dashboard:**
    ```bash
    streamlit run app.py
    ```

## Desafios para Testar

Acesse o [App Online](https://gradient-descent-lab.streamlit.app) e tente estas configurações:

1. **O "W" (Padrão):** `x**4 - 2*x**2 + 1`
   * *Desafio:* Ligue o Modo Comparação. Use o `SGD` de um lado e o `Adam` do outro. Mude o ponto inicial e veja qual deles consegue pular o "morro" central com a ajuda do Momentum!
2. **O Caos (Senoide):** `x * sin(x)`
   * *Desafio:* Tente encontrar o mínimo global. Veja como o algoritmo fica preso em vales locais (mínimos locais) dependendo de onde começa.
3. **Explosão de Gradiente:** `x**2`
   * *Desafio:* Aumente o Learning Rate para > 1.0. Veja o algoritmo oscilar loucamente pelas paredes da parábola ou divergir para o infinito (Status: Explosão).

---
*Desenvolvido por Danilo Ribeiro*
