def derivada_numerica(funcao, x, h=0.0001):
    y_atual = funcao(x)

    y_frente = funcao(x + h)

    inclinacao = (y_frente - y_atual)/h

    return inclinacao