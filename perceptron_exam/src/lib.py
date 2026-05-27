import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def layer_sizes(X,Y):
    """
    Argumentos:
    X -- conjunto de datos de entrada con dimensión (tamaño de entrada, número de ejemplos)
    Y -- etiquetas (valores objetivo) con dimensión (tamaño de salida, número de ejemplos)
    
    Retorna:
    n_x -- tamaño de la capa de entrada
    n_y -- tamaño de la capa de salida
    """
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    return (n_x, n_y)

def initialize_parameters(n_x, n_y):
    """
    Inicializa los parámetros del modelo (pesos y sesgos).

    Argumentos:
    n_x -- tamaño de la capa de entrada (número de características)
    n_y -- tamaño de la capa de salida

    Retorna:
    parameters -- diccionario de Python que contiene:
        W -- matriz de pesos de dimensión (n_y, n_x)
        b -- vector de sesgos de dimensión (n_y, 1)
    """
    
    # Inicialización aleatoria pequeña para evitar simetrías
    W = np.random.randn(n_y, n_x) * 0.01
    
    # Inicialización del sesgo en cero
    b = np.zeros((n_y, 1))
    
    parameters = {
        "W": W,
        "b": b
    }
    
    return parameters

def forward_propagation_regression(X, parameters):
    """
    Implementa la propagación hacia adelante (forward propagation).

    Argumentos:
    X -- datos de entrada de dimensión (n_x, m)
    parameters -- diccionario de Python que contiene los parámetros del modelo

    Retorna:
    Y_hat -- salida del modelo (predicciones)
    """
    
    W = parameters["W"]
    b = parameters["b"]
    
    # Propagación hacia adelante: cálculo de Z
    Z = np.matmul(W, X) + b
    
    # En este caso (regresión lineal), la salida es identidad
    Y_hat = Z

    return Y_hat

def forward_propagation_classification(X, parameters):
    """
    Implementa la propagación hacia adelante (forward propagation).

    Argumentos:
    X -- datos de entrada de dimensión (n_x, m)
    parameters -- diccionario de Python que contiene los parámetros del modelo

    Retorna:
    Y_hat -- salida del modelo (predicciones)
    """
    
    W = parameters["W"]
    b = parameters["b"]
    
    # Propagación hacia adelante: cálculo de Z
    Z = np.matmul(W, X) + b
    
    # En este caso (regresión lineal), la salida es identidad
    A = sigmoid(Z)

    return A

def compute_cost_mse(Y_hat, Y):
    """
    Calcula la función de costo utilizando la suma de cuadrados.

    Argumentos:
    Y_hat -- salida de la red neuronal de dimensión (n_y, número de ejemplos)
    Y -- valores reales (etiquetas) de dimensión (n_y, número de ejemplos)
    
    Retorna:
    cost -- valor de la función de costo, escalado por 1/(2*m)
    """
    
    # Número de ejemplos
    m = Y_hat.shape[1]

    # Cálculo de la función de costo
    cost = np.sum((Y_hat - Y)**2) / (2 * m)
    
    return cost

def compute_cost_bce(A, Y):
    """
    Calcula la función de costo utilizando la entropía cruzada binaria.

    Argumentos:
    Y_hat -- salida de la red neuronal de dimensión (n_y, número de ejemplos)
    Y -- valores reales (etiquetas) de dimensión (n_y, número de ejemplos)
    
    Retorna:
    cost -- valor de la función de costo, escalado por 1/(2*m)
    """
    
    # Número de ejemplos
    m = Y.shape[1]

    # Cálculo de la función de costo

    logprobs = (
        - np.multiply(np.log(A), Y)
        - np.multiply(np.log(1 - A), 1 - Y)
    )

    cost = 1 / m * np.sum(logprobs)
    
    return cost

def backward_propagation_regression(Y_hat, X, Y):
    """
    Implementa la propagación hacia atrás (backpropagation), calculando los gradientes.

    Argumentos:
    Y_hat -- salida de la red neuronal de dimensión (n_y, número de ejemplos)
    X -- datos de entrada de dimensión (n_x, número de ejemplos)
    Y -- valores reales (etiquetas) de dimensión (n_y, número de ejemplos)
    
    Retorna:
    grads -- diccionario de Python que contiene los gradientes respecto a los parámetros
    """
    
    m = X.shape[1]
    
    # Propagación hacia atrás: cálculo de derivadas parciales
    dZ = Y_hat - Y
    dW = (1 / m) * np.dot(dZ, X.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    
    grads = {
        "dW": dW,
        "db": db
    }
    
    return grads

def backward_propagation_classification(A, X, Y):
    """
    Implementa la propagación hacia atrás (backpropagation), calculando los gradientes.

    Argumentos:
    Y_hat -- salida de la red neuronal de dimensión (n_y, número de ejemplos)
    X -- datos de entrada de dimensión (n_x, número de ejemplos)
    Y -- valores reales (etiquetas) de dimensión (n_y, número de ejemplos)
    
    Retorna:
    grads -- diccionario de Python que contiene los gradientes respecto a los parámetros
    """
    
    m = X.shape[1]
    
    # Error del modelo
    dZ = A - Y

    # Gradiente respecto a los pesos
    dW = 1 / m * np.dot(dZ, X.T)

    # Gradiente respecto al sesgo
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    
    grads = {
        "dW": dW,
        "db": db
    }
    
    return grads

def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Actualiza los parámetros utilizando la regla de descenso del gradiente.

    Argumentos:
    parameters -- diccionario de Python que contiene los parámetros del modelo
    grads -- diccionario de Python que contiene los gradientes
    learning_rate -- tasa de aprendizaje (learning rate)

    Retorna:
    parameters -- diccionario con los parámetros actualizados
    """
    
    # Extraer parámetros
    W = parameters["W"]
    b = parameters["b"]
    
    # Extraer gradientes
    dW = grads["dW"]
    db = grads["db"]
    
    # Regla de actualización (gradient descent)
    W = W - learning_rate * dW
    b = b - learning_rate * db
    
    parameters = {
        "W": W,
        "b": b
    }
    
    return parameters

def nn_model_regression(X, Y, num_iterations=10, learning_rate=1.2, print_cost=False):
    """
    Construye y entrena un modelo de red neuronal simple.

    Argumentos:
    X -- conjunto de datos de dimensión (n_x, número de ejemplos)
    Y -- etiquetas o valores reales de dimensión (n_y, número de ejemplos)
    num_iterations -- número de iteraciones del ciclo de entrenamiento
    learning_rate -- tasa de aprendizaje para el descenso del gradiente
    print_cost -- si es True, imprime el costo en cada iteración

    Retorna:
    parameters -- parámetros aprendidos por el modelo. Posteriormente pueden usarse para hacer predicciones.
    """
    
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[1]
    
    parameters = initialize_parameters(n_x, n_y)
    
    # Ciclo de entrenamiento
    for i in range(0, num_iterations):
         
        # Propagación hacia adelante.
        # Entradas: X, parameters. Salida: Y_hat.
        Y_hat = forward_propagation_regression(X, parameters)
        
        # Función de costo.
        # Entradas: Y_hat, Y. Salida: cost.
        cost = compute_cost_mse(Y_hat, Y)
        
        # Propagación hacia atrás.
        # Entradas: Y_hat, X, Y. Salida: grads.
        grads = backward_propagation_regression(Y_hat, X, Y)
    
        # Actualización de parámetros mediante descenso del gradiente.
        # Entradas: parameters, grads, learning_rate. Salida: parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Imprimir el costo en cada iteración
        if print_cost:
            print("Costo después de la iteración %i: %f" % (i, cost))

    return parameters

def nn_model_classification(X, Y, num_iterations=10, learning_rate=1.2, print_cost=False):
    """
    Argumentos:
    X -- conjunto de datos con forma
         (n_x, número de ejemplos)

    Y -- etiquetas con forma
         (n_y, número de ejemplos)

    num_iterations -- número de iteraciones del entrenamiento

    learning_rate -- tasa de aprendizaje utilizada
                     en descenso del gradiente

    print_cost -- si es True, imprime el costo
                  en cada iteración

    Retorna:
    parameters -- parámetros aprendidos por el modelo.
                  Posteriormente pueden utilizarse
                  para hacer predicciones.
    """

    # Obtener tamaños de entrada y salida
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[1]

    # Inicializar parámetros
    parameters = initialize_parameters(n_x, n_y)

    # Loop de entrenamiento

    for i in range(0, num_iterations):

        # Forward propagation

        A = forward_propagation_classification(X, parameters)

        # Cálculo del costo

        cost = compute_cost_bce(A, Y)

        # Back propagation

        grads = backward_propagation_classification(A, X, Y)

        # Actualización de parámetros

        parameters = update_parameters(
            parameters,
            grads,
            learning_rate
        )

        # Imprimir costo en cada iteración
        if print_cost:
            print("Costo después de la iteración %i: %f" % (i, cost))

    return parameters