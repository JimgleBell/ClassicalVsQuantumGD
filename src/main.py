from neural_network.value import Value
from neural_network.visualize import draw
from neural_network.network import NN

# from neural_network.

x = [2, 3, -1]
n = NN(
    3, [3, 3, 3, 1]
)  # rede com 3 inputs, e camadas de 3 neurônios, 3 neurônios, 3 neurônios e 1 último neurônio
n(x)

draw(n(x))
