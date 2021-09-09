from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayeer, BiasUnit
from pybrain.structure import FullConnection
import pandas as pd
from sklearn.model_selection import train_test_split

csv_file = pd.read_csv("dadosTrabalhoRNA.csv")

i = pd.read_csv("dadosTrabalhoRNA.csv", usecols=[0])
o = pd.read_csv("dadosTrabalhoRNA.csv", usecols=[1])
x_train, x_test, y_train, y_test = train_test_split(i,o, test_size = 0.2, random_state=10)

x_test = np.array(x_test)
x_train = np.array(x_train)
y_test = np.array(y_test)
y_train = np.array(y_train)

rede = FeedForwardNetwork()
inputLayer = LinearLayer(1)
hiddenLayer1 = SigmoidLayer(5)
outputLayer = SigmoidLayer(1)

bias1 = BiasUnit()
bias2 = BiasUnit()

rede.addModule(inputLayer)
rede.addModule(hiddenLayer1)
rede.addModule(outputLayer)
rede.addModule(bias1)

connection1 = FullConnection(inputLayer, hiddenLayer1)
connection2 = FullConnection(hiddenLayer1, outputLayer)
bias1Connection = FullConnection(bias1, hiddenLayer1)
bias2Connection = FullConnection(bias2, outputLayer)

rede.sortModules()

print(rede)
print(connection1.params)
print(connection2.params)
print(bias1Connection)
print(bias2Connection)
