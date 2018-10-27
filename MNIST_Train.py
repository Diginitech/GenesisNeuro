import numpy
import scipy.special
import matplotlib.pyplot


class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inode = inputnodes
        self.hnode = hiddennodes
        self.onode = outputnodes
        #随机生成权重
        # W(hidden layer,output layer)
        self.wih = numpy.random.normal(0.0,pow(self.hnode,-0.5),(self.hnode,self.inode))
        # W(input layer,hidden layer)
        self.who = numpy.random.normal(0.0,pow(self.onode,-0.5),(self.onode,self.hnode))
        #学习率
        self.lr=learningrate
        #sigmoid函数定义
        self.activation_function = lambda x:scipy.special.expit(x)

        pass
    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T

        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        #误差计算
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T,output_errors)
        #权重更新
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),numpy.transpose(inputs))

        pass
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        #隐藏层输入输出
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        #输出层输入输出
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

input_nodes = 784
hidden_nodes = 200
output_nodes = 10

learning_rate = 0.1

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

training_data_file = open("mnist_dataset/mnist_train.csv","r")
training_data_list = training_data_file.readlines()
training_data_file.close()
#世代数目
epochs = 5
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes)+0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)
        pass

test_data_file = open("mnist_dataset/mnist_test.csv","r")
test_data_list = test_data_file.readlines()
test_data_file.close()
#计分板，代码自行体会
scoreboard = []

for record in  test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    if (label == correct_label):
        scoreboard.append(1)
    else:
        scoreboard.append(0)
        pass
    pass
#平均值
scoreboard_array = numpy.asfarray(scoreboard)
print("performance=", scoreboard_array.sum()/scoreboard_array.size)