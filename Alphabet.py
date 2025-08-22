import numpy as np
def sigmoid(x):
    e=np.exp(-1*x)
    return 1/(1+e)
def deriv_sigmoid(x):
    return (np.exp(x)/np.square(1+np.exp(x)))
def deriv_tanh(x):
    return 1 - np.tanh(x)**2
def generate_weight(x,y):
    """
        x:number of nodes in parent layer
        y:number of nodes in child layer
    """
    limit=np.sqrt(6/(x+y))
    return np.random.uniform(-limit,limit,(x,y))
def forward_pass(x,w1,w2):
    """
        To forward pass we need to have <x,w1> as an input of the hidden layer then we apply activation function which is sigmoid here
        then the output of the hidden layer we make as an input of the output layer with <outputh,w2> then apply sigmoid and we have our output
    """
    z1=x.dot(w1) # input of first layer
    a1=sigmoid(z1) #output of first layer
    z2=a1.dot(w2) # input of output layer
    a2=sigmoid(z2)  # output of output layer
    return a2
def loss(target,prediction,eps=1e-15):
    """
        We are going to apply Binary Cross Entropy : −(ylog(y^​)+(1−y)log(1−y^​))
    """
    prediction = np.clip(prediction, eps, 1 - eps)
    return -np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))
def back_prop(x,y,lr,w1,w2):
    """
        Implementing backpropagation 
        d1 : error at  hidder 
    """
    z1=x.dot(w1) 
    a1=sigmoid(z1)
    z2=a1.dot(w2)
    a2=sigmoid(z2)
    d2=np.multiply((a2-y),deriv_sigmoid(z2))
    d1=np.multiply(d2.dot(w2.T),deriv_tanh(z1))
    w1_adj = np.outer(x, d1)
    w2_adj = np.outer(a1, d2)
    w1-=lr*w1_adj
    w2-=lr*w2_adj
    return w1,w2
def classify(y):
    max=0
    k=0
    for i in range(len(y)):
        if max<y[i]:
            max=y[i]
            k=i
    ch=chr(65+k)
    print(f'The letter is {ch}')
from dict_alphabet import letters
x = np.array([letters[ch] for ch in letters.keys()]).reshape(-1, 30)
y = np.eye(26, dtype=int)
w1=generate_weight(30,28)
w2=generate_weight(28,26)
for epoch in range(1000):
    running_loss=0
    for i in range(len(x)):
        input=x[i]
        target=y[i]
        pred=forward_pass(input,w1,w2)
        running_loss+=loss(target,pred)
        w1,w2=back_prop(input,target,0.5,w1,w2)
    #print(f'Epoch {epoch}:{running_loss:.2g}')
for i in range(len(x)):
    input=x[i]
    pred=pred=forward_pass(input,w1,w2)
    classify(pred)

