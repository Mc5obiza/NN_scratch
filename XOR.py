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
def loss(target,prediction):
    """
        We are going to apply Mean Squared Error=1/n*Σ(^y-y)²
    """
    squared_error=np.square(target-prediction)
    mean_squared_error=np.sum(squared_error)/len(prediction)
    return mean_squared_error
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
    return 1 if y>0.5 else 0


x = np.array([[0,0], [0,1], [1,0], [1,1]])   
y = np.array([[0], [1], [1], [0]])           
w1=generate_weight(2,4)
w2=generate_weight(4,1)
print(w1)
print(w2)
for epoch in range(20000):
    running_loss=0
    for i in range (len(x)):
        input=x[i]
        pred=forward_pass(input,w1,w2)
        curr_loss=loss(pred,y[i])
        running_loss+=curr_loss
        w1,w2=back_prop(input,y[i],0.5,w1,w2)
        pred=classify(pred)
        
    epoch_loss=running_loss/len(x)
    if(epoch%1000==0):
        print(f'Loss for Epoch {epoch}: {epoch_loss:.2g}')

    
for i in range(len(x)):
    pred = forward_pass(x[i], w1, w2)
    print(f"Input: {x[i]}, Pred: {classify(pred):.0f}, Raw: {pred[0]:.3f}, True: {y[i][0]}")

        
