import mxnet as mx
from mxnet import autograd

w = mx.gluon.Parameter('w', shape=(1,), init=mx.init.One(), grad_req='write')
w.initialize()
trainer = mx.gluon.Trainer({'w': w}, 'sgd', optimizer_params={'learning_rate': 1.0})

while True:
    #w_data = w.data()

    dummy = mx.nd.ones(shape=(1,))
    dummy.attach_grad()
    with autograd.record():
        dw = autograd.grad(dummy*w.data(), dummy, retain_graph=True)[0]
        #dw = dummy*w.data()
        #print(dw)
        dw.backward()

    print(w.data())
    trainer.step(1)
    print(w.data())



