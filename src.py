import random

class Value:
    def __init__(self, value, children=()):
        self.data = value
        self.grad = 0.
        self._children = set(children)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        
        def _backward():            
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'only supporting int/float powers for now'
        out = Value(self.data ** other, (self,))

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __neg__(self):
        return self * Value(-1)

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)
    
    def __truediv__(self, other):
        return self * other ** -1
    
    def __rtruediv__(self, other):
        return other * self ** -1
    
    def relu(self):
        out = Value(max(0, self.data), (self,))
        
        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        dfs_stack = [(self, False)]

        while dfs_stack:
            v, processed = dfs_stack.pop()
            if processed:
                topo.append(v)
                continue
            if v not in visited:
                visited.add(v)
                dfs_stack.append((v, True))
                for child in v._children:
                    dfs_stack.append((child, False))
    
        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


def eval(x):
    return net([x / 100.]).data * 100


def compute_accuracy():
    cnt = 0
    for x in range(1, 101):
        pred = round(eval(x))
        if pred == x:
            cnt += 1
    return cnt


def solve():
    x = int(input())
    print(round(eval(x)))

if __name__ == '__main__':
    net = MLP(1, [4, 1])
    PARAMS = [-0.6090875544164814, -0.11634677302553537, 0.2700409771296602, -0.5403454710056752, -0.551194538405536, 1.0759167385989803, 0.6667990764607007, 0.14042287292399525, -0.4774162510674046, -0.40833519934153273, -1.2786064263693935, 0.44230560738792124, 1.3144184067367273]
    for p, v in zip(net.parameters(), PARAMS):
        p.data = v

    solve()