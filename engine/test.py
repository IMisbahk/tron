from node import Node
from noderoot import NodeRoot

a = Node(internalState=0.5, label="input.left_eye")
b = Node(internalState=0.0, label="logic.parse")

def hebbian_plasticity(w, x): return w + 0.01 * x

root = NodeRoot(a, b, weight=1.2, logicType="symbolic", symbolicTag="vision→parse")
root.applyPlasticity(hebbian_plasticity)

pulse = a.computePulseValue()
root.propagate(pulse)

print(b)
