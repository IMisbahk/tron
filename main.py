from engine.node import Node
from engine.registry import TronRegistryDashboard
import numpy as np

node = Node(internalState=0.75, label="token.hello", nodalType="embedding", owner="llm-agent")
node = Node(internalState=0.58, label="token.hi", nodalType="embedding", owner="llm-agent")
a = np.random.randint
node.mutate(lambda x: x + 3)

dash = TronRegistryDashboard()

dash.showAllNodes()
