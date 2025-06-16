import uuid
import time
import numpy as np
from types import FunctionType

from engine.registry import registry  # ⬅️ Added

class Node:
    """
    TRON Node: Dynamic, symbolic, traceable unit of cognition.
    """

    def __init__(
        self,
        internalState=None,
        activationThreshold=1.0,
        activationFn=None,
        pulseMode='accumulate',
        origin="architect",
        reason="init",
        owner="system",
        nodalType="general",
        label=None
    ):
        self.timestamp = time.time()
        self.stateHistory = []

        if label:
            # Pull metadata from registry or create new
            meta = registry.register(label, state=internalState, origin=origin, reason=reason, owner=owner, nodalType=nodalType)
            self.nodeID = meta['nodeID']
            self.origin = meta['origin']
            self.reason = meta['reason']
            self.owner = meta['owner']
            self.nodalType = meta['nodalType']
            self.tags = set(meta.get('tags', []))
            self.label = label
        else:
            self.nodeID = uuid.uuid4()
            self.origin = origin
            self.reason = reason
            self.owner = owner
            self.nodalType = nodalType
            self.tags = set()
            self.label = None

        self.internalState = internalState
        self.nodalActivity = 0.0
        self.activationThreshold = activationThreshold
        self.activationFn = activationFn or self.defaultActivation
        self.pulseMode = pulseMode

        self.incomingRoots = []
        self.outgoingRoots = []
        self.traceLog = []

    def defaultActivation(self, signal):
        return signal >= self.activationThreshold

    def receivePulse(self, signal):
        if self.pulseMode == "accumulate":
            self.nodalActivity += signal
        elif self.pulseMode == "overwrite":
            self.nodalActivity = signal
        elif isinstance(self.pulseMode, FunctionType):
            self.nodalActivity = self.pulseMode(self.nodalActivity, signal)

        self.traceLog.append({
            'time': time.time(),
            'event': 'pulse',
            'value': signal
        })

    def shouldFire(self):
        return self.activationFn(self.nodalActivity)

    def emitPulse(self):
        if self.shouldFire():
            pulseValue = self.computePulseValue()
            self.resetActivity()
            return pulseValue
        return None

    def computePulseValue(self):
        if isinstance(self.internalState, (int, float)):
            return float(self.internalState)
        elif isinstance(self.internalState, np.ndarray):
            return float(np.mean(self.internalState))
        elif isinstance(self.internalState, str):
            return 1.0
        return 1.0

    def resetActivity(self):
        self.nodalActivity = 0.0

    def mutate(self, func):
        self.internalState = func(self.internalState)
        self.stateHistory.append(self.internalState)
        if self.label:
            registry.updateState(self.label, self.internalState)

    def observe(self, includeTrace=False):
        obs = {
            'nodeID': str(self.nodeID),
            'timestamp': self.timestamp,
            'state': self.internalState,
            'activity': self.nodalActivity,
            'threshold': self.activationThreshold,
            'origin': self.origin,
            'reason': self.reason,
            'owner': self.owner,
            'type': self.nodalType,
            'tags': list(self.tags),
            'connections': {
                'in': len(self.incomingRoots),
                'out': len(self.outgoingRoots)
            }
        }
        if includeTrace:
            obs['traceLog'] = self.traceLog
        return obs

    def addTag(self, tag):
        self.tags.add(tag)

    def attachActivationFunction(self, fn):
        self.activationFn = fn

    def __repr__(self):
        labelInfo = f"{self.label}" if self.label else str(self.nodeID)[:8]
        return f"<Node {labelInfo} | {self.nodalType} | {type(self.internalState).__name__}>"
