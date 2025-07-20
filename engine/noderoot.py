# engine/nodeRoot.py

import uuid
import time
from types import FunctionType

class NodeRoot:
    """
    TRON NodeRoot: The cognitive artery of the TRON field.

    A NodeRoot is a directional, weighted, optionally delayed and symbolic
    link between two Node instances in the TRON network.

    This class enables dynamic signal propagation, learning drift,
    symbolic logic propagation, and network introspection.

    Roots are first-class objects: traceable, self-adjusting, auditable,
    and represent the *reasoning pathways* that evolve with experience.
    """

    def __init__(
        self,
        source,
        target,
        weight=1.0,
        delay=0,
        label=None,
        logicType="excitatory",
        propagationRule=None,
        plasticityRule=None,
        enabled=True,
        origin="system",
        owner="system",
        symbolicTag=None
    ):
        """
        Initialize a root from source to target.

        Args:
            source (Node): Origin node.
            target (Node): Destination node.
            weight (float): Influence multiplier.
            delay (int): Delay steps (not active in v0).
            label (str): Optional name (e.g. "vision → logic").
            logicType (str): excitatory | inhibitory | symbolic | etc.
            propagationRule (fn): Override for pulse behavior.
            plasticityRule (fn): Optional drift over time.
            symbolicTag (str): Reason this connection exists.
        """
        self.rootID = uuid.uuid4()
        self.source = source
        self.target = target

        self.weight = weight
        self.delay = delay
        self.label = label
        self.logicType = logicType
        self.enabled = enabled

        self.origin = origin
        self.owner = owner
        self.symbolicTag = symbolicTag

        self.propagationRule = propagationRule
        self.plasticityRule = plasticityRule

        self.activityLog = []  # Timestamped pulse trace
        self.metadata = {}

        # Automatically register to node connectivity
        if hasattr(source, 'outgoingRoots'):
            source.outgoingRoots.append(self)
        if hasattr(target, 'incomingRoots'):
            target.incomingRoots.append(self)

    def propagate(self, signalStrength):
        """
        Send a signal from source to target node.
        """
        if not self.enabled:
            return

        adjusted = self._computePulse(signalStrength)
        self._logPulse(signalStrength, adjusted)

        if self.delay == 0:
            self.target.receivePulse(adjusted)
        else:
            # Future: schedule-delayed pulse with Scheduler
            pass

        if self.plasticityRule:
            self.weight = self.plasticityRule(self.weight, signalStrength)

    def _computePulse(self, signal):
        if self.propagationRule and isinstance(self.propagationRule, FunctionType):
            return self.propagationRule(signal, self.weight)
        return signal * self.weight

    def _logPulse(self, original, adjusted):
        self.activityLog.append({
            "timestamp": time.time(),
            "source": str(self.source.nodeID),
            "target": str(self.target.nodeID),
            "original": original,
            "adjusted": adjusted,
            "weight": self.weight,
            "symbolic": self.symbolicTag,
            "label": self.label
        })

    def updateWeight(self, newWeight):
        self.weight = newWeight

    def applyPlasticity(self, rule):
        """
        Attach a plasticity function (e.g. Hebbian learning).
        """
        self.plasticityRule = rule

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def observe(self, includeLog=False):
        obs = {
            "rootID": str(self.rootID),
            "source": str(self.source.nodeID),
            "target": str(self.target.nodeID),
            "weight": self.weight,
            "enabled": self.enabled,
            "delay": self.delay,
            "logicType": self.logicType,
            "label": self.label,
            "symbolicTag": self.symbolicTag,
            "origin": self.origin,
            "owner": self.owner
        }
        if includeLog:
            obs["activityLog"] = self.activityLog
        return obs

    def __repr__(self):
        tag = f"[{self.symbolicTag}]" if self.symbolicTag else ""
        return (
            f"<Root {str(self.rootID)[:8]} | {self.logicType.upper()} "
            f"{tag} {self.label or ''} | W: {self.weight} | "
            f"{str(self.source.nodeID)[:6]} → {str(self.target.nodeID)[:6]}>"
        )
