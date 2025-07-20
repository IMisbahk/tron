# tron/engine/registry.py

import uuid
import json
import os
import time
from pathlib import Path
import rich
from rich.console import Console
from rich.table import Table

REGISTRY_PATH = Path(".tron_registry.json")
console = Console()

class TronRegistry:
    """
    Tracks all node registrations, symbolic metadata, and traceable lineage.
    Provides persistent UUID binding, human-readable keys, and node restoration.
    """

    def __init__(self, path=REGISTRY_PATH):
        self.path = path
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, "r") as f:
                self.registry = json.load(f)
        else:
            self.registry = {}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.registry, f, indent=2)

    def register(self, label, state=None, origin="user", reason="init", owner="unknown", nodalType="general"):
        """
        Register a node by symbolic label. Assigns deterministic UUID.
        """
        if label in self.registry:
            return self.registry[label]

        nodeID = str(uuid.uuid5(uuid.NAMESPACE_DNS, label))
        nodeData = {
            "label": label,
            "nodeID": nodeID,
            "state": state,
            "origin": origin,
            "reason": reason,
            "owner": owner,
            "nodalType": nodalType,
            "tags": [],
            "history": [state] if state is not None else [],
            "timestamp": time.time()
        }
        self.registry[label] = nodeData
        self._save()
        return nodeData

    def getByLabel(self, label):
        return self.registry.get(label)

    def getByID(self, nodeID):
        for meta in self.registry.values():
            if meta["nodeID"] == nodeID:
                return meta
        return None

    def updateState(self, label, newState):
        if label in self.registry:
            self.registry[label]["state"] = newState
            self.registry[label]["history"].append(newState)
            self._save()

    def allLabels(self):
        return list(self.registry.keys())

    def allNodes(self):
        return list(self.registry.values())


class TronRegistryDashboard:
    """
    CLI dashboard to inspect the current TRON node registry.
    """

    def __init__(self):
        self.registry = registry

    def showAllNodes(self):
        nodes = self.registry.allNodes()
        table = Table(title="Registered TRON Nodes")
        table.add_column("Label", style="cyan", no_wrap=True)
        table.add_column("Node ID", style="magenta")
        table.add_column("Owner", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("Current State", style="white")

        for node in nodes:
            state = str(node.get("state"))[:40]
            table.add_row(
                node["label"],
                node["nodeID"][:8],
                node["owner"],
                node["nodalType"],
                state
            )

        console.print(table)

    def showNode(self, label):
        node = self.registry.getByLabel(label)
        if not node:
            console.print(f"[red]No node found for label:[/] {label}")
            return
        console.print(f"\n[bold cyan]Node Details: {label}[/bold cyan]\n")
        console.print(json.dumps(node, indent=2))

    def searchByOwner(self, ownerName):
        nodes = [n for n in self.registry.allNodes() if n['owner'] == ownerName]
        if not nodes:
            console.print(f"[red]No nodes owned by:[/] {ownerName}")
            return
        console.print(f"[bold green]Nodes owned by: {ownerName}[/bold green]")
        for node in nodes:
            console.print(f"- {node['label']} ({node['nodeID'][:8]})")

    def export(self, path="tron_registry_export.json"):
        with open(path, "w") as f:
            json.dump(self.registry.registry, f, indent=2)
        console.print(f"[bold blue]Exported registry to:[/] {path}")


# Singleton
registry = TronRegistry()
