import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

class QuantumEntangler(nn.Module):
    """A PennyLane-based variational quantum circuit that:
    - angle-encodes a small-dim vector on num_qubits
    - applies a few layers of entangling rotations
    - measures expectation values to produce a small feature vector
    """
    def __init__(self, num_qubits: int = 6, depth: int = 2, out_dim: int = 8, device_str: str = 'default.qubit'):
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.out_dim = out_dim
        self.dev = qml.device(device_str, wires=num_qubits, shots=None)

        # variational parameters: rotations + entanglement angles
        self.theta = nn.Parameter(torch.randn(depth, num_qubits) * 0.01)
        self.phi   = nn.Parameter(torch.randn(depth, num_qubits) * 0.01)
        self.omega = nn.Parameter(torch.randn(depth, num_qubits) * 0.01)

        obs = [qml.PauliZ(i) for i in range(num_qubits)]
        while len(obs) < out_dim:
            a = len(obs) % num_qubits
            b = (len(obs) * 2) % num_qubits
            if a != b:
                obs.append(qml.PauliZ(a) @ qml.PauliZ(b))
            else:
                obs.append(qml.PauliX(a))
        self.obs_indices = list(range(out_dim))

        @qml.qnode(self.dev, interface="torch")
        def circuit(features, theta, phi, omega):
            # Angle encode features (pad/crop to num_qubits)
            if features.shape[0] < num_qubits:
                pad = torch.zeros(num_qubits - features.shape[0], dtype=features.dtype, device=features.device)
                feats = torch.cat([features, pad], dim=0)
            else:
                feats = features[:num_qubits]
            for i, f in enumerate(feats):
                qml.RY(f, wires=i)

            # Entangling layers
            for l in range(depth):
                for w in range(num_qubits):
                    qml.RZ(theta[l, w], wires=w)
                    qml.RY(phi[l, w], wires=w)
                    qml.RX(omega[l, w], wires=w)
                # simple ring entanglement
                for w in range(num_qubits):
                    qml.CNOT(wires=[w, (w+1) % num_qubits])

            # Measure expectations for first out_dim observables
            measures = []
            # Use simple set of observables deterministically
            for k in range(out_dim):
                idx = k % num_qubits
                measures.append(qml.expval(qml.PauliZ(idx)))
            return tuple(measures)

        self.circuit = circuit

    def forward(self, x):
        # x: (B, D) -> map to (B, out_dim)
        # scale/normalize to [-pi, pi]
        x = torch.tanh(x) * np.pi
        outputs = []
        for i in range(x.size(0)):
            y = self.circuit(x[i], self.theta, self.phi, self.omega)
            y = torch.stack(list(y))  # (out_dim,)
            outputs.append(y)
        return torch.stack(outputs, dim=0)
