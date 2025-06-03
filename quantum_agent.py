"""
Quantum AI Agent - A flexible interface for quantum computing with support for multiple backends.

This module provides the QuantumAIAgent class which simplifies quantum circuit creation,
execution, and result analysis across different quantum computing platforms.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import Aer, AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler, Options
from qiskit.visualization import plot_histogram, plot_bloch_multivector, plot_state_qsphere
from qiskit.circuit import Parameter, Gate
from qiskit.circuit.library import UnitaryGate, QFT
from qiskit.quantum_info import Statevector, Operator
# Modern qiskit_aer doesn't have execute function in __init__
from typing import Dict, List, Union, Optional, Tuple, Any, Callable, TypeVar, Generic
import numpy as np
import os
import json
from datetime import datetime
from dataclasses import dataclass, asdict, field
from dotenv import load_dotenv
import warnings
import logging
from enum import Enum, auto
import matplotlib.pyplot as plt
from typing import get_args, get_origin, get_type_hints
from abc import ABC, abstractmethod

# Type variable for generic quantum result type
T = TypeVar('T')

@dataclass
class QuantumResult:
    """Data class to store quantum computation results."""
    counts: Dict[str, int]
    job_id: str
    backend: str
    shots: int
    timestamp: str
    circuit_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantumResult':
        """Create a QuantumResult from a dictionary."""
        return cls(**data)

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('quantum_agent.log')
    ]
)
logger = logging.getLogger(__name__)

class BackendType(Enum):
    """Types of backends supported by the QuantumAIAgent."""
    SIMULATOR = auto()
    IONQ = auto()
    RIGETTI = auto()
    IBMQ = auto()
    IONQ_SIMULATOR = auto()
    RIGETTI_SIMULATOR = auto()

class QuantumAIAgent:
    def __init__(self, api_token: str = None, backend_name: str = None):
        """
        Initialize the Quantum AI Agent with optional backend configuration.
        
        Args:
            api_token: API token for IBM Quantum or other quantum providers
            backend_name: Name of the backend to use (e.g., 'ibmq_qasm_simulator', 'aer_simulator')
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize configuration
        self.api_token = api_token or os.getenv("IBM_QUANTUM_TOKEN")
        self.backend_name = backend_name or os.getenv("QUANTUM_BACKEND", "aer_simulator")
        self.backend = None
        self.backend_type = BackendType.SIMULATOR
        self.service = None
        self.session = None
        self.options = Options()
        self.shots = int(os.getenv("SHOTS", "1024"))
        self.optimization_level = int(os.getenv("OPTIMIZATION_LEVEL", "1"))
        
        # Initialize the quantum backend
        self._initialize_backend()
        
        # Initialize circuit tracking
        self.circuit = None
        self._circuit_history = []
        self._results = {}
        
        logger.info(f"Initialized QuantumAIAgent with backend: {self.backend_name}")
    
    def create_circuit(self, num_qubits: int, num_classical: int = None, name: str = None) -> QuantumCircuit:
        """
        Create a new quantum circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit
            num_classical: Number of classical bits (defaults to num_qubits if None)
            name: Name of the circuit (optional)
            
        Returns:
            QuantumCircuit: The created quantum circuit
            
        Raises:
            ValueError: If num_qubits is not a positive integer
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise ValueError("Number of qubits must be a positive integer")
            
        if num_classical is None:
            num_classical = num_qubits
            
        if not name:
            name = f"qc_{len(self._circuit_history) + 1}"
            
        self.circuit = QuantumCircuit(num_qubits, num_classical, name=name)
        self._circuit_history.append((name, self.circuit.copy()))
        logger.info(f"Created new circuit '{name}' with {num_qubits} qubits and {num_classical} classical bits")
        return self.circuit
    
    def apply_gate(self, gate_name: str, qubits: List[int], params: List[float] = None, **kwargs) -> QuantumCircuit:
        """
        Apply a quantum gate to the circuit.
        
        Args:
            gate_name: Name of the gate (e.g., 'h', 'x', 'cx', 'rx')
            qubits: List of qubit indices the gate acts on
            params: Optional parameters for parameterized gates
            **kwargs: Additional arguments specific to the gate
            
        Returns:
            QuantumCircuit: The modified quantum circuit
            
        Raises:
            ValueError: If no circuit exists or invalid gate/parameters provided
        """
        if self.circuit is None:
            raise ValueError("No active circuit. Create a circuit first using create_circuit().")
            
        try:
            gate_name = gate_name.lower()
            
            # Handle common gates
            if gate_name == 'h':
                self.circuit.h(qubits[0])
            elif gate_name == 'x':
                self.circuit.x(qubits[0])
            elif gate_name == 'y':
                self.circuit.y(qubits[0])
            elif gate_name == 'z':
                self.circuit.z(qubits[0])
            elif gate_name == 's':
                self.circuit.s(qubits[0])
            elif gate_name == 'sdg':
                self.circuit.sdg(qubits[0])
            elif gate_name == 't':
                self.circuit.t(qubits[0])
            elif gate_name == 'tdg':
                self.circuit.tdg(qubits[0])
            elif gate_name == 'cx':
                if len(qubits) < 2:
                    raise ValueError("CX gate requires at least 2 qubits")
                self.circuit.cx(qubits[0], qubits[1])
            elif gate_name == 'swap':
                if len(qubits) < 2:
                    raise ValueError("SWAP gate requires 2 qubits")
                self.circuit.swap(qubits[0], qubits[1])
            # Handle parameterized gates
            elif gate_name == 'rx':
                if not params or len(params) < 1:
                    raise ValueError("RX gate requires a rotation angle parameter")
                self.circuit.rx(params[0], qubits[0])
            elif gate_name == 'ry':
                if not params or len(params) < 1:
                    raise ValueError("RY gate requires a rotation angle parameter")
                self.circuit.ry(params[0], qubits[0])
            elif gate_name == 'rz':
                if not params or len(params) < 1:
                    raise ValueError("RZ gate requires a rotation angle parameter")
                self.circuit.rz(params[0], qubits[0])
            elif gate_name == 'u':
                if not params or len(params) < 3:
                    raise ValueError("U gate requires 3 parameters (theta, phi, lambda)")
                self.circuit.u(params[0], params[1], params[2], qubits[0])
            # Handle QFT
            elif gate_name == 'qft':
                if len(qubits) < 1:
                    raise ValueError("QFT requires at least 1 qubit")
                qft = QFT(num_qubits=len(qubits))
                self.circuit.append(qft, qubits)
            else:
                raise ValueError(f"Unsupported gate: {gate_name}")
                
            logger.debug(f"Applied {gate_name.upper()} gate to qubits {qubits}")
            return self.circuit
            
        except Exception as e:
            error_msg = f"Failed to apply gate {gate_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e
    
    def measure(self, qubits: List[int] = None, cbits: List[int] = None) -> QuantumCircuit:
        """
        Add measurement operations to the circuit.
        
        Args:
            qubits: List of qubits to measure (defaults to all qubits)
            cbits: List of classical bits to store results (defaults to same indices as qubits)
            
        Returns:
            QuantumCircuit: The modified quantum circuit
        """
        if self.circuit is None:
            raise ValueError("No active circuit. Create a circuit first using create_circuit().")
            
        if qubits is None:
            qubits = list(range(self.circuit.num_qubits))
        if cbits is None:
            cbits = qubits[:len(qubits)]
            
        if len(qubits) != len(cbits):
            raise ValueError("Number of qubits and classical bits must match")
            
        self.circuit.measure(qubits, cbits)
        logger.info(f"Added measurement operations for qubits {qubits} -> cbits {cbits}")
        return self.circuit
    
    def execute(self, shots: int = None, **kwargs) -> Dict:
        """
        Execute the current quantum circuit.
        
        Args:
            shots: Number of shots to run (overrides default if provided)
            **kwargs: Additional execution options
            
        Returns:
            Dict containing execution results
            
        Raises:
            RuntimeError: If execution fails
        """
        if self.circuit is None:
            raise RuntimeError("No circuit to execute. Create and build a circuit first.")
            
        try:
            # Use provided shots or instance default
            exec_shots = shots if shots is not None else self.shots
            
            # Transpile the circuit for the target backend
            transpiled_circuit = transpile(
                self.circuit,
                backend=self.backend,
                optimization_level=self.optimization_level
            )
            
            # Execute the circuit
            logger.info(f"Executing circuit '{self.circuit.name}' with {exec_shots} shots on {self.backend}")
            
            if self.backend_type == BackendType.IBMQ:
                # Use Qiskit Runtime for IBM Quantum backends
                with Session(service=self.service, backend=self.backend) as session:
                    sampler = Sampler(session=session, options={"shots": exec_shots, **self.options})
                    job = sampler.run(transpiled_circuit)
                    result = job.result()
                    counts = result.get_counts()
            else:
                # Use Aer simulator for local execution
                job = self.backend.run(transpiled_circuit, shots=exec_shots, **kwargs)
                result = job.result()
                counts = result.get_counts()
            
            # Store and return results
            self._results = {
                'counts': counts,
                'job_id': getattr(job, 'job_id', 'local_execution'),
                'backend': str(self.backend),
                'shots': exec_shots,
                'timestamp': datetime.now().isoformat(),
                'circuit_name': self.circuit.name
            }
            
            logger.info(f"Execution completed. Results: {counts}")
            return self._results
            
        except Exception as e:
            error_msg = f"Failed to execute circuit: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def get_statevector(self) -> np.ndarray:
        """
        Get the statevector of the current circuit.
        
        Returns:
            np.ndarray: The statevector
            
        Raises:
            RuntimeError: If the circuit is not compatible with statevector simulation
        """
        if self.circuit is None:
            raise RuntimeError("No circuit available")
            
        try:
            # Remove measurements for statevector simulation
            circuit_no_measure = self.circuit.remove_final_measurements(inplace=False)
            
            # Use Aer's statevector simulator
            backend = Aer.get_backend('statevector_simulator')
            result = backend.run(transpile(circuit_no_measure, backend), shots=1).result()
            statevector = result.get_statevector()
            
            logger.debug(f"Obtained statevector: {statevector}")
            return statevector
            
        except Exception as e:
            error_msg = f"Failed to get statevector: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def draw_circuit(self, output: str = None, **kwargs) -> Any:
        """
        Draw the current quantum circuit.
        
        Args:
            output: Output format ('text', 'mpl', 'latex', 'latex_source', 'qsphere')
            **kwargs: Additional arguments for the drawer
            
        Returns:
            The circuit visualization in the specified format
        """
        if self.circuit is None:
            raise RuntimeError("No circuit to draw")
            
        output = output or 'text'
        try:
            if output == 'qsphere':
                # Special handling for QSphere visualization
                from qiskit.visualization import plot_state_qsphere
                statevector = self.get_statevector()
                return plot_state_qsphere(statevector, **kwargs)
            else:
                # Standard circuit drawing
                return self.circuit.draw(output=output, **kwargs)
        except Exception as e:
            error_msg = f"Failed to draw circuit: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def _initialize_backend(self) -> None:
        """Initialize the quantum backend based on configuration."""
        try:
            if 'ibm' in self.backend_name.lower():
                self._init_ibmq_backend()
            elif 'ionq' in self.backend_name.lower():
                self._init_ionq_backend()
            elif 'rigetti' in self.backend_name.lower():
                self._init_rigetti_backend()
            else:
                self._init_simulator()
                
        except Exception as e:
            logger.warning(f"Failed to initialize {self.backend_name}: {str(e)}")
            logger.info("Falling back to local simulator")
            self._init_simulator('aer_simulator')
    
    def _init_ibmq_backend(self) -> None:
        """Initialize an IBM Quantum backend."""
        if not self.api_token:
            logger.warning("No IBM Quantum API token provided. Using local simulator instead.")
            self._init_simulator()
            return
            
        logger.info(f"Connecting to IBM Quantum backend: {self.backend_name}")
        self.service = QiskitRuntimeService(channel="ibm_quantum", token=self.api_token)
        
        if self.backend_name.lower() == 'ibmq_least_busy':
            # Get the least busy IBM Quantum backend
            backends = self.service.backends(operational=True, simulator=False)
            if not backends:
                raise RuntimeError("No operational IBM Quantum backends available.")
            self.backend = min(backends, key=lambda b: b.status().pending_jobs)
        else:
            self.backend = self.service.backend(self.backend_name)
            
        self.backend_type = BackendType.IBMQ
        self.shots = min(self.shots, 4000)  # Typical max for IBMQ
        # Configure options for IBM Quantum
        self.options.execution.shots = self.shots
        self.options.optimization_level = min(self.optimization_level, 3)  # IBMQ supports up to 3
        logger.info(f"Connected to IBM Quantum backend: {self.backend.name}")
    
    def _init_ionq_backend(self) -> None:
        """Initialize an IonQ backend."""
        try:
            from qiskit_ionq import IonQProvider
            
            if not self.api_token:
                logger.warning("No IonQ API token provided. Using local simulator instead.")
                self._init_simulator()
                return
                
            logger.info(f"Connecting to IonQ backend: {self.backend_name}")
            provider = IonQProvider(token=self.api_token)
            
            if 'simulator' in self.backend_name.lower():
                self.backend = provider.get_backend('ionq_simulator')
                self.backend_type = BackendType.IONQ_SIMULATOR
            else:
                self.backend = provider.get_backend('ionq_qpu')
                self.backend_type = BackendType.IONQ
                
            logger.info(f"Connected to IonQ backend: {self.backend.name}")
            
        except ImportError:
            logger.warning("qiskit-ionq package not found. Using local simulator instead.")
            self._init_simulator()
    
    def _init_rigetti_backend(self) -> None:
        """Initialize a Rigetti backend."""
        try:
            from qiskit_rigetti import RigettiProvider
            
            if not self.api_token:
                logger.warning("No Rigetti API token provided. Using local simulator instead.")
                self._init_simulator()
                return
                
            logger.info(f"Connecting to Rigetti backend: {self.backend_name}")
            provider = RigettiProvider(api_key=self.api_token)
            
            if 'simulator' in self.backend_name.lower():
                self.backend = provider.get_backend('Aspen-11')
                self.backend_type = BackendType.RIGETTI_SIMULATOR
            else:
                self.backend = provider.get_backend('Aspen-11')
                self.backend_type = BackendType.RIGETTI
                
            logger.info(f"Connected to Rigetti backend: {self.backend.name}")
            
        except ImportError:
            logger.warning("qiskit-rigetti package not found. Using local simulator instead.")
            self._init_simulator()
    
    def _init_simulator(self, backend_name: str = 'aer_simulator') -> None:
        """Initialize a local simulator."""
        logger.info(f"Initializing local simulator: {backend_name}")
        
        if 'aer' in backend_name.lower():
            self.backend = Aer.get_backend(backend_name)
        else:
            self.backend = AerSimulator()
            
        self.backend_type = BackendType.SIMULATOR
        self.shots = 1024  # Default for local simulation
        # Note: Options.shots is not available in newer versions, handle in execution
        logger.info(f"Initialized local simulator: {self.backend.name}")
    
    def get_backend_info(self) -> Dict[str, Any]:
        """
        Get information about the current backend.
        
        Returns:
            Dict containing backend information
        """
        if not self.backend:
            return {"status": "No backend initialized"}
            
        info = {
            "backend": str(self.backend),
            "backend_type": self.backend_type.name,
            "status": "connected",
            "shots": self.shots,
            "optimization_level": self.optimization_level
        }
        
        # Add provider-specific information
        if hasattr(self.backend, 'status'):
            try:
                status = self.backend.status()
                info["status"] = status.status_msg
                info["pending_jobs"] = status.pending_jobs
            except Exception as e:
                logger.warning(f"Could not get backend status: {str(e)}")
                info["status"] = "status_unknown"
            
        if hasattr(self.backend, 'configuration'):
            try:
                config = self.backend.configuration()
                info.update({
                    "num_qubits": getattr(config, 'n_qubits', 'N/A'),
                    "coupling_map": getattr(config, 'coupling_map', 'N/A'),
                    "basis_gates": getattr(config, 'basis_gates', 'N/A'),
                    "backend_version": getattr(config, 'backend_version', 'N/A')
                })
            except Exception as e:
                logger.warning(f"Could not get backend configuration: {str(e)}")
            
        return info
    
    def plot_results(self, results: Dict = None, title: str = None, **kwargs) -> plt.Figure:
        """
        Plot the results of a quantum circuit execution.
        
        Args:
            results: Results dictionary (uses last results if None)
            title: Plot title
            **kwargs: Additional arguments for plot_histogram
            
        Returns:
            matplotlib.figure.Figure: The plot figure
        """
        if results is None:
            results = self._results
            
        if not results or 'counts' not in results:
            raise ValueError("No results available to plot")
            
        counts = results['counts']
        title = title or f"Results from {results.get('circuit_name', 'circuit')}"
        
        fig = plot_histogram(counts, title=title, **kwargs)
        return fig
    
    def reset(self) -> None:
        """Reset the agent's state (circuit and results)."""
        self.circuit = None
        self._results = {}
        logger.info("Agent state has been reset")
    
    def create_quantum_circuit(self, num_qubits: int, gates: List[Dict]) -> QuantumCircuit:
        """
        Create a quantum circuit based on the specified gates.
        
        Args:
            num_qubits: Number of qubits in the circuit
            gates: List of gate operations with their parameters. Each gate is a dict with:
                  - 'name': str - Name of the gate (e.g., 'h', 'x', 'cx', 'rx')
                  - 'qubits': List[int] - Qubit indices the gate acts on
                  - 'params': List[float] - Optional parameters for parameterized gates
                  - 'cbits': List[int] - Optional classical bits for measurement
            
        Returns:
            QuantumCircuit: The constructed quantum circuit
            
        Raises:
            ValueError: If the gate specification is invalid
        """
        try:
            # Create a new circuit
            self.create_circuit(num_qubits, name="quantum_circuit")
            
            # Apply each gate in sequence
            for gate in gates:
                gate_name = gate.get('name', '').lower()
                qubits = gate.get('qubits', [])
                params = gate.get('params', [])
                cbits = gate.get('cbits', [])
                
                # Handle measurement separately
                if gate_name == 'measure':
                    if not qubits:
                        qubits = list(range(num_qubits))
                    if not cbits:
                        cbits = qubits[:len(qubits)]
                    self.measure(qubits, cbits)
                elif gate_name == 'barrier':
                    # Add barrier for visualization
                    self.circuit.barrier()
                else:
                    # Apply quantum gate
                    self.apply_gate(gate_name, qubits, params)
            
            logger.info(f"Created quantum circuit with {len(gates)} gates on {num_qubits} qubits")
            return self.circuit
            
        except Exception as e:
            error_msg = f"Failed to create quantum circuit: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e
    
    def run_circuit(self, circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, Any]:
        """
        Run the quantum circuit on the available backend.
        
        Args:
            circuit: The quantum circuit to run
            shots: Number of shots to run the circuit
            
        Returns:
            Dict containing the measurement results and status
        """
        if not hasattr(self, 'backend') or not self.backend:
            return {"error": "No quantum backend available", "status": "error"}
        
        try:
            logger.info(f"Running circuit with {shots} shots on {self.backend}")
            
            if isinstance(self.backend, str) and 'aer' in self.backend.lower():
                # Use local Aer simulator
                backend = Aer.get_backend('qasm_simulator')
                
                # Transpile the circuit for the backend
                transpiled_circuit = transpile(circuit, backend)
                
                # Execute the circuit
                job = backend.run(transpiled_circuit, shots=shots)
                result = job.result()
                counts = result.get_counts()
                
                return {
                    "counts": counts,
                    "status": "success",
                    "backend": str(backend),
                    "job_id": job.job_id() if hasattr(job, 'job_id') else None,
                    "execution_time": result.time_taken if hasattr(result, 'time_taken') else None
                }
                
            elif hasattr(self, 'service') and self.service is not None:
                # Use IBM Quantum Runtime or other cloud backends
                with Session(service=self.service, backend=self.backend) as session:
                    # Set up runtime options
                    options = Options()
                    options.optimization_level = self.optimization_level
                    options.execution.shots = shots
                    
                    # Create a Sampler instance
                    sampler = Sampler(session=session, options=options)
                    
                    # Run the job
                    job = sampler.run(circuit)
                    result = job.result()
                    
                    # Format the results
                    counts = {}
                    if hasattr(result, 'quasi_dists') and result.quasi_dists:
                        # Convert quasi-probability distribution to counts
                        quasi_dist = result.quasi_dists[0]
                        total_shots = sum(quasi_dist.values())
                        counts = {bin(k)[2:].zfill(circuit.num_qubits): int(v * total_shots) 
                                for k, v in quasi_dist.items()}
                    
                    return {
                        "counts": counts,
                        "status": "success",
                        "backend": str(self.backend),
                        "job_id": job.job_id(),
                        "execution_time": result.metadata[0].get('time_taken', None) if hasattr(result, 'metadata') else None
                    }
            else:
                # Fallback to basic execution for other backends
                job = self.backend.run(circuit, shots=shots)
                result = job.result()
                counts = result.get_counts()
                
                return {
                    "counts": counts,
                    "status": "success",
                    "backend": str(self.backend),
                    "job_id": job.job_id() if hasattr(job, 'job_id') else None,
                    "execution_time": result.time_taken if hasattr(result, 'time_taken') else None
                }
                
        except Exception as e:
            error_msg = f"Error running circuit: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "error": error_msg,
                "status": "error",
                "details": str(e)
            }
    
    def analyze_results(self, counts: Dict) -> Dict:
        """
        Analyze the measurement results and provide insights.
        
        Args:
            counts: Measurement results from the quantum circuit
            
        Returns:
            Dict containing analysis results
        """
        if not counts or "error" in counts:
            return counts
            
        total_shots = sum(counts.values())
        probabilities = {state: count/total_shots for state, count in counts.items()}
        
        # Calculate entropy as a measure of randomness
        entropy = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)
        
        # Find most probable state
        most_probable_state = max(probabilities, key=probabilities.get)
        
        return {
            "total_shots": total_shots,
            "probabilities": probabilities,
            "entropy": entropy,
            "most_probable_state": most_probable_state,
            "analysis": self._generate_insight(counts, probabilities, entropy)
        }
    
    def _generate_insight(self, counts: Dict, probabilities: Dict, entropy: float) -> str:
        """Generate human-readable insights from the quantum results."""
        if entropy > 2.5:  # High entropy indicates more uniform distribution
            return "The quantum state shows high entropy, indicating a nearly uniform superposition across states."
        else:
            most_probable = max(probabilities, key=probabilities.get)
            prob = probabilities[most_probable]
            return f"The most probable state is {most_probable} with probability {prob:.2f}. " \
                  f"The system shows {entropy:.2f} bits of entropy."

    def optimize_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Optimize the quantum circuit using built-in optimization passes.
        
        Args:
            circuit: The quantum circuit to optimize
            
        Returns:
            Optimized quantum circuit
        """
        try:
            from qiskit.transpiler import PassManager
            from qiskit.transpiler.passes import (
                Optimize1qGatesDecomposition,
                RemoveBarriers,
            )
            
            # Use only available optimization passes
            pass_manager = PassManager([
                RemoveBarriers(),
                Optimize1qGatesDecomposition(),
            ])
            
            return pass_manager.run(circuit)
        except Exception as e:
            logger.warning(f"Failed to optimize circuit: {str(e)}. Returning original circuit.")
            return circuit
