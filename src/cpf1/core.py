"""Core controller module for protein folding."""

from .physics import LatticeEngine, ConformationState
from .models import FoldingResult
import numpy as np


class ProteinFolder:
    """
    ANT Corp Core Controller.
    
    Translates natural language intents into thermodynamic parameters
    and manages the protein folding simulation.
    """

    def __init__(self, resonance_frequency: int = 528):
        """
        Initialize the ProteinFolder.

        Args:
            resonance_frequency: Resonance frequency in Hz (default 528 Hz)
        """
        self.resonance = resonance_frequency
        print(
            f"[SYSTEM] CPF-1 Lattice Physics Engine loaded. "
            f"Resonance: {self.resonance}Hz"
        )

    def _parse_intent(self, prompt: str) -> dict:
        """
        Parse user intent from natural language prompt.

        Translates user input into thermodynamic parameters:
        - Temperature (controls exploration vs exploitation)
        - Iteration count (controls simulation depth)
        - Mode (standard or besemer-enhanced)

        Args:
            prompt: Natural language user input

        Returns:
            dict: Settings dictionary with 'mode', 'temperature', 'iterations'
        """
        settings = {
            "mode": "standard_annealing",
            "temperature": 1.0,
            "iterations": 500,
            "cooling_rate": 0.995,
        }

        prompt_lower = prompt.lower()

        # BESEMER PROTOCOL: High temperature for exploration
        if "120" in prompt_lower or "high temp" in prompt_lower or "superconductiv" in prompt_lower:
            settings["temperature"] = 120.0
            settings["mode"] = "besemer_superconductive_search"
            settings["iterations"] = 1000
            settings["cooling_rate"] = 0.99

        # Enhanced stability settings
        if "stable" in prompt_lower or "focus" in prompt_lower:
            settings["iterations"] = 2000
            settings["cooling_rate"] = 0.995

        return settings

    def fold(self, sequence: str, prompt: str = "") -> FoldingResult:
        """
        Execute protein folding simulation.

        Args:
            sequence: Protein sequence (H for hydrophobic, P for polar)
            prompt: Natural language intent (optional)

        Returns:
            FoldingResult: Object containing folding results
        """
        # Parse intent
        settings = self._parse_intent(prompt)

        # Initialize engine
        engine = LatticeEngine(sequence)

        # Run simulated annealing
        best_state = engine.simulated_annealing(
            iterations=settings["iterations"],
            initial_temperature=settings["temperature"],
            cooling_rate=settings["cooling_rate"],
            verbose=False
        )

        # Extract coordinates as numpy array
        coordinates = np.array(
            [[pos.x, pos.y, 0] for pos in best_state.positions]
        )

        # Calculate confidence score
        # Theoretical minimum: -1.0 per residue for perfectly packed H-protein
        n = len(sequence)
        theoretical_min = -n

        if best_state.energy < 0:
            confidence = min(0.99, abs(best_state.energy) / abs(theoretical_min))
        else:
            confidence = 0.1

        # Create and return result
        return FoldingResult(
            sequence=sequence,
            coordinates=coordinates,
            confidence_score=max(0.1, confidence),
            energy_state=best_state.energy,
        )
