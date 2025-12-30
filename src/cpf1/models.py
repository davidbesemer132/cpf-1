"""Data models for protein folding results."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class FoldingResult:
    """
    Stores protein folding results including sequence, 3D structure, 
    confidence score, and thermodynamic energy state.
    
    Attributes:
        sequence (str): The amino acid sequence of the protein.
        coordinates (np.ndarray): 3D structure coordinates of shape (N, 3) 
            where N is the number of atoms/residues. Each row contains [x, y, z] coordinates.
        confidence_score (float): Confidence score for the folding prediction 
            (typically 0-1 or 0-100 depending on the source).
        energy_state (float): Thermodynamic energy state (e.g., free energy, 
            potential energy) of the folded structure.
    """
    
    sequence: str
    coordinates: np.ndarray
    confidence_score: float
    energy_state: float
    
    def __post_init__(self):
        """Validate the FoldingResult data after initialization."""
        if not isinstance(self.sequence, str) or not self.sequence:
            raise ValueError("Sequence must be a non-empty string")
        
        if not isinstance(self.coordinates, np.ndarray):
            raise TypeError("Coordinates must be a numpy array")
        
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 3:
            raise ValueError("Coordinates must be a 2D array with shape (N, 3)")
        
        if len(self.coordinates) == 0:
            raise ValueError("Coordinates array cannot be empty")
        
        if not isinstance(self.confidence_score, (int, float)):
            raise TypeError("Confidence score must be a numeric value")
        
        if not isinstance(self.energy_state, (int, float)):
            raise TypeError("Energy state must be a numeric value")
    
    def get_num_residues(self) -> int:
        """Get the number of residues in the protein structure."""
        return len(self.coordinates)
    
    def get_sequence_length(self) -> int:
        """Get the length of the amino acid sequence."""
        return len(self.sequence)
    
    def get_coordinate_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        """
        Get the bounding box of the 3D coordinates.
        
        Returns:
            Tuple of ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        """
        return (
            (float(np.min(self.coordinates[:, 0])), float(np.max(self.coordinates[:, 0]))),
            (float(np.min(self.coordinates[:, 1])), float(np.max(self.coordinates[:, 1]))),
            (float(np.min(self.coordinates[:, 2])), float(np.max(self.coordinates[:, 2])))
        )
