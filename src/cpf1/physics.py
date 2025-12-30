"""
HP-Model protein folding implementation with Pivot Moves and Simulated Annealing.

This module provides the LatticeEngine class for simulating protein folding
on a 2D square lattice using the HP-Model (Hydrophobic-Polar), with Pivot
Moves for conformation changes and Simulated Annealing using the Metropolis
Criterion for energy minimization.
"""

import math
import random
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass, field
from enum import Enum


class ResidueType(Enum):
    """Enumeration for HP-Model residue types."""
    HYDROPHOBIC = "H"
    POLAR = "P"


@dataclass
class Position:
    """Represents a 2D lattice position."""
    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if isinstance(other, Position):
            return self.x == other.x and self.y == other.y
        return False

    def __repr__(self):
        return f"Position({self.x}, {self.y})"

    def distance(self, other: "Position") -> int:
        """Manhattan distance between two positions."""
        return abs(self.x - other.x) + abs(self.y - other.y)


@dataclass
class ConformationState:
    """Represents the current conformation state of a protein."""
    positions: List[Position] = field(default_factory=list)
    energy: float = 0.0

    def copy(self) -> "ConformationState":
        """Create a deep copy of the conformation state."""
        return ConformationState(
            positions=[Position(p.x, p.y) for p in self.positions],
            energy=self.energy
        )


class LatticeEngine:
    """
    HP-Model protein folding engine with Pivot Moves and Simulated Annealing.
    
    The LatticeEngine simulates protein folding on a 2D square lattice using:
    - HP-Model (Hydrophobic-Polar) for energy calculations
    - Pivot Moves for conformation changes
    - Simulated Annealing with Metropolis Criterion for optimization
    """

    # Contact energy parameters (HH interactions are favorable)
    CONTACT_ENERGY_HH = -1.0  # Favorable interaction
    CONTACT_ENERGY_HP = 0.0   # Neutral interaction
    CONTACT_ENERGY_PP = 0.0   # Neutral interaction

    def __init__(self, sequence: str):
        """
        Initialize the LatticeEngine with a protein sequence.
        
        Args:
            sequence: String of 'H' (hydrophobic) and 'P' (polar) residues
            
        Raises:
            ValueError: If sequence contains invalid characters
        """
        self.sequence = sequence.upper()
        self.length = len(sequence)
        
        # Validate sequence
        for char in self.sequence:
            if char not in ['H', 'P']:
                raise ValueError(
                    f"Invalid residue: {char}. Must be 'H' or 'P'."
                )
        
        # Convert sequence to ResidueType
        self.residues: List[ResidueType] = [
            ResidueType.HYDROPHOBIC if char == 'H' else ResidueType.POLAR
            for char in self.sequence
        ]
        
        # Initialize with random conformation
        self.current_state = self._initialize_random_conformation()
        self.best_state = self.current_state.copy()
        self.temperature = 1.0
        self.cooling_rate = 0.995

    def _initialize_random_conformation(self) -> ConformationState:
        """
        Initialize a random valid conformation on the lattice.
        
        Starts with the first residue at the origin and randomly builds
        a valid (non-overlapping) conformation.
        
        Returns:
            ConformationState: A valid random conformation
        """
        positions = [Position(0, 0)]
        occupied = {Position(0, 0)}
        
        for i in range(1, self.length):
            # Get valid neighbors
            last_pos = positions[-1]
            neighbors = [
                Position(last_pos.x + 1, last_pos.y),
                Position(last_pos.x - 1, last_pos.y),
                Position(last_pos.x, last_pos.y + 1),
                Position(last_pos.x, last_pos.y - 1),
            ]
            
            # Filter to unoccupied neighbors
            valid_neighbors = [n for n in neighbors if n not in occupied]
            
            if not valid_neighbors:
                # Restart if we get stuck
                return self._initialize_random_conformation()
            
            next_pos = random.choice(valid_neighbors)
            positions.append(next_pos)
            occupied.add(next_pos)
        
        state = ConformationState(positions=positions)
        state.energy = self.calculate_energy(state)
        return state

    def _initialize_extended_conformation(self) -> ConformationState:
        """
        Initialize an extended (linear) conformation on the lattice.
        
        Returns:
            ConformationState: An extended conformation
        """
        positions = [Position(i, 0) for i in range(self.length)]
        state = ConformationState(positions=positions)
        state.energy = self.calculate_energy(state)
        return state

    def calculate_energy(self, state: ConformationState) -> float:
        """
        Calculate the energy of a conformation using the HP-Model.
        
        Energy is calculated based on hydrophobic-hydrophobic contacts.
        Only non-adjacent pairs are considered (topological contacts).
        
        Args:
            state: ConformationState to evaluate
            
        Returns:
            float: Total energy of the conformation
        """
        energy = 0.0
        positions = state.positions
        
        # Check all pairs for topological contacts (non-adjacent in sequence)
        for i in range(self.length):
            for j in range(i + 3, self.length):  # Skip adjacent residues
                # Check if positions are adjacent on lattice
                if positions[i].distance(positions[j]) == 1:
                    # Adjacent on lattice - calculate contact energy
                    residue_i = self.residues[i]
                    residue_j = self.residues[j]
                    
                    if (residue_i == ResidueType.HYDROPHOBIC and
                        residue_j == ResidueType.HYDROPHOBIC):
                        energy += self.CONTACT_ENERGY_HH
        
        return energy

    def _is_valid_conformation(self, positions: List[Position]) -> bool:
        """
        Check if a conformation is valid (no overlaps).
        
        Args:
            positions: List of positions to validate
            
        Returns:
            bool: True if conformation is valid, False otherwise
        """
        return len(positions) == len(set(positions))

    def _are_positions_connected(self, positions: List[Position]) -> bool:
        """
        Check if all positions form a connected chain.
        
        Args:
            positions: List of positions to check
            
        Returns:
            bool: True if chain is connected, False otherwise
        """
        if len(positions) < 2:
            return True
        
        for i in range(len(positions) - 1):
            if positions[i].distance(positions[i + 1]) != 1:
                return False
        
        return True

    def pivot_move(self, state: ConformationState) -> Optional[ConformationState]:
        """
        Perform a pivot move on the conformation.
        
        A pivot move:
        1. Selects a random pivot point
        2. Rotates a segment of the chain around that pivot
        3. Returns the new conformation if valid
        
        Args:
            state: Current ConformationState
            
        Returns:
            ConformationState: New state after pivot move, or None if invalid
        """
        if self.length < 2:
            return None
        
        # Make a copy to modify
        new_state = state.copy()
        positions = new_state.positions
        
        # Choose pivot point (0 to length-1)
        pivot_idx = random.randint(0, self.length - 1)
        
        # Choose segment to rotate (after pivot)
        # At least one residue must be rotated
        segment_start = pivot_idx + 1
        segment_end = random.randint(
            segment_start,
            self.length - 1
        )
        
        if segment_start > segment_end:
            return None
        
        # Get rotation angle (90, 180, or 270 degrees)
        angle = random.choice([1, 2, 3])  # 1=90°, 2=180°, 3=270°
        
        # Rotate segment around pivot
        pivot_pos = positions[pivot_idx]
        
        for idx in range(segment_start, segment_end + 1):
            # Translate to origin
            rel_x = positions[idx].x - pivot_pos.x
            rel_y = positions[idx].y - pivot_pos.y
            
            # Rotate
            for _ in range(angle):
                rel_x, rel_y = -rel_y, rel_x
            
            # Translate back
            positions[idx] = Position(
                pivot_pos.x + rel_x,
                pivot_pos.y + rel_y
            )
        
        # Validate new conformation
        if not self._is_valid_conformation(positions):
            return None
        
        if not self._are_positions_connected(positions):
            return None
        
        # Calculate energy
        new_state.energy = self.calculate_energy(new_state)
        return new_state

    def metropolis_criterion(self, energy_old: float, energy_new: float) -> bool:
        """
        Apply Metropolis criterion to accept/reject a move.
        
        Accepts moves with lower energy always. Accepts higher energy
        moves with probability exp(-(E_new - E_old) / T).
        
        Args:
            energy_old: Energy of current state
            energy_new: Energy of proposed state
            
        Returns:
            bool: True to accept the move, False to reject
        """
        if energy_new < energy_old:
            return True
        
        # Metropolis acceptance probability
        delta_e = energy_new - energy_old
        acceptance_prob = math.exp(-delta_e / self.temperature)
        
        return random.random() < acceptance_prob

    def simulated_annealing(
        self,
        iterations: int,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.995,
        verbose: bool = False
    ) -> ConformationState:
        """
        Perform simulated annealing optimization.
        
        Args:
            iterations: Number of annealing iterations
            initial_temperature: Starting temperature
            cooling_rate: Temperature cooling rate per iteration
            verbose: Print progress information
            
        Returns:
            ConformationState: Best conformation found
        """
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        accepted_moves = 0
        total_moves = 0
        
        for iteration in range(iterations):
            # Attempt pivot move
            new_state = self.pivot_move(self.current_state)
            
            if new_state is None:
                continue
            
            total_moves += 1
            
            # Apply Metropolis criterion
            if self.metropolis_criterion(
                self.current_state.energy,
                new_state.energy
            ):
                self.current_state = new_state
                accepted_moves += 1
                
                # Update best state if better
                if new_state.energy < self.best_state.energy:
                    self.best_state = new_state.copy()
                    if verbose:
                        print(
                            f"Iteration {iteration}: "
                            f"New best energy = {new_state.energy:.2f}"
                        )
            
            # Cool down
            self.temperature *= self.cooling_rate
            
            if verbose and (iteration + 1) % (iterations // 10) == 0:
                acceptance_rate = (
                    accepted_moves / total_moves * 100
                    if total_moves > 0
                    else 0
                )
                print(
                    f"Iteration {iteration + 1}/{iterations}: "
                    f"T={self.temperature:.4f}, "
                    f"Energy={self.current_state.energy:.2f}, "
                    f"Best={self.best_state.energy:.2f}, "
                    f"Acceptance Rate={acceptance_rate:.1f}%"
                )
        
        return self.best_state

    def get_conformation_as_string(self, state: ConformationState) -> str:
        """
        Convert conformation positions to a string representation.
        
        Args:
            state: ConformationState to represent
            
        Returns:
            str: String representation of positions
        """
        return " -> ".join(
            f"({pos.x},{pos.y})" for pos in state.positions
        )

    def reset(self):
        """Reset the engine to a new random conformation."""
        self.current_state = self._initialize_random_conformation()
        self.best_state = self.current_state.copy()
        self.temperature = 1.0

    def set_conformation(self, positions: List[Tuple[int, int]]):
        """
        Manually set the conformation.
        
        Args:
            positions: List of (x, y) coordinate tuples
            
        Raises:
            ValueError: If conformation is invalid or length doesn't match
        """
        if len(positions) != self.length:
            raise ValueError(
                f"Position list length {len(positions)} "
                f"doesn't match sequence length {self.length}"
            )
        
        pos_objects = [Position(x, y) for x, y in positions]
        
        if not self._is_valid_conformation(pos_objects):
            raise ValueError("Conformation contains overlapping positions")
        
        if not self._are_positions_connected(pos_objects):
            raise ValueError("Conformation chain is not connected")
        
        self.current_state = ConformationState(positions=pos_objects)
        self.current_state.energy = self.calculate_energy(self.current_state)
        self.best_state = self.current_state.copy()

    def get_statistics(self) -> dict:
        """
        Get statistics about the current simulation state.
        
        Returns:
            dict: Dictionary containing current statistics
        """
        return {
            "sequence": self.sequence,
            "sequence_length": self.length,
            "current_energy": self.current_state.energy,
            "best_energy": self.best_state.energy,
            "current_temperature": self.temperature,
            "current_conformation": self.get_conformation_as_string(
                self.current_state
            ),
            "best_conformation": self.get_conformation_as_string(
                self.best_state
            ),
        }
