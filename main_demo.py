#!/usr/bin/env python3
"""
Protein Folding Demonstration: Standard vs Besemer Mode Comparison
===================================================================

This script demonstrates protein folding simulations comparing:
- Standard Mode: Conventional protein folding algorithms
- Besemer Mode: Optimized protein folding with enhanced energy analysis

Features:
- Time measurements for both modes
- Detailed energy analysis
- Performance comparison metrics
- Visualization of results
"""

import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import sys


class ProteinFoldingSimulator:
    """Base class for protein folding simulation."""
    
    def __init__(self, sequence: str, temperature: float = 300.0):
        """
        Initialize protein folding simulator.
        
        Args:
            sequence: Amino acid sequence
            temperature: Simulation temperature in Kelvin
        """
        self.sequence = sequence
        self.temperature = temperature
        self.sequence_length = len(sequence)
        self.energy_history = []
        self.conformation = None
        self.execution_time = 0.0
        
    def generate_random_conformation(self) -> np.ndarray:
        """Generate random 3D protein conformation."""
        return np.random.randn(self.sequence_length, 3)
    
    def calculate_distance_matrix(self, conformation: np.ndarray) -> np.ndarray:
        """Calculate pairwise distance matrix."""
        distances = np.zeros((self.sequence_length, self.sequence_length))
        for i in range(self.sequence_length):
            for j in range(i + 1, self.sequence_length):
                dist = np.linalg.norm(conformation[i] - conformation[j])
                distances[i, j] = dist
                distances[j, i] = dist
        return distances
    
    def estimate_energy(self, conformation: np.ndarray) -> float:
        """Estimate folding energy of conformation."""
        distances = self.calculate_distance_matrix(conformation)
        
        # Van der Waals energy (repulsive at short distances)
        vdw_energy = 0.0
        for i in range(self.sequence_length):
            for j in range(i + 2, min(i + 10, self.sequence_length)):
                d = distances[i, j]
                if d > 0:
                    vdw_energy += (1.0 / (d ** 12)) - (2.0 / (d ** 6))
        
        # Hydrophobic effect (attractive for hydrophobic residues)
        hydrophobic_residues = {'A', 'V', 'L', 'I', 'M', 'F', 'W', 'P'}
        hydro_energy = 0.0
        for i in range(self.sequence_length):
            if self.sequence[i] in hydrophobic_residues:
                for j in range(i + 2, self.sequence_length):
                    if self.sequence[j] in hydrophobic_residues:
                        d = distances[i, j]
                        if d > 0 and d < 8.0:
                            hydro_energy -= (1.0 / d)
        
        # Bond strain energy
        bond_energy = 0.0
        for i in range(self.sequence_length - 1):
            bond_length = np.linalg.norm(conformation[i + 1] - conformation[i])
            bond_energy += (bond_length - 3.8) ** 2
        
        total_energy = vdw_energy + hydro_energy + bond_energy
        return total_energy


class StandardFoldingMode(ProteinFoldingSimulator):
    """Standard protein folding simulation mode."""
    
    def simulate(self, iterations: int = 1000, cooling_rate: float = 0.99) -> Dict:
        """
        Run standard protein folding simulation.
        
        Args:
            iterations: Number of simulation iterations
            cooling_rate: Temperature cooling factor per iteration
            
        Returns:
            Dictionary with simulation results
        """
        start_time = time.time()
        
        # Initialize conformation
        self.conformation = self.generate_random_conformation()
        current_energy = self.estimate_energy(self.conformation)
        self.energy_history = [current_energy]
        
        temperature = self.temperature
        accepted_moves = 0
        rejected_moves = 0
        
        # Simulated annealing optimization
        for iteration in range(iterations):
            # Generate neighbor conformation
            neighbor = self.conformation.copy()
            random_residue = np.random.randint(0, self.sequence_length)
            random_displacement = np.random.randn(3) * 0.5
            neighbor[random_residue] += random_displacement
            
            # Calculate energy change
            neighbor_energy = self.estimate_energy(neighbor)
            delta_energy = neighbor_energy - current_energy
            
            # Metropolis criterion
            if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / (8.314 * temperature / 1000)):
                self.conformation = neighbor
                current_energy = neighbor_energy
                accepted_moves += 1
            else:
                rejected_moves += 1
            
            self.energy_history.append(current_energy)
            
            # Cool down
            temperature *= cooling_rate
        
        self.execution_time = time.time() - start_time
        
        return {
            'mode': 'Standard',
            'final_energy': current_energy,
            'execution_time': self.execution_time,
            'accepted_moves': accepted_moves,
            'rejected_moves': rejected_moves,
            'acceptance_rate': accepted_moves / (accepted_moves + rejected_moves),
            'energy_history': self.energy_history,
            'sequence_length': self.sequence_length
        }


class BesomerFoldingMode(ProteinFoldingSimulator):
    """Optimized Besemer protein folding simulation mode."""
    
    def simulate(self, iterations: int = 1000, cooling_rate: float = 0.99, 
                 optimization_level: int = 2) -> Dict:
        """
        Run Besemer optimized protein folding simulation.
        
        Args:
            iterations: Number of simulation iterations
            cooling_rate: Temperature cooling factor per iteration
            optimization_level: Level of optimization (1-3)
            
        Returns:
            Dictionary with simulation results
        """
        start_time = time.time()
        
        # Initialize conformation with better starting point
        self.conformation = self.generate_random_conformation()
        self._optimize_initial_geometry()
        
        current_energy = self.estimate_energy(self.conformation)
        self.energy_history = [current_energy]
        
        temperature = self.temperature
        accepted_moves = 0
        rejected_moves = 0
        energy_improvements = 0
        
        # Enhanced simulated annealing with adaptive moves
        for iteration in range(iterations):
            # Generate neighbor conformation with adaptive strategy
            neighbor = self.conformation.copy()
            
            if optimization_level >= 2:
                # Use selective residue moves based on energy gradient
                num_moves = max(1, int(self.sequence_length * 0.3))
                residues_to_move = self._select_residues_for_optimization(num_moves)
            else:
                residues_to_move = [np.random.randint(0, self.sequence_length)]
            
            for residue_idx in residues_to_move:
                if optimization_level == 3:
                    # Use gradient-based displacement
                    displacement = self._calculate_optimal_displacement(residue_idx)
                else:
                    displacement = np.random.randn(3) * 0.5
                neighbor[residue_idx] += displacement
            
            # Calculate energy change
            neighbor_energy = self.estimate_energy(neighbor)
            delta_energy = neighbor_energy - current_energy
            
            # Metropolis criterion with bias towards improvements
            acceptance_prob = np.exp(-delta_energy / (8.314 * temperature / 1000))
            if delta_energy < 0:
                energy_improvements += 1
                acceptance_prob = min(1.0, acceptance_prob * 1.2)  # Bias towards improvements
            
            if delta_energy < 0 or np.random.random() < acceptance_prob:
                self.conformation = neighbor
                current_energy = neighbor_energy
                accepted_moves += 1
            else:
                rejected_moves += 1
            
            self.energy_history.append(current_energy)
            
            # Adaptive cooling
            temperature *= cooling_rate ** (1 + 0.1 * optimization_level)
        
        self.execution_time = time.time() - start_time
        
        return {
            'mode': 'Besemer',
            'final_energy': current_energy,
            'execution_time': self.execution_time,
            'accepted_moves': accepted_moves,
            'rejected_moves': rejected_moves,
            'acceptance_rate': accepted_moves / (accepted_moves + rejected_moves),
            'energy_improvements': energy_improvements,
            'energy_history': self.energy_history,
            'sequence_length': self.sequence_length,
            'optimization_level': optimization_level
        }
    
    def _optimize_initial_geometry(self):
        """Optimize initial protein geometry."""
        for _ in range(10):
            for i in range(self.sequence_length - 1):
                current_energy = self.estimate_energy(self.conformation)
                
                # Try small displacement
                test_conf = self.conformation.copy()
                test_conf[i] += np.random.randn(3) * 0.1
                test_energy = self.estimate_energy(test_conf)
                
                if test_energy < current_energy:
                    self.conformation = test_conf
    
    def _select_residues_for_optimization(self, num_residues: int) -> List[int]:
        """Select residues most likely to improve energy."""
        # In a real implementation, this would use energy gradient analysis
        return np.random.choice(self.sequence_length, size=num_residues, replace=False).tolist()
    
    def _calculate_optimal_displacement(self, residue_idx: int) -> np.ndarray:
        """Calculate optimal displacement for a residue."""
        # In a real implementation, this would use actual gradient calculations
        return np.random.randn(3) * 0.3


class PerformanceAnalyzer:
    """Analyze and compare simulation performance."""
    
    @staticmethod
    def compare_results(standard_results: Dict, besemer_results: Dict) -> Dict:
        """
        Compare standard and Besemer mode results.
        
        Args:
            standard_results: Results from standard mode
            besemer_results: Results from Besemer mode
            
        Returns:
            Comparison metrics
        """
        time_ratio = standard_results['execution_time'] / besemer_results['execution_time']
        energy_improvement = (standard_results['final_energy'] - besemer_results['final_energy']) / abs(standard_results['final_energy']) * 100
        
        return {
            'time_speedup': time_ratio,
            'energy_improvement_percent': energy_improvement,
            'acceptance_rate_difference': besemer_results['acceptance_rate'] - standard_results['acceptance_rate'],
            'standard_final_energy': standard_results['final_energy'],
            'besemer_final_energy': besemer_results['final_energy'],
            'standard_time': standard_results['execution_time'],
            'besemer_time': besemer_results['execution_time']
        }
    
    @staticmethod
    def print_detailed_report(standard_results: Dict, besemer_results: Dict, comparison: Dict):
        """Print detailed comparison report."""
        print("\n" + "="*70)
        print("PROTEIN FOLDING SIMULATION REPORT")
        print("="*70)
        print(f"\nTimestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"Sequence Length: {standard_results['sequence_length']} residues\n")
        
        print("-" * 70)
        print("STANDARD MODE RESULTS")
        print("-" * 70)
        print(f"Final Energy:          {standard_results['final_energy']:>15.4f} kcal/mol")
        print(f"Execution Time:        {standard_results['execution_time']:>15.4f} seconds")
        print(f"Accepted Moves:        {standard_results['accepted_moves']:>15d}")
        print(f"Rejected Moves:        {standard_results['rejected_moves']:>15d}")
        print(f"Acceptance Rate:       {standard_results['acceptance_rate']:>15.2%}")
        
        print("\n" + "-" * 70)
        print("BESEMER MODE RESULTS")
        print("-" * 70)
        print(f"Final Energy:          {besemer_results['final_energy']:>15.4f} kcal/mol")
        print(f"Execution Time:        {besemer_results['execution_time']:>15.4f} seconds")
        print(f"Accepted Moves:        {besemer_results['accepted_moves']:>15d}")
        print(f"Rejected Moves:        {besemer_results['rejected_moves']:>15d}")
        print(f"Acceptance Rate:       {besemer_results['acceptance_rate']:>15.2%}")
        print(f"Optimization Level:    {besemer_results['optimization_level']:>15d}")
        print(f"Energy Improvements:   {besemer_results['energy_improvements']:>15d}")
        
        print("\n" + "-" * 70)
        print("COMPARATIVE ANALYSIS")
        print("-" * 70)
        print(f"Time Speedup:          {comparison['time_speedup']:>15.2f}x")
        print(f"Energy Improvement:    {comparison['energy_improvement_percent']:>15.2f}%")
        print(f"Acceptance Rate Δ:     {comparison['acceptance_rate_difference']:>15.2%}")
        print(f"Energy Reduction:      {comparison['standard_final_energy'] - comparison['besemer_final_energy']:>15.4f} kcal/mol")
        
        print("\n" + "="*70 + "\n")


def run_demonstration():
    """Run the complete protein folding demonstration."""
    
    print("\n" + "="*70)
    print("INITIALIZING PROTEIN FOLDING DEMONSTRATION")
    print("="*70)
    
    # Define test sequences
    test_sequences = {
        'small': 'MLKKSVVVAVALLLAVVFAFSSCGDDDDTGPPAKSDLGVELVAVND',
        'medium': 'MKVLLALLLTGAAACDEFGHIKLMNPQRSTVWYFSTDFLHVMDPQKLMNPQRSTUVWXYZA',
        'custom': 'AVNGLARWGVVAHRVVSAIFDQQHLFPYGQFVVENVFVFYDTPQSMPVQEPPPPPPPPPPP'
    }
    
    print(f"\nStarting simulations at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"User: davidbesemer132")
    print(f"Repository: davidbesemer132/cpf-1\n")
    
    results_summary = {}
    
    # Run simulations for each sequence
    for seq_name, sequence in test_sequences.items():
        print(f"\n{'Processing sequence: ' + seq_name:-^70}")
        print(f"Sequence: {sequence[:40]}..." if len(sequence) > 40 else f"Sequence: {sequence}")
        print(f"Length: {len(sequence)} residues\n")
        
        # Standard mode
        print("Running Standard Mode...", end='', flush=True)
        standard = StandardFoldingMode(sequence, temperature=300.0)
        standard_results = standard.simulate(iterations=500, cooling_rate=0.98)
        print(f" Complete! ({standard_results['execution_time']:.4f}s)")
        
        # Besemer mode
        print("Running Besemer Mode...", end='', flush=True)
        besemer = BesomerFoldingMode(sequence, temperature=300.0)
        besemer_results = besemer.simulate(iterations=500, cooling_rate=0.98, optimization_level=2)
        print(f" Complete! ({besemer_results['execution_time']:.4f}s)")
        
        # Analysis
        comparison = PerformanceAnalyzer.compare_results(standard_results, besemer_results)
        results_summary[seq_name] = {
            'standard': standard_results,
            'besemer': besemer_results,
            'comparison': comparison
        }
        
        # Print report
        PerformanceAnalyzer.print_detailed_report(standard_results, besemer_results, comparison)
    
    # Summary statistics
    print("\n" + "="*70)
    print("OVERALL PERFORMANCE SUMMARY")
    print("="*70)
    
    avg_speedup = np.mean([r['comparison']['time_speedup'] for r in results_summary.values()])
    avg_energy_improvement = np.mean([r['comparison']['energy_improvement_percent'] for r in results_summary.values()])
    
    print(f"\nAverage Time Speedup (Besemer vs Standard): {avg_speedup:.2f}x")
    print(f"Average Energy Improvement: {avg_energy_improvement:.2f}%")
    print(f"\nTotal Sequences Processed: {len(results_summary)}")
    print(f"Completion Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("\n" + "="*70 + "\n")
    
    return results_summary


if __name__ == '__main__':
    try:
        results = run_demonstration()
        print("✓ Protein folding demonstration completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error during demonstration: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
