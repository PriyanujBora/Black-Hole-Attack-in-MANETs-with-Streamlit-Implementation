import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from collections import defaultdict
import time
from typing import List, Tuple, Dict, Set
import math

class MANETNode:
    """Represents a node in the MANET"""
    def __init__(self, node_id: int, x: float, y: float, is_blackhole: bool = False):
        self.id = node_id
        self.x = x
        self.y = y
        self.is_blackhole = is_blackhole
        self.neighbors = set()
        self.routing_table = {}
        self.packet_drop_rate = 0.0 if not is_blackhole else random.uniform(0.7, 1.0)
        self.energy = 100.0
        self.trust_value = 1.0
        self.packets_sent = 0
        self.packets_received = 0
        self.packets_forwarded = 0
        self.packets_dropped = 0
        
    def distance_to(self, other: 'MANETNode') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class MANETSimulator:
    """Simulates a Mobile Ad-hoc Network with black hole attacks"""
    def __init__(self, num_nodes: int, area_size: float, transmission_range: float, 
                 blackhole_percentage: float = 0.2):
        self.num_nodes = num_nodes
        self.area_size = area_size
        self.transmission_range = transmission_range
        self.nodes = []
        self.blackhole_nodes = set()
        self.network_graph = nx.Graph()
        
        # Create nodes
        num_blackholes = int(num_nodes * blackhole_percentage)
        blackhole_indices = random.sample(range(num_nodes), num_blackholes)
        
        for i in range(num_nodes):
            x = random.uniform(0, area_size)
            y = random.uniform(0, area_size)
            is_blackhole = i in blackhole_indices
            node = MANETNode(i, x, y, is_blackhole)
            self.nodes.append(node)
            if is_blackhole:
                self.blackhole_nodes.add(i)
        
        self._update_network_topology()
    
    def _update_network_topology(self):
        """Update network topology based on transmission range"""
        self.network_graph.clear()
        
        # Add all nodes to the graph first (including isolated nodes)
        for i in range(self.num_nodes):
            self.network_graph.add_node(i)
        
        # Clear neighbors
        for node in self.nodes:
            node.neighbors.clear()
        
        # Add edges based on transmission range
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes[i+1:], i+1):
                if node1.distance_to(node2) <= self.transmission_range:
                    node1.neighbors.add(j)
                    node2.neighbors.add(i)
                    self.network_graph.add_edge(i, j)
    
    def simulate_packet_transmission(self, source_id: int, dest_id: int) -> bool:
        """Simulate packet transmission from source to destination"""
        if source_id == dest_id:
            return True
        
        # Check if both nodes exist in the graph
        if source_id not in self.network_graph or dest_id not in self.network_graph:
            return False
        
        try:
            path = nx.shortest_path(self.network_graph, source_id, dest_id)
        except nx.NetworkXNoPath:
            return False
        
        # Simulate packet forwarding through the path
        for i in range(len(path) - 1):
            current_node = self.nodes[path[i]]
            next_node = self.nodes[path[i + 1]]
            
            current_node.packets_sent += 1
            
            # Check if next node is a black hole
            if next_node.is_blackhole:
                # Black hole drops packet based on its drop rate
                if random.random() < next_node.packet_drop_rate:
                    next_node.packets_dropped += 1
                    return False
            
            next_node.packets_received += 1
            if i < len(path) - 2:  # Not the final destination
                next_node.packets_forwarded += 1
        
        return True
    
    def collect_node_features(self) -> np.ndarray:
        """Collect features for each node for detection algorithms"""
        features = []
        for node in self.nodes:
            total_packets = node.packets_sent + node.packets_received + node.packets_forwarded
            if total_packets > 0:
                drop_ratio = node.packets_dropped / total_packets
            else:
                drop_ratio = 0
            
            # Calculate neighbor trust average
            neighbor_trust = []
            for neighbor_id in node.neighbors:
                neighbor_trust.append(self.nodes[neighbor_id].trust_value)
            avg_neighbor_trust = np.mean(neighbor_trust) if neighbor_trust else 1.0
            
            # Node features
            node_features = [
                drop_ratio,                    # Packet drop ratio
                node.energy / 100.0,          # Normalized energy
                node.trust_value,             # Trust value
                len(node.neighbors) / self.num_nodes,  # Normalized degree
                avg_neighbor_trust,           # Average neighbor trust
                node.packets_forwarded / (node.packets_received + 1),  # Forward ratio
            ]
            features.append(node_features)
        
        return np.array(features)

class DolphinEcholocationOptimizer:
    """Basic Dolphin Echolocation Algorithm for black hole detection"""
    def __init__(self, n_dolphins: int = 20, max_iterations: int = 100):
        self.n_dolphins = n_dolphins
        self.max_iterations = max_iterations
        self.detection_threshold = 0.5
    
    def _calculate_fitness(self, position: np.ndarray, features: np.ndarray) -> float:
        """Calculate fitness based on detection accuracy"""
        # Use position as weights for feature importance
        weights = np.abs(position) / (np.sum(np.abs(position)) + 1e-10)
        scores = np.dot(features, weights)
        return np.mean(scores)
    
    def detect_blackholes(self, features: np.ndarray) -> np.ndarray:
        """Detect black holes using Dolphin Echolocation"""
        n_features = features.shape[1]
        
        # Initialize dolphin positions
        dolphins = np.random.randn(self.n_dolphins, n_features)
        best_dolphin = None
        best_fitness = -np.inf
        
        for iteration in range(self.max_iterations):
            # Evaluate fitness for each dolphin
            fitness_values = []
            for dolphin in dolphins:
                fitness = self._calculate_fitness(dolphin, features)
                fitness_values.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_dolphin = dolphin.copy()
            
            # Update dolphin positions (echolocation)
            for i in range(self.n_dolphins):
                # Generate echolocation clicks
                frequency = np.random.uniform(0.1, 0.9)
                amplitude = np.random.uniform(0.5, 1.5)
                
                # Update position based on best dolphin
                dolphins[i] += frequency * (best_dolphin - dolphins[i]) * amplitude
                
                # Add random exploration
                dolphins[i] += np.random.randn(n_features) * 0.1
        
        # Calculate detection scores
        weights = np.abs(best_dolphin) / (np.sum(np.abs(best_dolphin)) + 1e-10)
        detection_scores = np.dot(features, weights)
        
        # Threshold-based detection
        predictions = (detection_scores > self.detection_threshold).astype(int)
        return predictions

class HybridDEABCOptimizer:
    """Original Hybrid Dolphin Echolocation and Artificial Bee Colony Algorithm"""
    def __init__(self, n_dolphins: int = 20, n_bees: int = 30, max_iterations: int = 100):
        self.n_dolphins = n_dolphins
        self.n_bees = n_bees
        self.max_iterations = max_iterations
        self.detection_threshold = 0.5
        self.employed_bees = n_bees // 2
        self.onlooker_bees = n_bees // 2
    
    def _calculate_fitness(self, position: np.ndarray, features: np.ndarray) -> float:
        """Calculate fitness based on detection accuracy"""
        weights = np.abs(position) / (np.sum(np.abs(position)) + 1e-10)
        scores = np.dot(features, weights)
        
        # Enhanced fitness with variance consideration
        score_variance = np.var(scores)
        return np.mean(scores) + 0.3 * score_variance
    
    def _abc_phase(self, food_sources: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Artificial Bee Colony optimization phase"""
        n_features = food_sources.shape[1]
        fitness_values = []
        
        # Calculate fitness for all food sources
        for source in food_sources:
            fitness = self._calculate_fitness(source, features)
            fitness_values.append(fitness)
        
        fitness_values = np.array(fitness_values)
        
        # Employed bee phase
        for i in range(self.employed_bees):
            # Select random dimension and neighbor
            j = random.randint(0, n_features - 1)
            k = random.choice([x for x in range(self.employed_bees) if x != i])
            
            # Generate new solution
            phi = random.uniform(-1, 1)
            new_source = food_sources[i].copy()
            new_source[j] = food_sources[i][j] + phi * (food_sources[i][j] - food_sources[k][j])
            
            # Greedy selection
            new_fitness = self._calculate_fitness(new_source, features)
            if new_fitness > fitness_values[i]:
                food_sources[i] = new_source
                fitness_values[i] = new_fitness
        
        # Calculate probabilities for onlooker bees
        total_fitness = np.sum(fitness_values)
        if total_fitness > 0:
            probabilities = fitness_values / total_fitness
        else:
            probabilities = np.ones(len(fitness_values)) / len(fitness_values)
        
        # Onlooker bee phase
        for _ in range(self.onlooker_bees):
            # Select food source based on probability
            i = np.random.choice(self.employed_bees, p=probabilities)
            
            # Similar to employed bee phase
            j = random.randint(0, n_features - 1)
            k = random.choice([x for x in range(self.employed_bees) if x != i])
            
            phi = random.uniform(-1, 1)
            new_source = food_sources[i].copy()
            new_source[j] = food_sources[i][j] + phi * (food_sources[i][j] - food_sources[k][j])
            
            new_fitness = self._calculate_fitness(new_source, features)
            if new_fitness > fitness_values[i]:
                food_sources[i] = new_source
                fitness_values[i] = new_fitness
        
        return food_sources
    
    def detect_blackholes(self, features: np.ndarray) -> np.ndarray:
        """Detect black holes using hybrid DE-ABC algorithm"""
        n_features = features.shape[1]
        
        # Initialize populations
        dolphins = np.random.randn(self.n_dolphins, n_features)
        food_sources = np.random.randn(self.employed_bees, n_features)
        
        best_solution = None
        best_fitness = -np.inf
        
        for iteration in range(self.max_iterations):
            # Dolphin Echolocation phase
            dolphin_fitness = []
            for dolphin in dolphins:
                fitness = self._calculate_fitness(dolphin, features)
                dolphin_fitness.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = dolphin.copy()
            
            # Update dolphin positions
            for i in range(self.n_dolphins):
                frequency = np.random.uniform(0.1, 0.9)
                amplitude = np.random.uniform(0.5, 1.5)
                
                # Echolocation update
                dolphins[i] += frequency * (best_solution - dolphins[i]) * amplitude
                
                # Information exchange with ABC
                if i < self.employed_bees:
                    dolphins[i] += 0.2 * (food_sources[i] - dolphins[i])
                
                # Random exploration
                dolphins[i] += np.random.randn(n_features) * 0.05
            
            # ABC phase
            food_sources = self._abc_phase(food_sources, features)
            
            # Update best solution from ABC
            for source in food_sources:
                fitness = self._calculate_fitness(source, features)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = source.copy()
            
            # Migration: Exchange best solutions between populations
            if iteration % 10 == 0 and len(dolphin_fitness) > 0:
                worst_dolphin_idx = np.argmin(dolphin_fitness)
                source_fitness = [self._calculate_fitness(s, features) for s in food_sources]
                if len(source_fitness) > 0:
                    best_source_idx = np.argmax(source_fitness)
                    dolphins[worst_dolphin_idx] = food_sources[best_source_idx].copy()
        
        # Calculate detection scores using best solution
        weights = np.abs(best_solution) / (np.sum(np.abs(best_solution)) + 1e-10)
        detection_scores = np.dot(features, weights)
        
        # Dynamic threshold adjustment
        mean_score = np.mean(detection_scores)
        std_score = np.std(detection_scores)
        adaptive_threshold = mean_score + 0.5 * std_score
        
        predictions = (detection_scores > adaptive_threshold).astype(int)
        return predictions

class ImprovedHybridDEABCOptimizer:
    """Enhanced Hybrid Dolphin Echolocation and Artificial Bee Colony Algorithm"""
    
    def __init__(self, n_dolphins: int = 30, n_bees: int = 40, max_iterations: int = 150):
        self.n_dolphins = n_dolphins
        self.n_bees = n_bees
        self.max_iterations = max_iterations
        self.employed_bees = n_bees // 2
        self.onlooker_bees = n_bees // 2
        self.abandonment_limit = 10
        self.trial_counters = None
        self.scaler = StandardScaler()
        
    def _enhanced_features(self, features: np.ndarray) -> np.ndarray:
        """Extract enhanced features for better detection"""
        n_samples, n_features = features.shape
        enhanced = np.zeros((n_samples, n_features + 4))
        
        # Original features
        enhanced[:, :n_features] = features
        
        # Additional statistical features
        enhanced[:, n_features] = np.mean(features[:, :3], axis=1)  # Mean of first 3 features
        enhanced[:, n_features + 1] = np.std(features[:, :3], axis=1)   # Std of first 3 features
        enhanced[:, n_features + 2] = features[:, 0] * features[:, 5]   # Drop ratio * forward ratio
        enhanced[:, n_features + 3] = features[:, 2] * features[:, 4]   # Trust * neighbor trust
        
        return enhanced
    
    def _calculate_adaptive_fitness(self, position: np.ndarray, features: np.ndarray, 
                                   labels_pred: np.ndarray = None) -> float:
        """Enhanced fitness calculation with adaptive weighting"""
        weights = np.abs(position) / (np.sum(np.abs(position)) + 1e-10)
        scores = np.dot(features, weights)
        
        # Statistical measures
        score_mean = np.mean(scores)
        score_std = np.std(scores)
        score_skew = np.mean(((scores - score_mean) / (score_std + 1e-10)) ** 3)
        
        # Clustering penalty - penalize if predicted blackholes are too clustered
        if labels_pred is not None:
            blackhole_indices = np.where(labels_pred == 1)[0]
            if len(blackhole_indices) > 1:
                clustering_penalty = 1.0 / (np.std(blackhole_indices) + 1)
            else:
                clustering_penalty = 0
        else:
            clustering_penalty = 0
        
        # Combined fitness with multiple objectives
        fitness = (score_mean + 
                  0.3 * score_std + 
                  0.2 * abs(score_skew) - 
                  0.1 * clustering_penalty)
        
        return fitness
    
    def _levy_flight(self, position: np.ndarray, best_position: np.ndarray, 
                    iteration: int) -> np.ndarray:
        """Levy flight for exploration in dolphin movement"""
        beta = 1.5
        sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        
        u = np.random.randn(len(position)) * sigma
        v = np.random.randn(len(position))
        step = u / (np.abs(v) ** (1 / beta))
        
        # Adaptive step size
        step_size = 0.01 * np.exp(-iteration / self.max_iterations) * step
        
        new_position = position + step_size * (best_position - position)
        return new_position
    
    def _adaptive_abc_phase(self, food_sources: np.ndarray, features: np.ndarray, 
                           iteration: int) -> np.ndarray:
        """Enhanced ABC phase with adaptive mechanisms"""
        n_features = food_sources.shape[1]
        fitness_values = []
        
        # Calculate fitness for all food sources
        for source in food_sources:
            fitness = self._calculate_adaptive_fitness(source, features)
            fitness_values.append(fitness)
        
        fitness_values = np.array(fitness_values)
        
        # Employed bee phase with adaptive neighborhood
        for i in range(self.employed_bees):
            # Adaptive neighborhood size
            neighborhood_size = max(2, int(self.employed_bees * (1 - iteration / self.max_iterations)))
            neighbors = random.sample([x for x in range(self.employed_bees) if x != i], 
                                    min(neighborhood_size, self.employed_bees - 1))
            
            # Multi-dimensional update
            new_source = food_sources[i].copy()
            n_dims = random.randint(1, max(1, n_features // 2))
            dims = random.sample(range(n_features), n_dims)
            
            for j in dims:
                k = random.choice(neighbors)
                phi = random.uniform(-1, 1) * (1 - 0.5 * iteration / self.max_iterations)
                new_source[j] = food_sources[i][j] + phi * (food_sources[i][j] - food_sources[k][j])
            
            # Boundary handling
            new_source = np.clip(new_source, -3, 3)
            
            # Greedy selection with probabilistic acceptance
            new_fitness = self._calculate_adaptive_fitness(new_source, features)
            if new_fitness > fitness_values[i]:
                food_sources[i] = new_source
                fitness_values[i] = new_fitness
                self.trial_counters[i] = 0
            else:
                self.trial_counters[i] += 1
        
        # Calculate adaptive probabilities
        min_fitness = np.min(fitness_values)
        max_fitness = np.max(fitness_values)
        if max_fitness > min_fitness:
            normalized_fitness = (fitness_values - min_fitness) / (max_fitness - min_fitness)
            probabilities = 0.1 + 0.9 * normalized_fitness
            probabilities = probabilities / np.sum(probabilities)
        else:
            probabilities = np.ones(len(fitness_values)) / len(fitness_values)
        
        # Onlooker bee phase with tournament selection
        for _ in range(self.onlooker_bees):
            # Tournament selection
            tournament_size = 3
            candidates = np.random.choice(self.employed_bees, tournament_size, p=probabilities)
            i = candidates[np.argmax(fitness_values[candidates])]
            
            # Similar update as employed bees
            new_source = food_sources[i].copy()
            n_dims = random.randint(1, max(1, n_features // 2))
            dims = random.sample(range(n_features), n_dims)
            
            for j in dims:
                k = random.choice([x for x in range(self.employed_bees) if x != i])
                phi = random.uniform(-1, 1) * (1 - 0.5 * iteration / self.max_iterations)
                new_source[j] = food_sources[i][j] + phi * (food_sources[i][j] - food_sources[k][j])
            
            new_source = np.clip(new_source, -3, 3)
            new_fitness = self._calculate_adaptive_fitness(new_source, features)
            
            if new_fitness > fitness_values[i]:
                food_sources[i] = new_source
                fitness_values[i] = new_fitness
                self.trial_counters[i] = 0
        
        # Scout bee phase - abandon exhausted sources
        for i in range(self.employed_bees):
            if self.trial_counters[i] > self.abandonment_limit:
                # Generate new source using Levy flight from best source
                best_idx = np.argmax(fitness_values)
                food_sources[i] = self._levy_flight(food_sources[i], food_sources[best_idx], iteration)
                self.trial_counters[i] = 0
        
        return food_sources
    
    def _ensemble_decision(self, detection_scores: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Ensemble decision making with multiple detection strategies"""
        n_samples = len(detection_scores)
        predictions = np.zeros((n_samples, 3))
        
        # Strategy 1: Adaptive threshold based on score distribution
        mean_score = np.mean(detection_scores)
        std_score = np.std(detection_scores)
        adaptive_threshold = mean_score + 0.5 * std_score
        predictions[:, 0] = (detection_scores > adaptive_threshold).astype(int)
        
        # Strategy 2: Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.2, random_state=42)
        predictions[:, 1] = (iso_forest.fit_predict(features) == -1).astype(int)
        
        # Strategy 3: Percentile-based detection
        percentile_threshold = np.percentile(detection_scores, 75)
        predictions[:, 2] = (detection_scores > percentile_threshold).astype(int)
        
        # Weighted voting
        weights = [0.3, 0.3, 0.4]  # Weights for each strategy
        final_predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            weighted_vote = sum(predictions[i, j] * weights[j] for j in range(3))
            final_predictions[i] = 1 if weighted_vote > 0.5 else 0
        
        return final_predictions.astype(int)
    
    def detect_blackholes(self, features: np.ndarray) -> np.ndarray:
        """Main detection method with enhanced hybrid algorithm"""
        # Feature enhancement
        enhanced_features = self._enhanced_features(features)
        scaled_features = self.scaler.fit_transform(enhanced_features)
        
        n_features = scaled_features.shape[1]
        
        # Initialize populations
        dolphins = np.random.randn(self.n_dolphins, n_features) * 0.5
        food_sources = np.random.randn(self.employed_bees, n_features) * 0.5
        self.trial_counters = np.zeros(self.employed_bees)
        
        best_solution = None
        best_fitness = -np.inf
        fitness_history = []
        
        # Early stopping parameters
        patience = 20
        no_improvement_count = 0
        
        for iteration in range(self.max_iterations):
            # Dolphin Echolocation phase with Levy flight
            dolphin_fitness = []
            for i, dolphin in enumerate(dolphins):
                fitness = self._calculate_adaptive_fitness(dolphin, scaled_features)
                dolphin_fitness.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = dolphin.copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            
            # Update dolphin positions with adaptive mechanisms
            for i in range(self.n_dolphins):
                # Adaptive frequency and amplitude
                frequency = 0.1 + 0.8 * np.exp(-iteration / self.max_iterations)
                amplitude = 1.5 * (1 - iteration / self.max_iterations)
                
                # Echolocation with Levy flight
                if random.random() < 0.3:  # 30% chance of Levy flight
                    dolphins[i] = self._levy_flight(dolphins[i], best_solution, iteration)
                else:
                    # Standard echolocation
                    dolphins[i] += frequency * (best_solution - dolphins[i]) * amplitude
                
                # Information exchange with ABC
                if i < self.employed_bees:
                    exchange_rate = 0.3 * (1 - iteration / self.max_iterations)
                    dolphins[i] += exchange_rate * (food_sources[i] - dolphins[i])
                
                # Mutation for diversity
                if random.random() < 0.1:
                    mutation_idx = random.randint(0, n_features - 1)
                    dolphins[i][mutation_idx] += np.random.randn() * 0.1
                
                # Boundary handling
                dolphins[i] = np.clip(dolphins[i], -3, 3)
            
            # Enhanced ABC phase
            food_sources = self._adaptive_abc_phase(food_sources, scaled_features, iteration)
            
            # Update best solution from ABC
            for source in food_sources:
                fitness = self._calculate_adaptive_fitness(source, scaled_features)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = source.copy()
                    no_improvement_count = 0
            
            # Population diversity maintenance
            if iteration % 20 == 0:
                # Replace worst performers with new random solutions
                all_solutions = np.vstack([dolphins, food_sources])
                all_fitness = [self._calculate_adaptive_fitness(s, scaled_features) for s in all_solutions]
                worst_indices = np.argsort(all_fitness)[:5]
                
                for idx in worst_indices:
                    if idx < self.n_dolphins:
                        dolphins[idx] = np.random.randn(n_features) * 0.5
                    else:
                        food_idx = idx - self.n_dolphins
                        if food_idx < len(food_sources):
                            food_sources[food_idx] = np.random.randn(n_features) * 0.5
            
            fitness_history.append(best_fitness)
            
            # Early stopping
            if no_improvement_count > patience:
                break
        
        # Final detection with ensemble decision
        weights = np.abs(best_solution) / (np.sum(np.abs(best_solution)) + 1e-10)
        detection_scores = np.dot(scaled_features, weights)
        
        # Use ensemble decision making
        predictions = self._ensemble_decision(detection_scores, scaled_features)
        
        return predictions

def run_comprehensive_simulation():
    """Run comprehensive simulation with multiple scenarios"""
    print("=== MANET Black Hole Attack Detection Simulation ===\n")
    
    # Test scenarios
    scenarios = [
        {"name": "Small Network", "nodes": 30, "area": 100, "range": 30, "blackhole_pct": 0.2},
        {"name": "Medium Network", "nodes": 50, "area": 150, "range": 35, "blackhole_pct": 0.25},
        {"name": "Large Network", "nodes": 100, "area": 200, "range": 40, "blackhole_pct": 0.15},
        {"name": "Dense Network", "nodes": 80, "area": 100, "range": 40, "blackhole_pct": 0.3},
        {"name": "Sparse Network", "nodes": 60, "area": 300, "range": 50, "blackhole_pct": 0.2}
    ]
    
    results = defaultdict(list)
    
    for scenario in scenarios:
        print(f"\n--- Testing {scenario['name']} ---")
        print(f"Nodes: {scenario['nodes']}, Area: {scenario['area']}x{scenario['area']}, "
              f"Range: {scenario['range']}, Black holes: {scenario['blackhole_pct']*100}%")
        
        # Create MANET
        manet = MANETSimulator(
            num_nodes=scenario['nodes'],
            area_size=scenario['area'],
            transmission_range=scenario['range'],
            blackhole_percentage=scenario['blackhole_pct']
        )
        
        # Simulate network traffic
        print("Simulating network traffic...")
        n_transmissions = scenario['nodes'] * 20
        successful_transmissions = 0
        
        for _ in range(n_transmissions):
            src = random.randint(0, scenario['nodes'] - 1)
            dst = random.randint(0, scenario['nodes'] - 1)
            if src != dst:
                if manet.simulate_packet_transmission(src, dst):
                    successful_transmissions += 1
        
        print(f"Successful transmissions: {successful_transmissions}/{n_transmissions}")
        
        # Collect features
        features = manet.collect_node_features()
        
        # Ground truth
        y_true = np.array([1 if node.is_blackhole else 0 for node in manet.nodes])
        
        # Test algorithms
        algorithms = [
            ("Dolphin Echolocation", DolphinEcholocationOptimizer()),
            ("Hybrid DE-ABC", HybridDEABCOptimizer()),
            ("Improved Hybrid DE-ABC", ImprovedHybridDEABCOptimizer())
        ]
        
        for algo_name, algorithm in algorithms:
            print(f"\nTesting {algo_name}...")
            
            # Time the detection
            start_time = time.time()
            y_pred = algorithm.detect_blackholes(features)
            detection_time = time.time() - start_time
            
            # Calculate metrics
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            results[algo_name].append({
                'scenario': scenario['name'],
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'detection_time': detection_time,
                'true_positives': np.sum((y_true == 1) & (y_pred == 1)),
                'false_positives': np.sum((y_true == 0) & (y_pred == 1)),
                'false_negatives': np.sum((y_true == 1) & (y_pred == 0)),
                'true_negatives': np.sum((y_true == 0) & (y_pred == 0))
            })
            
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"Detection Time: {detection_time:.4f}s")
    
    # Aggregate results
    print("\n=== OVERALL RESULTS ===\n")
    
    for algo_name in ["Dolphin Echolocation", "Hybrid DE-ABC", "Improved Hybrid DE-ABC"]:
        algo_results = results[algo_name]
        
        avg_precision = np.mean([r['precision'] for r in algo_results])
        avg_recall = np.mean([r['recall'] for r in algo_results])
        avg_f1 = np.mean([r['f1_score'] for r in algo_results])
        avg_time = np.mean([r['detection_time'] for r in algo_results])
        
        print(f"{algo_name}:")
        print(f"  Average Precision: {avg_precision:.4f}")
        print(f"  Average Recall: {avg_recall:.4f}")
        print(f"  Average F1-Score: {avg_f1:.4f}")
        print(f"  Average Detection Time: {avg_time:.4f}s")
        print()
    
    # Comparison visualization
    visualize_results(results)
    
    return results

def visualize_results(results: Dict):
    """Visualize the comparison results"""
    # First figure: Metrics comparison
    plt.figure(figsize=(20, 12))
    
    # Prepare data for visualization
    metrics = ['precision', 'recall', 'f1_score']
    scenarios = [r['scenario'] for r in results['Dolphin Echolocation']]
    
    # Create subplots for each metric
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        
        de_values = [r[metric] for r in results['Dolphin Echolocation']]
        hybrid_values = [r[metric] for r in results['Hybrid DE-ABC']]
        improved_values = [r[metric] for r in results['Improved Hybrid DE-ABC']]
        
        x = np.arange(len(scenarios))
        width = 0.25
        
        plt.bar(x - width, de_values, width, label='Dolphin Echolocation', alpha=0.8, color='blue')
        plt.bar(x, hybrid_values, width, label='Hybrid DE-ABC', alpha=0.8, color='orange')
        plt.bar(x + width, improved_values, width, label='Improved Hybrid DE-ABC', alpha=0.8, color='green')
        
        plt.xlabel('Scenario')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} Comparison')
        plt.xticks(x, scenarios, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
    
    # Detection time comparison
    plt.subplot(2, 2, 4)
    de_times = [r['detection_time'] for r in results['Dolphin Echolocation']]
    hybrid_times = [r['detection_time'] for r in results['Hybrid DE-ABC']]
    improved_times = [r['detection_time'] for r in results['Improved Hybrid DE-ABC']]
    
    plt.bar(x - width, de_times, width, label='Dolphin Echolocation', alpha=0.8, color='blue')
    plt.bar(x, hybrid_times, width, label='Hybrid DE-ABC', alpha=0.8, color='orange')
    plt.bar(x + width, improved_times, width, label='Improved Hybrid DE-ABC', alpha=0.8, color='green')
    
    plt.xlabel('Scenario')
    plt.ylabel('Detection Time (seconds)')
    plt.title('Detection Time Comparison')
    plt.xticks(x, scenarios, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Second figure: Average performance radar chart and box plot
    fig = plt.figure(figsize=(16, 8))
    
    # Create radar chart subplot
    ax1 = plt.subplot(1, 2, 1, projection='polar')
    
    # Radar chart for performance metrics
    metrics_avg = ['Avg Precision', 'Avg Recall', 'Avg F1-Score']
    
    de_avg = [
        np.mean([r['precision'] for r in results['Dolphin Echolocation']]),
        np.mean([r['recall'] for r in results['Dolphin Echolocation']]),
        np.mean([r['f1_score'] for r in results['Dolphin Echolocation']])
    ]
    
    hybrid_avg = [
        np.mean([r['precision'] for r in results['Hybrid DE-ABC']]),
        np.mean([r['recall'] for r in results['Hybrid DE-ABC']]),
        np.mean([r['f1_score'] for r in results['Hybrid DE-ABC']])
    ]
    
    improved_avg = [
        np.mean([r['precision'] for r in results['Improved Hybrid DE-ABC']]),
        np.mean([r['recall'] for r in results['Improved Hybrid DE-ABC']]),
        np.mean([r['f1_score'] for r in results['Improved Hybrid DE-ABC']])
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics_avg), endpoint=False).tolist()
    de_avg += de_avg[:1]
    hybrid_avg += hybrid_avg[:1]
    improved_avg += improved_avg[:1]
    angles += angles[:1]
    
    ax1.plot(angles, de_avg, 'o-', linewidth=2, label='Dolphin Echolocation', color='blue')
    ax1.fill(angles, de_avg, alpha=0.25, color='blue')
    ax1.plot(angles, hybrid_avg, 's-', linewidth=2, label='Hybrid DE-ABC', color='orange')
    ax1.fill(angles, hybrid_avg, alpha=0.25, color='orange')
    ax1.plot(angles, improved_avg, '^-', linewidth=2, label='Improved Hybrid DE-ABC', color='green')
    ax1.fill(angles, improved_avg, alpha=0.25, color='green')
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics_avg)
    ax1.set_ylim(0, 1)
    ax1.set_title('Average Performance Comparison', y=1.08)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax1.grid(True)
    
    # Box plot for F1-scores
    ax2 = plt.subplot(1, 2, 2)
    f1_data = [
        [r['f1_score'] for r in results['Dolphin Echolocation']],
        [r['f1_score'] for r in results['Hybrid DE-ABC']],
        [r['f1_score'] for r in results['Improved Hybrid DE-ABC']]
    ]
    
    bp = ax2.boxplot(f1_data, tick_labels=['DE', 'Hybrid', 'Improved'], patch_artist=True)
    colors = ['blue', 'orange', 'green']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax2.set_ylabel('F1-Score')
    ax2.set_title('F1-Score Distribution')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Third figure: Confusion matrices
    # fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # for idx, (algo_name, ax) in enumerate(zip(['Dolphin Echolocation', 'Hybrid DE-ABC', 'Improved Hybrid DE-ABC'], axes)):
    #     # Calculate total confusion matrix
    #     total_tp = sum(r['true_positives'] for r in results[algo_name])
    #     total_fp = sum(r['false_positives'] for r in results[algo_name])
    #     total_fn = sum(r['false_negatives'] for r in results[algo_name])
    #     total_tn = sum(r['true_negatives'] for r in results[algo_name])
        
    #     cm = np.array([[total_tn, total_fp], [total_fn, total_tp]])
        
    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
    #                xticklabels=['Normal', 'Black Hole'],
    #                yticklabels=['Normal', 'Black Hole'])
    #     ax.set_xlabel('Predicted')
    #     ax.set_ylabel('Actual')
    #     ax.set_title(f'{algo_name}\nConfusion Matrix')
    
    # plt.tight_layout()
    # plt.show()

def run_statistical_analysis(results: Dict):
    """Perform statistical analysis on results"""
    print("\n=== Statistical Analysis ===")
    
    # Compare improvements
    de_f1_scores = [r['f1_score'] for r in results['Dolphin Echolocation']]
    hybrid_f1_scores = [r['f1_score'] for r in results['Hybrid DE-ABC']]
    improved_f1_scores = [r['f1_score'] for r in results['Improved Hybrid DE-ABC']]
    
    # Calculate improvements
    hybrid_improvement = (np.mean(hybrid_f1_scores) - np.mean(de_f1_scores)) / np.mean(de_f1_scores) * 100
    improved_improvement = (np.mean(improved_f1_scores) - np.mean(de_f1_scores)) / np.mean(de_f1_scores) * 100
    improved_vs_hybrid = (np.mean(improved_f1_scores) - np.mean(hybrid_f1_scores)) / np.mean(hybrid_f1_scores) * 100
    
    print(f"\nHybrid DE-ABC shows {hybrid_improvement:.2f}% improvement over basic Dolphin Echolocation")
    print(f"Improved Hybrid DE-ABC shows {improved_improvement:.2f}% improvement over basic Dolphin Echolocation")
    print(f"Improved Hybrid DE-ABC shows {improved_vs_hybrid:.2f}% improvement over original Hybrid DE-ABC")
    
    # Performance summary
    print("\n=== Performance Summary ===")
    for algo_name in ["Dolphin Echolocation", "Hybrid DE-ABC", "Improved Hybrid DE-ABC"]:
        algo_results = results[algo_name]
        total_tp = sum(r['true_positives'] for r in algo_results)
        total_fp = sum(r['false_positives'] for r in algo_results)
        total_fn = sum(r['false_negatives'] for r in algo_results)
        total_tn = sum(r['true_negatives'] for r in algo_results)
        
        accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn)
        
        print(f"\n{algo_name}:")
        print(f"  Total True Positives: {total_tp}")
        print(f"  Total False Positives: {total_fp}")
        print(f"  Total False Negatives: {total_fn}")
        print(f"  Total True Negatives: {total_tn}")
        print(f"  Overall Accuracy: {accuracy:.4f}")
        
        # Additional metrics
        if total_tp + total_fp > 0:
            precision = total_tp / (total_tp + total_fp)
        else:
            precision = 0
            
        if total_tp + total_fn > 0:
            recall = total_tp / (total_tp + total_fn)
        else:
            recall = 0
            
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
            
        print(f"  Overall Precision: {precision:.4f}")
        print(f"  Overall Recall: {recall:.4f}")
        print(f"  Overall F1-Score: {f1:.4f}")

# Run the simulation
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Run comprehensive simulation
    results = run_comprehensive_simulation()
    
    # Run statistical analysis
    run_statistical_analysis(results)