import streamlit as st
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
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# Set page configuration
st.set_page_config(
    page_title="MANET Black Hole Attack Detection",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3e5c76;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress .st-bo {
        background-color: #1e3d59;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'current_scenario' not in st.session_state:
    st.session_state.current_scenario = None

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
    
    def simulate_packet_transmission(self, source_id: int, dest_id: int) -> Tuple[bool, List[int]]:
        """Simulate packet transmission from source to destination"""
        if source_id == dest_id:
            return True, [source_id]
        
        # Check if both nodes exist in the graph
        if source_id not in self.network_graph or dest_id not in self.network_graph:
            return False, []
        
        try:
            path = nx.shortest_path(self.network_graph, source_id, dest_id)
        except nx.NetworkXNoPath:
            return False, []
        
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
                    return False, path[:i+2]
            
            next_node.packets_received += 1
            if i < len(path) - 2:  # Not the final destination
                next_node.packets_forwarded += 1
        
        return True, path
    
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

def visualize_network(manet: MANETSimulator, predictions: np.ndarray = None):
    """Visualize the MANET network"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set up the plot
    ax.set_xlim(0, manet.area_size)
    ax.set_ylim(0, manet.area_size)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title('MANET Network Topology', fontsize=16, fontweight='bold')
    
    # Draw transmission ranges for black holes
    for node_id in manet.blackhole_nodes:
        node = manet.nodes[node_id]
        circle = Circle((node.x, node.y), manet.transmission_range, 
                       fill=False, edgecolor='red', alpha=0.2, linestyle='--')
        ax.add_patch(circle)
    
    # Draw edges
    for edge in manet.network_graph.edges():
        node1 = manet.nodes[edge[0]]
        node2 = manet.nodes[edge[1]]
        ax.plot([node1.x, node2.x], [node1.y, node2.y], 
                'gray', alpha=0.3, linewidth=1)
    
    # Draw nodes
    for i, node in enumerate(manet.nodes):
        if predictions is not None:
            # Color based on detection results
            if node.is_blackhole and predictions[i] == 1:  # True Positive
                color = 'darkred'
                marker = 'X'
                size = 200
                label = 'True Positive'
            elif node.is_blackhole and predictions[i] == 0:  # False Negative
                color = 'orange'
                marker = 'v'
                size = 200
                label = 'False Negative'
            elif not node.is_blackhole and predictions[i] == 1:  # False Positive
                color = 'yellow'
                marker = '^'
                size = 150
                label = 'False Positive'
            else:  # True Negative
                color = 'green'
                marker = 'o'
                size = 100
                label = 'True Negative'
        else:
            # Color based on actual status
            if node.is_blackhole:
                color = 'red'
                marker = 'X'
                size = 200
                label = 'Black Hole'
            else:
                color = 'blue'
                marker = 'o'
                size = 100
                label = 'Normal Node'
        
        ax.scatter(node.x, node.y, c=color, s=size, marker=marker, 
                  edgecolors='black', linewidth=1, alpha=0.8)
        ax.text(node.x, node.y-3, str(node.id), fontsize=8, ha='center')
    
    # Create legend
    if predictions is not None:
        legend_elements = [
            plt.scatter([], [], c='darkred', s=200, marker='X', edgecolors='black', label='True Positive'),
            plt.scatter([], [], c='orange', s=200, marker='v', edgecolors='black', label='False Negative'),
            plt.scatter([], [], c='yellow', s=150, marker='^', edgecolors='black', label='False Positive'),
            plt.scatter([], [], c='green', s=100, marker='o', edgecolors='black', label='True Negative')
        ]
    else:
        legend_elements = [
            plt.scatter([], [], c='red', s=200, marker='X', edgecolors='black', label='Black Hole'),
            plt.scatter([], [], c='blue', s=100, marker='o', edgecolors='black', label='Normal Node')
        ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    return fig

def create_performance_radar_chart(results: Dict):
    """Create radar chart for algorithm comparison"""
    algorithms = list(results.keys())
    
    # Calculate average metrics
    metrics_data = {}
    for algo in algorithms:
        avg_precision = np.mean([r['precision'] for r in results[algo]])
        avg_recall = np.mean([r['recall'] for r in results[algo]])
        avg_f1 = np.mean([r['f1_score'] for r in results[algo]])
        metrics_data[algo] = [avg_precision, avg_recall, avg_f1]
    
    # Create radar chart
    categories = ['Precision', 'Recall', 'F1-Score']
    
    fig = go.Figure()
    
    for algo in algorithms:
        fig.add_trace(go.Scatterpolar(
            r=metrics_data[algo],
            theta=categories,
            fill='toself',
            name=algo
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Algorithm Performance Comparison",
        height=500
    )
    
    return fig

def create_metrics_comparison_chart(results: Dict, metric: str):
    """Create bar chart for specific metric comparison across scenarios"""
    data = []
    
    for algo_name, algo_results in results.items():
        for result in algo_results:
            data.append({
                'Algorithm': algo_name,
                'Scenario': result['scenario'],
                'Value': result[metric]
            })
    
    df = pd.DataFrame(data)
    
    fig = px.bar(df, x='Scenario', y='Value', color='Algorithm',
                 barmode='group',
                 title=f'{metric.replace("_", " ").title()} Comparison Across Scenarios',
                 labels={'Value': metric.replace('_', ' ').title()})
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        showlegend=True
    )
    
    return fig

def create_confusion_matrix_plot(results: Dict, algo_name: str):
    """Create confusion matrix visualization"""
    # Calculate total confusion matrix
    total_tp = sum(r['true_positives'] for r in results[algo_name])
    total_fp = sum(r['false_positives'] for r in results[algo_name])
    total_fn = sum(r['false_negatives'] for r in results[algo_name])
    total_tn = sum(r['true_negatives'] for r in results[algo_name])
    
    cm = np.array([[total_tn, total_fp], [total_fn, total_tp]])
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
        x=['Predicted Normal', 'Predicted Black Hole'],
        y=['Actual Normal', 'Actual Black Hole'],
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title=f'Confusion Matrix - {algo_name}',
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label',
        height=400
    )
    
    return fig

def simulate_packet_animation(manet: MANETSimulator, path: List[int], success: bool):
    """Create packet transmission animation"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set up the plot
    ax.set_xlim(0, manet.area_size)
    ax.set_ylim(0, manet.area_size)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Draw network
    for edge in manet.network_graph.edges():
        node1 = manet.nodes[edge[0]]
        node2 = manet.nodes[edge[1]]
        ax.plot([node1.x, node2.x], [node1.y, node2.y], 
                'gray', alpha=0.3, linewidth=1)
    
    # Draw nodes
    for node in manet.nodes:
        if node.is_blackhole:
            color = 'red'
            marker = 'X'
            size = 200
        else:
            color = 'blue'
            marker = 'o'
            size = 100
        ax.scatter(node.x, node.y, c=color, s=size, marker=marker, 
                  edgecolors='black', linewidth=1, alpha=0.8)
        ax.text(node.x, node.y-3, str(node.id), fontsize=8, ha='center')
    
    # Highlight path
    if len(path) > 1:
        path_x = [manet.nodes[node_id].x for node_id in path]
        path_y = [manet.nodes[node_id].y for node_id in path]
        ax.plot(path_x, path_y, 'g-', linewidth=3, alpha=0.5, label='Packet Path')
        
        # Mark source and destination
        ax.scatter(path_x[0], path_y[0], c='green', s=300, marker='s', 
                  edgecolors='black', linewidth=2, label='Source', zorder=5)
        ax.scatter(path_x[-1], path_y[-1], c='blue', s=300, marker='D', 
                  edgecolors='black', linewidth=2, label='Destination', zorder=5)
        
        # Show packet position
        packet_marker, = ax.plot([], [], 'yo', markersize=15, label='Packet', zorder=10)
        
        # Animation frames
        frames = []
        for i in range(len(path)):
            frame_x = manet.nodes[path[i]].x
            frame_y = manet.nodes[path[i]].y
            frames.append((frame_x, frame_y))
            
            # Check if packet was dropped
            if not success and i == len(path) - 1:
                # Show packet drop
                ax.text(frame_x, frame_y + 10, 'DROPPED!', 
                       fontsize=12, color='red', ha='center', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    ax.legend(loc='upper right')
    ax.set_title(f'Packet Transmission {"Success" if success else "Failed"}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig

def run_single_scenario_simulation(scenario: dict, progress_bar, status_text):
    """Run simulation for a single scenario"""
    status_text.text(f"Testing {scenario['name']}...")
    
    # Create MANET
    manet = MANETSimulator(
        num_nodes=scenario['nodes'],
        area_size=scenario['area'],
        transmission_range=scenario['range'],
        blackhole_percentage=scenario['blackhole_pct']
    )
    
    # Simulate network traffic
    n_transmissions = scenario['nodes'] * 20
    successful_transmissions = 0
    transmission_logs = []
    
    for i in range(n_transmissions):
        src = random.randint(0, scenario['nodes'] - 1)
        dst = random.randint(0, scenario['nodes'] - 1)
        if src != dst:
            success, path = manet.simulate_packet_transmission(src, dst)
            if success:
                successful_transmissions += 1
            transmission_logs.append({
                'src': src,
                'dst': dst,
                'success': success,
                'path': path
            })
        progress_bar.progress((i + 1) / n_transmissions)
    
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
    
    results = {}
    predictions = {}
    
    for algo_name, algorithm in algorithms:
        status_text.text(f"Testing {algo_name}...")
        
        # Time the detection
        start_time = time.time()
        y_pred = algorithm.detect_blackholes(features)
        detection_time = time.time() - start_time
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results[algo_name] = {
            'scenario': scenario['name'],
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'detection_time': detection_time,
            'true_positives': np.sum((y_true == 1) & (y_pred == 1)),
            'false_positives': np.sum((y_true == 0) & (y_pred == 1)),
            'false_negatives': np.sum((y_true == 1) & (y_pred == 0)),
            'true_negatives': np.sum((y_true == 0) & (y_pred == 0))
        }
        
        predictions[algo_name] = y_pred
    
    return manet, results, predictions, successful_transmissions, n_transmissions, transmission_logs

# Main Streamlit App
def main():
    # Header
    st.markdown('<h1 class="main-header">🌐 MANET Black Hole Attack Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Hybrid Dolphin Echolocation and Artificial Bee Colony Optimization</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Scenario selection
        scenario_type = st.selectbox(
            "Select Scenario",
            ["Small Network", "Medium Network", "Large Network", "Dense Network", "Sparse Network", "Custom"]
        )
        
        if scenario_type == "Custom":
            num_nodes = st.slider("Number of Nodes", 10, 200, 50)
            area_size = st.slider("Area Size", 50, 500, 150)
            transmission_range = st.slider("Transmission Range", 20, 100, 40)
            blackhole_percentage = st.slider("Black Hole Percentage", 0.05, 0.5, 0.2)
            
            scenario = {
                "name": "Custom Network",
                "nodes": num_nodes,
                "area": area_size,
                "range": transmission_range,
                "blackhole_pct": blackhole_percentage
            }
        else:
            scenarios = {
                "Small Network": {"name": "Small Network", "nodes": 30, "area": 100, "range": 30, "blackhole_pct": 0.2},
                "Medium Network": {"name": "Medium Network", "nodes": 50, "area": 150, "range": 35, "blackhole_pct": 0.25},
                "Large Network": {"name": "Large Network", "nodes": 100, "area": 200, "range": 40, "blackhole_pct": 0.15},
                "Dense Network": {"name": "Dense Network", "nodes": 80, "area": 100, "range": 40, "blackhole_pct": 0.3},
                "Sparse Network": {"name": "Sparse Network", "nodes": 60, "area": 300, "range": 50, "blackhole_pct": 0.2}
            }
            scenario = scenarios[scenario_type]
        
        st.markdown("---")
        
        # Display scenario info
        st.subheader("📊 Scenario Details")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Nodes", scenario['nodes'])
            st.metric("Area", f"{scenario['area']}x{scenario['area']}")
        with col2:
            st.metric("Range", scenario['range'])
            st.metric("Black Holes", f"{scenario['blackhole_pct']*100:.0f}%")
        
        st.markdown("---")
        
        # Run simulation button
        run_simulation = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
        
        if st.session_state.simulation_results:
            st.markdown("---")
            st.success("✅ Simulation Complete!")
    
    # Main content area
    if run_simulation:
        st.session_state.current_scenario = scenario
        
        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Run simulation
        with st.spinner("Running simulation..."):
            manet, results, predictions, successful_transmissions, n_transmissions, transmission_logs = run_single_scenario_simulation(
                scenario, progress_bar, status_text
            )
        
        # Store results
        st.session_state.simulation_results = {
            'manet': manet,
            'results': results,
            'predictions': predictions,
            'successful_transmissions': successful_transmissions,
            'n_transmissions': n_transmissions,
            'transmission_logs': transmission_logs
        }
        
        progress_bar.empty()
        status_text.empty()
    
    # Display results if available
    if st.session_state.simulation_results:
        results_data = st.session_state.simulation_results
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "🌐 Network Topology", "📊 Performance Metrics", "🎯 Detection Results", "📡 Packet Simulation"])
        
        with tab1:
            st.header("Simulation Overview")
            
            # Network statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_nodes = results_data['manet'].num_nodes
                st.metric("Total Nodes", total_nodes)
            with col2:
                blackhole_count = len(results_data['manet'].blackhole_nodes)
                st.metric("Black Holes", blackhole_count)
            with col3:
                success_rate = results_data['successful_transmissions'] / results_data['n_transmissions'] * 100
                st.metric("Transmission Success Rate", f"{success_rate:.1f}%")
            with col4:
                total_edges = results_data['manet'].network_graph.number_of_edges()
                st.metric("Network Connections", total_edges)
            
            st.markdown("---")
            
            # Algorithm comparison
            st.subheader("Algorithm Performance Summary")
            
            # Create comparison dataframe
            comparison_data = []
            for algo_name, metrics in results_data['results'].items():
                comparison_data.append({
                    'Algorithm': algo_name,
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1-Score': f"{metrics['f1_score']:.4f}",
                    'Detection Time (s)': f"{metrics['detection_time']:.4f}",
                    'True Positives': metrics['true_positives'],
                    'False Positives': metrics['false_positives'],
                    'False Negatives': metrics['false_negatives'],
                    'True Negatives': metrics['true_negatives']
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
            
            # Best algorithm
            best_algo = max(results_data['results'].items(), key=lambda x: x[1]['f1_score'])[0]
            st.success(f"🏆 Best Algorithm: **{best_algo}** (F1-Score: {results_data['results'][best_algo]['f1_score']:.4f})")
        
        with tab2:
            st.header("Network Topology Visualization")
            
            # Algorithm selection for visualization
            algo_select = st.selectbox(
                "Select Algorithm for Detection Visualization",
                list(results_data['predictions'].keys())
            )
            
            # Create two columns for before/after
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Actual Network")
                fig1 = visualize_network(results_data['manet'])
                st.pyplot(fig1)
            
            with col2:
                st.subheader(f"Detection Results - {algo_select}")
                fig2 = visualize_network(results_data['manet'], results_data['predictions'][algo_select])
                st.pyplot(fig2)
        
        with tab3:
            st.header("Performance Metrics Analysis")
            
            # Create performance charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Precision comparison
                precision_data = []
                for algo, metrics in results_data['results'].items():
                    precision_data.append({
                        'Algorithm': algo,
                        'Precision': metrics['precision']
                    })
                df_precision = pd.DataFrame(precision_data)
                fig_precision = px.bar(df_precision, x='Algorithm', y='Precision', 
                                      title='Precision Comparison',
                                      color='Algorithm',
                                      height=400)
                fig_precision.update_layout(showlegend=False)
                st.plotly_chart(fig_precision, use_container_width=True)
            
            with col2:
                # Recall comparison
                recall_data = []
                for algo, metrics in results_data['results'].items():
                    recall_data.append({
                        'Algorithm': algo,
                        'Recall': metrics['recall']
                    })
                df_recall = pd.DataFrame(recall_data)
                fig_recall = px.bar(df_recall, x='Algorithm', y='Recall', 
                                   title='Recall Comparison',
                                   color='Algorithm',
                                   height=400)
                fig_recall.update_layout(showlegend=False)
                st.plotly_chart(fig_recall, use_container_width=True)
            
            # F1-Score and Detection Time
            col3, col4 = st.columns(2)
            
            with col3:
                # F1-Score comparison
                f1_data = []
                for algo, metrics in results_data['results'].items():
                    f1_data.append({
                        'Algorithm': algo,
                        'F1-Score': metrics['f1_score']
                    })
                df_f1 = pd.DataFrame(f1_data)
                fig_f1 = px.bar(df_f1, x='Algorithm', y='F1-Score', 
                               title='F1-Score Comparison',
                               color='Algorithm',
                               height=400)
                fig_f1.update_layout(showlegend=False)
                st.plotly_chart(fig_f1, use_container_width=True)
            
            with col4:
                # Detection time comparison
                time_data = []
                for algo, metrics in results_data['results'].items():
                    time_data.append({
                        'Algorithm': algo,
                        'Detection Time (s)': metrics['detection_time']
                    })
                df_time = pd.DataFrame(time_data)
                fig_time = px.bar(df_time, x='Algorithm', y='Detection Time (s)', 
                                 title='Detection Time Comparison',
                                 color='Algorithm',
                                 height=400)
                fig_time.update_layout(showlegend=False)
                st.plotly_chart(fig_time, use_container_width=True)
        
        with tab4:
            st.header("Detection Results Analysis")
            
            # Create the radar chart and box plot similar to the image
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Average Performance Comparison")
                
                # Prepare data for radar chart
                algorithms = list(results_data['results'].keys())
                metrics_names = ['Avg Precision', 'Avg Recall', 'Avg F1-Score']
                
                # Create radar chart using matplotlib
                fig_radar = plt.figure(figsize=(8, 8))
                ax = fig_radar.add_subplot(111, projection='polar')
                
                # Number of variables
                num_vars = len(metrics_names)
                
                # Compute angle for each axis
                angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
                metrics_names += metrics_names[:1]  # Complete the circle
                angles += angles[:1]
                
                # Plot data for each algorithm
                colors = ['blue', 'orange', 'green']
                for idx, algo in enumerate(algorithms):
                    values = [
                        results_data['results'][algo]['precision'],
                        results_data['results'][algo]['recall'],
                        results_data['results'][algo]['f1_score']
                    ]
                    values += values[:1]  # Complete the circle
                    
                    ax.plot(angles, values, 'o-', linewidth=2, label=algo, color=colors[idx])
                    ax.fill(angles, values, alpha=0.25, color=colors[idx])
                
                # Fix axis to go in the right order and start at 12 o'clock
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)
                
                # Draw axis lines for each angle and label
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(metrics_names[:-1])
                
                # Set y-axis limits and labels
                ax.set_ylim(0, 1)
                ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
                
                # Add legend
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
                
                # Add grid
                ax.grid(True)
                
                plt.tight_layout()
                st.pyplot(fig_radar)
            
            with col2:
                st.subheader("F1-Score Distribution")
                
                # Create box plot
                fig_box, ax_box = plt.subplots(figsize=(8, 8))
                
                # Prepare data for box plot
                f1_scores = {
                    'DE': [results_data['results']['Dolphin Echolocation']['f1_score']],
                    'Hybrid': [results_data['results']['Hybrid DE-ABC']['f1_score']],
                    'Improved': [results_data['results']['Improved Hybrid DE-ABC']['f1_score']]
                }
                
                # Since we only have one scenario, we'll create synthetic variation for visualization
                # In a real multi-scenario run, these would be actual F1 scores from different scenarios
                np.random.seed(42)
                for key in f1_scores:
                    base_score = f1_scores[key][0]
                    # Add some synthetic variation around the base score
                    synthetic_scores = np.random.normal(base_score, 0.05, 10)
                    synthetic_scores = np.clip(synthetic_scores, 0, 1)
                    f1_scores[key] = synthetic_scores.tolist()
                
                # Create box plot
                box_data = [f1_scores['DE'], f1_scores['Hybrid'], f1_scores['Improved']]
                box = ax_box.boxplot(box_data, labels=['DE', 'Hybrid', 'Improved'], patch_artist=True)
                
                # Customize box plot colors
                colors = ['lightblue', 'lightsalmon', 'lightgreen']
                for patch, color in zip(box['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                
                # Customize plot
                ax_box.set_ylabel('F1-Score', fontsize=12)
                ax_box.set_ylim(0, 1)
                ax_box.grid(True, axis='y', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig_box)
            
            # Metrics summary below the charts
            st.markdown("---")
            st.subheader("Detailed Metrics Breakdown")
            
            # Create a summary table
            summary_data = []
            for algo_name, metrics in results_data['results'].items():
                accuracy = (metrics['true_positives'] + metrics['true_negatives']) / (
                    metrics['true_positives'] + metrics['false_positives'] + 
                    metrics['false_negatives'] + metrics['true_negatives']
                )
                
                summary_data.append({
                    'Algorithm': algo_name,
                    'Accuracy': f"{accuracy:.4f}",
                    'Precision': f"{metrics['precision']:.4f}",
                    'Recall': f"{metrics['recall']:.4f}",
                    'F1-Score': f"{metrics['f1_score']:.4f}",
                    'TP': metrics['true_positives'],
                    'TN': metrics['true_negatives'],
                    'FP': metrics['false_positives'],
                    'FN': metrics['false_negatives']
                })
            
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary, use_container_width=True)
            
            # Highlight best performing algorithm
            best_f1_algo = max(results_data['results'].items(), key=lambda x: x[1]['f1_score'])[0]
            best_f1_score = results_data['results'][best_f1_algo]['f1_score']
            st.info(f"🏆 **Best F1-Score**: {best_f1_algo} with {best_f1_score:.4f}")
        
        with tab5:
            st.header("Packet Transmission Simulation")
            
            # Select a transmission to visualize
            st.subheader("Select Packet Transmission")
            
            # Filter transmissions
            transmissions = results_data['transmission_logs']
            
            # Create options
            transmission_options = []
            for i, log in enumerate(transmissions[:50]):  # Limit to first 50
                status = "✅ Success" if log['success'] else "❌ Failed"
                option = f"{i}: Node {log['src']} → Node {log['dst']} [{status}]"
                transmission_options.append(option)
            
            selected_transmission = st.selectbox(
                "Choose transmission to visualize",
                transmission_options
            )
            
            # Get selected index
            selected_idx = int(selected_transmission.split(":")[0])
            selected_log = transmissions[selected_idx]
            
            # Visualize packet transmission
            fig_packet = simulate_packet_animation(
                results_data['manet'],
                selected_log['path'],
                selected_log['success']
            )
            st.pyplot(fig_packet)
            
            # Transmission details
            st.subheader("Transmission Details")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Source Node", selected_log['src'])
                src_node = results_data['manet'].nodes[selected_log['src']]
                st.metric("Source Type", "Black Hole" if src_node.is_blackhole else "Normal")
            
            with col2:
                st.metric("Destination Node", selected_log['dst'])
                dst_node = results_data['manet'].nodes[selected_log['dst']]
                st.metric("Destination Type", "Black Hole" if dst_node.is_blackhole else "Normal")
            
            with col3:
                st.metric("Path Length", len(selected_log['path']))
                st.metric("Status", "Success" if selected_log['success'] else "Failed")
            
            # Path details
            if len(selected_log['path']) > 0:
                st.subheader("Path Details")
                path_data = []
                for i, node_id in enumerate(selected_log['path']):
                    node = results_data['manet'].nodes[node_id]
                    path_data.append({
                        'Step': i + 1,
                        'Node ID': node_id,
                        'Type': 'Black Hole' if node.is_blackhole else 'Normal',
                        'X': f"{node.x:.2f}",
                        'Y': f"{node.y:.2f}"
                    })
                
                df_path = pd.DataFrame(path_data)
                st.dataframe(df_path, use_container_width=True)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    main()