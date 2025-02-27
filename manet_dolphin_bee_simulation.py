import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from collections import defaultdict

class Node:
    def __init__(self, node_id, position, is_malicious=False, initial_energy=100):
        self.id = node_id
        self.position = position  # (x, y) coordinates
        self.energy = initial_energy
        self.is_malicious = is_malicious
        self.packets_received = 0
        self.packets_forwarded = 0
        self.trust_score = 1.0  # Initial trust score
        self.suspicious_count = 0  # Track how many times node is flagged as suspicious
        
    def update_energy(self, energy_consumed):
        self.energy -= energy_consumed
        if self.energy < 0:
            self.energy = 0
            
    def receive_packet(self):
        self.packets_received += 1
        
    def forward_packet(self):
        if not self.is_malicious:
            self.packets_forwarded += 1
            return True
        else:
            # Malicious node might drop packets (blackhole behavior)
            if random.random() < 0.9:  # 90% packet drop rate for malicious nodes
                return False
            self.packets_forwarded += 1
            return True
            
    def get_packet_delivery_ratio(self):
        if self.packets_received == 0:
            return 1.0
        return self.packets_forwarded / self.packets_received if self.packets_received > 0 else 0
        
    def calculate_trust(self, network):
        # Enhanced trust calculation based on packet delivery ratio
        pdr = self.get_packet_delivery_ratio()
        
        # Get recommendations from neighbor nodes
        neighbors = network.get_neighbors(self.id)
        neighbor_recommendations = 0
        total_neighbors = len(neighbors)
        
        if total_neighbors > 0:
            for neighbor_id in neighbors:
                neighbor = network.nodes[neighbor_id]
                if neighbor.get_packet_delivery_ratio() > 0.5:
                    neighbor_recommendations += 1
            
            neighbor_trust = neighbor_recommendations / total_neighbors
            
            # Improved trust formula with more weight on PDR for nodes with high traffic
            if self.packets_received > 10:
                # More emphasis on actual packet delivery for high-traffic nodes
                self.trust_score = 0.8 * pdr + 0.2 * neighbor_trust
            else:
                # More balanced for low-traffic nodes
                self.trust_score = 0.6 * pdr + 0.4 * neighbor_trust
        else:
            self.trust_score = pdr
            
        # Rapid trust decay for suspicious behavior
        if pdr < 0.4 and self.packets_received > 5:
            self.trust_score *= 0.8
            
        return self.trust_score

class MANETNetwork:
    def __init__(self, num_nodes, area_size, communication_range, malicious_percentage=10):
        self.num_nodes = num_nodes
        self.area_size = area_size
        self.communication_range = communication_range
        self.nodes = {}
        self.graph = nx.Graph()
        self.initialize_network(malicious_percentage)
        
    def initialize_network(self, malicious_percentage):
        # Create nodes with random positions
        for i in range(self.num_nodes):
            position = (random.uniform(0, self.area_size[0]), 
                       random.uniform(0, self.area_size[1]))
            
            # Determine if node is malicious
            is_malicious = random.random() < (malicious_percentage / 100)
            
            self.nodes[i] = Node(i, position, is_malicious)
            self.graph.add_node(i, pos=position)
            
        # Establish connections based on communication range
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                distance = np.sqrt((self.nodes[i].position[0] - self.nodes[j].position[0])**2 + 
                                  (self.nodes[i].position[1] - self.nodes[j].position[1])**2)
                if distance <= self.communication_range:
                    self.graph.add_edge(i, j, weight=distance)
                    
    def get_neighbors(self, node_id):
        return list(self.graph.neighbors(node_id))
    
    def simulate_traffic(self, num_packets=100):
        # Enhanced traffic simulation with targeted traffic
        # Create more traffic through potential bottleneck nodes
        
        # First, identify potential bottleneck nodes (high degree centrality)
        centrality = nx.degree_centrality(self.graph)
        high_centrality_nodes = [n for n, c in sorted(centrality.items(), 
                                                    key=lambda x: x[1], reverse=True)[:int(self.num_nodes/5)]]
        
        # Regular random traffic
        for _ in range(int(num_packets * 0.7)):  # 70% of traffic is random
            source = random.randint(0, self.num_nodes - 1)
            dest = random.randint(0, self.num_nodes - 1)
            while dest == source:
                dest = random.randint(0, self.num_nodes - 1)
                
            self._simulate_packet_transmission(source, dest)
        
        # Targeted traffic through high centrality nodes
        for _ in range(int(num_packets * 0.3)):  # 30% targeted traffic
            if high_centrality_nodes:
                # Either source or destination is a high centrality node
                if random.random() < 0.5:
                    source = random.choice(high_centrality_nodes)
                    dest = random.randint(0, self.num_nodes - 1)
                    while dest == source:
                        dest = random.randint(0, self.num_nodes - 1)
                else:
                    source = random.randint(0, self.num_nodes - 1)
                    dest = random.choice(high_centrality_nodes)
                    while dest == source:
                        source = random.randint(0, self.num_nodes - 1)
                        
                self._simulate_packet_transmission(source, dest)
                
        # Update trust scores after traffic simulation
        for node_id in self.nodes:
            self.nodes[node_id].calculate_trust(self)
    
    def _simulate_packet_transmission(self, source, dest):
        # Find shortest path
        if nx.has_path(self.graph, source, dest):
            path = nx.shortest_path(self.graph, source, dest)
            
            # Simulate packet forwarding along the path
            packet_delivered = True
            for i in range(len(path) - 1):
                current_node = self.nodes[path[i]]
                next_node = self.nodes[path[i + 1]]
                
                # Current node receives packet
                current_node.receive_packet()
                
                # Try to forward packet
                if not current_node.forward_packet():
                    packet_delivered = False
                    
                    # Flag suspicious drop behavior 
                    if current_node.packets_received > 5:
                        current_node.suspicious_count += 1
                    break
                
                # Next node receives packet
                next_node.receive_packet()
                
                # Energy consumption
                current_node.update_energy(0.1)  # Energy consumed for transmission
            
            if packet_delivered:
                self.nodes[dest].receive_packet()
    
    def visualize_network(self, title="MANET Network", detected_blackholes=None):
        plt.figure(figsize=(10, 8))
        pos = nx.get_node_attributes(self.graph, 'pos')
        
        # Draw regular nodes
        regular_nodes = [node_id for node_id, node in self.nodes.items() if not node.is_malicious]
        nx.draw_networkx_nodes(self.graph, pos, nodelist=regular_nodes, 
                              node_color='blue', node_size=300, alpha=0.8)
        
        # Draw actual malicious nodes
        malicious_nodes = [node_id for node_id, node in self.nodes.items() if node.is_malicious]
        nx.draw_networkx_nodes(self.graph, pos, nodelist=malicious_nodes, 
                              node_color='red', node_size=300, alpha=0.8)
        
        # Draw detected blackhole nodes (if provided)
        if detected_blackholes:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=detected_blackholes, 
                                  node_color='yellow', node_size=400, 
                                  node_shape='h', alpha=0.9)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, width=1.0, alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_family="sans-serif")
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def get_node_metrics(self):
        metrics = {}
        for node_id, node in self.nodes.items():
            metrics[node_id] = {
                'energy': node.energy,
                'trust': node.trust_score,
                'pdr': node.get_packet_delivery_ratio(),
                'is_malicious': node.is_malicious,
                'suspicious_count': node.suspicious_count
            }
        return metrics

class Route:
    def __init__(self, path, network):
        self.path = path  # List of node IDs representing the route
        self.network = network
        self.fitness = self.calculate_fitness()
        
    def calculate_fitness(self):
        # Improved fitness calculation
        if not self.path or len(self.path) < 2:
            return 0
            
        # Consider path length (shorter is better)
        length_factor = 1.0 / len(self.path)
        
        # Energy evaluation (higher energy nodes preferred)
        energy_values = [self.network.nodes[node_id].energy for node_id in self.path]
        energy_factor = sum(energy_values) / (100 * len(self.path))  # Normalize
        
        # Trust evaluation (higher trust preferred)
        trust_values = [self.network.nodes[node_id].trust_score for node_id in self.path]
        
        # Heavily penalize routes with low-trust nodes
        min_trust = min(trust_values)
        if min_trust < 0.4:  # Any node with very low trust significantly reduces route fitness
            trust_factor = min_trust
        else:
            trust_factor = sum(trust_values) / len(self.path)
        
        # Enhanced weighting with emphasis on trust
        fitness = 0.15 * length_factor + 0.25 * energy_factor + 0.60 * trust_factor
        
        return fitness
    
    def mutate(self):
        # Enhanced mutation strategy
        if len(self.path) <= 2:
            return
            
        # Either modify an intermediate node or try to optimize path
        if random.random() < 0.7:  # 70% chance to replace a low-trust node
            # Find nodes with lowest trust in the path
            trust_values = [(i, self.network.nodes[node_id].trust_score) 
                           for i, node_id in enumerate(self.path[1:-1], 1)]
            
            if trust_values:
                # Sort by trust score, ascending
                trust_values.sort(key=lambda x: x[1])
                pos = trust_values[0][0]  # Position of lowest trust node
                
                current_node = self.path[pos]
                prev_node = self.path[pos - 1]
                next_node = self.path[pos + 1]
                
                # Find alternative nodes with better trust
                alternatives = []
                for node_id in range(self.network.num_nodes):
                    if (node_id != current_node and 
                        node_id not in self.path and
                        self.network.graph.has_edge(prev_node, node_id) and 
                        self.network.graph.has_edge(node_id, next_node) and
                        self.network.nodes[node_id].trust_score > self.network.nodes[current_node].trust_score):
                        alternatives.append(node_id)
                        
                if alternatives:
                    # Replace with a random higher trust alternative
                    self.path[pos] = random.choice(alternatives)
        else:
            # Try to shorten the path if possible
            if len(self.path) > 3:
                for i in range(len(self.path) - 2):
                    for j in range(i + 2, len(self.path)):
                        # Check if we can skip some nodes
                        if self.network.graph.has_edge(self.path[i], self.path[j]):
                            # We can connect directly from path[i] to path[j]
                            self.path = self.path[:i+1] + self.path[j:]
                            break
                    else:
                        continue
                    break
                            
        # Recalculate fitness
        self.fitness = self.calculate_fitness()

class DolphinBeeOptimizer:
    def __init__(self, network, population_size=30, max_iterations=50):
        self.network = network
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.dolphin_population = []
        self.bee_population = []
        self.best_routes = []
        self.blackhole_nodes = set()
        self.suspicious_nodes = defaultdict(int)  # Track suspicion level across iterations
        
    def initialize_population(self):
        # Initialize dolphin population with diverse routes
        self.dolphin_population = []
        
        # Create routes between various node pairs
        for _ in range(self.population_size):
            source = random.randint(0, self.network.num_nodes - 1)
            dest = random.randint(0, self.network.num_nodes - 1)
            while dest == source:
                dest = random.randint(0, self.network.num_nodes - 1)
                
            # Find a path
            if nx.has_path(self.network.graph, source, dest):
                path = nx.shortest_path(self.network.graph, source, dest)
                route = Route(path, self.network)
                self.dolphin_population.append(route)
        
        # Add some longer routes to ensure more nodes are covered
        for _ in range(self.population_size // 4):
            if len(self.dolphin_population) > 0:
                # Take an existing route and extend it
                base_route = random.choice(self.dolphin_population)
                last_node = base_route.path[-1]
                
                # Find neighbors of the last node
                neighbors = self.network.get_neighbors(last_node)
                if neighbors:
                    next_node = random.choice(neighbors)
                    if next_node not in base_route.path:
                        new_path = base_route.path.copy() + [next_node]
                        route = Route(new_path, self.network)
                        self.dolphin_population.append(route)
        
    def dolphin_echolocation_phase(self):
        # Enhanced dolphin echolocation phase
        for route in self.dolphin_population:
            # Update fitness which includes trust evaluation
            route.fitness = route.calculate_fitness()
            
            # Analyze route quality for blackhole detection
            for node_id in route.path:
                node = self.network.nodes[node_id]
                
                # Flag suspicious nodes based on direct route analysis
                if node.trust_score < 0.6 and node.packets_received > 5:
                    self.suspicious_nodes[node_id] += 1
                
                # Check for energy drain patterns
                if node.energy < 40 and node.packets_forwarded < 0.4 * node.packets_received and node.packets_received > 10:
                    self.suspicious_nodes[node_id] += 2
            
        # Sort routes by fitness
        self.dolphin_population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Keep top 60% for bee phase
        top_routes = self.dolphin_population[:int(len(self.dolphin_population) * 0.6)]
        self.best_routes = top_routes
        
    def bee_colony_optimization(self):
        # Enhanced bee colony optimization
        employed_bees = self.best_routes.copy()
        onlooker_bees = []
        scout_bees = []
        
        # Employed bee phase - explore around top routes with more focused mutation
        for route in employed_bees:
            # Create multiple variations of each good route
            for _ in range(2):  # Increase exploration around good routes
                new_route = Route(route.path.copy(), self.network)
                new_route.mutate()
                onlooker_bees.append(new_route)
        
        # Onlooker bee phase - probabilistic selection based on fitness
        fitness_sum = sum(route.fitness for route in employed_bees)
        if fitness_sum > 0:
            for _ in range(len(employed_bees)):
                probabilities = [route.fitness / fitness_sum for route in employed_bees]
                selected_idx = np.random.choice(len(employed_bees), p=probabilities)
                selected_route = employed_bees[selected_idx]
                
                # Create modified version with more aggressive mutation
                new_route = Route(selected_route.path.copy(), self.network)
                new_route.mutate()
                
                # Add a second mutation for more diversity
                if random.random() < 0.5:
                    new_route.mutate()
                    
                onlooker_bees.append(new_route)
        
        # Scout bee phase - random exploration including high-centrality paths
        # Find high centrality nodes to ensure they're included in exploration
        centrality = nx.degree_centrality(self.network.graph)
        high_centrality_nodes = [n for n, c in sorted(centrality.items(), 
                                                    key=lambda x: x[1], reverse=True)[:5]]
        
        # Create some scout routes through high centrality nodes
        for high_node in high_centrality_nodes:
            for _ in range(2):
                source = random.randint(0, self.network.num_nodes - 1)
                while source == high_node:
                    source = random.randint(0, self.network.num_nodes - 1)
                    
                dest = random.randint(0, self.network.num_nodes - 1)
                while dest == source or dest == high_node:
                    dest = random.randint(0, self.network.num_nodes - 1)
                
                # Find path that goes through the high centrality node if possible
                if nx.has_path(self.network.graph, source, high_node) and nx.has_path(self.network.graph, high_node, dest):
                    path1 = nx.shortest_path(self.network.graph, source, high_node)
                    path2 = nx.shortest_path(self.network.graph, high_node, dest)
                    # Combine paths and remove duplicate
                    combined_path = path1 + path2[1:]
                    route = Route(combined_path, self.network)
                    scout_bees.append(route)
        
        # Add some completely random routes too
        for _ in range(max(1, len(employed_bees) // 3)):
            source = random.randint(0, self.network.num_nodes - 1)
            dest = random.randint(0, self.network.num_nodes - 1)
            while dest == source:
                dest = random.randint(0, self.network.num_nodes - 1)
                
            if nx.has_path(self.network.graph, source, dest):
                path = nx.shortest_path(self.network.graph, source, dest)
                route = Route(path, self.network)
                scout_bees.append(route)
        
        # Combine all populations and select best routes
        all_bees = employed_bees + onlooker_bees + scout_bees
        all_bees.sort(key=lambda x: x.fitness, reverse=True)
        
        # Update best routes with diversity
        self.best_routes = []
        
        # First, add top 50% by fitness
        top_half = int(self.population_size * 0.5)
        self.best_routes.extend(all_bees[:top_half])
        
        # Then add some routes for diversity (to avoid local optima)
        remaining_capacity = self.population_size - top_half
        if remaining_capacity > 0 and len(all_bees) > top_half:
            selected_indices = np.random.choice(
                range(top_half, len(all_bees)), 
                size=min(remaining_capacity, len(all_bees) - top_half),
                replace=False
            )
            for idx in selected_indices:
                self.best_routes.append(all_bees[idx])
                
        # Update dolphin population for next iteration
        self.dolphin_population = self.best_routes.copy()
    
    def detect_blackhole_nodes(self):
        # Multi-metric blackhole detection algorithm
        metrics = self.network.get_node_metrics()
        
        # 1. Analysis of direct network metrics
        for node_id, node_metrics in metrics.items():
            node = self.network.nodes[node_id]
            
            # Skip nodes with too little traffic for reliable detection
            if node.packets_received < 5:
                continue
                
            # Multiple detection criteria combined for better accuracy
            
            # Very low PDR is highly suspicious
            if node_metrics['pdr'] < 0.4 and node.packets_received > 10:
                self.suspicious_nodes[node_id] += 3
                
            # Low trust score is suspicious
            if node_metrics['trust'] < 0.5:
                self.suspicious_nodes[node_id] += 2
                
            # Energy doesn't match forwarding behavior (energy should decrease with forwarding)
            if node_metrics['energy'] > 50 and node_metrics['pdr'] < 0.5 and node.packets_received > 15:
                self.suspicious_nodes[node_id] += 2
                
            # Node's own suspicious behavior counter
            if node.suspicious_count > 2:
                self.suspicious_nodes[node_id] += node.suspicious_count
        
        # 2. Route-based analysis
        # Count node appearances in routes
        node_appearances = defaultdict(int)
        node_selections = defaultdict(int)
        
        # Count how many times each node appears in potential routes
        for route in self.dolphin_population:
            for node_id in route.path:
                node_appearances[node_id] += 1
        
        # Count how many times each node appears in best routes
        for route in self.best_routes[:max(1, len(self.best_routes)//4)]:  # Top 25%
            for node_id in route.path:
                node_selections[node_id] += 1
        
        # Identify nodes that appear often in potential routes but rarely in best routes
        for node_id in node_appearances:
            if (node_appearances[node_id] > 5 and 
                (node_id not in node_selections or 
                 node_selections[node_id] / node_appearances[node_id] < 0.3)):
                self.suspicious_nodes[node_id] += 2
                
        # 3. Neighborhood analysis
        for node_id in self.network.nodes:
            node = self.network.nodes[node_id]
            neighbors = self.network.get_neighbors(node_id)
            
            if len(neighbors) > 0:
                # Compare node's behavior to its neighbors
                neighbor_pdrs = [self.network.nodes[n].get_packet_delivery_ratio() for n in neighbors]
                avg_neighbor_pdr = sum(neighbor_pdrs) / len(neighbor_pdrs)
                
                # If node's PDR is significantly worse than its neighbors
                if node.get_packet_delivery_ratio() < 0.6 * avg_neighbor_pdr and node.packets_received > 10:
                    self.suspicious_nodes[node_id] += 2
        
        # 4. Consistency analysis
        for node_id, node in self.network.nodes.items():
            # Check for inconsistent forwarding behavior
            if node.packets_received > 15:
                segments = max(1, node.packets_received // 5)
                expected_forwarded = node.packets_received / segments
                
                # If node has periods of dropping nearly all packets
                if node.packets_forwarded < expected_forwarded * 0.4:
                    self.suspicious_nodes[node_id] += 2
        
        # Convert suspicion scores to blackhole detection
        # Use adaptive threshold based on the distribution of suspicion scores
        if self.suspicious_nodes:
            suspicion_scores = list(self.suspicious_nodes.values())
            if suspicion_scores:
                mean_suspicion = np.mean(suspicion_scores)
                std_suspicion = np.std(suspicion_scores)
                
                # Adaptive threshold: mean + 0.5*std ensures we catch more true positives
                threshold = max(4, mean_suspicion + 0.5 * std_suspicion)
                
                # Clear previous detections and set new ones
                self.blackhole_nodes = set()
                for node_id, suspicion in self.suspicious_nodes.items():
                    if suspicion >= threshold:
                        self.blackhole_nodes.add(node_id)
    
    def optimize(self):
        self.initialize_population()
        
        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration+1}/{self.max_iterations}")
            
            # Simulate more network traffic in each iteration for better behavioral data
            self.network.simulate_traffic(num_packets=100)
            
            # Dolphin phase
            self.dolphin_echolocation_phase()
            
            # Bee phase
            self.bee_colony_optimization()
            
            # Detect blackhole nodes
            self.detect_blackhole_nodes()
            
            # Print current best route and detections
            if self.best_routes:
                best_route = self.best_routes[0]
                print(f"Best route: {best_route.path}, Fitness: {best_route.fitness:.4f}")
                print(f"Detected blackhole nodes: {self.blackhole_nodes}")
        
        return self.best_routes, list(self.blackhole_nodes)

def run_simulation():
    # Initialize MANET network
    num_nodes = 30
    area_size = (1000, 1000)
    comm_range = 250
    malicious_percentage = 15
    
    # Create network
    print("Initializing MANET network...")
    network = MANETNetwork(num_nodes, area_size, comm_range, malicious_percentage)
    
    # Visualize initial network
    network.visualize_network(title="Initial MANET Network")
    
    # Run optimization algorithm with more iterations for better detection
    print("Starting Dolphin-Bee optimization...")
    optimizer = DolphinBeeOptimizer(network, population_size=40, max_iterations=50)
    best_routes, detected_blackholes = optimizer.optimize()
    
    # Print results
    print("\nOptimization complete!")
    print(f"Detected blackhole nodes: {detected_blackholes}")
    
    # Check accuracy of detection
    actual_blackholes = [node_id for node_id, node in network.nodes.items() if node.is_malicious]
    print(f"Actual blackhole nodes: {actual_blackholes}")
    
    # Calculate detection metrics
    true_positives = len(set(detected_blackholes) & set(actual_blackholes))
    false_positives = len(set(detected_blackholes) - set(actual_blackholes))
    false_negatives = len(set(actual_blackholes) - set(detected_blackholes))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nDetection Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    # Visualize final network with detected blackholes
    network.visualize_network(title="MANET Network with Detected Blackholes", 
                             detected_blackholes=detected_blackholes)
    
    # Print best route details
    if best_routes:
        print("\nTop 3 Optimized Routes:")
        for i, route in enumerate(best_routes[:3]):
            print(f"Route {i+1}: {route.path}")
            print(f"  Fitness: {route.fitness:.4f}")
            print(f"  Path Length: {len(route.path)}")
            print(f"  Average Trust: {sum(network.nodes[n].trust_score for n in route.path)/len(route.path):.4f}")
            print(f"  Average Energy: {sum(network.nodes[n].energy for n in route.path)/len(route.path):.2f}")
            print()

if __name__ == "__main__":
    run_simulation()