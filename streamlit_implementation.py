import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from collections import defaultdict
import time

class Node:
    def __init__(self, node_id, position, is_malicious=False, initial_energy=100):
        self.id = node_id
        self.position = position
        self.energy = initial_energy
        self.is_malicious = is_malicious
        self.packets_received = 0
        self.packets_forwarded = 0
        self.trust_score = 1.0
        self.suspicious_count = 0
        
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
            if random.random() < 0.9:
                return False
            self.packets_forwarded += 1
            return True
            
    def get_packet_delivery_ratio(self):
        if self.packets_received == 0:
            return 1.0
        return self.packets_forwarded / self.packets_received if self.packets_received > 0 else 0
        
    def calculate_trust(self, network):
        pdr = self.get_packet_delivery_ratio()
        
        neighbors = network.get_neighbors(self.id)
        neighbor_recommendations = 0
        total_neighbors = len(neighbors)
        
        if total_neighbors > 0:
            for neighbor_id in neighbors:
                neighbor = network.nodes[neighbor_id]
                if neighbor.get_packet_delivery_ratio() > 0.5:
                    neighbor_recommendations += 1
            
            neighbor_trust = neighbor_recommendations / total_neighbors
            
            if self.packets_received > 10:
                self.trust_score = 0.8 * pdr + 0.2 * neighbor_trust
            else:
                self.trust_score = 0.6 * pdr + 0.4 * neighbor_trust
        else:
            self.trust_score = pdr
            
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
        for i in range(self.num_nodes):
            position = (random.uniform(0, self.area_size[0]), 
                       random.uniform(0, self.area_size[1]))
            
            is_malicious = random.random() < (malicious_percentage / 100)
            
            self.nodes[i] = Node(i, position, is_malicious)
            self.graph.add_node(i, pos=position)
            
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                distance = np.sqrt((self.nodes[i].position[0] - self.nodes[j].position[0])**2 + 
                                  (self.nodes[i].position[1] - self.nodes[j].position[1])**2)
                if distance <= self.communication_range:
                    self.graph.add_edge(i, j, weight=distance)
                    
    def get_neighbors(self, node_id):
        return list(self.graph.neighbors(node_id))
    
    def simulate_traffic(self, num_packets=100):
        centrality = nx.degree_centrality(self.graph)
        high_centrality_nodes = [n for n, c in sorted(centrality.items(), 
                                                    key=lambda x: x[1], reverse=True)[:int(self.num_nodes/5)]]
        
        for _ in range(int(num_packets * 0.7)):
            source = random.randint(0, self.num_nodes - 1)
            dest = random.randint(0, self.num_nodes - 1)
            while dest == source:
                dest = random.randint(0, self.num_nodes - 1)
                
            self._simulate_packet_transmission(source, dest)
        
        for _ in range(int(num_packets * 0.3)):
            if high_centrality_nodes:
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
                
        for node_id in self.nodes:
            self.nodes[node_id].calculate_trust(self)
    
    def _simulate_packet_transmission(self, source, dest):
        if nx.has_path(self.graph, source, dest):
            path = nx.shortest_path(self.graph, source, dest)
            
            packet_delivered = True
            for i in range(len(path) - 1):
                current_node = self.nodes[path[i]]
                next_node = self.nodes[path[i + 1]]
                
                current_node.receive_packet()
                
                if not current_node.forward_packet():
                    packet_delivered = False
                    
                    if current_node.packets_received > 5:
                        current_node.suspicious_count += 1
                    break
                
                next_node.receive_packet()
                
                current_node.update_energy(0.1)
            
            if packet_delivered:
                self.nodes[dest].receive_packet()
    
    def get_network_visualization(self, title="MANET Network", detected_blackholes=None):
        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.get_node_attributes(self.graph, 'pos')
        
        regular_nodes = [node_id for node_id, node in self.nodes.items() if not node.is_malicious]
        nx.draw_networkx_nodes(self.graph, pos, nodelist=regular_nodes, 
                              node_color='blue', node_size=300, alpha=0.8, ax=ax)
        
        malicious_nodes = [node_id for node_id, node in self.nodes.items() if node.is_malicious]
        nx.draw_networkx_nodes(self.graph, pos, nodelist=malicious_nodes, 
                              node_color='red', node_size=300, alpha=0.8, ax=ax)
        
        if detected_blackholes:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=detected_blackholes, 
                                  node_color='yellow', node_size=400, 
                                  node_shape='h', alpha=0.9, ax=ax)
        
        nx.draw_networkx_edges(self.graph, pos, width=1.0, alpha=0.5, ax=ax)
        
        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_family="sans-serif", ax=ax)
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        return fig
    
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
        self.path = path
        self.network = network
        self.fitness = self.calculate_fitness()
        
    def calculate_fitness(self):
        if not self.path or len(self.path) < 2:
            return 0
            
        length_factor = 1.0 / len(self.path)
        
        energy_values = [self.network.nodes[node_id].energy for node_id in self.path]
        energy_factor = sum(energy_values) / (100 * len(self.path))
        
        trust_values = [self.network.nodes[node_id].trust_score for node_id in self.path]
        
        min_trust = min(trust_values)
        if min_trust < 0.4:
            trust_factor = min_trust
        else:
            trust_factor = sum(trust_values) / len(self.path)
        
        fitness = 0.15 * length_factor + 0.25 * energy_factor + 0.60 * trust_factor
        
        return fitness
    
    def mutate(self):
        if len(self.path) <= 2:
            return
            
        if random.random() < 0.7:
            trust_values = [(i, self.network.nodes[node_id].trust_score) 
                           for i, node_id in enumerate(self.path[1:-1], 1)]
            
            if trust_values:
                trust_values.sort(key=lambda x: x[1])
                pos = trust_values[0][0]
                
                current_node = self.path[pos]
                prev_node = self.path[pos - 1]
                next_node = self.path[pos + 1]
                
                alternatives = []
                for node_id in range(self.network.num_nodes):
                    if (node_id != current_node and 
                        node_id not in self.path and
                        self.network.graph.has_edge(prev_node, node_id) and 
                        self.network.graph.has_edge(node_id, next_node) and
                        self.network.nodes[node_id].trust_score > self.network.nodes[current_node].trust_score):
                        alternatives.append(node_id)
                        
                if alternatives:
                    self.path[pos] = random.choice(alternatives)
        else:
            if len(self.path) > 3:
                for i in range(len(self.path) - 2):
                    for j in range(i + 2, len(self.path)):
                        if self.network.graph.has_edge(self.path[i], self.path[j]):
                            self.path = self.path[:i+1] + self.path[j:]
                            break
                    else:
                        continue
                    break
                            
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
        self.suspicious_nodes = defaultdict(int)
        
    def initialize_population(self):
        self.dolphin_population = []
        
        for _ in range(self.population_size):
            source = random.randint(0, self.network.num_nodes - 1)
            dest = random.randint(0, self.network.num_nodes - 1)
            while dest == source:
                dest = random.randint(0, self.network.num_nodes - 1)
                
            if nx.has_path(self.network.graph, source, dest):
                path = nx.shortest_path(self.network.graph, source, dest)
                route = Route(path, self.network)
                self.dolphin_population.append(route)
        
        for _ in range(self.population_size // 4):
            if len(self.dolphin_population) > 0:
                base_route = random.choice(self.dolphin_population)
                last_node = base_route.path[-1]
                
                neighbors = self.network.get_neighbors(last_node)
                if neighbors:
                    next_node = random.choice(neighbors)
                    if next_node not in base_route.path:
                        new_path = base_route.path.copy() + [next_node]
                        route = Route(new_path, self.network)
                        self.dolphin_population.append(route)
        
    def dolphin_echolocation_phase(self):
        for route in self.dolphin_population:
            route.fitness = route.calculate_fitness()
            
            for node_id in route.path:
                node = self.network.nodes[node_id]
                
                if node.trust_score < 0.6 and node.packets_received > 5:
                    self.suspicious_nodes[node_id] += 1
                
                if node.energy < 40 and node.packets_forwarded < 0.4 * node.packets_received and node.packets_received > 10:
                    self.suspicious_nodes[node_id] += 2
            
        self.dolphin_population.sort(key=lambda x: x.fitness, reverse=True)
        
        top_routes = self.dolphin_population[:int(len(self.dolphin_population) * 0.6)]
        self.best_routes = top_routes
        
    def bee_colony_optimization(self):
        employed_bees = self.best_routes.copy()
        onlooker_bees = []
        scout_bees = []
        
        for route in employed_bees:
            for _ in range(2):
                new_route = Route(route.path.copy(), self.network)
                new_route.mutate()
                onlooker_bees.append(new_route)
        
        fitness_sum = sum(route.fitness for route in employed_bees)
        if fitness_sum > 0:
            for _ in range(len(employed_bees)):
                probabilities = [route.fitness / fitness_sum for route in employed_bees]
                selected_idx = np.random.choice(len(employed_bees), p=probabilities)
                selected_route = employed_bees[selected_idx]
                
                new_route = Route(selected_route.path.copy(), self.network)
                new_route.mutate()
                
                if random.random() < 0.5:
                    new_route.mutate()
                    
                onlooker_bees.append(new_route)
        
        centrality = nx.degree_centrality(self.network.graph)
        high_centrality_nodes = [n for n, c in sorted(centrality.items(), 
                                                    key=lambda x: x[1], reverse=True)[:5]]
        
        for high_node in high_centrality_nodes:
            for _ in range(2):
                source = random.randint(0, self.network.num_nodes - 1)
                while source == high_node:
                    source = random.randint(0, self.network.num_nodes - 1)
                    
                dest = random.randint(0, self.network.num_nodes - 1)
                while dest == source or dest == high_node:
                    dest = random.randint(0, self.network.num_nodes - 1)
                
                if nx.has_path(self.network.graph, source, high_node) and nx.has_path(self.network.graph, high_node, dest):
                    path1 = nx.shortest_path(self.network.graph, source, high_node)
                    path2 = nx.shortest_path(self.network.graph, high_node, dest)
                    combined_path = path1 + path2[1:]
                    route = Route(combined_path, self.network)
                    scout_bees.append(route)
        
        for _ in range(max(1, len(employed_bees) // 3)):
            source = random.randint(0, self.network.num_nodes - 1)
            dest = random.randint(0, self.network.num_nodes - 1)
            while dest == source:
                dest = random.randint(0, self.network.num_nodes - 1)
                
            if nx.has_path(self.network.graph, source, dest):
                path = nx.shortest_path(self.network.graph, source, dest)
                route = Route(path, self.network)
                scout_bees.append(route)
        
        all_bees = employed_bees + onlooker_bees + scout_bees
        all_bees.sort(key=lambda x: x.fitness, reverse=True)
        
        self.best_routes = []
        
        top_half = int(self.population_size * 0.5)
        self.best_routes.extend(all_bees[:top_half])
        
        remaining_capacity = self.population_size - top_half
        if remaining_capacity > 0 and len(all_bees) > top_half:
            selected_indices = np.random.choice(
                range(top_half, len(all_bees)), 
                size=min(remaining_capacity, len(all_bees) - top_half),
                replace=False
            )
            for idx in selected_indices:
                self.best_routes.append(all_bees[idx])
                
        self.dolphin_population = self.best_routes.copy()
    
    def detect_blackhole_nodes(self):
        metrics = self.network.get_node_metrics()
        
        for node_id, node_metrics in metrics.items():
            node = self.network.nodes[node_id]
            
            if node.packets_received < 5:
                continue
                
            if node_metrics['pdr'] < 0.4 and node.packets_received > 10:
                self.suspicious_nodes[node_id] += 3
                
            if node_metrics['trust'] < 0.5:
                self.suspicious_nodes[node_id] += 2
                
            if node_metrics['energy'] > 50 and node_metrics['pdr'] < 0.5 and node.packets_received > 15:
                self.suspicious_nodes[node_id] += 2
                
            if node.suspicious_count > 2:
                self.suspicious_nodes[node_id] += node.suspicious_count
        
        node_appearances = defaultdict(int)
        node_selections = defaultdict(int)
        
        for route in self.dolphin_population:
            for node_id in route.path:
                node_appearances[node_id] += 1
        
        for route in self.best_routes[:max(1, len(self.best_routes)//4)]:
            for node_id in route.path:
                node_selections[node_id] += 1
        
        for node_id in node_appearances:
            if (node_appearances[node_id] > 5 and 
                (node_id not in node_selections or 
                 node_selections[node_id] / node_appearances[node_id] < 0.3)):
                self.suspicious_nodes[node_id] += 2
                
        for node_id in self.network.nodes:
            node = self.network.nodes[node_id]
            neighbors = self.network.get_neighbors(node_id)
            
            if len(neighbors) > 0:
                neighbor_pdrs = [self.network.nodes[n].get_packet_delivery_ratio() for n in neighbors]
                avg_neighbor_pdr = sum(neighbor_pdrs) / len(neighbor_pdrs)
                
                if node.get_packet_delivery_ratio() < 0.6 * avg_neighbor_pdr and node.packets_received > 10:
                    self.suspicious_nodes[node_id] += 2
        
        for node_id, node in self.network.nodes.items():
            if node.packets_received > 15:
                segments = max(1, node.packets_received // 5)
                expected_forwarded = node.packets_received / segments
                
                if node.packets_forwarded < expected_forwarded * 0.4:
                    self.suspicious_nodes[node_id] += 2
        
        if self.suspicious_nodes:
            suspicion_scores = list(self.suspicious_nodes.values())
            if suspicion_scores:
                mean_suspicion = np.mean(suspicion_scores)
                std_suspicion = np.std(suspicion_scores)
                
                threshold = max(4, mean_suspicion + 0.5 * std_suspicion)
                
                self.blackhole_nodes = set()
                for node_id, suspicion in self.suspicious_nodes.items():
                    if suspicion >= threshold:
                        self.blackhole_nodes.add(node_id)
    
    def optimize(self, progress_bar=None, status_text=None):
        self.initialize_population()
        
        for iteration in range(self.max_iterations):
            if status_text:
                status_text.text(f"Running iteration {iteration+1}/{self.max_iterations}")
            if progress_bar:
                progress_bar.progress((iteration + 1) / self.max_iterations)
            
            self.network.simulate_traffic(num_packets=100)
            
            self.dolphin_echolocation_phase()
            
            self.bee_colony_optimization()
            
            self.detect_blackhole_nodes()
        
        return self.best_routes, list(self.blackhole_nodes)


def run_streamlit_app():
    st.set_page_config(page_title="MANET Blackhole Detection", layout="wide")
    
    st.title("MANET Network Blackhole Detection Simulator")
    st.write("This application simulates a Mobile Ad-hoc Network (MANET) and detects malicious blackhole nodes using a Dolphin-Bee optimization algorithm.")
    
    with st.sidebar:
        st.header("Network Parameters")
        
        col1, col2 = st.columns([10, 1])
        with col1:
            num_nodes = st.slider("Number of Nodes", 10, 50, 30)
        with col2:
            st.write("")
            st.write("")
            nodes_help = st.button("ℹ️", key="nodes_help")
        if nodes_help:
            st.info("The total number of devices in the network. More nodes increase complexity but provide more routing options.")
        
        col1, col2 = st.columns([10, 1])
        with col1:
            comm_range = st.slider("Communication Range", 100, 500, 250)
        with col2:
            st.write("")
            st.write("")
            range_help = st.button("ℹ️", key="range_help")
        if range_help:
            st.info("Maximum distance (in arbitrary units) that nodes can communicate with each other. Larger values create more connections in the network.")
        
        col1, col2 = st.columns([10, 1])
        with col1:
            malicious_percentage = st.slider("Malicious Node Percentage", 5, 30, 15)
        with col2:
            st.write("")
            st.write("")
            malicious_help = st.button("ℹ️", key="malicious_help")
        if malicious_help:
            st.info("Percentage of nodes that will act as blackholes (dropping packets instead of forwarding them). Higher values make detection more challenging.")
        
        st.header("Optimization Parameters")
        
        col1, col2 = st.columns([10, 1])
        with col1:
            population_size = st.slider("Population Size", 10, 60, 40)
        with col2:
            st.write("")
            st.write("")
            pop_help = st.button("ℹ️", key="pop_help")
        if pop_help:
            st.info("Number of potential routes evaluated in each iteration of the optimization algorithm. Larger population increases diversity of solutions but requires more computation.")
        
        col1, col2 = st.columns([10, 1])
        with col1:
            max_iterations = st.slider("Maximum Iterations", 10, 100, 50)
        with col2:
            st.write("")
            st.write("")
            iter_help = st.button("ℹ️", key="iter_help")
        if iter_help:
            st.info("Number of optimization cycles to run. More iterations allow the algorithm to refine its detection but take longer to complete.")
        
        run_button = st.button("Run Simulation")
    
    if 'network' not in st.session_state or run_button:
        with st.spinner("Initializing network..."):
            area_size = (1000, 1000)
            network = MANETNetwork(num_nodes, area_size, comm_range, malicious_percentage)
            st.session_state.network = network
            st.session_state.initial_fig = network.get_network_visualization(title="Initial MANET Network")
            st.session_state.has_run = False
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Initial Network")
        st.pyplot(st.session_state.initial_fig)
    
    if 'has_run' in st.session_state and st.session_state.has_run:
        with col2:
            st.subheader("Network with Detected Blackholes")
            st.pyplot(st.session_state.final_fig)
        
        st.subheader("Detection Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Blackhole Detection")
            detected_blackholes = st.session_state.detected_blackholes
            actual_blackholes = st.session_state.actual_blackholes
            
            st.write(f"**Detected blackhole nodes:** {detected_blackholes}")
            st.write(f"**Actual blackhole nodes:** {actual_blackholes}")
            
            true_positives = len(set(detected_blackholes) & set(actual_blackholes))
            false_positives = len(set(detected_blackholes) - set(actual_blackholes))
            false_negatives = len(set(actual_blackholes) - set(detected_blackholes))
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            st.write("### Metrics:")
            metrics_df = {
                "Metric": ["Precision", "Recall", "F1 Score"],
                "Value": [f"{precision:.4f}", f"{recall:.4f}", f"{f1_score:.4f}"]
            }
            st.table(metrics_df)
        
        with col2:
            st.write("### Top Optimized Routes")
            if st.session_state.best_routes:
                for i, route in enumerate(st.session_state.best_routes[:3]):
                    st.write(f"**Route {i+1}:** {route.path}")
                    st.write(f"- Fitness: {route.fitness:.4f}")
                    st.write(f"- Path Length: {len(route.path)}")
                    st.write(f"- Average Trust: {sum(st.session_state.network.nodes[n].trust_score for n in route.path)/len(route.path):.4f}")
                    st.write(f"- Average Energy: {sum(st.session_state.network.nodes[n].energy for n in route.path)/len(route.path):.2f}")
            else:
                st.write("No routes found.")
    
    if run_button or ('has_run' not in st.session_state or not st.session_state.has_run):
        st.subheader("Optimization Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Running Dolphin-Bee optimization..."):
            network = st.session_state.network
            optimizer = DolphinBeeOptimizer(network, population_size=population_size, max_iterations=max_iterations)
            best_routes, detected_blackholes = optimizer.optimize(progress_bar, status_text)
            
            actual_blackholes = [node_id for node_id, node in network.nodes.items() if node.is_malicious]
            
            st.session_state.best_routes = best_routes
            st.session_state.detected_blackholes = detected_blackholes
            st.session_state.actual_blackholes = actual_blackholes
            st.session_state.final_fig = network.get_network_visualization(
                title="MANET Network with Detected Blackholes", 
                detected_blackholes=detected_blackholes)
            st.session_state.has_run = True
        
        st.rerun()

if __name__ == "__main__":
    run_streamlit_app()