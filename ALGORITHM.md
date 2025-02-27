## Main Algorithm: Dolphin-Bee Optimization for MANET Blackhole Detection

### Initialization Phase
1. Create a MANET (Mobile Ad-hoc Network) with randomly positioned nodes
2. Designate a percentage of nodes as malicious (blackhole nodes)
3. Establish connections between nodes based on communication range
4. Initialize the Dolphin-Bee Optimizer with the network

### Network Simulation
1. Simulate network traffic by sending packets between nodes
   - 70% random traffic between arbitrary nodes
   - 30% targeted traffic through high-centrality nodes (potential bottlenecks)
2. For each packet transmission:
   - Find the shortest path between source and destination
   - Simulate packet forwarding along the path
   - Track packet delivery, energy consumption, and suspicious behavior
   - Regular nodes forward all packets
   - Malicious nodes drop 90% of packets

### Trust Calculation
1. Calculate trust scores for each node based on:
   - Packet Delivery Ratio (PDR): packets forwarded / packets received
   - Neighbor recommendations
   - Traffic volume (different weighting for high vs. low traffic nodes)
2. Apply trust decay for suspicious behavior

### Dolphin-Bee Optimization Loop
For each iteration:
1. **Dolphin Echolocation Phase**
   - Update fitness for each route based on: 
     - Path length (shorter is better)
     - Energy levels of nodes (higher is better)
     - Trust scores (heavily penalizes routes with low-trust nodes)
   - Flag suspicious nodes with low trust scores
   - Sort routes by fitness
   - Keep top 60% for the bee phase

2. **Bee Colony Optimization Phase**
   - Employed Bee Phase: Explore around top routes with focused mutations
   - Onlooker Bee Phase: Probabilistic selection of routes based on fitness
   - Scout Bee Phase: Random exploration including paths through high-centrality nodes
   - Combine and select the best routes with diversity preservation

3. **Blackhole Detection Phase**
   - Multi-metric detection algorithm:
     - Direct metric analysis: PDR, trust score, energy patterns
     - Route-based analysis: Compare node appearances in potential vs. best routes
     - Neighborhood analysis: Compare node behavior to neighbors
     - Consistency analysis: Check for inconsistent forwarding behavior
   - Calculate suspicion score for each node
   - Use adaptive threshold to identify blackhole nodes

4. **Evaluation Phase**
   - Calculate detection accuracy metrics (precision, recall, F1 score)
   - Visualize network with detected blackholes
   - Report top optimized routes

### Route Mutation Strategies
1. Replace low-trust nodes with higher-trust alternatives (70% chance)
2. Try to shorten path by skipping nodes when possible (30% chance)

This hybrid algorithm combines swarm intelligence techniques (dolphin echolocation and bee colony optimization) with trust-based routing to detect malicious nodes in a mobile ad-hoc network while optimizing secure communication routes.
