# MANET Black Hole Attack Detection Algorithm

## Overview

This algorithm implements three bio-inspired optimization techniques for detecting black hole attacks in Mobile Ad-hoc Networks (MANETs). The system combines **Dolphin Echolocation Algorithm (DEA)** and **Artificial Bee Colony (ABC)** optimization to identify malicious nodes that drop packets in wireless networks.

## System Architecture

### 1. MANET Network Simulation
- Creates a network of mobile nodes with random positions
- Establishes connectivity based on transmission range
- Simulates packet transmission and routing
- Monitors network behavior and collects node statistics

### 2. Feature Extraction
Each node is characterized by the following features:
- **Packet Drop Ratio**: Ratio of dropped packets to total packets
- **Energy Level**: Normalized energy consumption (0-1)
- **Trust Value**: Node trustworthiness score
- **Network Degree**: Number of neighboring nodes (normalized)
- **Average Neighbor Trust**: Mean trust value of neighboring nodes
- **Forward Ratio**: Ratio of forwarded to received packets

### 3. Detection Algorithms

## Algorithm 1: Basic Dolphin Echolocation Algorithm (DEA)

```
ALGORITHM: Dolphin Echolocation for Black Hole Detection
INPUT: Feature matrix F (n_nodes × n_features)
OUTPUT: Binary predictions (0: normal, 1: black hole)

1. INITIALIZE:
   - n_dolphins = 20
   - max_iterations = 100
   - dolphins[] = random positions in feature space
   - best_dolphin = null
   - best_fitness = -∞

2. FOR iteration = 1 to max_iterations:
   a. FOR each dolphin i:
      - Calculate fitness using weighted feature scores
      - IF fitness > best_fitness:
          best_fitness = fitness
          best_dolphin = dolphins[i]
   
   b. FOR each dolphin i:
      - Generate echolocation parameters:
          frequency = random(0.1, 0.9)
          amplitude = random(0.5, 1.5)
      - Update position:
          dolphins[i] += frequency × (best_dolphin - dolphins[i]) × amplitude
      - Add exploration noise:
          dolphins[i] += random_noise(0.1)

3. DETECTION:
   - weights = normalize(|best_dolphin|)
   - detection_scores = F × weights
   - predictions = (detection_scores > threshold) ? 1 : 0
   
4. RETURN predictions
```

## Algorithm 2: Hybrid Dolphin Echolocation-Artificial Bee Colony (DE-ABC)

```
ALGORITHM: Hybrid DE-ABC for Black Hole Detection
INPUT: Feature matrix F (n_nodes × n_features)
OUTPUT: Binary predictions (0: normal, 1: black hole)

1. INITIALIZE:
   - n_dolphins = 20, n_bees = 30
   - max_iterations = 100
   - dolphins[] = random positions
   - food_sources[] = random positions (employed bees)
   - best_solution = null, best_fitness = -∞

2. FOR iteration = 1 to max_iterations:
   
   a. DOLPHIN ECHOLOCATION PHASE:
      - FOR each dolphin i:
          fitness = calculate_enhanced_fitness(dolphins[i], F)
          IF fitness > best_fitness:
              best_fitness = fitness
              best_solution = dolphins[i]
      
      - FOR each dolphin i:
          frequency = random(0.1, 0.9)
          amplitude = random(0.5, 1.5)
          dolphins[i] += frequency × (best_solution - dolphins[i]) × amplitude
          
          // Information exchange with ABC
          IF i < employed_bees:
              dolphins[i] += 0.2 × (food_sources[i] - dolphins[i])
          
          dolphins[i] += exploration_noise(0.05)
   
   b. ARTIFICIAL BEE COLONY PHASE:
      
      b1. EMPLOYED BEE PHASE:
          FOR i = 1 to employed_bees:
              j = random_dimension()
              k = random_neighbor(i)
              phi = random(-1, 1)
              new_source[j] = food_sources[i][j] + phi × (food_sources[i][j] - food_sources[k][j])
              
              IF fitness(new_source) > fitness(food_sources[i]):
                  food_sources[i] = new_source
      
      b2. ONLOOKER BEE PHASE:
          probabilities = calculate_selection_probabilities(food_sources)
          FOR each onlooker bee:
              i = select_source_by_probability(probabilities)
              // Similar update as employed bees
              
      b3. UPDATE BEST SOLUTION:
          FOR each food_source:
              IF fitness(food_source) > best_fitness:
                  best_fitness = fitness(food_source)
                  best_solution = food_source
   
   c. MIGRATION PHASE (every 10 iterations):
      - Exchange worst dolphin with best food source

3. ADAPTIVE THRESHOLD DETECTION:
   - weights = normalize(|best_solution|)
   - detection_scores = F × weights
   - adaptive_threshold = mean(scores) + 0.5 × std(scores)
   - predictions = (detection_scores > adaptive_threshold) ? 1 : 0

4. RETURN predictions
```

## Algorithm 3: Improved Hybrid DE-ABC with Ensemble Learning

```
ALGORITHM: Enhanced Hybrid DE-ABC with Ensemble Decision
INPUT: Feature matrix F (n_nodes × n_features)
OUTPUT: Binary predictions (0: normal, 1: black hole)

1. FEATURE ENHANCEMENT:
   - enhanced_features = add_statistical_features(F)
   - scaled_features = standardize(enhanced_features)

2. INITIALIZE:
   - n_dolphins = 30, n_bees = 40
   - max_iterations = 150
   - abandonment_limit = 10
   - trial_counters[] = zeros(employed_bees)
   - populations = initialize_with_levy_flight()

3. MAIN OPTIMIZATION LOOP:
   FOR iteration = 1 to max_iterations:
   
   a. ENHANCED DOLPHIN PHASE:
      - FOR each dolphin i:
          fitness = calculate_adaptive_fitness(dolphins[i])
          
      - FOR each dolphin i:
          IF random() < 0.3:
              dolphins[i] = levy_flight_update(dolphins[i], best_solution)
          ELSE:
              // Standard echolocation with adaptive parameters
              frequency = 0.1 + 0.8 × exp(-iteration/max_iterations)
              amplitude = 1.5 × (1 - iteration/max_iterations)
              dolphins[i] += frequency × (best_solution - dolphins[i]) × amplitude
          
          // Information exchange and mutation
          IF i < employed_bees:
              exchange_rate = 0.3 × (1 - iteration/max_iterations)
              dolphins[i] += exchange_rate × (food_sources[i] - dolphins[i])
          
          IF random() < 0.1:
              mutate_random_dimension(dolphins[i])
   
   b. ADAPTIVE ABC PHASE:
      
      b1. EMPLOYED BEES WITH ADAPTIVE NEIGHBORHOOD:
          FOR i = 1 to employed_bees:
              neighborhood_size = max(2, employed_bees × (1 - iteration/max_iterations))
              neighbors = select_adaptive_neighborhood(i, neighborhood_size)
              
              n_dims = random(1, features/2)
              FOR each selected dimension j:
                  k = random_from(neighbors)
                  phi = random(-1,1) × (1 - 0.5×iteration/max_iterations)
                  new_source[j] = food_sources[i][j] + phi × (food_sources[i][j] - food_sources[k][j])
              
              IF fitness(new_source) > fitness(food_sources[i]):
                  food_sources[i] = new_source
                  trial_counters[i] = 0
              ELSE:
                  trial_counters[i]++
      
      b2. ONLOOKER BEES WITH TOURNAMENT SELECTION:
          probabilities = calculate_adaptive_probabilities(fitness_values)
          FOR each onlooker bee:
              i = tournament_selection(probabilities)
              // Similar multi-dimensional update
      
      b3. SCOUT BEES (Abandonment):
          FOR i = 1 to employed_bees:
              IF trial_counters[i] > abandonment_limit:
                  food_sources[i] = levy_flight_from_best(food_sources[best_idx])
                  trial_counters[i] = 0
   
   c. DIVERSITY MAINTENANCE (every 20 iterations):
      - Replace worst 5 solutions with random new solutions
   
   d. EARLY STOPPING:
      - IF no improvement for 20 iterations: BREAK

4. ENSEMBLE DECISION MAKING:
   - weights = normalize(|best_solution|)
   - detection_scores = scaled_features × weights
   
   a. STRATEGY 1 - Adaptive Threshold:
      pred1 = (scores > mean(scores) + 0.5×std(scores)) ? 1 : 0
   
   b. STRATEGY 2 - Isolation Forest:
      iso_forest = IsolationForest(contamination=0.2)
      pred2 = iso_forest.fit_predict(scaled_features)
   
   c. STRATEGY 3 - Percentile Threshold:
      pred3 = (scores > percentile(scores, 75)) ? 1 : 0
   
   d. WEIGHTED VOTING:
      weights = [0.3, 0.3, 0.4]
      final_predictions = weighted_vote(pred1, pred2, pred3, weights)

5. RETURN final_predictions
```

## Key Algorithmic Improvements

### 1. Lévy Flight Movement
- **Purpose**: Enhance exploration capability
- **Implementation**: `step = u / |v|^(1/β)` where β = 1.5
- **Benefit**: Better escape from local optima

### 2. Adaptive Parameters
- **Frequency**: `f = 0.1 + 0.8 × exp(-t/T_max)`
- **Amplitude**: `A = 1.5 × (1 - t/T_max)`
- **Exchange Rate**: `ER = 0.3 × (1 - t/T_max)`

### 3. Enhanced Fitness Function
```
fitness = score_mean + 0.3×score_std + 0.2×|score_skew| - 0.1×clustering_penalty
```

### 4. Multi-Strategy Ensemble
- Combines multiple detection approaches
- Reduces false positives through voting
- Adapts to different network conditions

## Complexity Analysis

- **Time Complexity**: O(I × (D × F + B × F²))
  - I: iterations, D: dolphins, B: bees, F: features
- **Space Complexity**: O((D + B) × F + N²)
  - N: number of nodes for network graph

## Performance Metrics

The algorithm is evaluated using:
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Detection Time**: Computational efficiency
- **Accuracy**: (TP + TN) / (TP + FP + TN + FN)

## Implementation Notes

1. **Feature Normalization**: All features are scaled to [0,1] range
2. **Boundary Handling**: Solutions are clipped to [-3,3] range
3. **Early Stopping**: Prevents overfitting with patience mechanism
4. **Population Diversity**: Maintains exploration through periodic reinitialization
5. **Parameter Adaptation**: Dynamic adjustment based on iteration progress

## Usage Example

```python
# Initialize MANET simulator
manet = MANETSimulator(num_nodes=50, area_size=150, 
                      transmission_range=35, blackhole_percentage=0.25)

# Simulate network traffic
for _ in range(1000):
    src, dst = random.sample(range(50), 2)
    manet.simulate_packet_transmission(src, dst)

# Extract features and detect black holes
features = manet.collect_node_features()
detector = ImprovedHybridDEABCOptimizer()
predictions = detector.detect_blackholes(features)
```

## References

- Dolphin Echolocation Algorithm: Bio-inspired optimization technique
- Artificial Bee Colony: Swarm intelligence optimization
- Lévy Flight: Random walk with heavy-tailed probability distribution
- Ensemble Learning: Multiple classifier combination for improved accuracy
