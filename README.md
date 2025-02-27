# Black Hole Attack Detection in MANETs using Dolphin-Bee Optimization

## Overview

This project implements a simulation of Mobile Ad-hoc Networks (MANETs) with black hole attack detection using a hybrid Dolphin-Bee optimization algorithm. The project includes both a simulation module and a Streamlit-based interactive web interface for visualizing and analyzing the detection process.

## Features

- **MANET Simulation**: Simulates a dynamic mobile ad-hoc network with configurable parameters
- **Black Hole Attack Modeling**: Implements malicious nodes that drop packets in the network
- **Hybrid Dolphin-Bee Optimization**: Novel approach combining dolphin echolocation and bee colony optimization for route optimization and attack detection
- **Interactive Visualization**: Streamlit-based UI to visualize network topology, attack detection, and performance metrics
- **Performance Analysis**: Tracks and displays network performance metrics including packet delivery ratio, energy consumption, and detection accuracy

## Technologies Used

- Python 3.x
- Streamlit for web interface
- NumPy for numerical computations
- Matplotlib for data visualization
- NetworkX for graph operations and network modeling

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PriyanujBora/Black-Hole-Attack-in-MANETs-with-Streamlit-Implementation.git
   cd Black-Hole-Attack-in-MANETs-with-Streamlit-Implementation
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Simulation

To run the basic simulation without the web interface:

```bash
python manet_dolphin_bee_simulation.py
```

### Running the Streamlit Web Application

To launch the interactive web application:

```bash
streamlit run streamlit_implementation.py
```

The web interface allows you to:
- Adjust network parameters (number of nodes, communication range, etc.)
- Control the percentage of malicious nodes
- Visualize the network in real-time
- Track detection metrics
- Analyze simulation results through various charts

## How It Works

1. **Network Initialization**: Creates a MANET with randomly placed nodes, some of which are malicious
2. **Traffic Simulation**: Simulates packet transmission between nodes
3. **Dolphin-Bee Optimization**:
   - Dolphin phase: Uses echolocation-inspired algorithm to explore potential routes
   - Bee phase: Uses bee colony optimization to exploit and refine routes
4. **Blackhole Detection**: Identifies malicious nodes based on behavioral analysis and network metrics

For a detailed explanation of the Dolphin-Bee Optimization algorithm and the blackhole detection process, see the [Algorithm Documentation](ALGORITHM.md).

## Project Structure

- `manet_dolphin_bee_simulation.py`: Core simulation module
- `streamlit_implementation.py`: Interactive web application
- `requirements.txt`: Project dependencies

## Future Work

- Implementation of other attack types (wormhole, sinkhole)
- Integration of additional optimization algorithms
- Performance comparison with traditional detection methods
- Real-time mobile device simulation

## License

[MIT License](LICENSE)

## Contributors

- [Priyanuj Bora](https://github.com/PriyanujBora), [Fahim Mashud Barbhuiyan](https://github.com/Fahim98), [Rohan Jaiswal](https://github.com/RohanJaiswall) - Main Developers
