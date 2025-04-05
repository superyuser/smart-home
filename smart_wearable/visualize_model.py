import graphviz
import os

def create_model_visualization():
    # Create a new directed graph
    dot = graphviz.Digraph(comment='LSTM Model Architecture')
    dot.attr(rankdir='TB')
    
    # Add input node
    dot.attr('node', shape='rectangle', style='rounded,filled', fillcolor='lightblue')
    dot.node('input', 'Input\n(10 timesteps × 5 features)\n[batch_size, 10, 5]')
    
    # Add LSTM layers
    dot.attr('node', shape='rectangle', style='rounded,filled', fillcolor='lightgreen')
    dot.node('lstm1', 'LSTM Layer 1\n64 units')
    dot.node('lstm2', 'LSTM Layer 2\n64 units')
    
    # Add dropout layer
    dot.attr('node', shape='rectangle', style='rounded,filled', fillcolor='lightpink')
    dot.node('dropout', 'Dropout Layer\n(20% dropout)')
    
    # Add fully connected layers
    dot.attr('node', shape='rectangle', style='rounded,filled', fillcolor='lightyellow')
    dot.node('fc1', 'Fully Connected\n64 → 16 units')
    
    # Add ReLU activation
    dot.attr('node', shape='rectangle', style='rounded,filled', fillcolor='lightgray')
    dot.node('relu', 'ReLU Activation')
    
    # Add output layer
    dot.attr('node', shape='rectangle', style='rounded,filled', fillcolor='lightcoral')
    dot.node('output', 'Output Layer\n16 → 5 classes\nSoftmax')
    
    # Add edges
    dot.edge('input', 'lstm1')
    dot.edge('lstm1', 'lstm2')
    dot.edge('lstm2', 'dropout')
    dot.edge('dropout', 'fc1')
    dot.edge('fc1', 'relu')
    dot.edge('relu', 'output')
    
    # Add feature labels
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Input Features')
        c.attr('node', shape='rectangle', style='filled', fillcolor='white')
        features = ['Heart Rate', 'HRV', 'Steps', 'Sleep Hours', 'Hour of Day']
        for i, feature in enumerate(features):
            c.node(f'feature_{i}', feature)
        c.attr(rank='same')
    
    # Add class labels
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='Output Classes')
        c.attr('node', shape='rectangle', style='filled', fillcolor='white')
        classes = ['Neutral', 'Focus', 'Fatigue', 'Stress', 'Emergency']
        for i, cls in enumerate(classes):
            c.node(f'class_{i}', cls)
        c.attr(rank='same')
    
    # Save the visualization
    dot.render('smart_wearable/model_architecture', format='png', cleanup=True)

if __name__ == "__main__":
    create_model_visualization() 