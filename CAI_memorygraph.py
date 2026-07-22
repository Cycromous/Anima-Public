import os
import webbrowser

def generate_interactive_map(G):
    """
    Receives a NetworkX graph (G) and generates a highly interactive Pyvis HTML map.
    """
    print("\n[VISUALIZER] Initializing Memory Subsystem...")
    
    try:
        from pyvis.network import Network
    except ImportError:
        print("[VISUALIZER ERROR] Pyvis is not installed. Please run: pip install pyvis")
        return
        
    if G is None or len(G.nodes) == 0:
        print("\n[VISUALIZER] The memory graph is currently empty. Teach Anima some facts first!")
        return
        
    print("[VISUALIZER] Compiling interactive memory map...")

    net = Network(height='800px', width='100%', bgcolor='#1a1a1a', font_color='white', directed=True)

    net.from_nx(G)

    net.repulsion(node_distance=200, spring_length=250)

    output_file = "CAI_MemoryMap.html"
    net.save_graph(output_file)
    
    print(f"[VISUALIZER] Graph saved to {output_file}. Launching browser...")

    try:
        webbrowser.open('file://' + os.path.realpath(output_file))
    except Exception as e:
        print(f"[VISUALIZER WARNING] Could not open browser automatically: {e}")
        print(f"Please manually open the '{output_file}' file in your folder.")
