import networkx as nx
import json
import os
from Prefrontal_curiosity import log_knowledge_gap_for_dreams

GRAPH_FILE = "./ai_memory/temporal_relational_graph.json"

def load_graph():
    """Loads the persistent graph from disk, or creates a new one."""
    if os.path.exists(GRAPH_FILE):
        try:
            with open(GRAPH_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return nx.node_link_graph(data)
        except json.JSONDecodeError:
            print("[GRAPH WARNING] Relational file corrupted. Booting blank graph.")
            return nx.DiGraph()
    return nx.DiGraph()

def save_graph(G):
    """Saves the current state of the graph to disk."""
    data = nx.node_link_data(G)
    with open(GRAPH_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def add_memory_synapses_batch(triples):
    """
    Wires multiple connections efficiently by loading and saving the graph only once.
    Expects a list of tuples: [(subject, relation, object, confidence), ...]
    """
    if not triples:
        return False

    G = load_graph()
    conflicts_detected = []

    for triple in triples:
        subject = triple[0]
        relation = triple[1]
        obj = triple[2]
        confidence = triple[3] if len(triple) > 3 else 0.5

        subj_norm = subject.lower().strip()
        obj_norm = obj.lower().strip()

        G.add_edge(subj_norm, obj_norm, relation=relation, confidence=confidence)

        if relation == "CONTRADICTS":
            conflicts_detected.append(f"{subj_norm} and {obj_norm}")

    save_graph(G)

    for conflict in conflicts_detected:
        print(f"  -> [GRAPH] Contradiction detected: {conflict}. Queuing for REM sleep resolution.")
        log_knowledge_gap_for_dreams(
            topic=f"Contradiction: {conflict}", 
            question=f"Research and resolve the logical contradiction between {conflict}. Determine which is correct."
        )

    return True

def get_multihop_context(topic, max_depth=2):
    """
    Performs 'Spreading Activation'. 
    Finds a topic and returns all connected concepts up to a certain depth.
    """
    G = load_graph()
    topic = topic.lower().strip()
    
    if topic not in G:
        return None
    subgraph = nx.ego_graph(G, topic, radius=max_depth)
    
    context_lines = []
    for u, v, data in subgraph.edges(data=True):
        relation = data.get('relation', 'is related to')
        context_lines.append(f"[{u}] -> ({relation}) -> [{v}]")
        
    return "\n".join(context_lines)

def find_logical_path(start_concept, end_concept):
    """Uses Dijkstra's algorithm to find the shortest logical bridge between two concepts."""
    G = load_graph()
    start = start_concept.lower().strip()
    end = end_concept.lower().strip()
    
    if start not in G or end not in G:
        return None
        
    try:
        path = nx.shortest_path(G, source=start, target=end)
        
        path_str = []
        for i in range(len(path)-1):
            u = path[i]
            v = path[i+1]
            relation = G[u][v].get('relation', 'relates to')
            path_str.append(f"[{u}] -({relation})->")
            
        path_str.append(f"[{path[-1]}]")
        return " ".join(path_str)
        
    except nx.NetworkXNoPath:
        return f"No logical bridge found between {start} and {end}."
