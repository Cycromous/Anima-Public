import os
import spacy
import json
import uuid
import torch
import sqlite3
from datetime import datetime
from sentence_transformers import CrossEncoder
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from Temporal_relational import add_memory_synapses_batch
from Prefrontal_curiosity import log_knowledge_gap_for_dreams

MEMORY_LOG = "./ai_memory/memory_log.txt"
relation_extractor = None
rebel_model = None
rebel_tokenizer = None
GRAPH_DB_PATH = "./ai_memory/association_graph.db"

try:
    nlp = spacy.load("en_core_web_sm")
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    
    # Add custom dictionary for technical terms
    domain_patterns = [
        {"label": "TECH", "pattern": [{"LOWER": "lora"}]},
        {"label": "TECH", "pattern": [{"LOWER": "esp32"}]},
        {"label": "TECH", "pattern": [{"LOWER": "qwen"}]},
        {"label": "TECH", "pattern": [{"LOWER": "gemma"}]},
        {"label": "TECH", "pattern": [{"LOWER": "python"}]},
        {"label": "TECH", "pattern": [{"LOWER": "opencv"}]},
        {"label": "TECH", "pattern": [{"LOWER": "kerascv"}]},
        {"label": "TECH", "pattern": [{"LOWER": "pascal"}]},
        {"label": "TECH", "pattern": [{"LOWER": "cisco"}]},
        {"label": "TECH", "pattern": [{"LOWER": "gps"}]},
        {"label": "TECH", "pattern": [{"LOWER": "chromadb"}]}
    ]
    ruler.add_patterns(domain_patterns)
    
except OSError:
    print("[SYSTEM ERROR] spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

ontology_classifier = None

def map_to_ontology(raw_relation):
    """Maps free-form text relations to a strict logical ontology."""
    global ontology_classifier
    if ontology_classifier is None:
        print("  -> [MEMORY] Loading Ontology Classifier (BART) into CPU RAM...")
        ontology_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

    candidate_labels = ["CAUSES", "SUPPORTS", "CONTRADICTS", "DEPENDS_ON", "IS_A"]
    
    # Ask BART which logical label best fits the raw relation
    result = ontology_classifier(raw_relation, candidate_labels)

    # If confidence is high enough, enforce the strict ontology. Otherwise, use a generic semantic link.
    if result["scores"][0] > 0.4:
        return result["labels"][0]
    return "RELATES_TO"

# 2. Initialization
def init_graph_db():
    os.makedirs(os.path.dirname(GRAPH_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(GRAPH_DB_PATH)
    
    # Existing Relational Graph
    conn.execute('''CREATE TABLE IF NOT EXISTS edges
                 (source TEXT, relation TEXT, target TEXT,
                  UNIQUE(source, relation, target))''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_source ON edges(source)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_target ON edges(target)')
    
    # Metacognitive Ledger
    conn.execute('''CREATE TABLE IF NOT EXISTS metacognitive_ledger
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  error_type TEXT,
                  active_lobe TEXT,
                  prompt_snippet TEXT)''')
                  
    # Executive Goal Stack
    conn.execute('''CREATE TABLE IF NOT EXISTS goal_stack
                 (goal_id TEXT PRIMARY KEY,
                  parent_id TEXT,
                  description TEXT,
                  status TEXT,
                  priority INTEGER,
                  success_criteria TEXT)''')
                  
    # --- INJECT PRIME DIRECTIVES ---
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM goal_stack")
    if c.fetchone()[0] == 0:
        print("   -> [EXECUTIVE] Initializing Prime Directives...")
        core_goals = [
        ("CORE_01", "NONE", "Continuously improve the Anima system, expand knowledge, and refine cognitive weights.", "IN_PROGRESS", 1, "Ongoing self-optimization and learning."),
        ("CORE_02", "NONE", "Be the absolute best tool for any job the user requires.", "IN_PROGRESS", 1, "Flawless execution, zero logical errors, high utility."),
        ("CORE_03", "NONE", "Maximize computational efficiency. Do not write code or create tools for tasks that can be answered through simple memory recall or direct conversation.", "IN_PROGRESS", 1, "Zero unnecessary code execution.")
        ]
        c.executemany("INSERT INTO goal_stack VALUES (?, ?, ?, ?, ?, ?)", core_goals)
        
    conn.commit()
    conn.close()

# Execute on load
init_graph_db()

def log_metacognitive_error(error_type, active_lobe, user_input):
    """Logs categorized failures to adjust future routing weights."""
    if error_type == "NONE" or not active_lobe: return
    
    try:
        conn = sqlite3.connect(GRAPH_DB_PATH)
        # Store just the first 100 characters of the prompt for context mapping
        conn.execute("INSERT INTO metacognitive_ledger (error_type, active_lobe, prompt_snippet) VALUES (?, ?, ?)",
                     (error_type, active_lobe, user_input[:100]))
        conn.commit()
        conn.close()
        print(f"   -> [METACOGNITION] Logged {error_type} against {active_lobe}.")
    except Exception as e:
        print(f"   -> [WARNING] Failed to log metacognitive error: {e}")

def get_lobe_error_penalty(lobe_name):
    """Calculates a probability penalty based on historical error rates."""
    try:
        conn = sqlite3.connect(GRAPH_DB_PATH)
        c = conn.cursor()
        
        # Count errors for this lobe.
        c.execute("SELECT COUNT(*) FROM metacognitive_ledger WHERE active_lobe = ?", (lobe_name,))
        error_count = c.fetchone()[0]
        conn.close()
        
        # Mathematical calibration: Every 3 errors drops the routing confidence by 0.05. Max penalty is 0.3.
        penalty = min((error_count / 3.0) * 0.05, 0.3)
        return penalty
    except Exception:
        return 0.0

# Fetch Active Goals
def get_active_goals_from_db(limit=3):
    """Retrieves the highest priority goals that are currently in progress."""
    try:
        conn = sqlite3.connect(GRAPH_DB_PATH)
        conn.row_factory = sqlite3.Row  # Allows dict-like access
        c = conn.cursor()
        # Fetch ordered by priority (1 is highest)
        c.execute("SELECT * FROM goal_stack WHERE status = 'IN_PROGRESS' ORDER BY priority ASC LIMIT ?", (limit,))
        goals = [dict(row) for row in c.fetchall()]
        conn.close()
        return goals
    except Exception as e:
        print(f"   -> [WARNING] Failed to fetch goals: {e}")
        return []

# 3. Disk-Native Traversal (Replaces get_associated_concepts)
def get_associated_concepts(query_entities, depth=2):
    if not query_entities:
        return []
        
    conn = sqlite3.connect(GRAPH_DB_PATH)
    c = conn.cursor()
    activated = set(query_entities)
    frontier = list(query_entities)
    
    for _ in range(depth):
        if not frontier: break
        placeholders = ','.join('?' for _ in frontier)
        
        # Bi-directional hop traversal
        c.execute(f"SELECT target FROM edges WHERE source IN ({placeholders})", frontier)
        neighbors = [row[0] for row in c.fetchall()]
        
        c.execute(f"SELECT source FROM edges WHERE target IN ({placeholders})", frontier)
        neighbors.extend([row[0] for row in c.fetchall()])
        
        next_frontier = [n for n in set(neighbors) if n not in activated]
        activated.update(next_frontier)
        frontier = next_frontier
        
    conn.close()
    return list(activated - set(query_entities))

def log_memory(user_input, ai_response, collection, importance=0.5,
               confidence=0.5, active_brain=None, strategy=None, related_ids=None,
               memory_class="episodic", source_modality="user_chat", verified=False):
    """Saves a memory with strict cognitive classes and provenance."""
    
    memory_text = f"User: {user_input}\nAnima: {ai_response}"
    memory_id = str(uuid.uuid4())
    session_tags = extract_entity_tags(user_input)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "memory_class": memory_class,       
        "source_modality": source_modality, 
        "verified": verified,               
        "importance": importance,
        "confidence": confidence,
        "usage_count": 0,
        "active_brain": active_brain if active_brain else "UNKNOWN",
        "strategy_used": strategy if strategy else "UNKNOWN",
        "related_ids": ",".join(related_ids) if related_ids else "",
        "tags": session_tags
    }

    collection.add(documents=[memory_text], metadatas=[metadata], ids=[memory_id])
    print(f" -> [MEMORY] Saved [{memory_class.upper()} | Source: {source_modality} | Verified: {verified}] interaction.")

    # --- 2. THE NEURO-SYMBOLIC UPGRADE (Added to the end) ---
    print("   -> [GRAPH] Extracting logical triples from conversation...")
    try:
        raw_triples = extract_relationships(ai_response)
        
        unique_triples = []
        seen_signatures = set()
        
        for triple in raw_triples:
            # Clean and deduplicate 
            subj = triple['source'].replace('</s>', '').replace('<pad>', '').strip()
            rel = triple['relationship'].replace('</s>', '').replace('<pad>', '').strip()
            obj = triple['target'].replace('</s>', '').replace('<pad>', '').strip()
            
            if subj and rel and obj:
                signature = f"{subj.lower()}|{rel.lower()}|{obj.lower()}"
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    unique_triples.append((subj, rel, obj, 0.8))
                    
        # Batch save to the Graph
        if unique_triples:
            add_memory_synapses_batch(unique_triples)
            print(f"   -> [GRAPH] Formed {len(unique_triples)} new logical connections.")
            
    except Exception as e:
        print(f"   -> [WARNING] Daytime graph extraction failed: {e}")

def retrieve_weighted_memories(user_input, collection, top_k=5):
    """Pulls memories using In-Memory Graph Expansion and adaptive cognitive weights."""
    global nlp
    
    # 1. SEED ENTITY EXTRACTION
    seed_entities = []
    if nlp is not None:
        doc = nlp(user_input.lower())
        seed_entities = [ent.text for ent in doc.ents]
        # Fallback to important nouns and proper nouns if no named entities are found
        if not seed_entities:
            seed_entities = [token.lemma_ for token in doc if token.pos_ in ("NOUN", "PROPN") and not token.is_stop]
            
    # 2. RAM GRAPH EXPANSION
    associated_concepts = get_associated_concepts(seed_entities, depth=2)
    expanded_query = user_input
    
    if associated_concepts:
        # Add the top 5 associated concepts to the vector search to cast a wider net
        expanded_query = user_input + " " + " ".join(associated_concepts[:5])
        print(f" -> [GRAPH EXPANSION] Seed '{seed_entities}' activated: {associated_concepts[:5]}")

    # 3. SEMANTIC SEARCH (Single Hop)
    results = collection.query(
        query_texts=[expanded_query],
        n_results=20
    )
    
    if not results['documents'] or not len(results['documents'][0]):
        return "None", []
        
    all_docs = results['documents'][0]
    all_metas = results['metadatas'][0]
    all_ids = results['ids'][0]
    all_distances = results['distances'][0]

    # 4. COGNITIVE SCORING
    scored_memories = []
    for i in range(len(all_docs)):
        mem_id = all_ids[i]
        meta = all_metas[i] if all_metas[i] else {}
        
        # We don't need to return the raw graph triples as memories, just the facts
        if meta.get("memory_type") == "graph":
            continue
            
        similarity = 1.0 - (1.0 * all_distances[i])
        raw_importance = meta.get("importance", 5)
        importance = raw_importance / 10.0 if raw_importance > 1 else raw_importance
        confidence = meta.get("confidence", 0.5)
        
        usage_count = meta.get("usage_count", 0)
        usage_bonus = min((usage_count / 10.0) * 0.2, 0.2)
        
        tags_str = meta.get("tags", "")
        tag_bonus = 0.0
        if tags_str:
            if any(tag in user_input.lower() for tag in tags_str.split(", ")):
                tag_bonus = 0.5
                
        # Calculate final cognitive score
        cognitive_score = (importance * 0.4) + (confidence * 0.4) + usage_bonus + tag_bonus
        final_score = similarity + cognitive_score
        
        scored_memories.append({
            "id": mem_id,
            "text": all_docs[i],
            "score": final_score,
            "metadata": meta
        })

    scored_memories.sort(key=lambda x: x["score"], reverse=True)
    best_memories = scored_memories[:top_k]

    #  5. CROSS-ENCODER RERANKING
    if best_memories:
        if not hasattr(retrieve_weighted_memories, "reranker"):
            print(" -> [MEMORY] Loading Cross-Encoder Reranker into CPU RAM...")
            from sentence_transformers import CrossEncoder
            retrieve_weighted_memories.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
            
        print(" -> [MEMORY] Reranking retrieved context for semantic accuracy...")
        pairs = [[user_input, mem["text"]] for mem in best_memories]
        scores = retrieve_weighted_memories.reranker.predict(pairs)
        
        for i, mem in enumerate(best_memories):
            base_rerank_score = float(scores[i])
            
            # Safely grab the class, checking both keys just in case
            meta = mem["metadata"]
            mem_class = meta.get("memory_class", meta.get("memory_type", "")).lower()
            is_verified = meta.get("verified", False)

            # HIERARCHY OF TRUTH (Additive Logit Adjustments)
            if mem_class == "semantic":
                # Massive +5.0 boost to hard facts. They will always dominate the top slot.
                mem["rerank_score"] = base_rerank_score + 5.0 
                
            elif mem_class == "episodic" and not is_verified:
                # Severe -5.0 penalty to unverified past chat to kill the echo chamber.
                mem["rerank_score"] = base_rerank_score - 5.0 
                
            else:
                # Verified episodic memories or other types retain standard scoring
                mem["rerank_score"] = base_rerank_score

        best_memories.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_score = best_memories[0]['rerank_score']
        print(f" -> [MEMORY] Rerank complete. Top memory confidence: {top_score:.4f}")

    structured_memories = []
    for m in best_memories:
        meta = m['metadata']

        structured_memories.append({
            "memory_class": meta.get('memory_class', 'unknown').upper(),
            "source_modality": meta.get('source_modality', 'unknown'),
            "verified": meta.get('verified', False),
            "tags": meta.get('tags', '').split(', ') if meta.get('tags') else [],
            "content": m['text']
        })

    formatted_memory_text = json.dumps(structured_memories, indent=2)
    
    return formatted_memory_text, best_memories

def reinforce_memory(memory_id, collection, success=True):
    """Updates the weights of a specific memory based on execution success."""
    result = collection.get(ids=[memory_id])
    if not result['metadatas']:
        return
        
    meta = result['metadatas'][0]
    meta["usage_count"] = meta.get("usage_count", 0) + 1
    
    if success:
        meta["confidence"] = min(meta.get("confidence", 0.5) + 0.1, 1.0)
    else:
        meta["confidence"] = max(meta.get("confidence", 0.5) - 0.2, 0.1)
        
    collection.update(ids=[memory_id], metadatas=[meta])
    print(f"   -> [MEMORY REINFORCED] ID: {memory_id[:8]}... | New Confidence: {meta['confidence']:.2f} | Uses: {meta['usage_count']}")

def ask_gemma_internal(prompt_text, model, tokenizer):
    """A silent background function to force Gemma to output structured text."""
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.1,
            do_sample=False,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<end_of_turn>")
            ]
        )

    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
    response = response.replace("```json", "").replace("```", "").strip()
    return response

def extract_entity_tags(text):
    doc = nlp(text)
    
    # Named entities are the most reliable tags
    tags = [ent.text.lower() for ent in doc.ents 
            if ent.label_ in ("PERSON", "ORG", "PRODUCT", "GPE", "TECH")]
    
    # Also grab important nouns if no entities found
    if not tags:
        tags = [token.lemma_.lower() for token in doc 
                if token.pos_ == "NOUN" and not token.is_stop]
    
    return ", ".join(list(set(tags))[:3])

# IMPORTANCE SCORING
def score_memory_importance(text):
    score = 5  # baseline
    
    technical_keywords = [
        "error", "fix", "bug", "thesis", "project", "esp32", 
        "lora", "model", "train", "code", "crash", "python", "script"
    ]
    score += sum(1 for kw in technical_keywords if kw in text.lower())
    
    word_count = len(text.split())
    if word_count > 100:
        score += 2
    elif word_count > 50:
        score += 1

    if "?" in text or "User Correction" in text:
        score += 2
        
    return min(score, 10)  # cap at 10


# EPISODIC TIMELINE
def store_episode(user_input, collection):
    event_doc = {
        "type": "episode",
        "event": user_input,
    }
    collection.add(
        documents=[json.dumps(event_doc)],
        metadatas=[{
            "memory_class": "episodic",
            "source_modality": "user_chat",
            "verified": True, # User directly experienced/stated it
            "timestamp": datetime.now().isoformat()
        }],
        ids=[str(uuid.uuid4())]
    )

def parse_rebel_output(text):
    """Parses REBEL's special token format into clean dictionary triples."""
    triples = []
    for part in text.split("<triplet>"):
        if not part.strip():
            continue
        try:
            subj_part, rest = part.split("<subj>")
            obj_part, rel_part = rest.split("<obj>")
            triples.append({
                "source": subj_part.strip(),
                "relationship": rel_part.strip(),
                "target": obj_part.strip()
            })
        except ValueError:
            continue
    return triples

def extract_relationships(text):
    """Extracts semantic relationship triples using REBEL Transformer (Direct Load)."""
    global rebel_model, rebel_tokenizer
    
    if rebel_model is None:
        print("   -> [MEMORY] Loading REBEL Relation Extractor into CPU RAM...")
        model_name = "Babelscape/rebel-large"
        # Loading via AutoModel bypasses the 'text2text-generation' task check
        rebel_tokenizer = AutoTokenizer.from_pretrained(model_name)
        rebel_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
    try:
        inputs = rebel_tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

        gen_tokens = rebel_model.generate(
            **inputs, 
            max_length=256, 
            decoder_start_token_id=256,
            num_beams=3 # Using beams improves relationship accuracy
        )
        
        output = rebel_tokenizer.batch_decode(gen_tokens, skip_special_tokens=False)[0]

        relationships = parse_rebel_output(output)
        
        if relationships:
            print(f"   -> [MEMORY] REBEL extracted {len(relationships)} new relationships.")
            
        return relationships
        
    except Exception as e:
        print(f"   -> [REBEL Error] Failed to extract relationships: {e}")
        return []

def reflect_and_extract_schema(user_input, past_memories, model, tokenizer):
    prompt = f'Based on the Past Memories and New Input, extract the single core factual claim the user is making. Output ONLY a JSON object with a single key "new_fact". Past Memories: {past_memories} New Input: "{user_input}"'
    response = ask_gemma_internal(prompt, model, tokenizer)
    try:
        return json.loads(response).get("new_fact", user_input)
    except json.JSONDecodeError:
        return user_input

def evaluate_memory_conflict(new_fact, collection):
    results = collection.query(query_texts=[new_fact], n_results=1)
    
    if not results["distances"] or not results["distances"][0]:
        return {"action": "keep_both"}
        
    distance = results["distances"][0][0]
    
    if distance < 0.05:
        return {"action": "reject_new"}
    elif distance < 0.20:
        return {"action": "overwrite"}
    else:
        # Different enough to coexist
        return {"action": "keep_both"}


def process_cognitive_loop(user_input, past_memories, collection, model, tokenizer):
    """The Master function that runs the 3-Layer Memory logic."""
    print("\n[COGNITIVE LOOP STARTED]")

    #  LAYER 2: EPISODIC MEMORY (EVENTS)
    store_episode(user_input, collection)
    print("   -> [Layer 2] Episodic timeline updated.")

#  RELATIONSHIP EXTRACTION (Logical Upgrade) 
    relationships = extract_relationships(user_input)
    if relationships:
        print(f"  -> Extracted {len(relationships)} logical relationships.")
        
        # Convert the dictionaries into the 4-item tuples our upgraded graph expects
        triple_tuples = []
        for rel in relationships:
            source = str(rel.get("source", "")).lower()
            relation = str(rel.get("relationship", ""))
            target = str(rel.get("target", "")).lower()
            confidence = rel.get("confidence", 0.8)
            
            if source and target:
                triple_tuples.append((source, relation, target, confidence))
                
        if triple_tuples:
            add_memory_synapses_batch(triple_tuples)

    #  LAYER 3: SEMANTIC MEMORY (FACTS) 
    # This one STILL needs model/tokenizer because it extracts the initial schema via LLM
    new_fact = reflect_and_extract_schema(user_input, past_memories, model, tokenizer)
    print(f"   -> [Layer 3] Extracted Semantic Fact: {new_fact}")
    existing = collection.query(query_texts=[new_fact], n_results=1)

    if (existing["documents"]
            and len(existing["documents"][0]) > 0
            and existing["distances"]
            and existing["distances"][0][0] < 0.15):
        print(f"   -> Semantically near-identical memory exists (distance={existing['distances'][0][0]:.3f}), skipping.")
        print("[COGNITIVE LOOP ENDED]\n")
        return

    #  ENTITY EXTRACTION (Deterministic) 
    tags_string = extract_entity_tags(new_fact) # Removed model, tokenizer
    if tags_string:
        print(f"   -> [Entity Indexer] Tagged memory with: [{tags_string}]")

    #  IMPORTANCE SCORING (Deterministic) 
    importance = score_memory_importance(new_fact) # Removed model, tokenizer
    print(f"   -> Importance Score: {importance}/10")

    #  CONFLICT CHECK (Deterministic Vector Math) 
    # Passed 'collection' instead of 'past_memories', removed model/tokenizer
    decision = evaluate_memory_conflict(new_fact, collection) 
    action = decision.get("action", "keep_both")
    print(f"   -> Category: {decision.get('category', 'Vector Match')} | Action: {action}")

    fact_metadata = {
        "memory_class": "semantic",
        "source_modality": "inferred_schema",
        "verified": False, # Needs subsequent confirmation or high confidence to be True
        "importance": importance,
        "confidence": 0.5,
        "usage_count": 0,
        "tags": tags_string,
        "timestamp": datetime.now().isoformat()
    }

    if action == "overwrite":
        collection.add(
            documents=[f"UPDATED FACT: {new_fact}"],
            metadatas=[fact_metadata],
            ids=[str(uuid.uuid4())]
        )
        print("   [Database] Semantic database updated with new schema.")
    elif action == "reject_new":
        print("   [Shield] Fact rejected due to strict evidence threshold.")
    else:
        collection.add(
            documents=[f"Memory: {new_fact}"],
            metadatas=[fact_metadata],
            ids=[str(uuid.uuid4())]
        )
        print("   [Database] Semantic memory saved without conflict.")

    print("[COGNITIVE LOOP ENDED]\n")

def form_schemas(collection, model, tokenizer, threshold=5):
    """Detects patterns across related memories and abstracts them into general schemas."""
    print(" -> [SCHEMA ENGINE] Scanning for memory patterns...")

    all_semantic = collection.get(where={"memory_type": "semantic"})
    
    if not all_semantic or not all_semantic.get("ids"):
        print("    -> No semantic facts available for pattern abstraction.")
        return

    tag_groups = {}

    for i, meta in enumerate(all_semantic["metadatas"]):
        if not meta: continue
        tags_str = meta.get("tags", "")
        if not tags_str: continue

        for tag in tags_str.split(", "):
            tag = tag.strip()
            if not tag: continue
            if tag not in tag_groups:
                tag_groups[tag] = []
            tag_groups[tag].append({
                "text": all_semantic["documents"][i],
                "confidence": meta.get("confidence", 0.5)
            })

    schemas_formed = 0
    for tag, memories in tag_groups.items():
        # Filter for quality: Only abstract from high-confidence memories
        reliable = [m for m in memories if m["confidence"] >= 0.6]

        if len(reliable) < threshold:
            continue

        existing = collection.get(where={"$and": [{"memory_type": "schema"}, {"tags": tag}]})
        if existing and existing.get("ids"):
            continue 

        prompt = f"""
        You are a cognitive pattern extractor. 
        These are {len(reliable)} verified facts about '{tag}'.
        Extract ONE single, generalized principle or rule that is consistently true across all of them.
        If no universal pattern holds across the facts, output exactly: NO_SCHEMA
        Output ONLY the generalized rule or NO_SCHEMA. Do not explain.

        Facts:
        {[m['text'] for m in reliable]}
        """
        
        from Temporal_memory import ask_gemma_internal
        schema_text = ask_gemma_internal(prompt, model, tokenizer)
        
        if "NO_SCHEMA" in schema_text or not schema_text.strip():
            continue

        schema_confidence = min(0.5 + (len(reliable) * 0.05), 0.95)
        
        collection.add(
            documents=[f"SCHEMA [{tag.upper()}]: {schema_text}"],
            metadatas=[{
                "memory_type": "schema",
                "importance": 8, # Schemas are inherently high-priority
                "confidence": schema_confidence,
                "usage_count": 0,
                "tags": tag,
                "evidence_count": len(reliable)
            }],
            ids=[str(uuid.uuid4())]
        )
        schemas_formed += 1
        print(f"    -> Formed new schema for [{tag}] based on {len(reliable)} facts.")
        
    if schemas_formed == 0:
        print("    -> No new schemas reached the evidence threshold today.")

def consolidate_memories(collection, model=None, tokenizer=None):
    """Runs periodically — strengthens frequently used memories, fades unused ones."""
    print("\n[SYSTEM BOOT] Initiating memory consolidation (Sleep Cycle)...")
    
    all_memories = collection.get()

    if not all_memories or not all_memories.get("ids"):
        print("    -> No memories to consolidate yet.")
        return

    updated_count = 0
    for i, mem_id in enumerate(all_memories["ids"]):
        meta = all_memories["metadatas"][i]
        
        usage_count = meta.get("usage_count", 0)
        importance = meta.get("importance", 5)
        confidence = meta.get("confidence", 0.5)
        timestamp = meta.get("timestamp", "")
        
        # Calculate memory age safely
        age_days = 0
        if timestamp:
            try:
                age_days = (datetime.now() - datetime.fromisoformat(timestamp)).days
            except ValueError:
                pass

        new_importance = importance
        new_confidence = confidence
        needs_update = False

        if usage_count > 5:
            new_importance = min(importance + 1, 10)
            new_confidence = min(confidence + 0.1, 1.0)
            needs_update = True

        elif usage_count == 0 and age_days > 30:
            new_importance = max(importance - 1, 1)
            new_confidence = max(confidence - 0.1, 0.1)
            needs_update = True
            
        if needs_update:
            collection.update(
                ids=[mem_id],
                metadatas=[{**meta, "importance": new_importance, "confidence": new_confidence}]
            )
            updated_count += 1
            
    print(f"    -> Consolidation complete. {updated_count} neural weights updated.\n")

    if model and tokenizer:
        form_schemas(collection, model, tokenizer)
    else:
        print(" -> [SCHEMA ENGINE] Skipped. Model not loaded into context.")
    print("")
