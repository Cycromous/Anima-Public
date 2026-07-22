import json
import time
import uuid
import requests
import re
from bs4 import BeautifulSoup
from ddgs import DDGS
from Temporal_memory import extract_relationships, ask_gemma_internal
from Temporal_relational import add_memory_synapses_batch, get_multihop_context

def embed_and_save_text(text, collection, source="Night Shift"):
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    for chunk in chunks:
        if chunk.strip():
            collection.add(
                documents=[chunk],
                metadatas=[{"memory_type": "raw_document", "source_file": source}],
                ids=[str(uuid.uuid4())]
            )
    print(f"      -> [VECTOR] Saved {len(chunks)} chunks to ChromaDB.")

def scrape_website(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        for script in soup(["script", "style", "nav", "footer"]):
            script.extract()
        text = soup.get_text(separator=' ', strip=True)
        return text[:5000]
    except Exception as e:
        print(f"      -> [ERROR] Could not scrape {url}: {e}")
        return None

def trigger_rem_sleep(collection, model, tokenizer):
    print("\n[HIPPOCAMPUS DREAMS] Initiating Active Inference REM Sleep...")
    filepath = "./ai_memory/dream_queue.json"
    
    try:
        with open(filepath, "r") as f:
            gaps = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("   -> [HIPPOCAMPUS] No pending dreams. System is resting.")
        return

    pending_gaps = [g for g in gaps if g['status'] == 'pending']
    if not pending_gaps:
        print("   -> [HIPPOCAMPUS] Dream queue is clear. System is resting.")
        return

    with DDGS() as ddgs:
        for gap in pending_gaps:
            topic = gap['topic']
            print(f"\n-> [DREAMING ABOUT]: {topic}")
            
            # PHASE 1: PREDICTION (Hypothesis Generation)
            print("   -> [PREDICTION] Polling Graph DB for context...")
            context = get_multihop_context(topic, max_depth=2) or "No prior context available."
            
            hyp_prompt = f"""
            You are a cognitive engine. You have detected a knowledge gap regarding: '{topic}'.
            Current Graph Context: {context}
            
            Based on the context, formulate 3 distinct hypotheses about what '{topic}' is, how it works, or how it solves the user's problem.
            Output ONLY a valid JSON list of 3 strings. Example: ["Hypothesis 1", "Hypothesis 2", "Hypothesis 3"]
            """
            
            raw_hypotheses = ask_gemma_internal(hyp_prompt, model, tokenizer)
            try:
                clean_hyp = re.sub(r'```json\n(.*?)\n```', r'\1', raw_hypotheses, flags=re.DOTALL)
                clean_hyp = clean_hyp.replace('```', '').strip()
                hypotheses = json.loads(clean_hyp)
                print(f"   -> [HYPOTHESES FORMED]:\n      1. {hypotheses[0][:60]}...\n      2. {hypotheses[1][:60]}...")
            except Exception as e:
                print(f"   -> [WARNING] Hypothesis generation failed: {e}. Falling back to blind search.")
                hypotheses = [f"I need to learn what {topic} is."]

            # PHASE 2: EXPERIMENT (Targeted Search Formulation)
            exp_prompt = f"""
            To prove or disprove these hypotheses about '{topic}': {hypotheses}
            What is the single best web search query to find the ground truth?
            Output ONLY the search query string.
            """
            search_query = ask_gemma_internal(exp_prompt, model, tokenizer).strip().replace('"', '')
            print(f"   -> [EXPERIMENT] Executing targeted search: '{search_query}'")
            
            try:
                results = list(ddgs.text(search_query, max_results=3))
            except Exception as e:
                print(f"   -> [ERROR] DuckDuckGo API failed: {e}")
                results = []
                
            if not results:
                print("   -> [WARNING] Search engine returned 0 results.")
                continue
                
            combined_research = ""
            for res in results:
                print(f"   -> Absorbing: {res['title']}")
                article_text = scrape_website(res['href'])
                if article_text:
                    combined_research += f"\n\nSource: {res['title']}\n{article_text}"
            
            # PHASE 3: EPISTEMIC UPDATE (Surprise Calculation)
            if combined_research.strip():
                print("   -> [EVALUATION] Calculating epistemic surprise metric...")
                
                eval_research = combined_research[:3000] 
                
                surprise_prompt = f"""
                You generated these hypotheses about a topic: {hypotheses}
                Here is the actual factual research retrieved: {eval_research}
                
                On a scale of 0.0 (The hypotheses were completely correct/expected) to 1.0 (The hypotheses were completely wrong/surprising), rate the outcome.
                Output ONLY a single float number.
                """
                
                try:
                    surprise_str = ask_gemma_internal(surprise_prompt, model, tokenizer)
                    match = re.search(r'0\.\d+|1\.0', surprise_str)
                    surprise_score = float(match.group(0)) if match else 0.5
                except Exception:
                    surprise_score = 0.5
                
                print(f"   -> [SURPRISE SCORE]: {surprise_score:.2f}")
                
                epistemic_weight = 5.0 + (surprise_score * 5.0)
                
                print("   -> [CONSOLIDATING] Forging new neural pathways...")
                embed_and_save_text(combined_research, collection, source=f"REM Cycle: {topic}", importance=epistemic_weight)
                
                try:
                    raw_triples = extract_relationships(combined_research[:1500])
                    unique_triples = []
                    seen_signatures = set()
                    
                    for triple in raw_triples:
                        subj = triple['source'].replace('</s>', '').replace('<pad>', '').strip()
                        rel = triple['relationship'].replace('</s>', '').replace('<pad>', '').strip()
                        obj = triple['target'].replace('</s>', '').replace('<pad>', '').strip()
                        
                        if not subj or not rel or not obj:
                            continue
                            
                        signature = f"{subj.lower()}|{rel.lower()}|{obj.lower()}"
                        if signature not in seen_signatures:
                            seen_signatures.add(signature)
                            unique_triples.append((subj, rel, obj))
                            
                    if unique_triples:
                        add_memory_synapses_batch(unique_triples)
                        print(f"   -> [GRAPH] {len(unique_triples)} new synapses formed based on research.")
                except Exception as e:
                    print(f"   -> [WARNING] Graph extraction failed: {e}")
                
                gap['status'] = 'resolved'
                print(f"   -> [SUCCESS] Concept Mastered: {topic}")
            else:
                gap['status'] = 'failed'
                print("   -> [FAILED] Dream collapsed. Could not extract data.")
                
            time.sleep(5)

        with open(filepath, "w") as f:
            json.dump(gaps, f, indent=4)
            
    print("\n[HIPPOCAMPUS DREAMS] REM Cycle Complete. Waking up smarter.")
