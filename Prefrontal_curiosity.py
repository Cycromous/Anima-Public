import os
import json
import random

# Attempt to load spacy for keyword extraction
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

QUESTION_TEMPLATES = [
    "I noticed I don't have much context on {topic}. Can you tell me more about it?",
    "How does {topic} fit into the broader scope of what you are working on?",
    "Could you clarify your experience with {topic} so I can tailor my tools better?",
    "Have you worked with {topic} before, or is this new territory for you?",
    "What specific aspect of {topic} are you focused on right now?",
    "Is {topic} something central to your current project or more peripheral?",
    "What outcome are you hoping to achieve with {topic}?",
    "Are there any constraints I should know about regarding {topic}?",
]

def log_knowledge_gap_for_dreams(topic, question):
    filepath = "./ai_memory/dream_queue.json"
    
    # Load existing dream queue
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                gaps = json.load(f)
            except json.JSONDecodeError:
                gaps = []
    else:
        gaps = []
        
    # Check if the topic already exists in the subconscious
    for g in gaps:
        if g['topic'].lower() == topic.lower():
            if g['status'] == 'pending':
                return # It's already waiting to be dreamed about
            else:
                # It failed or was resolved before. Force a re-evaluation!
                g['status'] = 'pending'
                g['question'] = question
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(gaps, f, indent=4)
                print(f"\n[CURIOSITY ENGINE] Re-activated '{topic}' in the Hippocampus Dream queue.")
                return
            
    # If it's completely new, append it
    gaps.append({
        "topic": topic, 
        "question": question, 
        "status": "pending"
    })
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(gaps, f, indent=4)
        
    print(f"\n[CURIOSITY ENGINE] Added '{topic}' to the Hippocampus Dream queue.")


def detect_knowledge_gap(user_input, collection):
    if nlp is None:
        return {"knowledge_gap": False}
        
    doc = nlp(user_input)
    input_keywords = [token.text.lower() for token in doc 
                      if token.pos_ in ("NOUN", "PROPN") and len(token.text) > 2]
    
    if not input_keywords:
        return {"knowledge_gap": False}
    
    results = collection.query(
        query_texts=[user_input],
        n_results=5
    )
    
    existing_tags = []
    if results["metadatas"] and results["metadatas"][0]:
        for meta in results["metadatas"][0]:
            if meta and "tags" in meta:
                existing_tags.extend(meta["tags"].split(", "))
    
    unknown_keywords = [kw for kw in input_keywords if kw not in existing_tags]
    
    if unknown_keywords:
        # Grab the most prominent unknown concept
        target_topic = unknown_keywords[0] 
        # Formulate a research question for the headless browser
        research_question = f"What is {target_topic} in the context of {user_input}?"
        
        # QUEUE IT FOR REM SLEEP!
        log_knowledge_gap_for_dreams(target_topic, research_question)
        
        return {
            "knowledge_gap": True,
            "missing_information": f"No memory found for: {', '.join(unknown_keywords)}"
        }
    
    return {"knowledge_gap": False}


def generate_curiosity_questions(gap_info, model=None, tokenizer=None):
    raw_info = gap_info.get("missing_information", "")
    if "No memory found for:" in raw_info:
        topic = raw_info.replace("No memory found for:", "").strip()
    else:
        topic = raw_info or "your current project"
    return [t.format(topic=topic) for t in QUESTION_TEMPLATES]


def curiosity_learning_action(question_list):
    if not question_list or not isinstance(question_list, list):
        return None
    return random.choice(question_list)