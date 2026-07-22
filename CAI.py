import os
import chromadb
import re
import json
import random
import uuid
import threading
import queue
import time
import warnings
import gc
import torch
import importlib.util
from unittest import result
from urllib import response
from datetime import datetime
from contextlib import nullcontext
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModelForCausalLM, NougatProcessor, VisionEncoderDecoderModel, pipeline, BitsAndBytesConfig
from peft import PeftModel
from transformers import logging

# Mute standard Python warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Mute Hugging Face transformers warnings (only show actual errors)
logging.set_verbosity_error()

# MODULAR ARCHITECTURE SWITCHBOARD
ARCHITECTURE_MODE = "NATIVE_4B"

if ARCHITECTURE_MODE == "NATIVE_4B":
    USE_ADAPTERS = False
    MODEL_PATH = "/home/johnray/Personal/gemma-3-transformers-gemma-3-4b-it-qat-int4-unquantized-v1"

brain_pipeline = queue.Queue()

# --- PRELOAD INTENT CLASSIFIER (BART) ---
print(" -> [SYSTEM BOOT] Loading BART Intent Classifier into CPU RAM...")
try:
    from transformers import pipeline
    # device=-1 forces it to use CPU RAM, keeping your GPU VRAM strictly for Gemma
    intent_classifier = pipeline(
        "zero-shot-classification", 
        model="facebook/bart-large-mnli", 
        device=-1 
    )
    print("    -> BART Classifier ready.")
except Exception as e:
    print(f" -> [WARNING] Failed to load BART. System will fall back to exact-match routing. Error: {e}")
    intent_classifier = None

# --- BIOLOGICAL LOBE IMPORTS ---
from Temporal_memory import log_memory, process_cognitive_loop, ask_gemma_internal
from Frontal_learning import learn_from_pdf, learn_from_images
from Parietal_skills import load_existing_skills, SKILL_REGISTRY, generate_and_save_skill
from Prefrontal_planner import run_planning_loop
from Hippocampus import reconstruct_memory
from Prefrontal_curiosity import detect_knowledge_gap, generate_curiosity_questions, curiosity_learning_action, log_knowledge_gap_for_dreams
from Temporal_memory import extract_relationships 
from Temporal_relational import add_memory_synapses_batch, load_graph

# --- CONVERSATIONAL STATE GLOBALS ---
SYSTEM_STATE = "NORMAL"
BASE_PROMPT = ""
ACCUMULATED_FEEDBACK = ""
CORRECTION_ATTEMPT_COUNT = 0 

# 1. Optimize Memory Allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 2. Setup Memory Brain
print("Initializing Memory Brain...")
chroma_client = chromadb.PersistentClient(path="./ai_memory")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2", device="cpu")
collection = chroma_client.get_or_create_collection(name="long_term_memory", embedding_function=emb_fn)

# 3. Load The Base Brain
print(f"\n--- Loading Model: {ARCHITECTURE_MODE} ---")

if ARCHITECTURE_MODE == "NATIVE_4B":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    print("--- Using Native 4B Logic (Adapters Disabled) ---")

def requires_calculation(prompt, classifier_pipeline=None):
    """
    A system-level deterministic gate to check if a prompt requires arithmetic.
    Avoids brittle substring matches like "total" triggering on "totalitarian".
    """
    prompt_lower = prompt.lower()
    
# 1. STRICT REGEX HEURISTIC
    math_patterns = [
        r"\bcalculate\b", r"\bcompute\b", r"\bmultiply\b", 
        r"\bdivide\b", r"\bsubtract\b"
    ]
    
    numeric_context_patterns = [
        r"\bhow much\b.*\d", r"\bhow many\b.*\d", 
        r"\btotal\b.*\d", r"\bsum\b.*\d",
        r"\bremaining\b.*\d", r"\baverage\b.*\d"
    ]
    
    for pattern in math_patterns:
        if re.search(pattern, prompt_lower): return True
        
    for pattern in numeric_context_patterns:
        if re.search(pattern, prompt_lower): return True
            
    # 2. NUMERIC HEURISTIC
    if re.search(r'\d+\s*[\+\-\*\/%]\s*\d+', prompt_lower) or "% of" in prompt_lower:
        return True

    # 3. SEMANTIC FALLBACK (Using your existing BART zero-shot pipeline)
    if classifier_pipeline:
        candidate_labels = ["mathematical calculation", "general factual question"]
        try:
            result = classifier_pipeline(prompt, candidate_labels)
            if result['labels'][0] == "mathematical calculation" and result['scores'][0] > 0.65:
                return True
        except Exception:
            pass

    return False  

# --- RUN THE SLEEP CYCLE NOW THAT THE BRAIN IS ONLINE ---
from Temporal_memory import consolidate_memories
import threading

def delayed_sleep_cycle():
    import time
    print(" -> [SYSTEM] Sleep cycle paused for 3 minutes to prioritize boot operations...")
    time.sleep(180) 
    print(" -> [SYSTEM] Waking up background consolidation thread.")
    consolidate_memories(collection, model, tokenizer)

print(" -> [SYSTEM BOOT] Launching Sleep Cycle in background thread...")
sleep_thread = threading.Thread(
    target=delayed_sleep_cycle,
    daemon=True
)
sleep_thread.start()

load_existing_skills()

def get_deterministic_intent(user_input):
    """
    Scans natural language for strict, deterministic routing patterns.
    """
    user_text = user_input.lower()
    
    routing_rules = {
        "CODE_BRAIN": [
            r"\b(need|build|create|make|write|generate|code)\b.{0,30}\b(tool|script|app|database|python|skill)\b",
            r"\b(fix|debug|repair|solve)\b.*\b(code|error|bug|traceback)\b"
        ],
        "DATABASE_INTENT": [
            r"\b(find|search|lookup|pull|open|details|info|tell|who|give|show|get|what)\b.*\b(client|file|record|case|smith|griffin|name)\b"
        ]
    }

    for intent, patterns in routing_rules.items():
        for pattern in patterns:
            if re.search(pattern, user_text):
                return intent
                
    return "UNKNOWN"

# --- SAFE ADAPTER UTILITIES ---
def safe_disable_adapter(model_instance):
    """Safely disables adapter if the model supports it."""
    if hasattr(model_instance, "disable_adapter"):
        return model_instance.disable_adapter()
    return nullcontext()

def switch_brain(model_instance, target_brain, use_adapters):
    """Modular function to handle brain switching regardless of base model."""
    if not use_adapters:
        return nullcontext()
        
    if target_brain == "MATH_BRAIN":
        model_instance.set_adapter("math_brain")
        return nullcontext()
    else:
        return safe_disable_adapter(model_instance)


def execute_reflex_action(state_json):
    """Instantly fires off a tool from the SKILL_REGISTRY based on sensory input."""
    tool_name = state_json.get("recommended_tool", "NONE")
    action_data = state_json.get("actionable_data", {})
    
    if tool_name == "NONE":
        return "No specific tool recommended."
        
    tool_path = f"./skills/{tool_name}.py"
    if not os.path.exists(tool_path):
        return f"Reflex failed: Tool '{tool_name}' does not exist in registry."
        
    try:
        # Dynamically load and run the tool
        spec = importlib.util.spec_from_file_location(tool_name, tool_path)
        tool_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool_module)
        
        # Pass the extracted actionable data directly into the tool
        result = tool_module.run_skill(json.dumps(action_data))
        return f"Executed {tool_name}: {result}"
    except Exception as e:
        return f"Reflex execution error: {e}"

class CognitiveScratchpad:
    def __init__(self, user_prompt):
        self.original_prompt = user_prompt
        self.thoughts = []
        self.tool_results = []
        self.iteration = 0
        self.is_resolved = False
        
    def add_thought(self, thought):
        self.thoughts.append(f"[Thought {self.iteration}]: {thought}")
        
    def add_tool_output(self, tool_name, output):
        self.tool_results.append(f"[Tool: {tool_name}]: {output}")
        
    def compile_workspace(self):
        """Compiles the internal state for the LLM to read."""
        workspace = f"--- INTERNAL WORKSPACE (Iteration {self.iteration}/5) ---\n"
        workspace += f"Goal: {self.original_prompt}\n\n"
        if self.thoughts:
            workspace += "Past Thoughts:\n" + "\n".join(self.thoughts) + "\n\n"
        if self.tool_results:
            workspace += "Tool Outputs:\n" + "\n".join(self.tool_results) + "\n\n"
        return workspace

def deliberate_and_execute(user_input, system_rules, model, tokenizer):
    """The System 2 Thinking Loop. Replaces the single-shot generation."""
    print("\n[PREFRONTAL CORTEX] Entering Deliberation Workspace...")
    scratchpad = CognitiveScratchpad(user_input)
    from Temporal_memory import ask_gemma_internal
    import json
    import re
    
    while scratchpad.iteration < 5 and not scratchpad.is_resolved:
        scratchpad.iteration += 1
        
        prompt = f"""
        {system_rules}
        
        {scratchpad.compile_workspace()}
        
        Analyze the workspace. What is your next move? 
        If you need to run a tool, output a JSON: {{"action": "run_tool", "tool": "tool_name", "args": {{"key": "value"}}}}
        If you need to think or plan, output a JSON: {{"action": "think", "thought": "your reasoning"}}
        If the goal is fully achieved, output a JSON: {{"action": "finalize", "answer": "The final response to the user."}}
        
        Output ONLY the JSON.
        """
        response = ask_gemma_internal(prompt, model, tokenizer)
        
        try:
            clean_json = re.sub(r'```json\n(.*?)\n```', r'\1', response, flags=re.DOTALL).replace('```', '').strip()
            decision = json.loads(clean_json)
            
            if decision.get("action") == "think":
                print(f"   -> [THINKING]: {decision.get('thought')[:80]}...")
                scratchpad.add_thought(decision.get("thought"))
                
            elif decision.get("action") == "run_tool":
                tool_name = decision.get("tool")
                print(f"   -> [EXECUTING]: {tool_name}")
                result = execute_reflex_action({"recommended_tool": tool_name, "actionable_data": decision.get("args")})
                scratchpad.add_tool_output(tool_name, result)
                
            elif decision.get("action") == "finalize":
                print("   -> [RESOLUTION REACHED]. Exiting workspace.")
                scratchpad.is_resolved = True
                return decision.get("answer")
                
        except Exception as e:
            print(f"   -> [WORKSPACE ERROR] Failed to parse internal state: {e}")
            scratchpad.add_thought(f"Error parsing previous step: {e}. I must strictly format my output as JSON.")
            
    if not scratchpad.is_resolved:
        return f"I deliberated for {scratchpad.iteration} cycles but could not reach a final resolution. Here is my last thought: {scratchpad.thoughts[-1] if scratchpad.thoughts else 'None'}"

# --- CORE ROUTING & COGNITIVE FUNCTIONS ---
def route_intent(user_input, has_image, model, tokenizer):
    lower_input = user_input.lower()
    
    synthesized_context = ""
    past_memories_list = []
    past_memories = ""

    # PRIORITY 0: Memory/storage commands
    memory_triggers = [
        "store this", "remember this", "save this", "note this",
        "store for my", "remember for my", "log this", "memorize this",
        "why did you", "what do you know", "what do you remember"
    ]
    if any(trigger in lower_input for trigger in memory_triggers):
        return "CHAT_BRAIN"
    
    # PRIORITY 1: THE DETERMINISTIC MATRIX 
    matrix_intent = get_deterministic_intent(user_input)
    if matrix_intent != "UNKNOWN":
        print(f"\n[ROUTER] Deterministic matrix match. Forcing {matrix_intent}.")
        return matrix_intent
    
    # PRIORITY 2: Logical Deduction & Conversation
    logic_triggers = [
        "is this going to work", "will this work", "is this connection",
        "does this mean", "should i", "can i", "what do you think",
        "based on", "given what i told", "given the", "will it work",
        "going to work", "out of range", "within range", "exceed",
        "is this within", "is this outside", "does this exceed"
    ]
    if any(trigger in lower_input for trigger in logic_triggers):
        return "CHAT_BRAIN"
        
    # PRIORITY 3: Vision Processing
    if has_image:
        return "BASE_BRAIN"

# --- PROBABILISTIC ROUTING (BART) ---
    global intent_classifier
    top_label = "UNKNOWN"
    top_score = 0.0  

    if intent_classifier is not None:
        print("\n[ROUTER] Intent ambiguous. BART classifying...")
        result = intent_classifier(
            user_input,
            candidate_labels=["write code or a script", "calculate or solve math", "answer a question or discuss"],
            hypothesis_template="The user is asking me to {}."
        )
        
        top_label = result["labels"][0]
        top_score = result["scores"][0]

    # PROBABILISTIC FALLBACK
    if top_score < 0.55:
        print(f"\n[ROUTER] BART confidence too low ({top_score:.2f}). Defaulting to CHAT_BRAIN.")
        return "CHAT_BRAIN"

    if "calculate" in top_label or "math" in top_label:
        return "MATH_BRAIN"
    elif "code" in top_label or "script" in top_label:
        return "CODE_BRAIN"
    else:
        return "CHAT_BRAIN"

def requires_planning(user_input, active_brain):
    PLANNING_TRIGGERS = [
        'create', 'build', 'write', 'generate', 'make', 'design',
        'plan', 'implement', 'fix', 'debug', 'analyze', 'compare',
        'find', 'search', 'calculate', 'convert', 'summarize', 'explain how'
    ]
    return any(trigger in user_input.lower() for trigger in PLANNING_TRIGGERS)

def generate_internal_thought(user_input, past_memories, model, tokenizer):
    prompt = f"""
    Think step-by-step about the user's message.
    User input: {user_input}
    Relevant memories: {past_memories}
    What is the best way to respond? Plan the response strictly based on the facts.
    """
    with safe_disable_adapter(model):
        thought = ask_gemma_internal(prompt, model, tokenizer)
    return thought


# --- GOAL-DRIVEN ENGINE ---
CORE_GOALS = [
    "Improve reasoning and problem-solving ability",
    "Learn new information about the user's hardware, software, and engineering projects",
    "Improve accuracy and reduce mathematical or logical mistakes",
    "Become more helpful, concise, and useful",
    "Expand knowledge of complex technical topics"
]

def evaluate_interaction_goals(user_input, ai_response, model, tokenizer):
    prompt = f"""You are a strict logic auditor. Your job is to catch ONLY serious failures.
User message: "{user_input}"
AI response: "{ai_response}"

Set improvement_needed to true ONLY if:
- The AI completely ignored the question
- The AI made a clear numeric error (e.g. said 7 is less than 5)
- The AI said it cannot answer when it clearly could

Set improvement_needed to false if the AI gave any reasonable on-topic answer.

Output ONLY this exact JSON and nothing else:
{{"improvement_needed": false, "improvement_action": ""}}
OR
{{"improvement_needed": true, "improvement_action": "one sentence describing the error"}}"""

    with safe_disable_adapter(model):
        result = ask_gemma_internal(prompt, model, tokenizer)
        
    try:
        parsed = json.loads(result)
        action = parsed.get("improvement_action", "")
        if parsed.get("improvement_needed") and len(action.split()) < 4:
            return {"improvement_needed": False}
        return parsed
    except:
        return {"improvement_needed": False}

def extract_numeric_comparison(problem, synthesized_context):
    import re
    combined = problem + " " + synthesized_context
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\s*(km|kilometers|meters|m|hz|mhz|ghz|km/h|dbm)\b', combined.lower())
    
    if len(numbers) < 2:
        return None
        
    range_patterns = ["will it work", "is this going to work", "will this work", 
                      "within range", "out of range", "exceed", "is this connection",
                      "can it reach", "will it reach", "does it exceed"]
                      
    if not any(p in problem.lower() for p in range_patterns):
        return None
        
    values = [float(n[0]) for n in numbers]
    unit = numbers[0][1]
    
    limit_keywords = ["maximum", "max", "limit", "range", "up to", "only"]
    context_lower = synthesized_context.lower()
    
    if any(kw in context_lower for kw in limit_keywords):
        compare_vals = values[-2:] if len(values) >= 2 else values
        limit = min(compare_vals)
        actual = max(compare_vals)
        
        if actual > limit:
            return f"[DETERMINISTIC RESULT: {actual}{unit} exceeds the {limit}{unit} limit. The answer is NO, it will NOT work.]"
        else:
            return f"[DETERMINISTIC RESULT: {actual}{unit} is within the {limit}{unit} limit. The answer is YES, it will work.]"
            
    return None

def simulate_outcomes(problem, past_memories, skill_registry, active_goals, model, tokenizer):
    options = []
    lower_p = problem.lower()

    # 1. Deterministic Override (Declarative Learning)
    learning_triggers = ["store this", "remember this", "save this", "note this", 
                         "note that", "note down", "store for my", "remember for my", 
                         "log this", "memorize this", "fact for my", "keep this", 
                         "remember for later", "note for my", "save for my"]
    if any(t in lower_p for t in learning_triggers):
        return [{"approach": "Declarative Learning: Purely store information in long-term memory.", "confidence": 1.0, "source": "memory_system"}]

    factual_starters = ["what", "how many", "how much", "who", "when", 
                        "where", "which", "tell me", "do you remember",
                        "what do you know"]
    code_keywords = ["code", "script", "calculate", "program", 
                     "function", "tool", "python", "build", "create", "make"]
    
    is_factual_question = any(lower_p.startswith(w) for w in factual_starters)
    wants_code = any(w in lower_p for w in code_keywords)
    has_memories = (past_memories and 
                    past_memories != "None" and 
                    "No relevant memories" not in str(past_memories) and
                    "Past memories contain different domain details" not in str(past_memories))
    
# Grab the intent classifier safely to check for math
    classifier = globals().get('intent_classifier')
    
    is_math = requires_calculation(problem, classifier)

    if is_factual_question and has_memories and not wants_code and not is_math:
        print("\n[WORLD MODEL] Factual query + memories detected. Forcing memory recall. Skipping simulation.")
        return [{"approach": "Direct Memory Recall: Apply solution from past experience.", "confidence": 1.0, "source": "memory"}]
    elif is_factual_question and has_memories and is_math:
        print("\n[WORLD MODEL] Factual query + memories detected, but mathematical logic required. Proceeding to simulation.")

    # 2. Gather Potential Tools
    SKILL_STOPWORDS = ["the", "this", "that", "with", "from", "your", "have", "will", "what", "when", "where", "which", "skill", "tool", "does", "there", "their", "about", "store", "note", "save", "did", "you", "use", "for", "and", "why", "how", "my"]
    matching_skills = [
        name for name in skill_registry.keys()
        if any(word.lower() in problem.lower() for word in name.lower().split("_") if len(word) > 4 and word.lower() not in SKILL_STOPWORDS)
    ]
    tool_target = matching_skills[0] if matching_skills else "None'
    
    # 2.5 The Semantic Efficiency Gate
    global intent_classifier
    if intent_classifier is not None:
        gate_result = intent_classifier(
            lower_p,
            candidate_labels=["ask a factual question or recall memory", "execute a complex task or build a tool"],
            hypothesis_template="This text is trying to {}."
        )
        top_intent = gate_result["labels"][0]
        top_confidence = gate_result["scores"][0]
        
        if top_intent == "ask a factual question or recall memory" and top_confidence > 0.65:
            print(f"\n[SYSTEM GATE] Semantic classification: {top_intent} ({top_confidence:.2f}). Bypassing World Model simulation.")
            options = []
            if has_memories:
                options.append({"approach": "Direct Memory Recall: Apply solution from past experience.", "confidence": 0.95, "source": "memory"})
            else:
                options.append({"approach": "Reason from first principles and step-by-step logic.", "confidence": 0.85, "source": "base_reasoning"})
            return options

    # 3. INTRINSIC GOAL FORMATTING
    goal_context = ""
    if active_goals:
        goal_context = "INTRINSIC SYSTEM GOALS:\n"
        for g in active_goals:
            goal_context += f"- {g.get('description', '')} (Target: {g.get('success_criteria', '')})\n"
    else:
        goal_context = "INTRINSIC SYSTEM GOALS: None active.\n"

# 4. Build the Intrinsic Simulation Prompt
    sim_prompt = f"""
    You are Anima's Intrinsic World Model. Evaluate potential execution paths for a task.

    Task: "{problem}"
    Available Context: "{past_memories}"
    Matched Tool: "{tool_target}"

    {goal_context}

    Evaluate the 'hybrid score' (from 0.0 to 1.0) for each approach based on TWO criteria:
    1. Task Efficacy (50%): Does it efficiently and correctly solve the user's immediate task?
    2. Intrinsic Alignment (50%): Does it actively advance our Intrinsic Goals?

    Approaches:
    1. 'tool_execution': Using the Matched Tool. (Score 0.0 if no tool matched).
    2. 'tool_creation': Writing a brand new Python script. IMPORTANT: Only score >0.5 if task EXPLICITLY asks to build a script, app, or tool. Do NOT score high for math equations.
    3. 'pure_math': Calculating numbers, solving equations, or doing math word problems. IMPORTANT: Score high if the user asks to calculate or compute a value.
    4. 'memory_recall': Relying purely on the Available Context. IMPORTANT: Score >0.7 if context contains relevant facts.
    5. 'first_principles': Reasoning step-by-step from scratch.

    Output strictly in JSON format like this, with no other text:
    {{
        "tool_execution": 0.0,
        "tool_creation": 0.0,
        "pure_math": 0.8,
        "memory_recall": 0.4,
        "first_principles": 0.5,
        "reasoning": "Briefly explain which approach best aligns with the intrinsic goals."
    }}
    """

    # 5. Run the Simulation via Gemma
    try:
        print("\n[WORLD MODEL] Running Intrinsic Motivation simulations...")
        ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
        with ctx:
            response = ask_gemma_internal(sim_prompt, model, tokenizer)
            response = response.replace("```json", "").replace("```", "").strip()
            simulated_scores = json.loads(response)
            print(f"   -> [INTRINSIC REASONING] {simulated_scores.get('reasoning', 'No reasoning provided.')}")
    except Exception as e:
        print(f"\n[SIMULATOR ERROR] World Model failed to generate scores. Error: {e}")
        print("   -> [INTRINSIC REASONING] Falling back to neutral baseline. Deferring to standard Router.")
        
        simulated_scores = {
            "tool_execution": 0.0,
            "tool_creation": 0.0,
            "memory_recall": 0.0,
            "first_principles": 0.0
        }

# 6. Map Dynamic Scores to Options
    if matching_skills:
        options.append({"approach": f"Execute tool: {matching_skills[0]}", "confidence": float(simulated_scores.get("tool_execution", 0.0)), "source": "skill_registry"})
    
    options.append({"approach": "Write a new Python tool to automate this.", "confidence": float(simulated_scores.get("tool_creation", 0.0)), "source": "CODE_BRAIN"})
    
    options.append({"approach": "Solve using the Dedicated Math Brain.", "confidence": float(simulated_scores.get("pure_math", 0.0)), "source": "MATH_BRAIN"})
    
    if has_memories:
        options.append({"approach": "Apply solution from similar past experience using recalled context.", "confidence": float(simulated_scores.get("memory_recall", 0.0)), "source": "memory"})
    
    options.append({"approach": "Reason from first principles and step-by-step logic.", "confidence": float(simulated_scores.get("first_principles", 0.0)), "source": "base_reasoning"})
    
    return options

def pick_best_option(options):    
    return max(options, key=lambda x: x["confidence"])

# --- MAIN CHAT FUNCTION ---
def get_gemma_response(user_input, chat_history, image_path=None, pdf_path=None):
    global code_model, code_tokenizer, model, tokenizer, intent_classifier
    
    if len(chat_history) > 4:
        working_history = chat_history[-10:]
        print(f" -> [PREFRONTAL CORTEX] Chat history truncated. Retaining last {len(working_history)} messages.")
    else:
        working_history = chat_history

    # 1. FAILSAFE INITIALIZATION & AMNESIA PROTOCOL
    synthesized_context = ""
    past_memories = ""
    past_memories_list = []
    is_factual_recall = False
    top_intent = ""
    
    active_brain = "CHAT_BRAIN"
    best_option = {
        "approach": "Default Chat Routing (or Teaching Phase)", 
        "confidence": 1.0, 
        "source": "CHAT_BRAIN"
    }

    # 2. CHIT-CHAT BYPASS
    chit_chat_triggers = ["hi", "hello", "hey", "good morning", "how are you", "what's up"]
    if user_input.strip().lower() in chit_chat_triggers:
        reply = "Hello I am Anima, how can I assist you today?"
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": reply})
        return reply, chat_history

# --- VISUALIZER COMMAND INTERCEPTOR ---
    if user_input.strip().lower() == "/visualize":
        reply = "[SYSTEM] Launching interactive memory visualizer in your browser..."
        
        try:
            G = load_graph() 
            
            from CAI_memorygraph import generate_interactive_map
            
            generate_interactive_map(G)
        except Exception as e:
            reply = f"[ERROR] Could not launch visualizer: {e}"

        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": reply})
        
        return reply, chat_history

# --- SLEEP COMMAND INTERCEPTOR ---
    if "/sleep" in user_input.lower():
        # 1. Extract everything before the '/sleep' command
        raw_topic = user_input.lower().split("/sleep")[0].strip()
        
        # 2. Strip out common conversational filler words
        filler_words = [
            "research about", "research", 
            "give me details on", "give me details about", 
            "tell me about", "what is", "explain", "look up"
        ]
        
        target_topic = raw_topic
        for filler in filler_words:
            if target_topic.startswith(filler):
                target_topic = target_topic[len(filler):].strip()
                break
        
        # 3. Handle the UI response and queuing
        if target_topic:
            research_question = f"{target_topic} comprehensive overview explained"
            log_knowledge_gap_for_dreams(target_topic, research_question)
            reply = f"[FRONTAL LOBE] REM Override authorized. Queuing '{target_topic}' for deep consolidation. Entering sleep cycle..."
        else:
            reply = "[SYSTEM] Entering REM sleep with existing dream queue."
            
        from Hippocampus_dreams import trigger_rem_sleep
        
        # 4. Update the UI Chat History safely
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": reply})
        
        # 5. Trigger the cycle
        trigger_rem_sleep(collection, model, tokenizer)
        
        return reply, chat_history
    global SYSTEM_STATE, BASE_PROMPT, ACCUMULATED_FEEDBACK, CORRECTION_ATTEMPT_COUNT

    if not chat_history:
        chat_history = []
    if image_path or pdf_path:
        working_history = []

    audit_triggers = ["check your tools", "are your tools working", "check current tools", "audit your skills", "test your code"]
    if any(trigger in user_input.lower() for trigger in audit_triggers):
        from Parietal_skills import audit_and_fix_skills
        audit_report = audit_and_fix_skills(model, tokenizer)
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "model", "content": audit_report})
        return audit_report, chat_history

    extracted_text = ""
    ocr_context = ""

    if pdf_path:
        print(f"\n[FRONTAL LOBE] Deep studying PDF: {pdf_path}")
        from Frontal_learning import learn_from_pdf
        learn_from_pdf(pdf_path, collection, nougat_model, nougat_processor)
        ocr_context = f"\n\n[SYSTEM: You just read a PDF at {pdf_path} and saved its contents. Acknowledge this.]"

    # The NEW Image Reflex Arc
    if image_path:
        print(f"\n[OCCIPITAL VISION LOBE] Processing sensory input: {image_path}")
        from Frontal_learning import learn_from_images
        from Occipital_vision import extract_actionable_state
        from Temporal_memory import get_active_goals_from_db
        
        try:
            # 1. Passive Ingestion
            raw_text = learn_from_images([image_path], collection, nougat_model, nougat_processor)
            
            if not raw_text:
                ocr_context = "\n\n[CRITICAL INSTRUCTION: The image has no readable text. Tell the user.]"
            else:
                # 2. Strategic Cross-Reference
                active_goals = get_active_goals_from_db(limit=2)
                
                print("   -> [PERCEPTION] Analyzing state against Active Goals...")
                state_json = extract_actionable_state(raw_text, active_goals, model, tokenizer)
                
                # 3. The Reflex Arc
                if state_json.get("triggers_action"):
                    target_tool = state_json.get('recommended_tool')
                    print(f"   -> [REFLEX TRIGGERED] Sensory anomaly detected. Mandating immediate action: {target_tool}")
                    
                    tool_result = execute_reflex_action(state_json)
                    
                    ocr_context = (
                        f"\n\n[SYSTEM: You perceived the environment and extracted this state: {state_json.get('detected_state', 'Unknown')}. "
                        f"You automatically triggered the '{target_tool}' reflex. Result: {tool_result}]\n"
                    )
                    print(f"   -> [ACTION COMPLETE] {tool_result}")
                else:
                    print("   -> [PERCEPTION] Nominal state. No immediate reflexes required.")
                    ocr_context = f"\n\n[SYSTEM: Extracted Image Text: {raw_text}]\n"
                    
        except Exception as e:
            print(f"\n[OCR FATAL ERROR]: {e}\n")
            ocr_context = "\n\n[CRITICAL INSTRUCTION: The image scanner crashed. Tell the user.]"

    if SYSTEM_STATE == "AWAITING_CORRECTION":
        print("\n   -> [STUDENT] Receiving Master's feedback from UI...")
        ACCUMULATED_FEEDBACK += f"\n- {user_input}"
        CORRECTION_ATTEMPT_COUNT += 1
        
        if CORRECTION_ATTEMPT_COUNT >= 2:
            current_prompt = f"{BASE_PROMPT}\n\n[CORRECTION: {user_input}]\nAnswer directly and concisely."
            ACCUMULATED_FEEDBACK = ""
            CORRECTION_ATTEMPT_COUNT = 0 
        else:
            current_prompt = f"{BASE_PROMPT}\n\n[MASTER'S CORRECTION ON PREVIOUS ATTEMPT]: {ACCUMULATED_FEEDBACK}\nFix your mistakes and do it again."
            
        SYSTEM_STATE = "NORMAL" 
    else:
        current_prompt = user_input
        BASE_PROMPT = current_prompt
        ACCUMULATED_FEEDBACK = ""
    
    # Wipe the slate before searching
    synthesized_context = ""
    past_memories_list = []
    past_memories = ""
    
    from Temporal_memory import retrieve_weighted_memories, process_cognitive_loop
    past_memories, recalled_data = retrieve_weighted_memories(current_prompt, collection)
    
    if not image_path and current_prompt.strip():
        process_cognitive_loop(current_prompt, past_memories, collection, model, tokenizer)

    if past_memories and past_memories != "None":
        from Hippocampus import reconstruct_memory
        from Temporal_memory import ask_gemma_internal

        # GLOBAL ROUTING DEFAULTS (Failsafe)
        active_brain = "CHAT_BRAIN"
        best_option = {
            "approach": "Default Chat Routing (or Teaching Phase)", 
            "confidence": 1.0, 
            "source": "CHAT_BRAIN"
        }

        # --- THE UNIVERSAL REFLEX ARC (NORMALIZED) ---
        is_factual_recall = False
        is_perfect_match = False
        
        if past_memories_list and 'scores' in locals() and len(scores) > 0:
            if len(scores) >= 2:
                gap = abs(scores[0] - scores[1])
                score_spread = abs(max(scores) - min(scores)) if max(scores) != min(scores) else 1.0
                dominance_ratio = gap / score_spread
                if dominance_ratio > 0.40:
                    is_perfect_match = True
                    print(f"\n-> [SYSTEM GATE] Reflex Arc Engaged! Top memory dominates by {dominance_ratio*100:.1f}%.")
            elif len(scores) == 1:
                is_perfect_match = True
                print("\n-> [SYSTEM GATE] Reflex Arc Engaged! Sole memory match found.")

        # GENERAL USE / PROCEDURAL ROUTING 
        print("-> [ROUTER] Booting procedural evaluation...")
        
        if past_memories:
            with safe_disable_adapter(model):
                synthesized_context = reconstruct_memory(past_memories, current_prompt, model, tokenizer, ask_gemma_internal, working_history)
        else:
            if 'recalled_data' in locals() and recalled_data:
                top_hit = recalled_data[0]
                synthesized_context = top_hit.get('text', top_hit.get('content', top_hit.get('document', "")))
            else:
                synthesized_context = "No relevant memories recalled."

        from Parietal_skills import SKILL_REGISTRY
        
        # 1. FETCH INTRINSIC GOALS EARLY
        from Temporal_memory import get_active_goals_from_db
        active_goals = get_active_goals_from_db(limit=3)

        # 2. RUN INTRINSIC WORLD MODEL 
        world_model_options = simulate_outcomes(current_prompt, synthesized_context, SKILL_REGISTRY, active_goals, model, tokenizer)
        
        print("\n[SIMULATION ENGINE] Evaluated Paths:")
        for opt in world_model_options:
            print(f" -> {opt['source'].upper()}: {opt['confidence']:.2f} | {opt['approach'][:60]}...")
            
        best_option = pick_best_option(world_model_options)
        print(f"\n[WORLD MODEL] Selected Strategy: {best_option['approach']} (Confidence: {best_option['confidence']})")

        # 1. PARIETAL LOBE: RIGHT-OF-WAY CHECK
        print("\n[ROUTER] Checking Parietal Lobe for registered tools...")
        from Parietal_skills import select_and_execute_skill
        tool_result = select_and_execute_skill(current_prompt, model, tokenizer)

        if tool_result:
            print("[ROUTER] Tool executed successfully. Bypassing Simulation Engine.")
            
            # Patch 2: Data Protection. If the tool outputs numbers/math, output directly.
            if any(char.isdigit() for char in tool_result):
                print(f"\n[ANIMA]: {tool_result.strip()}")
                active_brain = "DIRECT_OUTPUT" 
                
            else:
                active_brain = "MEMORY"
                current_prompt = f"{current_prompt}\n\n[SYSTEM CONTEXT - TOOL OUTPUT]: {tool_result}"

# 2. STANDARD FALLBACK & DATABASE BYPASS (THE ARBITRATION ENGINE)
        else:
            print("\n[ROUTER] Initiating Confidence-Based Arbitration...")
            
            # 1. Initialize the Bid Ledger
            bids = {
                "CHAT_BRAIN": 0.0,
                "MATH_BRAIN": 0.0,
                "CODE_BRAIN": 0.0
            }

            # --- BIDDING ROUND 0: THE SMART CIRCUIT BREAKER (Unified) ---
            if 'recalled_data' in locals() and recalled_data:
                top_hit = recalled_data[0]
                top_mem_score = top_hit.get('rerank_score', 0.0)
                is_verified = top_hit.get('metadata', {}).get('verified', False) or top_hit.get('verified', False)
            else:
                top_mem_score = 0.0
                is_verified = False
            if is_verified and (top_mem_score > 3.0 or ('is_perfect_match' in locals() and is_perfect_match)):
                print(f" -> [ARBITRATION] Verified high-confidence memory detected. Bidding heavily to protect CHAT_BRAIN.")
                bids["CHAT_BRAIN"] += 5.0 
                
                best_option = {
                    "approach": "Authoritative Memory Override: Verified Direct Answer.",
                    "confidence": 1.0,
                    "source": "memory"
                }
            elif top_mem_score > 3.0:
                print(f" -> [ARBITRATION] High memory confidence ({top_mem_score:.2f}), but memory is UNVERIFIED. Standing down circuit breaker.")
    
            # 2. BIDDING ROUND 1: THE WORLD MODEL (System 2 - Deep Context)
            wm_source = best_option.get("source", "")
            wm_conf = float(best_option.get("confidence", 0.0))
            wm_approach = best_option.get("approach", "")
            
            if "Declarative Learning" in wm_approach:
                bids["CHAT_BRAIN"] += 2.0  
                print(" -> [ARBITRATION] World Model bids 2.00 for CHAT_BRAIN (Declarative Learning).")
            else:
                if wm_source == "memory":
                    bids["CHAT_BRAIN"] += wm_conf * 1.2
                elif wm_source == "CODE_BRAIN":
                    bids["CODE_BRAIN"] += wm_conf * 1.2
                elif wm_source == "base_reasoning":
                    bids["CHAT_BRAIN"] += wm_conf * 1.0
                
                print(f" -> [ARBITRATION] World Model bids for {wm_source} with weighted score {(wm_conf * 1.2):.2f}.")
            
            # 1. RUN DETERMINISTIC COMPARISON FIRST
            deterministic_result = extract_numeric_comparison(current_prompt, synthesized_context)

            # BIDDING ROUND 1.5: THE REGEX HEURISTIC & SHIELD ROUTING
            import re
            
            math_patterns = [
                r"\bcalculate\b", r"\bcompute\b", r"\bmultiply\b", 
                r"\bdivide\b", r"\bsubtract\b", r"\bequation\b", r"\bformula\b"
            ]
            
            # Non-greedy (.{0,40}?) prevents ReDoS CPU spikes on massive prompts
            numeric_context_patterns = [
                r"\bhow much\b.{0,40}?\d", r"\bhow many\b.{0,40}?\d", 
                r"\btotal\b.{0,40}?\d", r"\bsum\b.{0,40}?\d",
                r"\bremaining\b.{0,40}?\d", r"\baverage\b.{0,40}?\d",
                r"\bhow long\b.{0,40}?\b(travel|take|reach)\b.{0,40}?\d",
                r"\btime\b.{0,40}?\b(travel|distance|speed)\b.{0,40}?\d"
            ]
            
            is_regex_math = any(re.search(p, current_prompt.lower()) for p in math_patterns + numeric_context_patterns)

            if deterministic_result:
                print("  -> [ARBITRATION] Deterministic comparison solved the problem. Bypassing Math Brain.")
                bids["CHAT_BRAIN"] += 5.0
            elif is_regex_math:
                print("  -> [ARBITRATION] Complex numeric/word problem detected. Mandating MATH_BRAIN (+5.0).")
                bids["MATH_BRAIN"] += 5.0
                
            # 3. BIDDING ROUND 2: SEMANTIC INTENT
            if 'intent_classifier' in globals() and intent_classifier is not None:
                gate_result = intent_classifier(
                    current_prompt.lower(),
                    candidate_labels=[
                        "perform a mathematical calculation",
                        "write code or a script",
                        "recall a factual number or statistic",
                        "engage in general conversation"
                    ],
                    hypothesis_template="This text is trying to {}."
                )
                top_intent = gate_result["labels"][0]
                confidence = gate_result["scores"][0]
                
                if top_intent == "perform a mathematical calculation":
                    bids["MATH_BRAIN"] += confidence * 0.8
                    print(f" -> [ARBITRATION] BART bids for MATH_BRAIN with weighted score {(confidence * 0.8):.2f}.")
                elif top_intent == "write code or a script":
                    bids["CODE_BRAIN"] += confidence * 0.8
                    print(f" -> [ARBITRATION] BART bids for CODE_BRAIN with weighted score {(confidence * 0.8):.2f}.")
            else:
                print(" -> [ARBITRATION WARNING] BART offline. Skipping System 1 bids.")

            # ROUND 3: FINAL VERDICT
            if "CHAT_BRAIN" not in bids:
                bids["CHAT_BRAIN"] = 0.01
                
            active_brain = max(bids, key=bids.get)
            winning_score = bids[active_brain]
            
            print(f"\n  [ROUTER VERDICT] {active_brain} won arbitration with a score of {winning_score:.2f}.")

            # 5. THE VERDICT
            winning_brain = max(bids, key=bids.get)
            winning_score = bids[winning_brain]
            
            if winning_score < 0.4:
                active_brain = "CHAT_BRAIN"
                print("\n [ROUTER VERDICT] All bids too low. Defaulting to CHAT_BRAIN.")
            else:
                active_brain = winning_brain
                print(f"\n [ROUTER VERDICT] {winning_brain} won arbitration with a score of {winning_score:.2f}.")
                    
# PHASE 2: THE PRIMARY EXECUTION
    if active_brain == "CODE_BRAIN":
        print("\n[PARIETAL LOBE] Bypassing base model. Booting Qwen Code Specialist...")
        
        # 1. FREE UP VRAM (Unload Gemma)
        if 'model' in globals():
            del model
        if 'tokenizer' in globals():
            del tokenizer
            

        gc.collect()
        torch.cuda.empty_cache()

        from Parietal_skills import generate_and_save_skill
        import glob
        import time
        
        build_start_time = time.time()
            
        # 2. START THE THREAD
        worker_thread = threading.Thread(
            target=generate_and_save_skill,
            args=(current_prompt, synthesized_context, collection, brain_pipeline.put)
        )
        worker_thread.start()
        
        # CODE BRAIN WAITING LOOP
        print("\n================ [LIVE MONOLOGUE] ================")
        try:
            while worker_thread.is_alive() or not brain_pipeline.empty():
                try:
                    message = brain_pipeline.get(timeout=0.5)
                    print(message)
                except queue.Empty:
                    pass
        except KeyboardInterrupt:
            print("...", end="")
        print("==================================================\n")
        
        #3. RESTORE VRAM (Reload Gemma)
        print("\n[SYSTEM] Code Brain finished. Restoring Chat Brain to VRAM...")
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
            local_files_only=True
        )

        # 4. ORIGINAL RESPONSE GENERATION
        list_of_files = glob.glob('./skills/*.py')
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getctime)
            if os.path.getctime(latest_file) > build_start_time:
                with open(latest_file, "r", encoding="utf-8") as f:
                    generated_code = f.read()
                response = f"I have successfully built and tested the tool in the sandbox. Here is the code I generated:\n\n```python\n{generated_code}\n```\n\nIt has been saved to `{os.path.basename(latest_file)}` and is ready to use!"
            else:
                response = "I attempted to build the tool, but it failed to pass the isolated sandbox tests. I have aborted the save."
        else:
            response = "I attempted to build the tool, but it failed to pass the isolated sandbox tests. I have aborted the save."

    # --- NEW MATH ROUTING BLOCK ---
    elif active_brain == "MATH_BRAIN":
        print("\n[PARIETAL MATH LOBE] Bypassing base model. Booting Dedicated Math Specialist...")
        from Parietal_math import solve_math_problem
        import re
        math_response_container = []

        def math_worker():
            res = solve_math_problem(current_prompt, past_memories, working_history, ocr_context, collection, brain_pipeline.put)
            math_response_container.append(res)
        
        worker_thread = threading.Thread(target=math_worker)
        worker_thread.start()
        
        # MATH BRAIN WAITING LOOP
        print("\n================ [LIVE MONOLOGUE] ================")
        try:
            while worker_thread.is_alive() or not brain_pipeline.empty():
                try:
                    message = brain_pipeline.get(timeout=0.5)
                    print(message)
                except queue.Empty:
                    pass
        except KeyboardInterrupt:
            print("...", end="")
        print("==================================================\n")
        
        # MATH BRAIN RESPONSE GENERATION
        raw_response = math_response_container[0] if math_response_container else "Math Lobe failed to return a response."
        
        # --- THE MATH OUTPUT SCRUBBER ---
        if "Math Lobe failed" not in raw_response:
            print(" -> [PARIETAL LOBE] Scrubbing internal Python code from final output...")
            
            # 1. Delete the Python code blocks entirely
            scrubbed = re.sub(r'```python.*?```', '', raw_response, flags=re.DOTALL)
            
            # 2. Delete the terminal output blocks
            scrubbed = re.sub(r'```output.*?```', '', scrubbed, flags=re.DOTALL)
            
            # 3. Remove transition phrases
            scrubbed = re.sub(r"Let's write the Python code.*?:", "", scrubbed, flags=re.IGNORECASE)
            
            # 4. Strip out LaTeX formatting
            scrubbed = re.sub(r'\\boxed\{(.*?)\}', r'\1', scrubbed)
            scrubbed = scrubbed.replace('\\(', '').replace('\\)', '')
            
            # 5. Clean up any massive blank spaces left behind
            response = re.sub(r'\n{3,}', '\n\n', scrubbed).strip()
        else:
            response = raw_response

    else:
        # 1. PLANNING
        if "Declarative Learning" in best_option['approach']:
            plan_execution_results = "[SYSTEM: Intent identified as Learning. Skipping Procedural Planning.]"
        elif active_brain == "CHAT_BRAIN":
            plan_execution_results = "[SYSTEM: Conversational query. Reasoning directly from recalled context.]"
        else:
            from Prefrontal_planner import run_planning_loop
            plan_execution_results = run_planning_loop(current_prompt, SKILL_REGISTRY, model, tokenizer)
        
        SKIP_MESSAGES = ["Skipping Procedural Planning", "Reasoning directly from recalled context"]
        if plan_execution_results.strip() and not any(s in plan_execution_results for s in SKIP_MESSAGES):
            ocr_context += f"\n\n[SYSTEM: The Prefrontal Planner executed a step-by-step plan. Use these results to format your final answer:]\n{plan_execution_results}"

        # 2. DETERMINISTIC COMPARISON
        deterministic_result = extract_numeric_comparison(current_prompt, synthesized_context)

# 3. SYSTEM RULES & DYNAMIC ROUTING
        if "Declarative Learning" in best_option['approach']:
            system_rules = (
                "You are Anima. The user has just given you a fact to remember. "
                "Acknowledge it in ONE short sentence. Confirm what you stored. "
                "Do NOT calculate anything. Do NOT list steps. Do NOT use bullet points."
            )
        else:
            from Temporal_memory import get_active_goals_from_db
            active_goals = get_active_goals_from_db(limit=3)
            
            goal_context = ""
            if active_goals:
                goal_context = "--- ACTIVE DIRECTIVES & GOALS ---\n"
                for g in active_goals:
                    goal_context += f"- [Priority {g['priority']}] {g['description']} (Target: {g['success_criteria']})\n"
                goal_context += "--------------------\n\n"

            if 'is_factual_recall' in locals() and is_factual_recall:
                print("\n-> [SYSTEM GATE] High confidence fact retrieved. Extracting primary content...")
                
                try:
                    import json
                    memories = json.loads(synthesized_context)
                    if isinstance(memories, list) and len(memories) > 0:
                        final_response = memories[0].get("content", "").replace("Memory: ", "").strip()
                    else:
                        final_response = synthesized_context.strip()
                except Exception:
                    final_response = synthesized_context.strip()
                
                chat_history.append({"role": "assistant", "content": final_response})
                return final_response, chat_history

            system_rules = (
                "You are Anima, a direct and highly logical AI assistant. "
                f"CURRENT ACTIVE NEURAL PATHWAY: {active_brain}\n\n"
                "CRITICAL RULES:\n"
                "1. NEVER use emojis, emoticons, or weird unicode characters. Use plain text.\n"
                f"2. You are currently operating as the {active_brain}. You MUST strictly adhere to this domain.\n"
                "3. Answer the user's question directly in 1-3 sentences.\n"
                "4. Review the Past Memories below. IF they are irrelevant to the user's current prompt, COMPLETELY IGNORE THEM.\n"
                "5. NUMERIC STRICTNESS: You MUST preserve all numbers, metrics, and units EXACTLY as they appear in the Recalled Context.\n\n"
                f"{goal_context}"
                f"--- SELECTED STRATEGY ---\nApproach: {best_option['approach']}\nConfidence: {best_option['confidence']}\nSource: {best_option['source']}\n--------------------\n\n"
                f"--- RECALLED CONTEXT ---\n{synthesized_context}\n--------------------\n"
                + (f"\n--- RESOLVED ANSWER ---\n{deterministic_result}\n--------------------\n" if deterministic_result else "")
            )
            
        # 4. CURIOSITY ENGINE — runs before augmented_input so question reaches the prompt
        gap_info = detect_knowledge_gap(current_prompt, collection)
        if gap_info.get("knowledge_gap"):
            questions = generate_curiosity_questions(gap_info, model, tokenizer)
            curiosity_q = curiosity_learning_action(questions)
            if curiosity_q:
                ocr_context += f"\n\n[CURIOSITY NOTE: You noticed a knowledge gap. If appropriate, ask the user: {curiosity_q}]"

        # 5. BUILD PROMPT
        augmented_input = f"{system_rules}\n\nUser's message: {current_prompt}\n{ocr_context}" if ocr_context else f"{system_rules}\n\nUser's message: {current_prompt}"
        temp_history = working_history.copy()
        temp_history.append({"role": "user", "content": augmented_input})
            
        #6. EXECUTION: SYSTEM 2 DELIBERATION OR SYSTEM 1 FAST RESPONSE
        if "Declarative Learning" in best_option.get('approach', '') or "Authoritative Memory Override" in best_option.get('approach', ''):
            print("\n[SYSTEM 1] Fast response engaged for strict preservation...")
            from Temporal_memory import ask_gemma_internal
            final_response = ask_gemma_internal(
            f"{system_rules}\nUser: {current_prompt}", model, tokenizer
        )
            
        else:
            final_response = deliberate_and_execute(
            current_prompt, system_rules, model, tokenizer
        )

        response = final_response.strip()

# --- PHASE 3: HYBRID GOAL EVALUATION & RLHF ---
    if len(response.split()) > 4 and user_input.strip():
        from Temporal_memory import log_memory
        
        related_mem_ids = [mem["id"] for mem in recalled_data] if recalled_data else []
        
        # Determine if this was a factual conversation or a task/tool execution
        is_procedural = active_brain in ["CODE_BRAIN", "MATH_BRAIN"] or "tool" in str(best_option).lower()
        mem_class = "procedural" if is_procedural else "episodic"
        
        # Determine source modality based on the active brain
        modality = active_brain.lower() if active_brain else "base_brain"
        
        # Determine if it is verified
        is_verified = True if is_procedural else False

        log_memory(
            user_input=user_input, 
            ai_response=response, 
            collection=collection,
            active_brain=active_brain,
            strategy=best_option['approach'] if best_option and 'approach' in best_option else "UNKNOWN",
            related_ids=related_mem_ids,
            memory_class=mem_class,
            source_modality=modality,
            verified=is_verified
        )

    if CORRECTION_ATTEMPT_COUNT > 0 or "Declarative Learning" in best_option['approach'] or active_brain in ["MATH_BRAIN", "CODE_BRAIN"]:
        print("\n[AI SELF-EVALUATION] Skipped — handled by specialist lobes or correction applied.")
        CORRECTION_ATTEMPT_COUNT = 0
        if recalled_data:
            from Temporal_memory import reinforce_memory
            for mem in recalled_data:
                reinforce_memory(mem["id"], collection, success=True)
        
        display_user_input = f"[Image Uploaded] {user_input}" if image_path else user_input
        chat_history.append({"role": "user", "content": display_user_input})
        chat_history.append({"role": "model", "content": response})
        
        return response, chat_history

    print("\n[AI SELF-EVALUATION RUNNING...]")
    ai_goal_feedback = evaluate_interaction_goals(current_prompt, response, model, tokenizer)
    ai_improvement_needed = ai_goal_feedback.get("improvement_needed", False)
    ai_action = ai_goal_feedback.get("improvement_action", "Review logic").strip()

    ignore_phrases = ["one line", "one line if true", "empty string", "empty string if false", "one sentence describing the error", ""]
    if ai_action.lower() in ignore_phrases:
        ai_improvement_needed = False

    if ai_improvement_needed or "Error" in response:
        print(f" -> AI suspects it made a mistake: {ai_action}")
        if recalled_data:
            from Temporal_memory import reinforce_memory
            for mem in recalled_data:
                reinforce_memory(mem["id"], collection, success=False) 
        
        
        display_user_input = f"[Image Uploaded] {user_input}" if 'image_path' in locals() and image_path else user_input
        chat_history.append({"role": "user", "content": display_user_input})
        chat_history.append({"role": "model", "content": response})
        return response, chat_history
        
    else:
        print(" -> AI believes it executed the task perfectly. Finalizing.")
        if recalled_data:
            from Temporal_memory import reinforce_memory
            for mem in recalled_data:
                reinforce_memory(mem["id"], collection, success=True)
        if ACCUMULATED_FEEDBACK:
            from datetime import datetime
            import json
            import uuid
            reflection_doc = {"type": "reflection", "action": f"User corrected AI: {ACCUMULATED_FEEDBACK}", "timestamp": datetime.now().isoformat()}
            collection.add(
                documents=[json.dumps(reflection_doc)],
                metadatas=[{"memory_type": "goal_reflection", "importance": 10, "confidence": 0.5, "usage_count": 0}],
                ids=[str(uuid.uuid4())]
            )
            
        display_user_input = f"[Image Uploaded] {user_input}" if 'image_path' in locals() and image_path else user_input
        chat_history.append({"role": "user", "content": display_user_input})
        chat_history.append({"role": "model", "content": response})
        return response, chat_history
