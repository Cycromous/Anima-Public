def reconstruct_memory(fragments, user_input, model, tokenizer, ask_fn, working_history=None):
    if not fragments or fragments == "None":
        return ""
        
    print(" -> [HIPPOCAMPUS] Performing analogical synthesis on memory fragments...")
    
    # 1. Format the short-term memory cleanly
    recent_context = ""
    if working_history:
        for msg in working_history:
            recent_context += f"{msg['role'].upper()}: {msg['content']}\n"
    
    # 2. Inject it into the prompt so the Hippocampus knows what "it" or "that" refers to
    prompt = f"""
    You are the cognitive Hippocampus. Your job is to perform active inference, not just passive recall.

    Recent Conversation Context (To resolve pronouns/references):
    {recent_context}

    Current Query: "{user_input}"
    Retrieved Memory Fragments:
    {fragments}

    CRITICAL RULES:
    1. Do not just repeat the fragments. Analyze them in relation to the query.
    2. Identify any underlying patterns, lessons, or analogies from these past experiences that apply to the current situation.
    3. Synthesize a coherent thought connecting the past to the present.
    4. IF the query is a simple factual question and the memory fragments contain the direct factual answer, DO NOT create an analogy. Output ONLY: "Factual Context: [The exact fact and units from memory]".

    Output ONLY a 1-2 sentence synthesized insight.
    If the memories are completely unrelated and offer no analogical value, output exactly: "No relevant memories found."
    Do NOT explain your reasoning. Output only the final result.
    """
    
    return ask_fn(prompt, model, tokenizer).strip()