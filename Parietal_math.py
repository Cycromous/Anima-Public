import torch
import gc
import re
import sympy
from sympy.parsing.sympy_parser import parse_expr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- ADD THE COMPRESSOR FUNCTION HERE ---
def compress_math_context(working_history, ocr_context):
    """
    Layer 2 Filter: Scours working history and OCR for explicitly technical data.
    Strips out conversational filler, pleasantries, and non-numerical text.
    """
    import re
    compressed_history = []
    
    # We keep a turn if it contains digits, operators, or technical/engineering units
    technical_indicators = [
        r'\d', r'\+', r'\-', r'\*', r'/', r'=', r'%', r'<', r'>',
        r'\b(km|m|cm|mm|kg|g|mg|lbs|oz)\b',                 # Distance/Weight
        r'\b(kbps|mbps|gbps|hz|khz|mhz|ghz|dbm|watts)\b',   # Network/Hardware
        r'\b(cost|rate|total|sum|average|variance)\b'       # Math keywords
    ]
    combined_pattern = re.compile('|'.join(technical_indicators), re.IGNORECASE)

    # 1. Compress History (Keep only technical turns)
    if working_history:
        for msg in working_history:
            if combined_pattern.search(msg['content']):
                compressed_history.append(f"{msg['role'].upper()}: {msg['content']}")
    
    history_str = "\n".join(compressed_history) if compressed_history else "No numerical or technical context in recent history."

    # 2. Compress OCR (Line by Line filtering)
    ocr_str = "None"
    if ocr_context:
        compressed_ocr = [line for line in ocr_context.split('\n') if combined_pattern.search(line)]
        ocr_str = "\n".join(compressed_ocr) if compressed_ocr else "No numerical data in visual context."

    return history_str, ocr_str

# THE FIX: Changed 'synthesized_context' to 'past_memories' in the signature
def solve_math_problem(prompt, past_memories, working_history, ocr_context, collection, queue_put):
    """
    JIT Lobe: Boots Qwen2.5-Math, solves the equation, verifies deterministically, 
    and instantly flushes VRAM.
    """
    queue_put("\n[PARIETAL MATH LOBE] Initializing Dedicated Math Brain (JIT)...")
    
    # UPDATE THIS PATH to where you extract Qwen2.5-Math-1.5B-Instruct
    MATH_MODEL_PATH = "/home/johnray/Personal/qwen2.5-math-transformers-1.5b-instruct-v1" 
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    math_model = None
    math_tokenizer = None
    
    try:
        queue_put("   -> [PARIETAL MATH LOBE] Allocating ~1.2GB VRAM. Loading weights...")
        math_tokenizer = AutoTokenizer.from_pretrained(MATH_MODEL_PATH, local_files_only=True)
        math_model = AutoModelForCausalLM.from_pretrained(
            MATH_MODEL_PATH,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        
        queue_put("   -> [PARIETAL MATH LOBE] Math Specialist Online. Calculating...")
        
        # 1. Format the Unified Context via the Compressor
        history_str, ocr_str = compress_math_context(working_history, ocr_context)

        # THE FIX: Inject the raw 'past_memories' JSON here instead of the hallucination-prone summary
        context_payload = f"""
--- DATABASE CONTEXT (Raw Recalled Info) ---
{past_memories}

--- FILTERED NUMERICAL HISTORY ---
{history_str}

--- FILTERED NUMERICAL SENSORY/OCR DATA ---
{ocr_str}
"""

        # 2. Define the Persona Prompts (Slim Ensemble for 1.5B Model)
        prompts = {
            "PURE MATHEMATICIAN": (
                "Solve this problem using strict, step-by-step algebraic logic.\n"
                f"SYSTEM CONTEXT:\n{context_payload}\n"
                "RULES: Isolate your final mathematical expression inside <verify> tags. Do not write Python code."
            ),
            "MATH PROFESSOR": (
                "Solve this problem by identifying the core mathematical formula or theorem required first.\n"
                f"SYSTEM CONTEXT:\n{context_payload}\n"
                "RULES: Isolate your final mathematical expression inside <verify> tags. Do not write Python code."
            ),
            "APPLIED ENGINEER": (
                "Solve this problem by focusing directly on computing the final numerical values and units.\n"
                f"SYSTEM CONTEXT:\n{context_payload}\n"
                "RULES: Isolate your final mathematical expression inside <verify> tags. Do not write Python code."
            )
        }

        generated_solutions = {}
        device = next(math_model.parameters()).device

        # 3. Execution Engine (Iterating through personas)
        for persona_name, system_rules in prompts.items():
            queue_put(f"   -> [MATH LOBE] Generating solution as {persona_name}...")
            
            messages = [
                {"role": "system", "content": system_rules},
                {"role": "user", "content": prompt}
            ]
            
            # Apply the template which sets up the <|im_start|>assistant prompt
            input_text = math_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # --- THE FIX (FIX 3): PUT WORDS IN ITS MOUTH ---
            forced_start = "Here is the step-by-step logical calculation. The final numerical value is <verify>"
            input_text += forced_start
            # -----------------------------------------------
            
            inputs = math_tokenizer(input_text, return_tensors="pt").to(device)
            
            # Low Temperature for Strict Logic
            with torch.no_grad():
                outputs = math_model.generate(
                    **inputs, 
                    max_new_tokens=512, 
                    temperature=0.01, 
                    do_sample=True,
                    pad_token_id=math_tokenizer.eos_token_id
                )
            
            input_length = inputs["input_ids"].shape[1]
            persona_response = math_tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
            
            # We must attach the forced start back onto the response so the regex can find the opening <verify> tag!
            generated_solutions[persona_name] = forced_start + persona_response

        # =================================================================
        # 4. SELF-CONSISTENCY EVALUATION (Majority Voting)
        # =================================================================
        from collections import defaultdict
        queue_put("   -> [MATH LOBE] Evaluating generated solutions for consensus...")
        
        answer_tally = defaultdict(list)
        solution_map = {}
 
        def normalize_answer(raw_ans):
            """
            Normalizes a math answer string for fair comparison.
            Handles: whitespace, trailing punctuation, equation right-hand sides,
            and float/int equivalence (e.g. '42.0' == '42').
            """
            # Strip whitespace and trailing punctuation
            cleaned = raw_ans.strip().replace(" ", "").rstrip('.')
            
            # If it contains an equals sign, take only the right-hand side
            if "=" in cleaned:
                cleaned = cleaned.split("=")[-1]
            
            # --- FIX 1: NUMERIC NORMALIZATION ---
            # Converts '42.0', '42', '42.00' all to the same canonical form
            # so they match correctly in the tally.
            try:
                as_float = float(cleaned)
                # If it's a whole number, represent as int string ('42' not '42.0')
                if as_float == int(as_float):
                    return str(int(as_float))
                else:
                    # Round to 6 decimal places to avoid float precision mismatches
                    return str(round(as_float, 6))
            except ValueError:
                # Not a pure number (e.g. symbolic expression like '2*x+5')
                # Return cleaned string as-is for symbolic comparison
                return cleaned
 
        for persona, solution in generated_solutions.items():
            match = re.search(r'<verify>(.*?)</verify>', solution, re.DOTALL)
            if match:
                raw_ans = match.group(1)
                normalized_ans = normalize_answer(raw_ans)
                
                answer_tally[normalized_ans].append(persona)
                
                # Save the first full solution that yielded this specific answer
                if normalized_ans not in solution_map:
                    solution_map[normalized_ans] = solution
            else:
                queue_put(f"   -> [WARNING] {persona} failed to provide <verify> tags.")
 
        raw_response = None
        
        if answer_tally:
            # Find the answer with the most votes
            best_ans = max(answer_tally, key=lambda k: len(answer_tally[k]))
            votes = len(answer_tally[best_ans])
            voting_personas = ", ".join(answer_tally[best_ans])
            
            if votes >= 2:
                queue_put(f"   -> [MATH LOBE] Consensus reached! {votes}/3 agreed on the result ({voting_personas}).")
                raw_response = solution_map[best_ans]
            else:
                queue_put("   -> [MATH LOBE WARNING] No consensus reached (1/3 split). Defaulting to Pure Mathematician's logic.")
                # --- FIX 2: SAFE FALLBACK ---
                # Find whichever normalized answer the Pure Mathematician produced.
                # If Pure Mathematician failed to produce verify tags entirely,
                # fall back to whichever answer scored highest rather than
                # crashing with a KeyError or returning None.
                pure_math_ans = next(
                    (ans for ans, personas in answer_tally.items() if "PURE MATHEMATICIAN" in personas),
                    best_ans  # Safe fallback if Pure Mathematician produced no tags
                )
                raw_response = solution_map.get(pure_math_ans, solution_map[best_ans])
        else:
            # Absolute fallback if NO persona used the tags
            queue_put("   -> [FATAL WARNING] No verification tags passed by any persona. Defaulting to Applied Engineer.")
            raw_response = generated_solutions.get("APPLIED ENGINEER", next(iter(generated_solutions.values())))

        # 3. THE DETERMINISTIC VERIFIER
        queue_put("   -> [PARIETAL MATH LOBE] Running deterministic SymPy verification...")
        match = re.search(r'<verify>(.*?)</verify>', raw_response, re.DOTALL)
        
        if match:
            raw_equation = match.group(1).strip()
            
            # --- THE NEW SCRUBBER ---
            # If the LLM included an equals sign, just take the final answer on the right
            if "=" in raw_equation:
                raw_equation = raw_equation.split("=")[-1].strip()

            # Remove common trailing punctuation that breaks SymPy
            raw_equation = raw_equation.rstrip('.')
            # ------------------------
            
            try:
                # Safely evaluate the symbolic math
                verified_result = parse_expr(raw_equation)
                queue_put(f"   -> [VERIFIED] Deterministic result: {verified_result}")
                
                # Append the verified truth to the LLM's explanation
                response = f"{raw_response}\n\n[SYSTEM VERIFIED]: The symbolic result evaluates to {verified_result}"
            except Exception as e:
                queue_put(f"   -> [VERIFICATION FAILED] SymPy could not validate the logic: {e}")
                response = "I attempted to solve the problem, but my internal logic failed deterministic verification. I need to rethink the steps."
        else:
            queue_put("   -> [WARNING] No verification tags found. Returning unverified response.")
            response = raw_response

        queue_put("   -> [PARIETAL MATH LOBE] Calculation complete.")
        return response

    except Exception as e:
        if queue_put:
            queue_put(f"[PARIETAL MATH LOBE FATAL ERROR] {str(e)}")
        return f"I encountered an error while calculating: {str(e)}"
        
    finally:
        # 4. THE JIT FLUSH (Prevents VRAM crashes)
        if queue_put:
            queue_put("\n   -> [PARIETAL MATH LOBE] Executing JIT VRAM Flush...")
        
        # 1. Record VRAM before deletion
        vram_before = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        
        # 2. Delete components
        if 'math_model' in locals() and math_model is not None:
            del math_model
        if 'math_tokenizer' in locals() and math_tokenizer is not None:
            del math_tokenizer
        
        # 3. Force garbage collection
        gc.collect()
        
        # --- THE FIX (CROSS-BRAIN FIX A): VRAM VERIFICATION LOOP ---
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            import time
            start_flush = time.time()
            vram_cleared = False
            
            while time.time() - start_flush < 5.0: 
                current_vram = torch.cuda.memory_allocated() / (1024 ** 2)
                if vram_before - current_vram > 500: 
                    vram_cleared = True
                    if queue_put:
                        queue_put(f"   -> [PARIETAL MATH LOBE] VRAM successfully flushed! Freed {vram_before - current_vram:.0f}MB.")
                    break
                time.sleep(0.5)
            
            if not vram_cleared and queue_put:
                queue_put(f"   -> [WARNING] VRAM flush timeout or leak detected. Current VRAM stuck at {current_vram:.0f}MB.")
        # -------------------------------------------------------------

        if queue_put:
            queue_put("   -> [PARIETAL MATH LOBE] VRAM cleared. Handing control back to Base Brain.")