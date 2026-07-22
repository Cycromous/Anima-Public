import os
import sys
import subprocess
import tempfile
import glob
import torch
import json
import re
import gc
import uuid
import time
import importlib.util
import shutil
from Temporal_memory import ask_gemma_internal
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path

try:
    import resource
except ImportError:
    resource = None

SKILLS_DIR = "./skills"
os.makedirs(SKILLS_DIR, exist_ok=True)
VENV_DIR = os.path.join(os.path.dirname(__file__), "ai_venv")
SKILL_REGISTRY = {}
SKILL_METADATA_FILE = "./skills/skill_metadata.json"
coder_loaded = False
code_model = None
code_tokenizer = None
CODER_MODEL_PATH = "/home/johnray/Personal/qwen2.5-coder-transformers-3b-instruct-v1"

def compress_code_context(working_history, ocr_context, current_prompt):
    """
    Keeps only the last 3 chat turns and filters OCR data to prevent 
    the 3B model from drowning in visual noise unless explicitly requested.
    """
    # 1. Truncate History (Last 3 turns max)
    recent_history = working_history[-3:] if working_history else []
    history_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in recent_history]) if recent_history else "None"
    
    # 2. Filter OCR Context
    # Only pass OCR if the user explicitly references the file/data in their prompt
    ocr_keywords = ['image', 'picture', 'pdf', 'document', 'file', 'data', 'text', 'read', 'extract']
    prompt_lower = current_prompt.lower()
    
    if ocr_context and any(kw in prompt_lower for kw in ocr_keywords):
        # Cap OCR at ~1500 characters to protect the 3B model's context window
        ocr_str = ocr_context[:1500] + "\n...[TRUNCATED TO PROTECT VRAM]" if len(ocr_context) > 1500 else ocr_context
    else:
        ocr_str = "[SYSTEM: No relevant document data requested for this task.]"
        
    return history_str, ocr_str

# ==============================================================================
# SKILL METADATA & REGISTRY
# ==============================================================================

def load_skill_metadata():
    if os.path.exists(SKILL_METADATA_FILE):
        with open(SKILL_METADATA_FILE, "r") as f:
            return json.load(f)
    return {}


def save_skill_metadata(metadata):
    with open(SKILL_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)


def archive_stale_skills():
    metadata = load_skill_metadata()
    archive_dir = "./skills/archive"
    os.makedirs(archive_dir, exist_ok=True)

    now = datetime.now()
    archived_count = 0

    for tool_name, stats in list(metadata.items()):
        last_used = datetime.fromisoformat(stats.get("last_used", now.isoformat()))
        days_inactive = (now - last_used).days

        if days_inactive >= 30 or stats.get("fail_count", 0) >= 3:
            src = os.path.join(SKILLS_DIR, f"{tool_name}.py")
            dst = os.path.join(archive_dir, f"{tool_name}.py")
            if os.path.exists(src):
                shutil.move(src, dst)
                print(f" -> [ARCHIVE] Moved {tool_name}.py (Inactive: {days_inactive}d, Fails: {stats.get('fail_count', 0)})")
            del metadata[tool_name]
            archived_count += 1

    if archived_count > 0:
        save_skill_metadata(metadata)
        load_existing_skills()
        print(f" -> [SKILL GOVERNANCE] Archived {archived_count} stale tools.")


def load_existing_skills():
    """Scans the skills folder and loads them into the registry."""
    global SKILL_REGISTRY
    SKILL_REGISTRY.clear()
    metadata = load_skill_metadata()

    def make_tracked_skill(name, func):
        def tracked_wrapper(*args, **kwargs):
            meta = load_skill_metadata()
            if name in meta:
                meta[name]["usage_count"] = meta[name].get("usage_count", 0) + 1
                meta[name]["last_used"] = datetime.now().isoformat()
                save_skill_metadata(meta)
            return func(*args, **kwargs)
        return tracked_wrapper

    for filename in os.listdir(SKILLS_DIR):
        if filename.endswith(".py"):
            module_name = filename[:-3]
            file_path = os.path.join(SKILLS_DIR, filename)
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                if hasattr(module, 'run_skill'):
                    if module_name not in metadata:
                        metadata[module_name] = {
                            "usage_count": 0,
                            "last_used": datetime.now().isoformat(),
                            "fail_count": 0
                        }
                    SKILL_REGISTRY[module_name] = make_tracked_skill(module_name, module.run_skill)
                else:
                    print(f"[PARIETAL LOBE] Warning: {filename} has no run_skill() function, skipping.")
            except Exception as e:
                print(f"[PARIETAL LOBE] Failed to load {filename}: {e}")

    save_skill_metadata(metadata)
    print(f"[PARIETAL LOBE] Loaded {len(SKILL_REGISTRY)} custom skills into registry.")


def select_and_execute_skill(user_input, model, tokenizer):
    """Asks the AI if it should use a tool, and executes it if needed."""
    if not SKILL_REGISTRY:
        return None
    
    # --- PATCH 1: THE INFERENCE GATE ---
    # Only wake up Gemma if the user uses a trigger word or names a tool
    user_text_lower = user_input.lower()
    trigger_words = ["calculate", "use tool", "compute", "run"]
    tool_names = [name.replace('_', ' ') for name in SKILL_REGISTRY.keys()]
    
    needs_tool = any(word in user_text_lower for word in trigger_words + tool_names)
    
    if not needs_tool:
        return None # Instantly skip to the World Model, saving VRAM!
    # -----------------------------------

    available_tools = list(SKILL_REGISTRY.keys())
    prompt = f"""
    Determine if a tool should be used to help answer the user.
    User request: {user_input}
    Available tools: {available_tools}

    INSTRUCTIONS:
    - If the user explicitly asks to calculate, compute, or use a specific tool by name (e.g., "use the lora range calculator"), output true and provide the exact tool_name.
    - If the user asks a general question, asks for an opinion, or asks you to remember something (e.g., "what is my favorite car"), output false.

    Output ONLY a JSON object: {{"use_tool": true/false, "tool_name": "..."}}
    """

    with (model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()):
        result_str = ask_gemma_internal(prompt, model, tokenizer)

    try:
        decision = json.loads(result_str)
        if decision.get("use_tool") and decision.get("tool_name") in SKILL_REGISTRY:
            tool_name = decision["tool_name"]
            print(f"[PARIETAL LOBE] Executing skill: {tool_name}")
            result = SKILL_REGISTRY[tool_name](user_input)
            return f"\n[TOOL OUTPUT from {tool_name}]: {result}\n"
    except Exception as e:
        print(f"[PARIETAL LOBE] Failed to execute skill. Error: {e}")
        return None

    return None


# ==============================================================================
# MODULE-LEVEL QWEN HELPERS  (must be defined before any function calls them)
# ==============================================================================

def load_auditor():
    global coder_loaded, code_model, code_tokenizer
    if not coder_loaded:
        print("   -> [SYSTEM] Booting Qwen Auditor Brain...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        # Path() bypasses huggingface_hub's repo ID string validation
        code_tokenizer = AutoTokenizer.from_pretrained(Path(CODER_MODEL_PATH))
        code_model = AutoModelForCausalLM.from_pretrained(
            Path(CODER_MODEL_PATH),
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        coder_loaded = True


def ask_qwen(prompt, system_prompt="You are an elite Python engineer.", temperature=0.1):
    """Routes a prompt to the Qwen2.5-Coder specialist. Assumes load_auditor() was called first."""
    global code_model, code_tokenizer

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    input_text = code_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = next(code_model.parameters()).device
    inputs = code_tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = code_model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=temperature,
            do_sample=True,
            pad_token_id=code_tokenizer.eos_token_id
        )

    raw_output = code_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return raw_output.strip()


def generate_and_save_skill(current_prompt, synthesized_context, collection, pipeline_put):
    global code_model, code_tokenizer, coder_loaded
    pipeline_put(f"\n[MULTI-AGENT BOOT] Task received: {current_prompt}")
    pipeline_put("[CODE BRAIN] Booting Collaborative Design Council + Hostile QA Pipeline...")
 
    # 1. JIT-load Qwen2.5-Coder
    load_auditor()
 
    is_success = False
    successful_candidates = []
 
    # Define the 3 personas (lean — style only, no filler)
    persona_styles = {
        "SOFTWARE ENGINEER": "Optimize for clean, readable, Pythonic code.",
        "SYSTEMS ENGINEER":  "Optimize for fault tolerance and strict edge-case handling.",
        "COMPUTER ENGINEER":  "Optimize for raw computational efficiency.",
    }
 
    try:
        # =================================================================
        # PHASE 1 — ISOLATED QA: Generate hostile mock data BEFORE
        # any persona sees the task. Completely blind to any solution.
        # =================================================================
        pipeline_put("\n[PHASE 1] Generating Hostile QA Dataset (blind — no code exists yet)...")
        qa_prompt = (
            f"You are a destructive QA tester. Your only job is to break Python scripts.\n"
            f"A script will be written to handle this task: '{current_prompt}'\n"
            f"Generate a hostile mock input string designed to expose edge cases, "
            f"unexpected formatting, extreme values, empty inputs, and malformed data.\n"
            f"Output ONLY the raw hostile string. No markdown, no explanation, nothing else."
        )
        hostile_mock_data = ask_qwen(
            qa_prompt,
            system_prompt="You are a destructive QA tester. Output raw hostile test input only."
        )
        pipeline_put(f"   -> [QA] Hostile input locked in: {hostile_mock_data[:80]}...")
 
        # =================================================================
        # PHASE 2 — DESIGN DISCUSSION: Each persona proposes an approach.
        # No code written yet — proposals only.
        # =================================================================
        pipeline_put("\n[PHASE 2] Design Council — Gathering independent proposals...")
        proposals = {}
        for persona_name, style_note in persona_styles.items():
            pipeline_put(f"   -> [{persona_name}] Submitting design proposal...")
            proposal_prompt = (
                f"You are a {persona_name}. {style_note}\n"
                f"You have been asked to build a Python tool for this task:\n"
                f"Task: '{current_prompt}'\n"
                f"Context from memory: {synthesized_context}\n\n"
                f"DO NOT write any code yet.\n"
                f"Output a short design proposal covering:\n"
                f"1. Your chosen approach and why\n"
                f"2. The key edge cases you will handle\n"
                f"3. The biggest risk of failure and how you will prevent it\n"
                f"Keep it under 150 words."
            )
            proposals[persona_name] = ask_qwen(
                proposal_prompt,
                system_prompt=f"You are a {persona_name}. Output a concise design proposal only. No code."
            )
            pipeline_put(f"   -> [{persona_name}] Proposal received.")
 
        # =================================================================
        # PHASE 3 — CROSS-CRITIQUE: Each persona reviews the other two
        # proposals and identifies weaknesses. Output is a consensus doc.
        # =================================================================
        pipeline_put("\n[PHASE 3] Cross-Critique — Building consensus design document...")
        all_proposals_text = "\n\n".join(
            [f"[{name} PROPOSAL]:\n{proposal}" for name, proposal in proposals.items()]
        )
        consensus_prompt = (
            f"Three engineers have proposed approaches for this task: '{current_prompt}'\n\n"
            f"{all_proposals_text}\n\n"
            f"Review all three proposals. Identify:\n"
            f"1. The strongest idea from each proposal\n"
            f"2. The weaknesses or blind spots in each\n"
            f"3. A final consensus approach that combines the best elements\n"
            f"4. A definitive list of edge cases ALL implementations must handle\n"
            f"Output this as a structured consensus design document. No code."
        )
        consensus_doc = ask_qwen(
            consensus_prompt,
            system_prompt="You are a senior technical architect. Output a consensus design document only. No code."
        )
        pipeline_put("   -> [COUNCIL] Consensus design document established.")
 
        # =================================================================
        # PHASE 4 — CODE GENERATION: Each persona writes code based on
        # the consensus document. Shared error ledger carries forward.
        # =================================================================
        pipeline_put("\n[PHASE 4] Code Generation — Each engineer implements the consensus...")
        shared_error_ledger = ""
 
        for branch_name, style_note in persona_styles.items():
            pipeline_put(f"\n   -> [{branch_name}] Writing implementation...")
 
            # Build the generation prompt from the consensus doc
            generation_prompt = (
                f"You are a {branch_name}. {style_note}\n"
                f"Task: '{current_prompt}'\n"
                f"Memory context: {synthesized_context}\n\n"
                f"--- AGREED DESIGN DOCUMENT (you must follow this) ---\n"
                f"{consensus_doc}\n"
                f"-----------------------------------------------------\n\n"
                f"Write a Python function: run_skill(user_input: str) -> str\n"
                f"that implements the agreed design.\n"
                f"Output ONLY the code inside a ```python block."
            )
 
            # Inject shared failure ledger if prior branches failed
            if shared_error_ledger:
                generation_prompt += (
                    f"\n\n--- CRITICAL: PREVIOUS IMPLEMENTATIONS FAILED QA ---\n"
                    f"The engineers before you wrote code that failed hostile testing.\n"
                    f"YOU MUST USE A DIFFERENT APPROACH to avoid the same failures.\n"
                    f"{shared_error_ledger}"
                )
 
            raw_code_response = ask_qwen(generation_prompt)
 
            # Clean output
            match = re.search(r'```python\n(.*?)\n```', raw_code_response, re.DOTALL)
            clean_code = (
                match.group(1).strip() if match
                else raw_code_response.replace("```python", "").replace("```", "").strip()
            )
 
            # =============================================================
            # PHASE 5 — SANDBOX TEST using the pre-generated hostile data
            # =============================================================
            pipeline_put(f"   -> [{branch_name}] Entering Sandbox with hostile QA data...")
            start_time = time.time()
            success, output = isolated_sandbox_test(
                clean_code, test_input=hostile_mock_data,
                model=code_model, tokenizer=code_tokenizer
            )
            execution_time = time.time() - start_time
 
            # --- 1-RETRY LIFELINE ---
            if not success or len(output.strip()) == 0:
                pipeline_put(f"   -> [{branch_name}] FAILED initial QA. Error: {output[:120]}. Initiating self-correction...")
                fix_prompt = (
                    f"Your code failed hostile QA testing with this error:\n{output}\n\n"
                    f"Original Code:\n{clean_code}\n\n"
                    f"Fix the exact error. You must still satisfy all requirements "
                    f"from the consensus design document.\n"
                    f"Output ONLY the fixed code in a ```python ``` block."
                )
                raw_fixed_response = ask_qwen(fix_prompt)
                match = re.search(r'```python\n(.*?)\n```', raw_fixed_response, re.DOTALL)
                clean_code = (
                    match.group(1).strip() if match
                    else raw_fixed_response.replace("```python", "").replace("```", "").strip()
                )
 
                pipeline_put(f"   -> [{branch_name}] Re-entering Sandbox after self-correction...")
                start_time = time.time()
                success, output = isolated_sandbox_test(
                    clean_code, test_input=hostile_mock_data,
                    model=code_model, tokenizer=code_tokenizer
                )
                execution_time = time.time() - start_time
 
            # --- LOG OUTCOME ---
            if not success or len(output.strip()) == 0:
                pipeline_put(f"   -> [WARNING] {branch_name} failed even after self-correction. Logging to shared ledger...")
                shared_error_ledger += (
                    f"\n[FAILED ATTEMPT BY {branch_name}]\n"
                    f"Approach summary: {proposals.get(branch_name, 'N/A')[:200]}\n"
                    f"Code written:\n```python\n{clean_code}\n```\n"
                    f"Hostile test input used:\n{hostile_mock_data}\n"
                    f"Error output:\n{output}\n"
                    f"-------------------------------------------\n"
                )
            else:
                lines_of_code = len(clean_code.split('\n'))
                pipeline_put(f"   -> [{branch_name}] PASSED hostile QA! (Time: {execution_time:.4f}s | Lines: {lines_of_code})")
                successful_candidates.append({
                    "branch": branch_name,
                    "code": clean_code,
                    "time": execution_time,
                    "lines": lines_of_code
                })
 
        # =================================================================
        # PHASE 6 — TOURNAMENT: Score all passing candidates and pick best
        # =================================================================
        if not successful_candidates:
            pipeline_put("\n   -> [CRITICAL ERROR] All branches failed hostile QA even with self-correction. Aborting save.")
            return False
 
        pipeline_put(f"\n[PHASE 6] Tournament Evaluation — {len(successful_candidates)}/3 branches passed hostile QA.")
 
        # Sort: fastest execution first, then fewest lines as tiebreaker
        successful_candidates.sort(key=lambda x: (x['time'], x['lines']))
        ultimate_winner = successful_candidates[0]
        pipeline_put(f"   -> [WINNER] {ultimate_winner['branch']} selected. "
                     f"(Time: {ultimate_winner['time']:.4f}s | Lines: {ultimate_winner['lines']})")
 
        # =================================================================
        # PHASE 7 — SAVE: Generate semantic filename and write to disk
        # =================================================================
        pipeline_put("\n[PHASE 7] Generating semantic filename...")
        name_prompt = (
            f"Task: '{current_prompt}'. Generate a short descriptive snake_case filename "
            f"for a Python script that does this (max 3 words). "
            f"Do NOT include the .py extension. Output ONLY the snake_case name."
        )
        raw_name = ask_qwen(
            name_prompt,
            system_prompt="You are a strict file naming utility. Output ONLY valid snake_case text, nothing else."
        )
        safe_filename = re.sub(r'[^a-z0-9_]', '', raw_name.replace('.py', ''))
        if len(safe_filename) < 2:
            safe_filename = "custom_tool_" + str(int(time.time()))
        safe_filename = safe_filename + ".py"
 
        skills_folder = globals().get('SKILLS_DIR', './skills')
        filepath = os.path.join(skills_folder, safe_filename)
 
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(ultimate_winner['code'])
 
        pipeline_put(f"   -> [SAVED] Optimal skill saved to {filepath}")
        is_success = True
 
    finally:
        pipeline_put("\n   -> [SYSTEM] Initiating Qwen 3B VRAM Flush...")
        
        # 1. Record VRAM before deletion (if CUDA is available)
        vram_before = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        
        # 2. Delete components
        if 'code_model' in globals() and code_model is not None:
            del code_model
        if 'code_tokenizer' in globals() and code_tokenizer is not None:
            del code_tokenizer
        
        # 3. Force garbage collection
        gc.collect()
        
        # --- THE FIX (CROSS-BRAIN FIX A): VRAM VERIFICATION LOOP ---
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            import time
            start_flush = time.time()
            vram_cleared = False
            
            # Wait up to 5 seconds for the GPU to physically release the memory
            while time.time() - start_flush < 5.0: 
                current_vram = torch.cuda.memory_allocated() / (1024 ** 2)
                # If VRAM dropped by at least 500MB, we know the model was successfully evicted
                if vram_before - current_vram > 500: 
                    vram_cleared = True
                    pipeline_put(f"   -> [SYSTEM] VRAM successfully flushed! Freed {vram_before - current_vram:.0f}MB.")
                    break
                time.sleep(0.5)
            
            if not vram_cleared:
                pipeline_put(f"   -> [WARNING] VRAM flush timeout or leak detected. Current VRAM stuck at {current_vram:.0f}MB.")
        # -------------------------------------------------------------

        global coder_loaded
        code_model = None
        code_tokenizer = None
        coder_loaded = False
        pipeline_put("   -> [SYSTEM] Execution Complete. Returning to Base Brain.")

# ==============================================================================
# SANDBOX
# ==============================================================================

def get_venv_executables():
    """Returns the correct paths for python and pip inside the isolated venv."""
    if os.name == 'nt':
        return (
            os.path.join(VENV_DIR, "Scripts", "python.exe"),
            os.path.join(VENV_DIR, "Scripts", "pip.exe")
        )
    else:
        return (
            os.path.join(VENV_DIR, "bin", "python"),
            os.path.join(VENV_DIR, "bin", "pip")
        )


def isolated_sandbox_test(code_string, test_input="test_data", model=None, tokenizer=None, retry_depth=0):
    """
    Runs AI code in an isolated venv, catches missing PIP packages,
    and prevents infinite loops using a 15-second timeout and recursion depth limits.
    """
    # 1. Prevent infinite pip loops
    if retry_depth > 2:
        return False, "Execution halted: Maximum auto-install retries exceeded. Check module names."

    # 2. Initialize walled garden if it doesn't exist
    if not os.path.exists(VENV_DIR):
        print("\n[SYSTEM ALERT] Initializing isolated AI virtual environment (ai_venv)...")
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR])
        print("   -> Isolation barrier established.")

    python_exe, pip_exe = get_venv_executables()

    trigger_block = f"\n\nif __name__ == '__main__':\n    print(run_skill('{test_input}'))\n"
    full_code = code_string + trigger_block

    def limit_resources():
        if resource:
            max_mem = 2000 * 1024 * 1024  # 2 GB
            resource.setrlimit(resource.RLIMIT_AS, (max_mem, max_mem))

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
        temp_file.write(full_code)
        temp_file_path = temp_file.name

    try:
        if test_input == "test_data":
            test_dir = os.path.join(os.path.dirname(temp_file_path), "test_data")
            os.makedirs(test_dir, exist_ok=True)
            open(os.path.join(test_dir, "dummy_file.txt"), 'w').close()

        sandbox_env = os.environ.copy()
        sandbox_env["OPENBLAS_NUM_THREADS"] = "1"
        sandbox_env["MKL_NUM_THREADS"] = "1"

        result = subprocess.run(
            [python_exe, temp_file_path],
            capture_output=True,
            text=True,
            timeout=15,
            preexec_fn=limit_resources if resource else None,
            env=sandbox_env
        )

        # Auto-pip interceptor
        if "ModuleNotFoundError" in result.stderr:
            match = re.search(r"No module named '(\w+)'", result.stderr)
            if match and model and tokenizer:
                missing_module = match.group(1)
                print(f"\n[SYSTEM ALERT] Tool execution halted. Missing library: '{missing_module}'")

                exp_prompt = f"What does the Python pip library '{missing_module}' do? Explain in one short sentence."
                inputs = tokenizer(exp_prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    exp_out = model.generate(**inputs, max_new_tokens=50, temperature=0.3)
                lib_explanation = tokenizer.decode(
                    exp_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
                print(f"   -> Library Function: {lib_explanation}")

                approve_pip = input(
                    f"   -> [SUPERVISOR OVERRIDE] Allow AI to install '{missing_module}' into its ISOLATED venv? (y/n): "
                ).strip().lower()

                if approve_pip in ['y', 'yes']:
                    print(f"   -> Installing {missing_module} into ai_venv...")
                    subprocess.run([pip_exe, "install", missing_module])
                    print("   -> Installation complete. Restarting tool execution...")
                    os.remove(temp_file_path)
                    return isolated_sandbox_test(code_string, test_input, model, tokenizer, retry_depth + 1)
                else:
                    return False, f"[SYSTEM] Execution failed: Human denied permission to install {missing_module}."

        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr

    except subprocess.TimeoutExpired:
        return False, "TimeoutError: The code took longer than 15 seconds. You likely wrote an infinite loop."
    except Exception as e:
        return False, f"System Error: {str(e)}"
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# ==============================================================================
# SKILL AUDITOR
# ==============================================================================

def audit_and_fix_skills(model, tokenizer):
    """
    Scans the ./skills directory, tests every python file in the sandbox,
    and forces the Qwen Code Brain to fix any broken scripts.
    """
    print("\n[SYSTEM AUDIT] Initiating Parietal Lobe Tool Inspection...")
    skill_files = glob.glob(os.path.join(SKILLS_DIR, "*.py"))

    if not skill_files:
        print("   -> No skills found to audit.")
        return "No skills found. The tool directory is empty."

    repair_log = []

    for filepath in skill_files:
        filename = os.path.basename(filepath)
        print(f"   -> Auditing {filename}...")

        with open(filepath, "r", encoding="utf-8") as f:
            original_code = f.read()

        success, output = isolated_sandbox_test(original_code)

        if success:
            print(f"      [PASS] {filename} is fully operational.")
            repair_log.append(f"PASS: {filename}")
        else:
            print(f"      [FAIL] {filename} crashed. Triggering Code Brain repair...")
            load_auditor()  # Boot Qwen only if something is broken

            repair_prompt = f"""
The following system tool named '{filename}' is currently broken.

Broken Code:
{original_code}

Sandbox Error Traceback:
{output}

Fix the code so it executes properly without crashing. Ensure the 'run_skill(user_input)'
function is intact. Output the fixed Python code strictly inside a ```python ``` markdown block.
No explanations.
"""
            messages = [
                {"role": "system", "content": "You are an elite Python engineer. Output ONLY valid Python code enclosed in a ```python block. No explanations."},
                {"role": "user", "content": repair_prompt}
            ]

            formatted = code_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = code_tokenizer(formatted, return_tensors="pt").to(code_model.device)

            with torch.no_grad():
                outputs = code_model.generate(
                    **inputs,
                    max_new_tokens=1500,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=code_tokenizer.eos_token_id
                )

            raw_response = code_tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()

            match = re.search(r'```python\s*(.*?)\s*```', raw_response, re.DOTALL)
            repaired_code = (
                match.group(1).strip() if match
                else raw_response.replace("```python", "").replace("```", "").strip()
            )

            repair_success, repair_output = isolated_sandbox_test(repaired_code)

            if repair_success:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(repaired_code)
                print(f"      [REPAIRED] {filename} successfully debugged and overwritten.")
                repair_log.append(f"REPAIRED: {filename}")
            else:
                print(f" -> [CRITICAL] Repair failed. Manual intervention required for {filename}.")
                repair_log.append(f"FAILED TO REPAIR: {filename}")

                meta = load_skill_metadata()
                mod_name = filename[:-3]
                if mod_name in meta:
                    meta[mod_name]["fail_count"] = meta[mod_name].get("fail_count", 0) + 1
                    save_skill_metadata(meta)

    if coder_loaded:
        print("   -> [SYSTEM] Audit complete. Flushing Qwen Auditor from VRAM...")
        del code_model
        del code_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print("[SYSTEM AUDIT COMPLETE]\n")
    return "System Audit Complete. Here are the results:\n\n" + "\n".join(repair_log)