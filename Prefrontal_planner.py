import json
from Temporal_memory import ask_gemma_internal
from contextlib import nullcontext


def generate_plan(user_input, skill_registry, model, tokenizer):
    """Breaks the user request into logical steps, utilizing available skills."""
   
    # Look inside the toolbox
    available_tools = ", ".join(skill_registry.keys()) if skill_registry else "None"
   
    # The Upgraded Prompt (No JSON required!)
    prompt = f"""
    Break the following task into a maximum of 3 logical steps.
   
    CRITICAL INSTRUCTION:
    You have access to the following Python tools in your Skill Registry: [{available_tools}]
    If the task requires complex math, coding, or data processing, you MUST explicitly plan to use the relevant tool from that list. Do not try to calculate complex math in your head.
   
    Task: {user_input}
   
    Output ONLY a numbered list. Example:
    1. Analyze the parameters of the problem.
    2. Execute the [Tool Name] tool.
    3. Return the final answer.
    """


    ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
    with ctx:
        response = ask_gemma_internal(prompt, model, tokenizer)
       
    try:
        # Robust Text Parser: Extracts steps from a numbered or bulleted list
        steps = []
        for line in response.split('\n'):
            line = line.strip()
           
            # Check for numbered lists (e.g., "1. " or "1) ")
            if line and line[0].isdigit() and ('.' in line[:3] or ')' in line[:3]):
                delimiter = '.' if '.' in line[:3] else ')'
                step_text = line.split(delimiter, 1)[-1].strip()
                if step_text:
                    steps.append(step_text)
                   
            # Check for bullet points (e.g., "- " or "* ")
            elif line.startswith('- ') or line.startswith('* '):
                steps.append(line[2:].strip())
       
        # If the AI somehow failed to write a list, fallback to raw text
        if not steps:
            print("   -> [PLANNER FALLBACK] Failed to parse list. Using raw text.")
            return [user_input]
           
        return steps[:3] # Cap at 3 steps maximum
       
    except Exception as e:
        print(f"   -> [PLANNER FALLBACK] Parser error: {e}")
        return [user_input]


def run_skill_for_step(step, skill_registry):
    """Checks if a step matches an available tool, otherwise returns the step as a reasoning prompt."""
    if not skill_registry:
        return "No external tools available. Requires general reasoning."
       
    for name, skill in skill_registry.items():
        # Simple keyword matching to see if the tool name is in the plan step
        if name.lower() in step.lower():
            try:
                return skill(step)
            except Exception as e:
                return f"Error running {name}: {e}"
               
    return "No specific tool matched. Requires general LLM reasoning."


def execute_plan(steps, skill_registry):
    """Executes the steps sequentially."""
    results = []
    for step in steps:
        print(f"   [PLAN STEP] {step}")
        result = run_skill_for_step(step, skill_registry)
        results.append(f"Step: {step} | Execution Result: {result}")
    return results


def evaluate_plan(task, results, model, tokenizer):
    """Deterministically evaluates if the plan successfully utilized a tool."""
   
    # If any step actually executed a real tool, consider it successful
    success = any("No specific tool matched" not in str(result) for result in results)
   
    improvement_msg = "None" if success else "Failed to utilize available tools. Require explicit tool creation or utilization instructions."
   
    return {
        "success": success,
        "improvement": improvement_msg
    }


def run_planning_loop(user_input, skill_registry, model, tokenizer):
    """The master loop that orchestrates the prefrontal cortex."""
    print("\n[PREFRONTAL PLANNER STARTED]")
    steps = generate_plan(user_input, skill_registry, model, tokenizer)
   
    # --- THE CIRCUIT BREAKER ---
    # If the 1B model hallucinates an endless list of steps, slice it down to 5.
    if len(steps) > 5:
        print(f"   -> [CIRCUIT BREAKER] Model generated {len(steps)} steps. Truncating to 5 to prevent execution loops.")
        steps = steps[:5]
       
    print(f"   -> Generated Plan: {steps}")
   
    results = execute_plan(steps, skill_registry)
   
    evaluation = evaluate_plan(user_input, results, model, tokenizer)
    print(f"   -> Plan Evaluation: Success={evaluation.get('success')}")
   
    # If the plan failed, do exactly ONE retry to prevent infinite loops
    if not evaluation.get("success") and evaluation.get("improvement"):
        print(f"   -> Retrying with improvement: {evaluation.get('improvement')}")
        new_task = f"{user_input}. Note: {evaluation.get('improvement')}"
        steps = generate_plan(new_task, skill_registry, model, tokenizer)
       
        # --- APPLY BREAKER TO THE RETRY AS WELL ---
        if len(steps) > 5:
            print(f"   -> [CIRCUIT BREAKER] Truncating retry steps to 5.")
            steps = steps[:5]
           
        results = execute_plan(steps, skill_registry)
       
    print("[PREFRONTAL PLANNER ENDED]\n")
    return "\n".join(results)