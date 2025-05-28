import json
import os
import datetime
import uuid

LOG_DIR = "manual_interaction_logs"

def get_input(prompt_message):
    """Gets multi-line input from the user."""
    print(f"{prompt_message} (Type END when done):")
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        except EOFError: # Handles Ctrl+D
             break
    return "\n".join(lines)

def log_event(log_file_path, event_data):
    """Appends a JSON event to the log file."""
    event_data['timestamp'] = datetime.datetime.utcnow().isoformat() + "Z"
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event_data) + '\n')
    except IOError as e:
        print(f"Error writing to log file {log_file_path}: {e}")

if __name__ == "__main__":
    print("--- Manual Interaction Logger ---")
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- Initialize Run ---
    run_tag = input("Enter a unique tag for this run (e.g., P1-Task1-FullSystem): ")
    persona_id = input("Enter Persona ID (e.g., P1): ")
    task_id = input("Enter Task ID (e.g., Task1): ")
    simulator_name = input("Enter Your Name/ID as Simulator: ")
    ablation_condition = input("Enter Ablation Condition (e.g., FullSystem): ")
    run_id = f"{run_tag}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:4]}"
    log_file = os.path.join(LOG_DIR, f"{run_id}.jsonl")

    print(f"\nLogging to file: {log_file}")
    print("Starting run. Enter 'quit' at any prompt to end the session.")

    base_log_data = {
        'run_id': run_id,
        'persona_id': persona_id,
        'task_id': task_id,
        'llm_simulator': simulator_name, # Indicates manual entry
        'ablation_condition': ablation_condition,
    }
    log_event(log_file, {**base_log_data, 'turn_number': 0, 'event_type': 'MANUAL_SIMULATION_START'})

    turn_number = 0
    while True:
        turn_number += 1
        print(f"\n--- Turn {turn_number} ---")

        # --- Log User Utterance ---
        user_utterance = get_input("Enter USER Utterance")
        if user_utterance.strip().lower() == 'quit':
            break
        log_event(log_file, {
            **base_log_data,
            'turn_number': turn_number,
            'event_type': 'USER_UTTERANCE',
            'event_data': {'text': user_utterance}
        })

        # --- Log System Response ---
        system_response_text = get_input("Enter SYSTEM Response Text")
        if system_response_text.strip().lower() == 'quit':
            break

        system_products_str = get_input("Paste SYSTEM Products JSON/Data (or leave blank then END)")
        if system_products_str.strip().lower() == 'quit':
            break
        parsed_products = None
        if system_products_str.strip():
             try:
                 # Try parsing as JSON list, fallback to storing as string
                 parsed_products = json.loads(system_products_str)
             except json.JSONDecodeError:
                 print("Warning: Could not parse products as JSON, storing as raw string.")
                 parsed_products = system_products_str # Store as string if not valid JSON

        turn_notes = input("Enter any Notes for this turn (optional): ")
        if turn_notes.strip().lower() == 'quit':
             break

        log_event(log_file, {
            **base_log_data,
            'turn_number': turn_number,
            'event_type': 'SYSTEM_RESPONSE',
            'event_data': {
                'text': system_response_text,
                'products': parsed_products,
                'notes': turn_notes
            }
        })

    log_event(log_file, {**base_log_data, 'turn_number': turn_number, 'event_type': 'MANUAL_SIMULATION_END'})
    print(f"\nSession ended. Log saved to {log_file}")