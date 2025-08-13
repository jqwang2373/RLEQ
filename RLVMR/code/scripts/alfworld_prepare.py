from datasets import load_dataset, load_from_disk
import logging
import numpy as np
import time
import json
from agent_system.environments.prompts import *
import re
import random
import openai


NUM_TRAJS = 300
openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI API key
MODEL = "gpt-4o"  # Replace with your desired model
SAVE_PATH = "data/alfworld_cold-start.json"

alfworld_dataset = load_from_disk("data/alfworld_expert_traj")

trajs = []
for traj in alfworld_dataset:
    conversations = traj["conversations"][2:]
    traj_log = []
    task = conversations[0]['value'].split("Your task is to:")[-1].split("AVAILABLE ACTIONS:")[0].strip()
    for i in range(0, len(conversations), 2):
        obs = conversations[i]['value'].split("AVAILABLE ACTIONS:")[0]
        llm_action = conversations[i + 1]['value']
        # Extracting the action from the llm_action: Action: xxx
        action = llm_action.split("Action:")[-1].strip()
        traj_log.append({
            "observation": obs,
            "action": action
        })
    if len(traj_log) < 30:
        trajs.append({
            "task": task,
            "traj": traj_log
        })

random.shuffle(trajs)

def llm(prompt, model, temperature=0.0, max_tokens=1024, retries=3):
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            time.sleep(2 ** attempt)
    return None

def llm_json(prompt, model, temperature=0.0, max_tokens=1024, retries=5):
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response['choices'][0]['message']['content'].replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            print(f"Error: {e}. Retrying... (Attempt {attempt + 1}/{retries})")
            time.sleep(2 ** attempt)
    
    return []


def merge(res, traj):
    if (len(res) != len(traj)):
        print(f"Length mismatch: {len(res)} vs {len(traj)}")
        return False, None

    if "action" not in res[0] or "reason" not in res[0]:
        print("Missing 'action' or 'reason' in the response structure.")
        return False, None
    
    output = []
    for a, b in zip(res, traj):
        if a["action"] != b["action"]:
            print(f"Action mismatch: {a['action']} vs {b['action']}")
            return False, None
        if not a["reason"].startswith("<"):
            print(f"Invalid reason format: {a['reason']}")
            return False, None
        output.append({
            "obs": b["observation"],
            "reason": a["reason"],
            "action": a["action"],
        })
    return True, output

meta_traj = []
sft_data = []

for traj in trajs[:NUM_TRAJS]:
    prompt = ALFWORLD_TAGGING_TEMPLATE.format(traj=json.dumps(traj["traj"], ensure_ascii=False))
    res = llm_json(prompt, MODEL)
    valid, res = merge(res, traj["traj"])

    if not valid:
        print(f"Invalid response for task: {traj['task']}")
        print(f"Response: {res}")
        continue

    if valid:
        meta_traj.append({
            "task": traj["task"],
            "traj": res
        })
        
        step_level_data = []
        latest_planning = "No plan."
        for i, item in enumerate(res):
            if i == 0:
                prompt = ALFWORLD_TEMPLATE_NO_HIS_CS.format(
                    current_observation=item["obs"],
                )
            else:
                action_history = "\n".join([f"[Observation {j + 1}: '{res[j]['obs']}', Action {j + 1}: '{res[j]['action']}']" for j in range(i)])
                history_think_length = min(3, i)
                action_history += "\n- recent reasoning process: \n"
                for j in range(i - history_think_length, i):
                    action_history += f"[Observation {j + 1}: {res[j]['obs']}, output: '{res[j]['reason']} <action>{res[j]['action']}</action>']\n"
                prompt = ALFWORLD_TEMPLATE_CS.format(
                    task_description=traj["task"],
                    step_count=i,
                    history_length=i,
                    action_history=action_history,
                    current_step=i + 1,
                    current_observation=item["obs"],
                    planning=latest_planning
                )

            # update current planning
            if '<planning>' in item["reason"]:
                current_planning = re.search(r'<planning>(.*?)</planning>', item["reason"], re.DOTALL)
                if current_planning:
                    latest_planning = current_planning.group(1).strip()
                else:
                    pass
            response = f"{item['reason']}\n<action>{item['action']}</action>\n"

            step_level_data.append({
                "step": i + 1,
                "obs": item["obs"],
                "prompt": prompt,
                "response": response,
            })

        sft_data.append({
            "task": traj["task"],
            "done": "True",
            "data": step_level_data
        })

with open(SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(sft_data, f, ensure_ascii=False, indent=4)