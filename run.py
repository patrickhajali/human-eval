
import asyncio
import json
import os
import re
import tqdm
import time
import openai
from prompts import (
    gpt_zeroshot_prompt,
    gpt_system_prompt, 
    gpt_prompt_with_formatting
)

# from dotenv import load_dotenv
# load_dotenv()

openai.api_key = ''


HUMAN_EVAL = os.environ['PWD'] + '/data/HumanEval.jsonl'
OUT_FILE = os.environ['PWD'] + '/results/{}_{}.jsonl'

async def retry(sem, fn):
    for i in range(5, 10):
        try:
            async with sem:
                return await fn()
        except Exception as e:
            print(e)
            print('retrying')
            time.sleep(i) 
    return await fn()

async def get_completion(sem, prompt, num_tries=1, model='gpt-3.5-turbo'):
    '''Get a completion from a model.'''

    if num_tries == 1:
        temperature = 0.0
    elif num_tries == 10:
        temperature = 0.6
    elif num_tries == 100:
        temperature = 0.8
    else:
        temperature = 0.5
  
    if model in {'gpt-3.5-turbo', 'gpt-4'}:
        completion = await retry(sem, lambda: openai.ChatCompletion.acreate(messages= prompt, model=model, temperature=temperature, max_tokens=1000, n=num_tries))
        choices = completion.choices
        return [choice['message']['content'] for choice in choices]
    else:
        return 'Invalid Model'


def iter_hval():
    all_lines = []
    with open(HUMAN_EVAL) as f:
        for line in f:
            all_lines.append(json.loads(line))

    return all_lines

async def get_results(num_tries=10, model='gpt-3.5-turbo', prompt_type = 'zero-shot', ids = []):
    '''Generates prompt, gets completion, and formats output for all question ids. Writes results to out_file.'''

    out_file = OUT_FILE.format(model, prompt_type)

    with open(out_file, 'w') as f:
        pass

    sem = asyncio.Semaphore(10)

    async def output(prompt, task_id):
        '''Builds prompt (for a single question) based on the model and calls get_completion. Parses the completions and
         returns result as dictionary'''

        # PROMPT SELECTION
        if prompt_type == 'zero-shot':
            full_prompt = gpt_zeroshot_prompt(prompt)
        elif prompt_type == 'system-only':
            full_prompt = gpt_system_prompt(prompt)
        elif prompt_type == 'system-with-parser':
            full_prompt = gpt_prompt_with_formatting(prompt)

        async with sem:
            completions = await get_completion(sem, full_prompt, num_tries=num_tries, model=model)

        outs = []

        for idx, completion in enumerate(completions):

            if prompt_type == 'system-with-parser':
                    # Parser 
                    completion_key = "# START OF COMPLETION"
                    if completion_key in completion:
                        start_index = completion.find(completion_key)
                        completion = completion[start_index+len(completion_key):]
                    # End Parser
            else:
                completion = completion
                
            outs.append({'task_id': task_id, 'completion': completion})

        return outs


    futures = []
    for line in tqdm.tqdm(iter_hval()):
        prompt = line['prompt']
        task_id = line['task_id']
        id = int(task_id.split("/")[1])
        if (id in ids) or ids == []:
            futures.append(output(prompt, task_id))


    with open(out_file, 'a') as out_f:
        for future in tqdm.tqdm(asyncio.as_completed(futures), total=len(futures)):
            outs = await future
            for out in outs:
                out_f.write(json.dumps(out) + '\n')


if __name__ == '__main__':
    # Select k -> pass@k 
    num_tries=1

    # Select Model ('gpt-3.5-turbo', 'gpt-4')
    model = 'gpt-3.5-turbo'

    # Select prompt ('zero-shot', 'system-only', 'system-with-parser' )
    prompt_type = 'zero-shot'

    # Run only specific task-ids (if needed), leave ids = [] to test all
    ids = []

    asyncio.run(get_results(num_tries=num_tries, model=model, prompt_type=prompt_type, ids=ids)) 

    out_f = OUT_FILE.format(model, num_tries)
    print(out_f)
