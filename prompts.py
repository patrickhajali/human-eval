from typing import List
import json


def gpt_zeroshot_prompt(func: str) -> List[dict]:
    return [
        {
            'role': 'system',
            'content': ''''''
        },
        {
            'role': 'user',
            'content':  
f'''
{func}
'''
        }
    ]


def gpt_system_prompt(func: str) -> List[dict]:
    return [
        {
            'role': 'system',
            'content': '''Complete the code.'''
        },
        {
            'role': 'user',
            'content': 
f'''            
{func}
'''
        }
    ]


def gpt_prompt_with_formatting(func: str) -> List[dict]:
    return [
        {
            'role': 'system',
            'content': f'''
Complete the code using the following format:

# FUNCTION HEADER
... 
# START OF COMPLETION 
...
 '''
        },
        {
            'role': 'user',
            'content':  
f'''
{func}
'''
        }
    ]
