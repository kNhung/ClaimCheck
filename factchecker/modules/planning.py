from .llm import prompt_ollama

plan_prompt = """Instructions
The available knowledge is insufficient to assess the Claim.
Therefore, propose a set of actions to retrieve new and helpful evidence. Adhere to the following rules:
The actions available are listed under Valid Actions, including a short description for each action. No other actions are possible at this moment.
For each action, use the formatting as specified in Valid Actions.
Include all actions in a single Markdown code block at the end of your answer.
Propose as few actions as possible but as much as needed. Do not propose similar or previously used actions.
Consider Both Modalities Equally: Avoid focusing too much on one modality at the expense of the other, but always check whether the text claim is true or false.
Compare Image and Caption: Verify the context of the image and caption.

Valid Actions:
{valid_actions}

Examples:
{examples}

Record:
{record}

Claim: {claim}
Your Actions:
"""

decompose_prompt = """Instructions
Decompose the claim into smaller, manageable sub-claims or questions that can be addressed individually. Each sub-claim should be specific and focused.
There should be no more than 5 sub-claims.

Claim: {claim}
Your Sub-Claims:

"""

def plan(claim, record="", examples="", actions=None, think=True):
    action_definitions = {
        "geolocate": {"desc": "Determine the country where an image was taken by providing an image ID.", "example": "geolocate(<image:k>)"},
        "reverse_search": {"desc": "Perform a reverse image search on the web for similar images.", "example": "reverse_search(<image:k>)"},
        "web_search": {"desc": "Run an open web search for related webpages.", "example": 'web_search("New Zealand Food Bill 2020")'},
        "image_search": {"desc": "Retrieve related images for a given query.", "example": 'image_search("China officials white suits carry people")'}
    }
    if not actions:
        actions = ["web_search", "image_search"]
    elif actions == "All":
        actions = list(action_definitions.keys())
    valid_actions = "\n".join([f"{a}: {action_definitions[a]['desc']}" for a in actions])
    examples = "\n".join([f"{action_definitions[a]['example']}" for a in actions])
    prompt = plan_prompt.format(valid_actions=valid_actions, examples=examples, record=record, claim=claim)
    response = prompt_ollama(prompt, think=think)
    return response
