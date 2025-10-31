from .llm import prompt_ollama

summarize_prompt = """
Instructions
In order to find evidence that helps your fact-check, you just ran a web search, which yielded a Search Result.
Your task right now is to summarize the Search Result concisely in at most 5 sentences, only including information that is relevant to the Claim you are checking.
What to include:
Information that might be useful for fact-checking the claim (see Record).
If available: the release date as well as the author or the publisher (e.g., the media company) of the search result.
Do NOT include:
Advertisements.
Any other information unrelated to the Record or the Claim.
Additional Rules:
Do not add any additional information besides the information in the Search Result. Also, do not add any information that is not related to the claim, even if it is mentioned in the Search Result.
If the Search Result doesn't contain any relevant information for the fact-checking work, print only one word in capital letters, do not include anything else: NONE.
Keep your writing style consistent with the provided Examples.
Try to filter out relevant information even if the Search Result is in a different language.

Claim: {claim}

Evidence:
{url}
{search_result}

Record:
{record}

Important: Write your summary in Vietnamese.
Your Summary:
"""

def summarize(claim, search_result, url, record, think=True):
    prompt = summarize_prompt.format(
        claim=claim,
        search_result=search_result,
        record=record,
        url=url
    )
    return prompt_ollama(prompt, think=think)
    