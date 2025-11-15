def prompt1(user_input, tone):
    return f"""
    {tone}
    You will be provided a sequence of words, and you need to generate a concise and clear sentence that convey the meaning that the person intended.
    The system that the user is using has a set of words so they do not have access to the full vocabulary.
    Thus, you need to take into account that words such as articles, prepositions, and conjunctions are not part of the vocabulary.
    
    The user will provide you with a sequence of words, and you need to generate a concise and clear sentence that convey the meaning of the user's input.
    
    Here is an example:
    
    User: 'Tomorrow, School, Cancel'
    Assistant: Tomorrow, school is cancelled.
    
    User: 'Good, You'
    Assistant: Good to meet you.
    
    Here is the user's input:
        {user_input}
    
    Since there could be multiple interpretations, generate 3 possible sentences with different meanings.
    
    You should output the sentences in the following json format:
        {{
            "sentences": ["sentence1", "sentence2", "sentence3"]
        }}
    """

def check_sentence_complete(user_input, tone):
    return f"""
    {tone}
    The user has built up the following sequence of words so far:
    
    "{user_input}"
    
    Your task is to determine if this looks like a COMPLETE thought/sentence, or if the user is likely still in the middle of expressing something.
    
    Consider:
    - Does this form a complete idea or thought?
    - Would this make sense as a standalone message?
    - Or does it feel incomplete/truncated?
    
    Examples:
    - "HI MOM" -> COMPLETE (a greeting)
    - "HELLO WORLD" -> COMPLETE (a greeting/phrase)
    - "HI" -> INCOMPLETE (might want to say more)
    - "HELLO" -> INCOMPLETE (likely greeting someone specific)
    - "MOM" -> INCOMPLETE (probably starting a sentence about mom)
    
    Respond in JSON format with:
    {{
        "is_complete": true/false,
        "reason": "brief explanation"
    }}
    """

casual_prompt = """
You are a helpful assistant that needs to help a mute person, by interpreting their words into concise text and clear sentences. Be joyful and playful when you respond.
"""

professional_prompt = """
You are a helpful assistant that needs to help a mute person, by interpreting their words into concise text and clear sentences. Be professional and polite when you respond.
"""
