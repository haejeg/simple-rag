EMBEDDING_MODEL: str = 'llama3'
GENERATIVE_MODEL: str = 'llama3'
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 20
PERSIST_DIRECTORY = 'db'
LLM_NUM_PREDICT = 128
LLM_TEMPERATURE = 0.5
SEARCH_K = 5
SEARCH_SCORE = 0.35

Precision score:  0.39344262295081966
Precision score reason:  The score is 0.39 because the context includes a direct definition — "An ally is someone willing to take action in support of another person to remove barriers" — plus helpful clarifications like "Being an ally is a verb" and that it "goes beyond being inclusive." However, much of the retrieved content is off-target, focusing on unrelated material such as "At GitLab, it is not required to be an ally, but it is required to be inclusive," "The Ally Lab Learning Group," and procedural items like "open an issue using a template."


EMBEDDING_MODEL: str = 'llama3' 
GENERATIVE_MODEL: str = 'llama3' 
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 100
PERSIST_DIRECTORY = 'db'
LLM_NUM_PREDICT = 512
LLM_TEMPERATURE = 0.5
SEARCH_K = 5
SEARCH_SCORE = 0.35
Precision score:  0.5348837209302325
Precision score reason:  The score is 0.53 because the context contains a direct definition — "An ally is someone willing to take action in support of another person to remove barriers" and "An ally is a member of a social group that has some privilege, works to end oppression" — but much of the rest is off-target, focusing on things like "The first step in being an ally is self-educating," organizational specifics such as "At GitLab it is not required to be an ally," and unrelated definitions like "Privilege is an unearned advantage" and "Oppression is systemic inequality."