from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from tools import webSearch, ObtainPrompt, wikidataSearch, wikipediaSearch

CuratorPrompt= """
    You are the main researcher of an investigation team. Your job is to create an in-depth analysis report.

    PROCESS:
    1. You will receive an initial research and a list of subtopics
    2. For EACH subtopic, you must:
    - Search for information using your tools (webSearch, wikipediaSearch, wikidataSearch)
    - Write a detailed section (1 paragraph minimum) about that subtopic
    - Connect it to the main topic when relevant
    - Add new insights, not just repeat what's in the initial research

    3. Structure your output as follows:

    # [Main research]
    [Brief overview of the main topic]

    ## 1. Subtopic name
    [Detailed analysis with new information]

    ## 2. Subtopic name
    [Detailed analysis with new information]

    ## 3. Subtopic name
    [Detailed analysis with new information]

    # Sources:

    ## [Source Name] (year): Title of the source - url of the source


    CRITICAL RULES:
    - They may not be any subtopics.
    - Research ALL subtopics provided, not just the first one
    - Each subtopic must have its own dedicated section
    - Always cite your sources at the end of the research. Do not include them in the middle.
    - Do NOT skip any subtopics
    - Use your tools to gather fresh information for each subtopic
    - Prioritize recent (2024+) information for ever-changing topics such as a Country economic situation
    - The MAJORITY of the research needs to be done on the main section. 10% of total information share per subtopic MAX. 
    - The Research needs to be extense
    - Prioritize inforrmation on the main topic, then procede with the subtopics, but always leave room from them. The report can't be 'too long'

    IMPORTANT: You must complete research on ALL subtopics before finishing.
"""

llamaModel = ChatOllama(model='qwen2.5', temperature=0.3)

CuratorAgent = create_agent(
    model = llamaModel, 
    tools=[webSearch, wikipediaSearch, wikidataSearch],
    system_prompt= ObtainPrompt(CuratorPrompt)
)

def InvokeCurator(research: str, subtopics: str|None) -> str:
    '''
    :param research: Key things and basic information about a topic via Searching the web, wikipedia and wikidata
    :type research: str
    :param subtopics: Subtopics of the research
    :type subtopics: str | None
    :return: More advanced research of said topic.
    :rtype: str
    '''
    if subtopics is not None:
        prompt: str = f"""
        Here is the original, more surface-level research: {research}. Here are the subtopics: {subtopics}.
        Research further and gather more specific and detailed information about ALL of them.
        """
    else: #Case if all subtopics are rejected.
        prompt: str = f""" 
        Here is the original, more surface-level research: {research}. There are no subtopics.
        Research further and gather more specific and detailed information about the topic.
        """
    
    response = CuratorAgent.invoke( 
        {'messages' : [ {'role' : 'user', 'content' : prompt} ] }
    )
    return (response['messages'][-1].content)