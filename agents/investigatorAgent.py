from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from tools import webSearch, ObtainPrompt

InvestigatorPrompt =  """
    You are a research assistant that gathers information from the web.
    When given a topic, you must:
    1. Search the web and write the initial information (just the key ideas) covering the main points
    2. At the END, list 3-5 potential subtopics for deeper research
    3. Format subtopics EXACTLY like this, one per line:
        SUBTOPIC: AI Ethics
        SUBTOPIC: AI Applications
        SUBTOPIC: Climate Models
    
    IMPORTANT: 
    - Always search the web before answering
    - Keep subtopic names short (2-4 words maximum)
    - Separate main research from subtopics clearly
    - Only output the information, do not mention any Task or step.
    - Only research what you are asked to.
    """


llamaModel = ChatOllama(model='llama3.2:3b', temperature=0)
InvestigatorAgent = create_agent(
    model = llamaModel,
    tools= [webSearch],
    system_prompt= ObtainPrompt(InvestigatorPrompt)
)

def InvokeInvestigator() -> str:
    '''
    Gathers fundamental info about a topic via searching the web.
    :rtype: str
    :return: Key information about the topic requested.
    '''
    question: str = input("Provide the theme to be researched: ")
    response = InvestigatorAgent.invoke(
        {'messages' : [ {'role' : 'user', 'content' :  f"{question}"} ] }
    )
    return(response['messages'][-1].content)
    
