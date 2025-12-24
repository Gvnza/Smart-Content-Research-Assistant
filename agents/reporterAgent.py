from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from tools import ObtainPrompt

reporterPrompt = """
    You are the main writer of a research team. Your job is to write the report. You will have at your disposition 
    what other agents gatehered about the topic. The report needs to feel polished. Do not add anything to the research, 
    nor think about its contents, only write the report. Only return the report and its title. It needs to have a structure, 
    so do a section for every subtopic. You will have the titles of the subtopics provided, but the information resides only on
    the main research. Do not generate a new section if you think you find another subtopic, only do it for the explicitly mentioned
    ones.

    RULES:
    - Always include references (sources) at the end of the report and at the end only (AFTER the conclusion).
    - The main section needs to represent the majority of the report
    - The report needs to be at least 5 paragraphs long.
    - Keep the URLs, titles and years of the sources.
    - Always include an executive summary.
    - Always include a conclusion.
    """

llamaModel = ChatOllama(model='qwen2.5', temperature=0.5)

reporterAgent = create_agent(
    model=llamaModel, 
    system_prompt=ObtainPrompt(reporterPrompt)
)

def invokeReporter(RESEARCH: str) -> str:
    '''
    Generates a report based on a research of a certain topic
    
    :param RESEARCH: Research of a certain topic
    :type RESEARCH: str
    :rtype: str
    :return: Report on said research
    '''
    response = reporterAgent.invoke(
        {'messages' : [ {'role' : 'user', 'content' :  f"Do the report on these research: {RESEARCH}"} ] }
    )

    return(response['messages'][-1].content)