from langchain.tools import tool
from textwrap import dedent
from ddgs import DDGS
import wikipedia
from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun

def ObtainPrompt(system_prompt: str) -> str:
    '''
    Dedents a prompt of multiple lines and facilitates usage of it
    :param system_prompt: Multi-line prompt 
    :type system_prompt: str
    :return: said prompt dedented and facilitated for an agent
    :rtype: str
    '''
    return dedent(system_prompt).strip()


@tool('webSearch', description='Surf the web for information on a given topic')
def webSearch(query: str) -> str:
    '''
    Searches on the web with a certain query
    
    :param query: Topic to search for
    :type query: str
    :return: Web result for search 
    :rtype: str
    '''
    try:
        with DDGS() as search:
            results = list(search.text(query, max_results=3))

            if not results:
                return "Did not find any information in the web"
            
            output = ""
            for result in results:
                output += f"Title: {result['title']}\n"
                output += f"Source: {result['href']}\n"
                output += f"Content: {result['body']}\n\n"
            
            return output
    
    except Exception as error:
        return f"An error has occurred when surfin in the web: {error}"



@tool('wikipediaSearch', description="Get a summary of a page with a certain name on wikipedia")
def wikipediaSearch(query: str) -> str:
    '''
    Searches Wikipedia and returns a summary of the topic
    
    :param query: Topic to search for
    :type query: str
    :return: Wikipedias summary 
    :rtype: str
    '''
    try:
        wikipedia.set_lang("en") 
        
        summary = str(wikipedia.summary(query, sentences=10))
        page = wikipedia.page(query)
        return f"Title: {page.title}\nSummary: {summary}"
    
    except wikipedia.exceptions.DisambiguationError as e:
        options = e.options[:5]
        options_text = "\n".join(f"- {opt}" for opt in options)
        return f"Multiple results founded for: '{query}'. Specify an option:\n{options_text}"
    
    except wikipedia.exceptions.PageError:
        return f"There is no wikipedia page for '{query}'"
    
    except Exception as error:
        return f"An error occurred while searching Wikipedia: {error}"
    

wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())

@tool('wikidataSearch', description="Get the information of a page with a certain name on wikidata")
def wikidataSearch(query: str) -> str:
    '''
    Searches Wikidata and returns information on the topic
    
    :param query: Topic to search for
    :type query: str
    :return: Wikidatas info 
    :rtype: str
    '''
    try:
        return f"Information: {wikidata.run(query)}"
    
    except Exception as error:
        return f"An error occurred while searching Wikipedia: {error}"
    
