import sys
sys.dont_write_bytecode = True #Avoiding pycache

from dotenv import load_dotenv 
import gc

from agents.reporterAgent import invokeReporter
from agents.curatorAgent import InvokeCurator
from agents.investigatorAgent import InvokeInvestigator
from agents.supervisorAgent import supervisorAgent

load_dotenv()
def main():
    print("-------- Initial research starts --------")
    research: str = InvokeInvestigator()
    parts: list[str] = research.split('SUBTOPIC:') #Separate the main research and each subtopic. 

    initalResearch: str = parts[0].strip() 
    parts.pop(0) #Save the main research in another variable and get rid of it in the list.

    subtopics: list[str]= [] #Remove the strays of descriptions we don't want
    for part in parts:
        subtopic = part.strip().split('\n')[0].strip() 
        if subtopic: 
            subtopics.append(subtopic) 
    
    if len(subtopics) != 0: 

        print('=' * 70)
        print("PROPOSED SUBTOPICS:")
        print('=' * 70)

        for i, subtopic in enumerate(subtopics, 1): 
            print(f"{i}. {subtopic}")

        print('=' * 70)
        election: str = input("What do you wanna do with the subtopics? (approve [default], reject, modify, add) ").strip()

        subtopics_formatted = "\n".join([f"{i}. {s}" for i, s in enumerate(subtopics, 1)]) 
        #Format the subtopics for easier understanding of the LLM
        
        if len(subtopics_formatted) != 0: 
            prompt = f"""Proposed subtopics:
                {subtopics_formatted}

                User decision: {election}

                Execute Step 1 and 2. Return ONLY the approved subtopics with the formatting rule.
                """

            approvedSubtopics: str = str(supervisorAgent.invoke(prompt).content)
            gc.collect() #Cleaning afther calling an OLlama agent 

            finalResearch = InvokeCurator(initalResearch, approvedSubtopics)
        
        else: #Alternative when there is no approved subtopic
            finalResearch = InvokeCurator(initalResearch, None)

    else: #Alternative when there is no proposed subtopic
        finalResearch = InvokeCurator(initalResearch, None)

    report = invokeReporter(finalResearch)

    print(report)

    return;

if __name__ == "__main__":
    main()
