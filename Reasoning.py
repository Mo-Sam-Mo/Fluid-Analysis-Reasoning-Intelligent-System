from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from Secrets import RETRIVAL_DOC, PROMPT


GEMINI_API_KEY = str(os.getenv("GEMINI_API_KEY"))

class Reasoning_Model():
    
    def __init__(self):
        self.prompt = PromptTemplate(
            input_variables=["class", "sample", "reasoning"],
            template=PROMPT,
        )

        self.gemini = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0
        )

        self.chain = LLMChain(
            llm=self.gemini,
            prompt=self.prompt,
            verbose=False
        )

    def generate_response(self, cls, sample):
        return self.chain.run(cls=cls, sample=sample, reasoning=RETRIVAL_DOC[cls])
