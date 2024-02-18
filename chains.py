from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from utils import format_docs
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def complexity_eval_chain():
    llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0)
    prompt = PromptTemplate.from_template(
        """Evaluate the complexity of the user's question and decide the optimal number of hint levels required for guiding the learner from understanding the basic concept to complete mastery. 
        For instance, if a user asks 'How to implement a quicksort algorithm in Python?', you would respond with '5' indicating the number of hint levels.
        Given the question: {question}
        The ideal number of hint levels is
        """
    )

    complexity_eval = (
        {"question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    )
    return complexity_eval


def hints_generation_chain():
    db = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 1},
    )

    llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0)

    prompt = PromptTemplate(
        template="""
    Given the Python coding question: <question> {question} </question>
    and context <context> {context} </context>, craft {num_hints} interconnected hints that sequentially guide the learner from a basic understanding to a comprehensive solution. Each hint should build upon the last, gradually revealing more detail and depth:

    Level 1: Start with a basic tip that gently points the learner in the right direction, laying the foundation for subsequent hints.
    Level 2: Expand on the first hint by introducing a key concept or tool necessary for solving the problem, providing a clearer direction and deeper insight.
    Intermediate Levels: Each subsequent level should elaborate on the previous hints, offering more specific guidance, including implementation details or logic, and gradually leading the learner towards the solution.
    Final Level: Culminate with a comprehensive explanation that not only outlines the full solution but also integrates all previous hints into a cohesive understanding of the concept, including its application and the rationale behind it.
    Ensure continuity between hints, with each hint building on the previous ones to develop a full understanding of the solution. When using code in hints, it should be concise and incrementally expand on the code introduced in earlier hints, wrapped in a code block using triple backticks (```Python)
    {format_instructions}
""",
        input_variables=["question", "num_hints", "format_instructions"],
    )
    hints_generator = (
        {
            "question": itemgetter("question"),
            "context": itemgetter("question") | retriever | format_docs,
            "num_hints": itemgetter("num_hints"),
            "format_instructions": itemgetter("format_instructions"),
        }
        | prompt
        | llm
    )
    return hints_generator


def coverage_chain():
    llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0)
    prompt3 = PromptTemplate.from_template(
        """Given a Python coding question <question> {question} </question> and hints <hints> {hints} </hints>, estimate the coverage of the question by the hints. Output a coverage estimate as a decimal number between 0 (no coverage) and 1 (full coverage), without any additional text.
        """
    )

    coverage = prompt3 | llm | StrOutputParser()
    return coverage
