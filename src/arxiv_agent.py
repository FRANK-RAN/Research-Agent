from llama_index.tools.arxiv.base import ArxivToolSpec
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI

arxiv_tool = ArxivToolSpec(max_results=20)

llm = OpenAI(model="o3-mini")
# Set up the OpenAI agent with the Arxiv tool
agent = OpenAIAgent.from_tools(
    arxiv_tool.to_tool_list(),
    llm=llm,
    verbose=True,
    system_prompt="Always use tools to answer questions. Do not rely on your own knowledge. Your answer should have a reference list to specify the arxiv paper you refer to, with paper title and arxiv id. If you don't know the answer, just say you don't know.",
)



def print_equal_lab():
    letters = {
        'E': [
            "#######",
            "#      ",
            "#      ",
            "#####  ",
            "#      ",
            "#######"
        ],
        'Q': [
            " ##### ",
            "#     #",
            "#     #",
            "#     #",
            "#  #  #",
            " ######"
        ],
        'U': [
            "#     #",
            "#     #",
            "#     #",
            "#     #",
            "#     #",
            " ##### "
        ],
        'A': [
            "   #   ",
            "  # #  ",
            " #   # ",
            "#######",
            "#     #",
            "#     #"
        ],
        'L': [
            "#      ",
            "#      ",
            "#      ",
            "#      ",
            "#      ",
            "#######"
        ],
        'B': [
            "###### ",
            "#     #",
            "#     #",
            "###### ",
            "#     #",
            "###### "
        ],
        ' ': [
            "       ",
            "       ",
            "       ",
            "       ",
            "       ",
            "       "
        ]
    }

    text = "EQUAL LAB"
    # Each letter is 6 rows high.
    for row in range(6):
        # Join each letter's row with 4 spaces in between.
        line = "    ".join(letters[ch][row] for ch in text)
        print(line)

print_equal_lab()

print("Hi, I am a chatbot built using Retrieval Augmented Generation (RAG) connected with arXiv with scientific textbooks and papers.")
print("I am powered by GPT-o3-mini for question answering, and I use research papers as references to provide evidence-backed answers.")
print("Please ask your question!\n")

# Prompt the user for a question
question = input("Enter your question: ")

response = agent.chat(question)

print("\033[1;31m\nRAG with arXiv Answer:\n\033[0m")
print(f"\033[31m{response}\033[0m")



