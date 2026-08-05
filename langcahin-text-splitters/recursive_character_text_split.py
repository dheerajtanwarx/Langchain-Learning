# RecursiveCharacterTextSplitter - Chhota Summary
# Ye bhi text ko chhote chunks me todta hai (jaise CharacterTextSplitter), lekin smart tareeke se — words aur sentences ko beech me se tootne se bachata hai.
# Kaise kaam karta hai?
# Ye ek priority list of separators use karta hai (default):
# Matlab ye pehle try karega:
# Paragraph pe todna (\n\n)
# Agar chunk phir bhi bada hai → line pe todna (\n)
# Agar phir bhi bada hai → space pe todna (words ke beech)
# Sabse last me → character-by-character ("")
# Yani ye pehle "natural" jagah pe todne ki koshish karta hai, aur sirf tab tak chhota karta jaata hai jab tak chunk_size ke andar na aa jaaye.


from langchain_text_splitters import RecursiveCharacterTextSplitter
text = """SECTION 1: THE BIRTH OF ARTIFICIAL INTELLIGENCE (1940s–1950s)
The formal foundations of Artificial Intelligence (AI) were established in the mid-20th century. In 1950, Alan Turing published his seminal paper "Computing Machinery and Intelligence," where he introduced the "Turing Test" as a benchmark for machine intelligence. The actual term "Artificial Intelligence" was coined a few years later in 1956 during the Dartmouth Summer Research Project on Artificial Intelligence. Organized by John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon, this historic workshop brought together scientists to discuss the potential of creating thinking machines. Early AI research focused heavily on symbolic logic, mathematical problem solving, and basic game-playing programs.

SECTION 2: THE FIRST AI WINTER AND REBORN EXPERT SYSTEMS (1970s–1980s)
Initial optimism ran high, with researchers predicting that human-level intelligence would be achieved within a generation. However, these projections proved overly ambitious. By the mid-1970s, funding dried up due to the severe computation limits of hardware and the inability of algorithms to handle combinatoric complexity, marking the first "AI Winter." AI experienced a dramatic resurgence in the 1980s with the commercial success of "Expert Systems." These specialized programs used explicit rules derived from human specialists to solve complex domain-specific tasks, such as corporate decision-making or medical diagnosis. Unfortunately, because these systems were expensive to update and highly fragile when encountering edge cases, they triggered a second AI Winter.

SECTION 3: THE EMERGENCE OF MACHINE LEARNING AND DATA (1990s–2000s)
In the late 1990s, the AI paradigm shifted away from rigid knowledge engineering toward probabilistic systems. Instead of hand-writing strict rules, scientists began feeding algorithms vast pools of data to let them learn patterns independently. In 1997, IBM's Deep Blue defeated World Chess Champion Garry Kasparov, proving that rule-based processing combined with massive computing power could outperform human intellect in specific constraints. During the 2000s, the explosion of the internet and web scraping provided researchers with unprecedented volumes of unstructured text, images, and user data, laying the foundation for modern data-driven architectures.

SECTION 4: THE DEEP LEARNING REVOLUTION AND LARGE LANGUAGE MODELS (2010s–Present)
The current golden era of AI began around 2012, fueled by the availability of Graphics Processing Units (GPUs) and massive datasets like ImageNet. Multi-layered Artificial Neural Networks, rebranded as Deep Learning, successfully solved complex computer vision, speech recognition, and translation tasks. In 2017, researchers introduced the Transformer architecture, which relied on self-attention mechanisms to process sequential data simultaneously. This structural breakthrough directly enabled the creation of modern Large Language Models (LLMs). These models generate remarkably human-like text and reason across disparate contexts, making them ideal engines for semantic search applications and framework orchestrators like LangChain."""


splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 0,

)

chunks = splitter.split_text(text)


print(len(chunks))
print(chunks)