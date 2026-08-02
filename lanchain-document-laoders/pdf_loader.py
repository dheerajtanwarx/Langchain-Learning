from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('sample.pdf') #Yahan hum PyPDFLoader ka ek object (instance) bana rahe hain, aur usse batate hain ki kaunsi file load karni hai — yani 'sample.pdf'. Is step me file abhi tak actually read nahi hui hai, bas loader ko file ka path bata diya gaya hai.

docs = loader.load()#Ye asli kaam karne wali line hai. .load() method call hone par:
#PDF file open hoti hai
#Har page ka text nikal (extract) kiya jata hai
#Har page ek Document object banta hai (jisme page_content aur metadata hota hai)
#Ye sab Documents ek list me docs variable me store ho jate hain
#Yani agar PDF me 3 pages hain, to docs ek list hogi jisme 3 Document objects honge.

print(len(docs)) #Ye docs list ki length print karega — matlab PDF me total kitne pages hain, wahi number aayega (jaise 3).

print(docs[0].page_content)#docs[0] list ka pehla element hai (Python me indexing 0 se start hoti hai), yani PDF ka pehla page. .page_content uss page ka actual text hota hai jo PDF se extract kiya gaya. Ye line pehle page ka text print karegi.

print(docs[1].metadata)#docs[1] doosra element hai, yani PDF ka doosra page. .metadata us page ki extra jaankari deta hai — jaise file ka source path (sample.pdf) aur page number (page: 1). Text ke bajaye ye sirf information deta hai, jaisे: {'source': 'sample.pdf', 'page': 1}.