import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader



import urllib.parse

def clean_youtube_url(url):
    parsed = urllib.parse.urlparse(url)
    
    if "youtu.be" in parsed.netloc:
        video_id = parsed.path.lstrip('/')
        return f"https://www.youtube.com/watch?v={video_id}"

    query = urllib.parse.parse_qs(parsed.query)
    video_id = query.get("v", [None])[0]
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    
    return url


#Streamlit App
st.set_page_config(page_title="Langchain: Summarize Text from YT or website")
st.title("Langchain: Summarize Text From YT or Website")

with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", type = "password")

generic_url = st.text_input("URL", label_visibility="collapsed")


llm = ChatGroq(model = 'Llama3-8b-8192', groq_api_key = groq_api_key)


prompt_template = '''
Provide summary of the following content:
Content : {text}
'''

prompt = PromptTemplate(
    template = prompt_template,
    input_variables = ["text"]
)


if st.button("Summarize the Content from YT or website"):
    #Validate all the inputs
    if not groq_api_key.strip():
        st.warning("Please enter your Groq API Key to continue.")
    elif not generic_url.strip():
        st.warning("Please enter a valid YouTube or website URL.")

    else:
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in generic_url:
                    clean_url = clean_youtube_url(generic_url)
                    loader = YoutubeLoader(clean_url, add_video_info=True)
                else:
                    loader  =UnstructuredURLLoader(urls= [generic_url], ssl_verify = False)

                docs = loader.load()

                chain = load_summarize_chain(
                    llm,
                    chain_type = 'stuff',
                    prompt = prompt
                )

                output = chain.run(docs)

                st.success(output)

        except Exception as e:
            st.markdown(f"Exception: {e}")
