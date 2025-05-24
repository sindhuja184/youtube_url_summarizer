import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain 
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader


st.set_page_config(
    page_title= "Langchain: ummarize Text from YT or Webite"
)
st.title("Summarize content from youtube or website")
st.subheader("Summarize URL")



#Get the groq api key
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key",type = "password", value = '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')

generic_url = st.text_input("URL", label_visibility= "collapsed")

llm = ChatGroq(model = "llama3-8b-8192", groq_api_key = groq_api_key)

prompt_template = '''
Provide a ummary of the following content in 300 words

Context {text}
'''

prompt = PromptTemplate(
    template= prompt_template, input_variables=['text']
)


if st.button("Summarize"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")

    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")

    else :
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info = True)
                else :
                    loader = UnstructuredURLLoader(urls = [generic_url], ssl_verify = False)


                docs = loader.load()

                chain = load_summarize_chain(llm, chain_type = "stuff", prompt = prompt)

                output_summary = chain.run(docs)

                st.success(output_summary)
    
        except Exception as e:
            import traceback
            st.error(f"Exception: {e}")
            st.text(traceback.format_exc())
