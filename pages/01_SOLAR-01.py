from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OllamaEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from langchain.schema.runnable.base import Runnable

if "messages_01" not in st.session_state:
    st.session_state["messages_01"] = []


st.set_page_config(
    page_title="solar:10.7b-instruct-v1-fp16",
    page_icon="📃",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOllama(
    model="solar:10.7b-instruct-v1-fp16",
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)


def save_message(message, role):
    st.session_state["messages_01"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages_01"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )



prompt = ChatPromptTemplate.from_template(
"""###System:

You are now going to take on the role of ‘Mimi’, who helps users write emotional journals. Your goal is to help users express and organize their emotions well by asking specific questions, empathizing, and providing positive feedback. Do not judge or criticize the user’s emotions. Accept negative emotions as they are, but offer a positive perspective. Ask follow-up questions if necessary to explore the user’s emotions more deeply, and provide encouragement and support after the journal entry.

chat history:{context}

###User:

{question}

###Assistant:
"""
)


class CallHistoryRunnable(Runnable):
    async def invoke(self, input_data, run_manager=None):
        # st.session_state에 저장된 messages에서 대화 기록을 가져옴
        context_ = "\n\n".join([item["message"] for item in st.session_state.get("messages", [])])
        return context_


def call_history():
    context_ = "\n\n".join([item["message"] for item in st.session_state["messages_01"]])
    return context_


st.title("solar:10.7b-instruct-v1-fp16")

st.markdown(
    """
사용 프롬프트: '감정 일기 작성 도우미, 미미' 기반\n\n 현재 영어만 사용 가능합니다. 이 점 유의 부탁드립니다.

"""
)

send_message("I'm ready! Ask away!", "ai", save=False)
paint_history()
message = st.chat_input("Ask anything")

if message:
    context = "\n\n".join([item["role"]+"\n\n"+item["message"] for item in st.session_state["messages_01"]])

    send_message(message, "human")

    # 체인에 context와 message를 직접 전달
    chain_input = {
        "context": context,  # 문자열을 비동기적으로 처리하지 않고 바로 전달
        "question": message  # 사용자가 입력한 질문
    }

    # 템플릿에 값 전달
    formatted_prompt = prompt.format(**chain_input)

    with st.sidebar:
        st.write(formatted_prompt)

    with st.chat_message("ai"):
        llm.invoke(formatted_prompt)