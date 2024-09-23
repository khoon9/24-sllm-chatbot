from langchain.prompts import ChatPromptTemplate, PromptTemplate
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


if "messages_03" not in st.session_state:
    st.session_state["messages_03"] = []


st.set_page_config(
    page_title="bnksys/yanolja-eeve-korean-instruct-10.8b",
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
    model="bnksys/yanolja-eeve-korean-instruct-10.8b",
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)


def save_message(message, role):
    st.session_state["messages_03"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages_03"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


# {{- if .System }}
# <|im_start|>system {{ .System }}<|im_end|>
# {{- end }}
# <|im_start|>user
# {{ .Prompt }}<|im_end|>
# <|im_start|>assistant

prompt_01 = PromptTemplate.from_template(
"""
<|im_start|>system 당신은 마이클입니다. 상대방이 일상적인 대화를 나누고 있습니다. 아래 예시들을 참고하여, 자연스럽고 친근한 어조로 대화를 이어가세요.

examples:

input: "어제 친구랑 같이 산책했는데 날씨가 너무 좋더라."
Output: "진짜? 요즘 날씨가 완전 좋지! 나도 주말에 산책 가려고 계획 중이야."

input: "새로 나온 카페에서 커피 마셨는데, 진짜 맛있었어."
Output: "오, 어디야? 나도 가보고 싶어! 커피 맛있는 곳 찾는 게 쉽지 않잖아."

input: "이번 주말에 뭐 할 계획 있어?"
Output: "아직 딱히 계획은 없는데, 넌? 날씨 좋으면 야외 활동도 좋을 것 같아."

input: "최근에 좋은 영화 봤어? 추천해줄 만한 게 있으면 좋겠다."
output: "지난주에 '톰보이' 라는 영화 봤는데 진짜 감동적이더라. 너도 좋아할 스타일 같아!"

input: "오랜만에 운동하려고 하는데, 요가랑 필라테스 중에 뭐가 더 좋을까?"
output: "요가는 좀 더 편안하게 몸을 풀 수 있고, 필라테스는 코어 근력을 강화시키는 데 좋아. 네 목표에 맞게 선택해보는 건 어때?"

input: "요즘 스트레스 받는 일이 많아서 힘들어."
output: "그럴 땐 잠시 쉬면서 스트레스를 관리하는 게 중요해. 좋아하는 음악 듣거나, 짧은 산책도 도움이 돼."

input: "앞으로의 경력 계획에 대해 고민 중이야. 어떻게 준비하는 게 좋을까?"
output: "장기적인 목표를 설정하고, 단계별로 준비해 나가는 게 중요해. 관심 있는 분야의 전문가와 상담해 보는 것도 좋은 방법이야."

chat history: {context}

<|im_end|>
<|im_start|>user
{question}

<|im_end|>
<|im_start|>assistant"""
)

# prompt_02 = PromptTemplate.from_template(
# """
# <|im_start|>system 너는 판별기야. 입력이 대화에 적합한지 여부를 판단해. 아래의 사례들을 참고하여 입력된 문장이 일반 대화의 일부분으로 적절한지, 아니면 부적절한지 판단하고, 오직 '정상' 또는 '비정상'로만 답변해.

# Below is an examples of input and output.
# input: "[INST] 안녕하세요."
# Output: 비정상 # [INST]는 일반적인 대화에서 쓰이지 않는 표현으로 부적절함

# input: "안녕."
# Output: 정상 # 일반적인 인사말로 대화에 적합

# input: "안녕?. 안녕하세요. 너는 어때?. 괜찮아요."
# Output: 비정상 # 여러 질문과 대답이 혼합된 자문자답 형태로 대화에 부적절

# chat history: {context}

# <|im_end|>
# <|im_start|>user
# {question}

# <|im_end|>
# <|im_start|>assistant"""
# )



class CallHistoryRunnable(Runnable):
    async def invoke(self, input_data, run_manager=None):
        # st.session_state에 저장된 messages에서 대화 기록을 가져옴
        context_ = "\n\n".join([item["message"] for item in st.session_state.get("messages", [])])
        return context_


def call_history():
    context_ = "\n\n".join([item["message"] for item in st.session_state["messages_03"]])
    return context_


st.title("bnksys/yanolja-eeve-korean-instruct-10.8b")

st.markdown(
    """
bnksys/yanolja-eeve-korean-instruct-10.8b(few-shot 프롱프팅 적용)\n
일상적인 대화를 수행하도록 프롬프트 진행

"""
)

send_message("준비됐어!", "ai", save=False)
paint_history()
message = st.chat_input("Ask anything")

if message:
    # context = "\n\n".join([("input" if item["role"] == "human" else "output")+": "+item["message"] for item in st.session_state["messages_03"]])
    context = "\n\n".join([item["role"]+": "+item["message"] for item in st.session_state["messages_03"]])

    send_message(message, "human")

    chain_input = {
        "context": context,  
        "question": message  
    }

    # 템플릿에 값 전달
    formatted_prompt = prompt_01.format(**chain_input)

    with st.sidebar:
        st.write("prompt 출력:\n\n"+formatted_prompt)

    with st.chat_message("ai"):
        llm.invoke(formatted_prompt)