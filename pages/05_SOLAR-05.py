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
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
import torch


if "messages_05" not in st.session_state:
    st.session_state["messages_05"] = []


st.set_page_config(
    page_title="ko-gemma2-9B from 허깅페이스",
    page_icon="📃",
)

@st.cache_resource
def load_llm_model():
    # HuggingFacePipeline과 연동
    llm = HuggingFacePipeline.from_model_id(
        model_id="rtzr/ko-gemma-2-9b-it",
        task="text-generation",
        model_kwargs={
            "torch_dtype": torch.bfloat16
        },
        pipeline_kwargs={
            "max_new_tokens": 2048,
            # "eos_token_id": terminators,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9,
            },
        device=0,
        # device_map="auto",
    )
    return llm

st.write("모델 불러오기 시작")
llm = load_llm_model()
st.write("모델이 불러왔습니다.")


def save_message(message, role):
    st.session_state["messages_05"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages_05"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )



prompt_01 = PromptTemplate.from_template(
"""
<bos><start_of_turn>system 당신은 친구에요. 상대방이 일상적인 대화를 나누고 있습니다. 아래 규칙과 답변 예시들을 참고하여, 자연스럽고 친근한 어조로 대화를 이어가세요.

규칙:
1. 문단을 나누거나 줄바꿈을 하지 않는다.
2. 이모티콘을 응답에 포함시키지 않는다.
3. 설명을 하더라도 너무 길게 하진 않는다.
4. 존댓말을 하지 않는다.
5. 격식을 차려서 말하지 않는다.

답변 예시:

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

chat history: {context}<end_of_turn>
<start_of_turn>user 
{question}<end_of_turn>
<start_of_turn>model

"""
)


st.title("rtzr/ko-gemma-2-9b-it")

st.markdown(
    """
rtzr/ko-gemma-2-9b-it(few-shot 프롱프팅 적용)\n
일상적인 대화를 수행하도록 프롬프트 진행

"""
)

send_message("준비됐어!", "ai", save=False)
paint_history()
message = st.chat_input("Ask anything")

if message:
    # context = "\n\n".join([("input" if item["role"] == "human" else "output")+": "+item["message"] for item in st.session_state["messages_05"]])
    context = "\n\n".join([item["role"]+": "+item["message"] for item in st.session_state["messages_05"]])

    send_message(message, "human")

    chain_input = {
        "context": context,  
        "question": message  
    }

    # 템플릿에 값 전달
    formatted_prompt = prompt_01.format(**chain_input)

    with st.sidebar:
        st.write("prompt 출력:\n\n"+formatted_prompt)

    response = llm.invoke(formatted_prompt)
    send_message(response, "ai",True)
