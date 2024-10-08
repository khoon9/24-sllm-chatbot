# 24-sllm-chatbot

- 향후 사용 가능성이 있는 모든 의존성 설정을 포함하여 기재하였음

0. python: 3.11

1. cuda 11.8 기준 의존성

```cmd
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 transformers accelerate langchain=0.0 streamlit ipykernel -c pytorch -c nvidia -y
```

2. 실행

```cmd
streamlit run Home.py
```

3. 환경변수 설정("./.streamlit/secrets.toml"와 "./.env")

```
OPENAI_API_KEY="your key"

LANGCHAIN_TRACING_V2 = True
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = "your key"

HUGGINGFACEHUB_API_TOKEN = "your token"
UPSTAGE_API_KEY = "your key"
```
