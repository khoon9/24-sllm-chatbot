# 24-sllm-chatbot

0. python: 3.11
1. LangChain 의존성

```cmd
pip install -r requirements.txt
```

2. cuda 11.8 기준 pytorch 의존성

```cmd
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. 실행

```cmd
streamlit run Home.py
```

4. 환경변수 설정("./.streamlit/secrets.toml"와 "./.env")

```
OPENAI_API_KEY="your key"

LANGCHAIN_TRACING_V2 = True
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = "your key"

HUGGINGFACEHUB_API_TOKEN = "your token"
UPSTAGE_API_KEY = "your key"
```
