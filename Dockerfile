FROM python:3.11.1

WORKDIR /pythondir

EXPOSE 8501

COPY . /pythondir

RUN pip install -r requirements.txt

CMD streamlit run heart_disease_server.py