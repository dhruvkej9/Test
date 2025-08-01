FROM python:3.10

WORKDIR /code

COPY backend/requirements.txt /code/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY backend/app.py /code/app.py

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]