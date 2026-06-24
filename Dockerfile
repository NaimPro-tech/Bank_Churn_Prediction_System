#use official image of python
FROM python:3.10-slim
#install python into container

#set working directory into container
WORKDIR /app
#in that container create a folder named app

#copy requirement file into container
COPY requirements.txt .
#copy requirements file into this container

#install libraries into contaier
RUN pip install --no-cache-dir -r requirements.txt
#now lets install all the dependencies from that file into container


#copy al file from local project to container
COPY . .
#this cmd will copy all the files from local machine to docker container


#command to start API server while running docker
 CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
 #this will cmd to docker that when anyone start this container run the fastapi server into backend.