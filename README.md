
# go

install requirements:

    python -m pip install -r requirements.txt

run the program and view help:

    python train.py -h

load the trained and committed model. it also shows some predictions using this model:

    python train.py load model_parameters/r18_lo3.pth

run the webserver:

    python -m flask run

then navigate to http://localhost:5000/ on a web browser. you should
see a barebones website.
