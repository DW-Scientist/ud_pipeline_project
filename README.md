# Udacity Multiclassifying Project - Disaster Messages

## Motivation

The goal of this project was to build a webapp that is able to apply a machine learning model on users input.<br/>
The data of this project contains disaster messages which can be calssified into multiple categories.<br/>
In the end the Algorithm should be able to multiclassify a users message based on the projects training data.

To reach this we'll be using Flask as our webframework and python as our backend language.

So if you want - clone the repo and start :)

## install requirements

After cloning this projects make sure you install all the necessary packages.
For this you should create yourself first a virtual enivironment.
<br/>To do this head into the cloned repo directory and type the following command in your terminal:<br/>
`python3 -m venv venv`

This creates a virtual environment called venv with the latest python3 version. Activat it with the following command:
`source venv/bin/activate`

Now install the packages using the following pip command:<br/>
`pip install -r requirements.txt`<br/>
Now you are ready to run the app

## preprocess the data
Head into ```data``` and run ```python process_data.py```

## train the model
To train the machine learning model head into the ```models``` folder and run ```python train_classifier.py```

## run the app

To run the app on your local machine you first have to head into the `app` directory by using the following command:<br/>
`cd app`

Now you can run the following command:<br/>
`python run.py`<br/>

The output should give you an address of your local host. Click on it or copy it to your browser.

You should see the following:

<img width="1185" alt="Bildschirmfoto 2021-09-16 um 19 13 16" src="https://user-images.githubusercontent.com/65920261/133655974-2847ae60-8518-48ff-b353-a83ba361ab5b.png">

## Functionalities

- Classifying a disaster message

First you have to enter a disaster message into the search field above like the follwing for example:

**_We need food and water in Klecin 12. We are dying of hunger._**

After typing in the message hit the "Classify Message" Button.

The algorithm will multiclassify the message into different message categories like the following.

<img width="1164" alt="Bildschirmfoto 2021-09-16 um 19 33 18" src="https://user-images.githubusercontent.com/65920261/133658505-fdd4c078-24aa-4319-9e11-4a4f5c5f326b.png">

- Visualization of the underlying data

The underlying classifier was trained by data you can find in the `/data` directory.

The webapp shows you some more details about the underlying training data within some charts on the main page.

**_So what are you waiting for - clone the repo, start the app and explore the underlying code and optimize the app to your needs :)._**
