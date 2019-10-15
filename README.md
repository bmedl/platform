# BME Deep Learning Trading Platform

[![Build Status](https://cicd.bmedl.soothingblue.space/api/badges/bmedl/platform/status.svg)](https://cicd.bmedl.soothingblue.space/bmedl/platform)

## Overview

This repo is the home of a Homework Project for the [Deep Learning course of BME](http://smartlab.tmit.bme.hu/oktatas-deep-learning-nagy-hazi). 

The goal is to build intelligent software based on neural networks that will predict the fluctuations of stock prices.

### The stack

The heart of the project is written in Python. The application will constantly monitor the stocks via various public REST APIs. At the end of each day, a neural network is trained for each stock with a subset of the most recent data.
The most important library is [Keras](https://keras.io/), which is used for training and then running predictions using the neural networks, while [BlazingSQL](https://blazingsql.com/) ensures that that the data is always there when it is needed.

The application also exposes a REST API of its own via the popular [Django](https://www.djangoproject.com/) library, it is available [here](https://api.bmedl.soothingblue.space).

Finally an [Angular](https://angular.io/) UI is available that consumes the Django API and shows various statistics [here](https://app.bmedl.soothingblue.space).

### Hosting, Continuous Integration and Deployment

The project is hosted on a single [Google Compute](https://cloud.google.com/compute/) n1-standard-4 instance with 4 NVIDIA Tesla K80 GPUs. The data for training is stored in a [Google Cloud Storage](https://cloud.google.com/storage/) bucket of CSV files.

The deployments are made easier with [Docker](https://docker.com), while [Drone](https://drone.io) is used for continuous integration and deployment.

![Drone Pipelines](/assets/drone.png?raw=true "Drone Pipeline")

## Development

### Data - exploratory study
https://colab.research.google.com/drive/1DIjR-VT-jJJTLXeQp6Gvd4kVu9Lr1F3J#scrollTo=zdjpqoSSSaiZ

### Django Backend

#### Quick start

- Install [Pipenv](https://docs.pipenv.org/en/latest/install/#installing-pipenv)
- Run `pipenv sync` and then `pipenv run python3 app/manage.py runserver` in the backend directory.

### Angular SPA

#### Quick start

- Start the backend first
- Install [Angular CLI](https://cli.angular.io/)
- Run `npm install` and then `ng serve` in the frontend directory.

## License

The project is licensed under GNU GPLv3.
