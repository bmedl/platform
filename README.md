# BME Deep Learning Trading Platform

[![Build Status](https://cicd.bmedl.soothingblue.space/api/badges/bmedl/platform/status.svg)](https://cicd.bmedl.soothingblue.space/bmedl/platform)

## Overview

This repo is the home of a Homework Project for the [Deep Learning course of BME](http://smartlab.tmit.bme.hu/oktatas-deep-learning-nagy-hazi). 

The goal is to build intelligent software based on neural networks that will predict the fluctuations of stock prices.

### The stack

**All the related code is in `backend/stocks/neuralnet`**

The heart of the project is written in Python. The application will constantly monitor the stocks via various public REST APIs. At the end of each day, a neural network is trained for each stock with a subset of the most recent data.
The most important library is [Keras](https://keras.io/), which is used for training and then running predictions using the neural networks.

The application also exposes a REST API of its own via the popular [Django](https://www.djangoproject.com/) library, it is available [here](https://api.bmedl.soothingblue.space).

Finally a [VueJS](https://vuejs.org/) UI is available that consumes the Django API and shows various statistics [here](https://app.bmedl.soothingblue.space).

### Hosting, Continuous Integration and Deployment

The project is hosted on a single [Google Compute](https://cloud.google.com/compute/) n1-standard-4 instance with a NVIDIA TESLA P4 GPU. The data for training is stored in a Google Cloud SQL database.

The deployments are made easier with [Docker](https://docker.com), while [Drone](https://drone.io) is used for continuous integration and deployment.

![Drone Pipelines](/assets/drone.png?raw=true "Drone Pipeline")

## Development

### Data - exploratory study
It's available in first_milestone.ipynb

### Django Backend

TODO - no time left.

### VueJS SPA

TODO - no time left.


## License

The project is licensed under GNU GPLv3.
