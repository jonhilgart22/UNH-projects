## This project creates daily predictions of BART capacity for each station.

##### What if you could know how full the next BART train would be before it arrived?
  - The overarching idea was to combine historical weather and BART ridership information, alongside live BART train arrival, BART train size, and live weather information to generate near real-time capacity predictions. This project ended up with a daily prediction for each station, but future versions can decrease this time latency.

- **For a project overview**, start with the notebook Daily_Bart_Ridership_Predictions.ipynb. This gives an overview of the project including the architecture.

- The airflow scripts folder contains the airflow scripts to automate the batch jobs for data ingestion, data normalization, and bart ridership predictions.

- The spark submit scripts conatins the code used to normalize the bart and weather data as well as the MLlib model used to create the daily rider predictions.

- The data ingestion folder contains the code to
  - 1) Pull live bart arrival data every ten minutes
  - 2) Pull live weather information for San Francisco
  - 3) Push data to Mongodb

- The system dependencies highlights the external libraries used in this project.

- To look at the final output, visit <a href="http://bart-capacity-predictions.com.s3-website-us-east-1.amazonaws.com/">Bart Predictions</a>


