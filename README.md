# Blast Furnace Temperature Prediction Model

This project was made during the training programme under the Vizag Steel Plant under the IT & ERP Department.

This project's goal is to help and detect the average skin temperature of the blast furnace for the next 1, 2 , 3 and 4 hours. The decision factor for the output parameter is due to the time lag that happens between the input parameters and the effects of it on the blast furnace. Average skin temperature can help in deciding factor for minimizing the operational cost, reduce fuel consumption, and optimize the overall efficiency of the blast furnace and also improve the productivity of the blast furnace. The project is made in python and uses Scikit and Tensorflow.

The input data is stored in BF_data.csv and has about 25 columns of input parameters and last 4 columns are the output parameters. The project looks into various algorithms to find out the most optimial model for the temperature detection. Some of the algorithms used to find the optimal includes:
- Multiple Linear Regression
- Random Forest Regression
- Deep Neural Network

Each algorithm was also run with different parameter to find the most optimal version of that algorithm. Those models are then saved and to be deployed currently in localhost in a interactive webapp which can help to run and predict the output parameters without going through the code. This front-end interactive webapp is built using Streamlit.

In order to run and deploy the webapp, first install streamlit package from pip and go in the root of this folder and run:
```
streamlit run streamlit_app.py
```

## Note
. Pkl files can't be added to Github due to file size limit. Run the python notebook in order to save the model into a pkl file.