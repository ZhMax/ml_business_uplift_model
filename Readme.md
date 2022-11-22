The project is devoted to development an uplift model of X5 retail dataset.
https://www.kaggle.com/competitions/x5-uplift-valid/overview

In the project feature engeneering is performed such
that some of initially existing features are transformed and   
new features are constructed.
Then a solo model based on the skilift and catboost library is developed.


Jyputer notebook uplift_model_constructing.ipynb contains code
for feature engineering and learning of uplift models.
As the solo model shows the best prediction efficency,
it is saved and used in the Flask server.

Text file cmd_commands.txt contains console commands
which have to be done virtual environment and docker container

Python file flask_server.py contains api implemented in Flask 
which allows processing user data and provide uplift estimation.

Jyputer notebook prediction_using_api.ipynb contains code
for sending data to Flask server and obtaining predictions.
