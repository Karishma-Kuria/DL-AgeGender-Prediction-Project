# Heroku App Deployment Notes

This folder contains all the dependent files needed for deployment in Heroku.

1. requirements.txt: contains all the dependent libraries to be installed during deployment.
2. setup.sh: this is a shell file which contains shell commands. In this file we will create a streamlit folder with credentials.toml and a config.toml file
3. Procfile: this files specifies the commands once you run the app in Heroku. It first run the setup file and then call streamlit run to run the application.
