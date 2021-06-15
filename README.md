# Machine Learning Baby Monitor Sleep Schedule
### What is it?
Convolutional neural network trained to track our infant’s sleep schedule using our Google Nest camera.
This project is wrapped in a Flask web app to provide:
- Live video feed
- Live prediction from the neural network
- UI for classifying new training data
- Interactive plots for exploring the various models I've tried

### What's the status?
Web app is currently live on my local network. The most recent model appears robust enough to put into production.

### What's next?
I will have the model make periodic predictions and record the infant's sleep time in an SQLite database.
