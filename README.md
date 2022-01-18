#  **Mini-Siri**
## Prerequisites
In order to run this program, you need to install Python on your computer. You can do this by following the instructions below:

https://www.python.org/downloads/

Note: This script was tested using Python version 3.8.10

After installing Python, you need to create an instance of a Python package manager like Anaconda or Python virtual environment.
Once this package manager is activated, execute the following command from the project directory.
```
pip install -r requirements.txt
```
This will install the necessary dependencies for the project.
## Getting Started

To run the Python script, execute the following command while in the project directory:
```
$ python3 main.py
```
The Python script `main.py` finetunes a pretrained BERT transformer for classifying the intent of digital voice assistant commands into one of `AddToPlaylist`, `BookRestaurant`, `GetWeather`, `PlayMusic`, `RateBook`, `SearchCreativeWork`, and `SearchScreeningEvent`. The script also performs named entity recognition on the command to determine the nouns that are relevant to the classified intent.

After finetuning, the user can input example voice assistant commands to test the finetuned model, as shown in the acreenshot below:

![image](https://user-images.githubusercontent.com/63531728/149874223-b78cf24f-50a7-4869-800c-4ae8c267404a.png)


## Credits

The following dataset was used to to train the NLP model:
https://github.com/sz128/slot_filling_and_intent_detection_of_SLU/tree/master/data/snips
