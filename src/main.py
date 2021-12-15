import os

import load_data
import analyze_data
import model_training

def main():
  data = load_data.Data()

  os.system("clear")
  print("#################\n## VOICEDRIVEN ##\n#################\n")
  print("Type 'help' to print commands")
  command = ""
  while command != "exit": 
    command = input(">")
    if command == "help":
      print("#clear: clears the console")
      print("#exit: exits the programm")
      print("#init : initializes Datasets")
      print("#size : prints size of Datasets")
      print("#analyze : shows simple Analysis of Datasets")
      print("#build : build and train the model")
      print("#confusion_matrix : shows a confusion matrix")
    elif command == "clear":
      os.system("clear")
      print("#################\n## VOICEDRIVEN ##\n#################\n")
      print("Type 'help' to print commands")
    elif command == "init":
      data.getData()
    elif command == "size":
      data.size()
    elif command == "analyze":
      graphics = analyze_data.analyze_data(data.train_files, data.commands)
    elif command == "confusion_matrix":
      model = model_training.Model(data, graphics)
      model.confusion_matrix()
    elif command == "build":
      model = model_training.Model(data, graphics)
      model.build_train_model()
    else:
      print("Command not recognized! Try typing 'help' to show a command list")

if __name__ == '__main__':
  main()