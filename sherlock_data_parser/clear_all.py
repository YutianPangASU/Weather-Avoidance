import os, shutil

folder = '/mnt/data/WeatherCNN/sherlock/flight_plan_plot'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)  # this line will clear the directory as well
    except Exception as e:
        print(e)

folder = '/mnt/data/WeatherCNN/sherlock/flight_plan_coords'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

folder = '/mnt/data/WeatherCNN/sherlock/traj_csv'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

folder = '/mnt/data/WeatherCNN/sherlock/traj_plot'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

folder = '/mnt/data/WeatherCNN/sherlock/cache'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

folder = '/mnt/data/WeatherCNN/sherlock/EchoTopPic'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

folder = '/mnt/data/WeatherCNN/sherlock/x_train'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

folder = '/mnt/data/WeatherCNN/sherlock/x_train_npy'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

# remove y_train.csv
if os.path.exists("y_train.csv"):
  os.remove("y_train.csv")
else:
  print("The file y_train.csv does not exist")


# remove start_and_end.csv
if os.path.exists("start_and_end.csv"):
  os.remove("start_and_end.csv")
else:
  print("The file start_and_end.csv does not exist")


# remove call_sign.csv
# if os.path.exists("call_sign.csv"):
#     os.remove("call_sign.txt")
# else:
#     print("The file does not exist")