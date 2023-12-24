import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import os
import sys
import getopt
import shutil
# 0 - ART
# 1 - EOS
# 2 - LIN
# 3 - LYM
# 4 - MAC
# 5 - MON
# 6 - NEU
# 7 - NRBC
# 8 - PLA
# 9 - RBC

# Загрузить изображение по переданному пути.
def load_image(img_path):
		img = image.load_img(img_path, target_size=(249,249))
		img_tensor = image.img_to_array(img)
		img_tensor = np.expand_dims(img_tensor,axis=0)
		img_tensor /= 255.
		
		return img_tensor

def classify(model_path : str, input_path : str, output_path:str):
	# Загрузка модели
	model = tf.keras.models.load_model(model_path)
	model.compile()
		
	# Перечисление файлов (входных данных)
	inputdir = os.listdir(input_path)
	files = [f for f in inputdir if os.path.isfile(os.path.join(input_path, f))]
		
	# Если нету директорий для выходных данных - создаём их. 
	if not os.path.exists(output_path):
		os.mkdir(output_path)
		
	target_folders = ['ART','EOS','LIN','LYM','MAC','MON','NEU','NRBC','PLA','RBC']
	for target_folder in target_folders:
		target_directory_path = os.path.join(output_path, target_folder)
		if not os.path.exists(target_directory_path):
			os.mkdir(target_directory_path)
	
	for file in files:
		# Загрузка изображения
		file_path = os.path.join(input_path, file)
		img = load_image(file_path)
		
		# классификация
		pred = list(model.predict(img)[0])
		index_max = max(range(len(pred)), key=pred.__getitem__)
		
		# Копируем файл в папку назначения по результатам классификации.
		subfolder = target_folders[index_max]
		target_path = os.path.join(output_path, subfolder)
		shutil.copy2(file_path,target_path)


if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:],'hi:o:m:',['input','output', 'model'])
    model_path = './model.keras'
    input_path = './input'
    output_path = './output'
    for opt, arg in opts:
        if opt == '-h':
            print('classificator -i[--input] <INPUT_PATH> -o[--output] <OUTPUT_PATH> -m[--model] <MODEL_PATH>')
            sys.exit()
        elif opt in ('--input', '-i'):
            input_path = arg
        elif opt in ('--output', '-i'):
            output_path = arg
        elif opt in ('--model', '-m'):
            model_path = arg
    classify(model_path=model_path,
             input_path=input_path,
             output_path=output_path)
    print('Classification completed.')

			
			
			

