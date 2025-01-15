import os
import math
from skimage import io
from skimage import color
from skimage.transform import resize
from skimage.feature import hog
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

num_of_histogram_bins = 9
size_of_each_histogram_bin = 180 / num_of_histogram_bins

def compute_gradients_magnitude_and_orientation(image):
    # Listas para armazenar as magnitudes e os ângulos dos gradientes de cada pixel da imagem
    image_gradients_magnitudes = []
    image_gradients_angles = []

    # Iterando por cada uma das 128 linhas da imagem
    for i in range(128):

        # Listas para armazenar as magnitudes e os ângulos dos gradientes de cada pixel por linha da imagem
        current_row_gradients_magnitudes = []
        current_row_gradients_angles = []

        # Iterando por cada uma das 64 colunas da imagem
        for j in range(64):

            # Calculando o gradiente de cada pixel na direção horizontal
            # Tratando casos especiais, onde estamos na primeira ou última coluna da imagem
            if j == 0:
              x_gradient = image[i][j+1] - 0
            elif j == len(image[0])-1:
              x_gradient = 0 - image[i][j-1]
            else:
              x_gradient = image[i][j+1] - image[i][j-1]

            # Calculando o gradiente de cada pixel na direção vertical
            # Tratando casos especiais, onde estamos na primeira ou última linha da imagem
            if i == 0:
              y_gradient = image[i+1][j] - 0
            elif i == len(image)-1:
              y_gradient = 0 - image[i-1][j]
            else:
              y_gradient = image[i+1][j] - image[i-1][j]

            x_gradient = np.abs(x_gradient)
            y_gradient = np.abs(y_gradient)

            # Calculando a magnitude do gradiente do pixel atual
            magnitude = math.sqrt(pow(x_gradient, 2) + pow(y_gradient, 2))

            # Calculando o ângulo do gradiente do pixel atual
            if x_gradient == 0:
                angle = math.degrees(0.0)
            else:
                angle = math.degrees(abs(math.atan(y_gradient / x_gradient)))

            current_row_gradients_magnitudes.append(round(magnitude, 9))
            current_row_gradients_angles.append(round(angle, 9))

        # Registrando os valores calculados acima para a linha atual
        image_gradients_magnitudes.append(current_row_gradients_magnitudes)
        image_gradients_angles.append(current_row_gradients_angles)

    return np.array(image_gradients_magnitudes), np.array(image_gradients_angles)

def compute_gradients_histograms(image_gradients_magnitudes, image_gradients_angles):
    # Lista para armazenar os histogramas de todos os blocos da imagem
    image_histograms = []

    # Iterando por todas as linhas com blocos de tamanho 8x8 pixels
    for i in range(0, 128, 8):
        # Lista para armazenar os histogramas do bloco atual
        current_row_block_histograms = []

        # Iterando por todas as colunas com blocos de tamanho 8x8 pixels
        for j in range(0, 64, 8):
            # Separando os valores das magnitudes e dos ângulos dos gradientes dos pixels do bloco atual
            current_block_gradients_magnitudes = [[image_gradients_magnitudes[x][y] for y in range(j, j+8)] for x in range(i, i+8)]
            current_block_gradients_angles = [[image_gradients_angles[x][y] for y in range(j, j+8)] for x in range(i, i+8)]

            # Criando um histograma para o bloco atual
            current_block_histogram = [0.0 for _ in range(num_of_histogram_bins)]

            # Iterando por todos os valores do bloco atual
            for x in range(len(current_block_gradients_magnitudes)):
                for y in range(len(current_block_gradients_magnitudes[0])):

                    # Calculando em qual bin estamos de acordo com o ângulo do gradiente
                    current_bin_index = math.floor((current_block_gradients_angles[x][y] / size_of_each_histogram_bin) - 0.5)

                    current_bin_center_value = round(size_of_each_histogram_bin * ((current_bin_index + 1) + 0.5))

                    # Calculando quanto adicionar no bin atual e quanto adicionar no bin seguinte
                    value_to_add_current_bin = round(current_block_gradients_magnitudes[x][y] * ((current_bin_center_value - current_block_gradients_angles[x][y] / size_of_each_histogram_bin)), 9)
                    value_to_add_to_next_bin = current_block_gradients_magnitudes[x][y] - value_to_add_current_bin

                    # Somando-se os valores nos dois bins alvo
                    current_block_histogram[current_bin_index] += value_to_add_current_bin
                    current_block_histogram[current_bin_index + 1] += value_to_add_to_next_bin

            current_block_histogram = [round(x, 9) for x in current_block_histogram]
            current_row_block_histograms.append(current_block_histogram)

        image_histograms.append(current_row_block_histograms)

    return np.array(image_histograms)

def compute_image_feature_vector(image_blocks_histograms):
  # Lista que irá guardar o vetor de características da imagem
  feature_vector = []

  # Iterando por todos os blocos de dimensão 2x2 da imagem na direção vertical
  for i in range(0, len(image_blocks_histograms) - 1):

      # Lista que irá guardar o vetor de características do bloco atual (e que irá compor o vetor final)
      current_row_blocks_feature_vector = []

      # Iterando por todos os blocos de dimensão 2x2 da imagem na direção horizontal
      for j in range(0, len(image_blocks_histograms[0]) - 1):

          # Selecionando os 4 histogramas do bloco atual
          current_block_histograms = [[image_blocks_histograms[x][y] for y in range(j, j+2)] for x in range(i, i+2)]

          # Transformando os 4 histogramas do bloco em um vetor unidimensional
          current_block_1d_feature_vector = []
          for histogram in current_block_histograms:
              for bin in histogram:
                  for value in bin:
                      current_block_1d_feature_vector.append(value)

          # Normalizando o vetor
          normalization_factor = round(math.sqrt(sum([pow(val, 2) for val in current_block_1d_feature_vector])), 9)
          current_block_1d_feature_vector = [round(val / (normalization_factor + 1e-6), 9) for val in current_block_1d_feature_vector]

          current_row_blocks_feature_vector.append(current_block_1d_feature_vector)

      feature_vector.append(current_row_blocks_feature_vector)

  return np.array(feature_vector)

def compute_hog_features(image):
    # Unindo todas as chamadas para computar o feature vector do método HOG
    gradients_magnitude, gradients_angle = compute_gradients_magnitude_and_orientation(image)
    histograms = compute_gradients_histograms(gradients_magnitude, gradients_angle)
    feature_vector = compute_image_feature_vector(histograms)

    return feature_vector

def split_images(images_folder_path, output_folder):
    regions = [
        (10, 37),   
        (37, 68),   
        (68, 96),  
        (98, 127),  
        (130, 155), 
        (153, 180)  
    ]

    files = sorted([f for f in os.listdir(images_folder_path) if f.endswith('.jpg')])
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Realizando a divisão no eixo x das imagens fornecidas em 6 regiões descritas pelos intervalos de 'regions' 
    # Cada imagem deve conter um caracter em sua quase totalidade
    for file in files:
        image_path = os.path.join(images_folder_path, file)

        image = io.imread(image_path)

        resized_images = []
        for idx, (start, end) in enumerate(regions):
            split_image = image[:, start:end]
            resized_split_image = resize(split_image, (128, 64), anti_aliasing=False)
            resized_images.append(resized_split_image)

            output_image_path = os.path.join(output_folder, f"{file[:-4]}_part_{idx}.jpg")
            io.imsave(output_image_path, (resized_split_image * 255).astype('uint8'))

def read_and_strip_chars_from_files(labels_folder_path):
    final_vector = []

    files = sorted([f for f in os.listdir(labels_folder_path) if f.endswith('.txt')])

    # Acessando a pasta contendo as labels e separando o CAPTCHA de cada '.txt' em 6 caracteres (mantendo a ordem)
    for file in files:
        file_path = os.path.join(labels_folder_path, file)
        with open(file_path, 'r') as f:
            first_six_chars = f.read(6)
            final_vector.extend(char for char in first_six_chars)

    return final_vector

def extract_features_from_folder(images_folder_path):
    all_features = []

    images = sorted([f for f in os.listdir(images_folder_path) if f.endswith('.jpg')])

    # Iterando por todas as imagens (já contendo as letras individuais) e obtendo o feature vector de cada uma delas
    for file in images:
        image_path = os.path.join(images_folder_path, file)
        image = io.imread(image_path)
        features = compute_hog_features(color.rgb2gray(image))
        all_features.append(features)

    return np.array(all_features)

def clean_folder(image_folder, text_folder):
    # Função que faz o pré-processamento dos dados, removendo pares de CAPTCHAS e labels contendo '?' ou um número de caracteres difrente de 6 (nas labels
    image_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_folder) if f.endswith('.jpg')}
    text_files = {os.path.splitext(f)[0]: f for f in os.listdir(text_folder) if f.endswith('.txt')}

    for text_name, text_file in text_files.items():
        text_path = os.path.join(text_folder, text_file)
        image_path = os.path.join(image_folder, image_files.get(text_name, ''))

        if os.path.isfile(text_path) and image_files.get(text_name):
            with open(text_path, 'r') as file:
                content = file.read().strip()

            if len(content) != 6 or '?' in content:
                os.remove(text_path)
                os.remove(image_path)
                print(f"Removendo {text_file} e a imagem {image_files[text_name]}")

def count_files_in_folder(folder_path):
    # Retorna o número de arquivos na pasta de parâmetro
    return len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

training_images_folder = './treinamento'
training_splitted_images_folder = './treinamento_final'

validation_images_folder = './validacao'
validation_splitted_images_folder = './validacao_final'

test_images_folder = './teste'
test_splitted_images_folder = './teste_final'

labels_folder = './labels10k'

clean_folder(training_images_folder, labels_folder)
clean_folder(validation_images_folder, labels_folder)
clean_folder(test_images_folder, labels_folder)

# As 3 linhas abaixo geram as letras individualmente (apenas rodar se você não já possuir os CAPTCHAS divididos APÓS a limpeza dos dados)
#training_images = split_images(training_images_folder, training_splitted_images_folder)
#validation_images = split_images(validation_images_folder, validation_splitted_images_folder)
#test_images = split_images(test_images_folder, test_splitted_images_folder)

labels = read_and_strip_chars_from_files(labels_folder)

i = count_files_in_folder(training_splitted_images_folder)
j = count_files_in_folder(validation_splitted_images_folder)
training_labels = labels[0:i]
validation_labels = labels[i:i+j]
test_labels = labels[i+j:]

# Extraindo os feature vectors para todas as 60 mil imagens
X_train = extract_features_from_folder(training_splitted_images_folder)
X_train = X_train.reshape(X_train.shape[0], -1)

X_val = extract_features_from_folder(validation_splitted_images_folder)
X_val = X_val.reshape(X_val.shape[0], -1)

X_test = extract_features_from_folder(test_splitted_images_folder)
X_test = X_test.reshape(X_test.shape[0], -1)

# Utilizando GridSearchCV para escolher melhor conjunto de hiperparâmetros com base no conjunto de validação
X_combined = np.concatenate((X_train, X_val), axis=0)
y_combined = np.concatenate((training_labels, validation_labels), axis=0)

split_index = np.concatenate(([-1] * len(X_train), [0] * len(X_val)))
predefined_split = PredefinedSplit(test_fold=split_index)

param_grid = {'C': [0.1, 1, 5], 'kernel': ['linear']}

# Normalizando os dados empregados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

grid_search = GridSearchCV(SVC(), param_grid, cv=predefined_split)
grid_search.fit(np.concatenate((X_train_scaled, X_val_scaled), axis=0), y_combined.ravel())

best_model = grid_search.best_estimator_

# Salvando o melhor modelo obtido
joblib.dump(best_model, 'best_svc_model.pkl')