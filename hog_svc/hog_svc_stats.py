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
import matplotlib.pyplot as plt
import joblib
from PIL import Image

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

def preprocess_and_extract_features(image_path):
    # Fazendo o pré-processamento de cada letra e gerando seu feature vector
    image = io.imread(image_path)
    grayscale_image = color.rgb2gray(image)
    resized_image = resize(grayscale_image, (128, 64), anti_aliasing=False)
    feature_vector = compute_hog_features(resized_image)

    return feature_vector.reshape(1, -1)

def get_captchas_paths(folder_path):
    captchas_paths = []

    captcha_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])

    # Carregando o path de cada CAPTCHA completo
    for filename in captcha_files:
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            captchas_paths.append(file_path)

    return captchas_paths

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

# Carregando o melhor modelo encontrado
svc_model = joblib.load('best_model.pkl')

# Carregando as imagens de teste e as labels ground-truth
training_images_folder = './treinamento'
training_splitted_images_folder = './treinamento_final'

validation_images_folder = './validacao'
validation_splitted_images_folder = './validacao_final'

test_images_folder = './teste'
test_splitted_images_folder = './teste_final'

labels_folder = './labels10k'

labels_array = read_and_strip_chars_from_files(labels_folder)

# Os labels de teste são os que restam após a remoção dos labels de treino de validação
test_labels_array = labels_array[count_files_in_folder(training_splitted_images_folder) + count_files_in_folder(validation_splitted_images_folder):]

captcha_size = 6
correct_count_per_captcha = []
incorrect_count_per_captcha = []

captchas = get_captchas_paths(test_images_folder)

image_files = sorted([f for f in os.listdir(test_splitted_images_folder) if f.endswith('.jpg')])

num_captchas = len(test_labels_array) // captcha_size

# Lista que irá guardar quantos CAPTCHAS terão ao menos 1, 2, 3... de seus caracteres deduzidos corretamente
correct_bins = [0, 0, 0, 0, 0, 0]

output_folder = './results'
os.makedirs(output_folder, exist_ok=True)

# Fazendo as predições no conjunto de testes com base no modelo treinado
for i in range(num_captchas):
    true_label = test_labels_array[i * captcha_size:(i + 1) * captcha_size]
    predicted_label = []

    for j in range(captcha_size):
        image_index = i * captcha_size + j
        image_path = os.path.join(test_splitted_images_folder, image_files[image_index])
        features = preprocess_and_extract_features(image_path)
        predicted_char = svc_model.predict(features)[0]
        predicted_label.append(predicted_char)

    predicted_label_str = ''.join(predicted_label)
    
    # Registrando a taxa de acerto da predição gerada pelo classificador para o CAPTCHA atual ground-truth
    correct_count = sum(1 for t, p in zip(true_label, predicted_label) if t == p)

    if correct_count >= 1:
        correct_bins[0] += 1
    if correct_count >= 2:
        correct_bins[1] += 1
    if correct_count >= 3:
        correct_bins[2] += 1
    if correct_count >= 4:
        correct_bins[3] += 1
    if correct_count >= 5:
        correct_bins[4] += 1
    if correct_count == 6:
        correct_bins[5] += 1

    # Gerando a imagem contendo o CAPTCHA atual ground-truth e a string gerada pelo classificador
    image = Image.open(captchas[i])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off') 
    ax.imshow(image)

    fig.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)

    plt.figtext(0.5, 0.18, predicted_label_str, ha="center", va="top", fontsize=35)
    
    output_path = os.path.join(output_folder, f'resultado_captcha_{i:04d}.png')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

# Montando um gráfico de barras com os resultados encontrados
correct_labels = ['1', '2', '3', '4', '5', '6']
correct_percentages = [(count / num_captchas) * 100 for count in correct_bins]

plt.figure(figsize=(8, 6))
plt.bar(correct_labels, correct_percentages, color='deepskyblue')
plt.xlabel('Número mínimo de caracteres corretos')
plt.ylabel('Taxa de reconhecimento')
plt.title('HOG + SVC\nTaxa de Reconhecimento por Número de Caracteres Corretos')

# Salvando o gráfico gerado com o desempenho do modelo nos dados de teste
output_image_path = './results/desempenho_geral.png'
plt.savefig(output_image_path)

