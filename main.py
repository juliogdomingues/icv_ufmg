import numpy as np
import cv2 as cv
import os
import pickle
from sklearn.svm import LinearSVC
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from datetime import datetime

# Diretórios
TRAIN_DIR = "dados/CAPTCHA-10k/treinamento"
VALIDATION_DIR = "dados/CAPTCHA-10k/validacao" 
TEST_DIR = "dados/CAPTCHA-10k/teste"
LABELS_DIR = "dados/CAPTCHA-10k/labels10k/"

def pad_image(img, target_width):
    """Adiciona padding para padronizar a largura das imagens"""
    height, width = img.shape
    pad_width = target_width - width
    if pad_width <= 0:
        return img
    return np.pad(img, ((0,0), (0,pad_width)), mode='constant')

def segment_characters(captcha):
    """Segmenta o captcha em 6 caracteres individuais"""
    chars = []
    regions = [
        (10, 37),   # 1o char
        (37, 68),   # 2o char
        (68, 96),   # 3o char
        (98, 127),  # 4o char
        (130, 155), # 5o char
        (153, None) # 6o char
    ]
    for left, right in regions:
        if right is None:
            char = captcha[:, left:]
        else:
            char = captcha[:, left:right]
        char = pad_image(char, 32)
        chars.append(char)
    return chars

# ainda em teste: segmentação automática
def segment_characters_advanced(captcha):
    """Segmentação avançada considerando ruído e sobreposição"""
    def preprocess(img):
        denoised = cv.medianBlur(img, 3)
        _, thresh = cv.threshold(denoised, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
        return cleaned

    def find_regions(binary_img):
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(binary_img)
        valid_regions = []
        min_width, max_width, min_height = 15, 45, 20
        for i in range(1, num_labels):
            x, y, w, h = stats[i, cv.CC_STAT_LEFT], stats[i, cv.CC_STAT_TOP], stats[i, cv.CC_STAT_WIDTH], stats[i, cv.CC_STAT_HEIGHT]
            if w >= min_width and w <= max_width and h >= min_height:
                valid_regions.append((x, y, w, h))
        valid_regions.sort(key=lambda r: r[0])
        merged_regions = []
        i = 0
        while i < len(valid_regions):
            current = list(valid_regions[i])
            while (i + 1 < len(valid_regions) and valid_regions[i+1][0] - (current[0] + current[2]) < 5):
                next_region = valid_regions[i+1]
                current[2] = next_region[0] + next_region[2] - current[0]
                i += 1
            merged_regions.append(tuple(current))
            i += 1
        return merged_regions

    processed = preprocess(captcha)
    regions = find_regions(processed)
    chars = []
    expected_chars = 6
    if len(regions) == expected_chars:
        for x, y, w, h in regions:
            char = captcha[:, x:x+w]
            char = pad_image(char, 32)
            chars.append(char)
    else:
        width = captcha.shape[1]
        char_width = width // expected_chars
        for i in range(expected_chars):
            left = i * char_width
            right = (i + 1) * char_width
            char = captcha[:, left:right]
            char = pad_image(char, 32)
            chars.append(char)
    return chars

def compute_hog(img):
    """Calcula as features HOG para uma imagem"""
    win_size = (32, 32)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    features = hog.compute(img)
    return features.flatten()

def load_data_hog():
    """Carrega e prepara os dados com barra de progresso"""
    X, y = [], []
    files = os.listdir(TRAIN_DIR)
    print(f"Processando {len(files)} imagens...")
    for filename in tqdm(files, desc="Carregando dados"):
        img_path = os.path.join(TRAIN_DIR, filename)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        label_path = os.path.join(LABELS_DIR, filename.replace('.jpg','.txt'))
        with open(label_path) as f:
            label = f.read().strip()
        chars = segment_characters(img)
        for char, true_char in zip(chars, label):
            features = compute_hog(char)
            X.append(features)
            y.append(true_char)
    return np.array(X), np.array(y)

def load_training_data_cnn():
    """Carrega e prepara os dados de treinamento para CNN"""
    X, y = [], []
    files = os.listdir(TRAIN_DIR)
    print(f"Processando {len(files)} imagens de treinamento...")
    for filename in tqdm(files, desc="Carregando dados de treinamento"):
        img_path = os.path.join(TRAIN_DIR, filename)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        label_path = os.path.join(LABELS_DIR, filename.replace('.jpg','.txt'))
        with open(label_path) as f:
            label = f.read().strip()
        chars = segment_characters(img)
        for char, true_char in zip(chars, label):
            resized_char = cv.resize(char, (32, 32))
            X.append(resized_char)
            y.append(ord(true_char) - ord('A'))
    X = np.array(X).astype('float32') / 255.0
    X = np.expand_dims(X, axis=-1)
    y = to_categorical(y, num_classes=26)
    return X, y

def load_test_data_cnn():
    """Carrega e prepara os dados de teste para CNN"""
    X, y = [], []
    files = os.listdir(TEST_DIR)
    print(f"Processando {len(files)} imagens de teste...")
    for filename in tqdm(files, desc="Carregando dados de teste"):
        img_path = os.path.join(TEST_DIR, filename)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        label_path = os.path.join(LABELS_DIR, filename.replace('.jpg','.txt'))
        with open(label_path) as f:
            label = f.read().strip()
        chars = segment_characters(img)
        for char, true_char in zip(chars, label):
            resized_char = cv.resize(char, (32, 32))
            X.append(resized_char)
            y.append(ord(true_char) - ord('A'))
    X = np.array(X).astype('float32') / 255.0
    X = np.expand_dims(X, axis=-1)
    y = to_categorical(y, num_classes=26)
    return X, y

def evaluate_model(model, data_dir):
    """Avalia o desempenho do modelo"""
    correct_counts = np.zeros(7)
    total = 0
    files = os.listdir(data_dir)
    print(f"\nAvaliando {len(files)} imagens de teste...")
    for filename in tqdm(files, desc="Testando"):
        total += 1
        img_path = os.path.join(data_dir, filename)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        label_path = os.path.join(LABELS_DIR, filename.replace('.jpg','.txt'))
        with open(label_path) as f:
            true_label = f.read().strip()
        chars = segment_characters(img)
        pred_label = ''
        for char in chars:
            features = compute_hog(char)
            pred_char = model.predict([features])[0]
            pred_label += pred_char
        num_correct = sum(1 for x,y in zip(pred_label, true_label) if x == y)
        correct_counts[num_correct] += 1
    rates = []
    for i in range(7):
        rate = sum(correct_counts[i:]) / total
        rates.append(rate)
    return np.array(rates)

def evaluate_model_cnn(model, data_dir):
    """Avalia o desempenho do modelo CNN"""
    correct_counts = np.zeros(7)
    total = 0
    files = os.listdir(data_dir)
    print(f"\nAvaliando {len(files)} imagens de teste...")
    for filename in tqdm(files, desc="Testando", leave=False, ncols=100, ascii=True):  # Adicionar leave=False para não deixar barras de progresso
        total += 1
        img_path = os.path.join(data_dir, filename)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        label_path = os.path.join(LABELS_DIR, filename.replace('.jpg','.txt'))
        with open(label_path) as f:
            true_label = f.read().strip()
        chars = segment_characters(img)
        pred_label = ''
        for char in chars:
            resized_char = cv.resize(char, (32, 32))
            resized_char = resized_char.astype('float32') / 255.0
            resized_char = np.expand_dims(resized_char, axis=-1)
            resized_char = np.expand_dims(resized_char, axis=0)  # Add batch dimension
            pred_char = model.predict(resized_char, verbose=0)  # Adicionar verbose=0 para suprimir a saída
            pred_char = chr(np.argmax(pred_char) + ord('A'))
            pred_label += pred_char
        num_correct = sum(1 for x, y in zip(pred_label, true_label) if x == y)
        correct_counts[num_correct] += 1
    rates = []
    for i in range(7):
        rate = sum(correct_counts[i:]) / total
        rates.append(rate)
    return np.array(rates)

def plot_accuracy_rates(rates, method):
    """Plota o gráfico de taxas de acurácia"""
    plt.figure(figsize=(10,6))
    x = range(len(rates))
    plt.plot(x, rates, 'b-', marker='o')
    plt.xlabel('Número mínimo de caracteres corretos')
    plt.ylabel('Taxa de reconhecimento')
    plt.grid(True)
    plt.title('Desempenho do Reconhecimento de Caracteres')
    for i, rate in enumerate(rates):
        plt.annotate(f'{rate:.2%}', (i, rate), textcoords="offset points", xytext=(0,10), ha='center')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'recognition_rates_{method}_{timestamp}.png')
    plt.show()

def main_svc():
    try:
        print("Iniciando preparação dos dados...")
        X, y = load_data_hog()
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        print("\nTreinando classificador SVC...")
        print(f"Número de amostras de treinamento: {len(X_train)}")
        print(f"Número de amostras de validação: {len(X_val)}")
        print(f"Número de amostras de teste: {len(X_test)}")
        print(f"Dimensão das features: {X_train.shape[1]}")
        from sklearn.svm import SVC
        clf = SVC(kernel='linear', verbose=0)
        with tqdm(total=100, desc="Treinando SVC", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [Tempo: {elapsed}<{remaining}]') as pbar:
            clf.fit(X_train, y_train)
            pbar.n = 100
            pbar.refresh()
        val_accuracy = clf.score(X_val, y_val)
        print(f"Acurácia na validação: {val_accuracy:.2f}")
        test_rates = evaluate_model(clf, TEST_DIR)
        plot_accuracy_rates(test_rates, "hog_svc")
        print("\nSalvando modelo SVC...")
        with open('captcha_model_svc.pkl', 'wb') as f:
            pickle.dump(clf, f)
        print("Concluído!")
    except Exception as e:
        print(f"\nOcorreu um erro: {str(e)}")
        raise

def create_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def main_cnn():
    try:
        print("Iniciando preparação dos dados para CNN...")
        X_train, y_train = load_training_data_cnn()
        input_shape = (32, 32, 1)
        num_classes = 26

        model = create_cnn(input_shape, num_classes)
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # EarlyStopping
        early_stopping = EarlyStopping(
            monitor='val_accuracy', 
            patience=3, 
            verbose=1, 
            restore_best_weights=True
        )
        
        print("\nTreinando modelo CNN...")
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            verbose=1,
            callbacks=[early_stopping]
        )

        X_test, y_test = load_test_data_cnn()
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"Acurácia no teste: {test_acc:.2f}")
        test_rates = evaluate_model_cnn(model, TEST_DIR)
        plot_accuracy_rates(test_rates, "cnn")
        print("\nSalvando modelo CNN...")
        model.save('cnn_model.h5')
        print("Concluído!")
    except Exception as e:
        print(f"\nOcorreu um erro: {str(e)}")
        raise

def load_and_evaluate_cnn():
    try:
        print("Carregando modelo CNN salvo...")
        model = tf.keras.models.load_model('cnn_model.h5')
        test_rates = evaluate_model_cnn(model, TEST_DIR)
        plot_accuracy_rates(test_rates, "cnn")
        print("Avaliação concluída!")
    except Exception as e:
        print(f"\nOcorreu um erro: {str(e)}")
        raise

if __name__ == "__main__":
    print("Executando abordagem SVC...")
    main_svc()
    print("\nExecutando abordagem CNN...")
    main_cnn()
    print("\nCarregando e avaliando modelo CNN salvo...")
    load_and_evaluate_cnn()
