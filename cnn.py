import argparse
import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Diretórios
TRAIN_DIR = "dados/CAPTCHA-10k/treinamento"
VALIDATION_DIR = "dados/CAPTCHA-10k/validacao"
TEST_DIR = "dados/CAPTCHA-10k/teste"
LABELS_DIR = "dados/CAPTCHA-10k/labels10k/"

# Definições
NUM_CLASSES = 37  # Letras (A-Z) + Números (0-9) + ?
INPUT_SHAPE_SIMPLE = (32, 32, 1)  # Modelo simples
INPUT_SHAPE_BRANCHES = (128, 64, 1)  # Modelo com branches
NUM_CHARS = 6  # Número de caracteres em cada CAPTCHA


def char_to_index(char):
    if 'A' <= char <= 'Z':
        return ord(char) - ord('A')
    elif '0' <= char <= '9':
        return ord(char) - ord('0') + 26
    elif char == '?':
        return 36
    raise ValueError(f"Caractere inválido: {char}")


def index_to_char(index):
    if 0 <= index < 26:
        return chr(index + ord('A'))
    elif 26 <= index < 36:
        return chr(index - 26 + ord('0'))
    elif index == 36:
        return '?'
    raise ValueError(f"Índice inválido: {index}")


# Funções do modelo simples (baseado em caracteres segmentados)
def segment_characters(img):
    regions = [(10, 37), (37, 68), (68, 96), (98, 127), (130, 155), (153, None)]
    chars = []
    for left, right in regions:
        char = img[:, left:] if right is None else img[:, left:right]
        char = np.pad(char, ((0, 0), (0, max(0, 32 - char.shape[1]))), mode='constant')
        chars.append(char)
    return chars


def load_data_simple(data_dir, label_dir):
    X, y = [], []
    files = os.listdir(data_dir)
    for filename in tqdm(files, desc=f"Carregando {data_dir}"):
        img_path = os.path.join(data_dir, filename)
        label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        with open(label_path) as f:
            label = f.read().strip()

        chars = segment_characters(img)
        for char, true_char in zip(chars, label):
            resized_char = cv.resize(char, (32, 32)).astype('float32') / 255.0
            X.append(np.expand_dims(resized_char, axis=-1))
            y.append(char_to_index(true_char))

    X = np.array(X)
    y = to_categorical(y, num_classes=NUM_CLASSES)
    return X, y

def create_cnn_simple(input_shape, num_classes):
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
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# Funções do modelo com branches
def preprocess_labels(labels, num_classes, num_chars):
    y = [[] for _ in range(num_chars)]
    for label in labels:
        for i, char in enumerate(label):
            one_hot = np.zeros((num_classes,), dtype='float32')
            one_hot[char_to_index(char)] = 1
            y[i].append(one_hot)
    y = [np.array(branch, dtype='float32') for branch in y]
    return y


def load_data_with_branches(data_dir, label_dir, img_size=(64, 128)):
    X, labels = [], []
    files = os.listdir(data_dir)
    for filename in tqdm(files, desc=f"Carregando {data_dir}"):
        img_path = os.path.join(data_dir, filename)
        label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))

        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        img_resized = cv.resize(img, img_size).astype('float32') / 255.0
        with open(label_path) as f:
            label = f.read().strip()

        if len(label) != NUM_CHARS:
            print(f"Atenção: ignorando {filename} (rótulo inválido: {label})")
            continue

        X.append(np.expand_dims(img_resized, axis=-1))
        labels.append(label)

    X = np.array(X)
    y = preprocess_labels(labels, num_classes=NUM_CLASSES, num_chars=NUM_CHARS)
    return X, y


def create_cnn_branches(input_shape, num_classes, num_chars):
    img = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(img)
    mp1 = layers.MaxPooling2D((2, 2), padding='same')(conv1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(mp1)
    mp2 = layers.MaxPooling2D((2, 2), padding='same')(conv2)
    flat = layers.Flatten()(mp2)

    outs = []
    for i in range(num_chars):
        dense = layers.Dense(128, activation='relu')(flat)
        out = layers.Dense(num_classes, activation='softmax', name=f'char_{i}')(dense)
        outs.append(out)

    model = models.Model(inputs=img, outputs=outs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'] * num_chars
    )
    return model

def evaluate_with_graph(model, data_dir, label_dir, num_chars):
    """
    Avalia o modelo e gera o gráfico de taxa de reconhecimento.

    Args:
        model: Modelo treinado.
        data_dir: Diretório contendo as imagens de teste.
        label_dir: Diretório contendo os rótulos ground-truth.
        num_chars: Número de caracteres em cada CAPTCHA.
    """
    print("Avaliando o modelo e gerando gráfico...")

    # Carregar os dados de teste
    X_test, y_test = load_data_with_branches(data_dir, label_dir)
    predictions = model.predict(X_test)

    # Avaliar o desempenho para cada CAPTCHA
    correct_counts = np.zeros(num_chars + 1)
    total_captchas = len(X_test)

    for i in range(total_captchas):
        pred_label = ''.join(index_to_char(np.argmax(pred[i])) for pred in predictions)
        true_label = ''.join(index_to_char(np.argmax(y_test[branch][i])) for branch in range(num_chars))

        # Contar o número de caracteres corretos
        num_correct = sum(1 for p, t in zip(pred_label, true_label) if p == t)
        correct_counts[num_correct] += 1

    # Calcular a taxa de reconhecimento
    recognition_rates = [sum(correct_counts[i:]) / total_captchas for i in range(num_chars + 1)]

    # Gerar o gráfico de taxa de reconhecimento
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_chars + 1), recognition_rates, marker='o')
    plt.xlabel('Número mínimo de caracteres corretos')
    plt.ylabel('Taxa de reconhecimento')
    plt.title('Taxa de Reconhecimento por Número de Caracteres Corretos')
    plt.grid(True)
    plt.show()

    # Exibir a taxa para CAPTCHAs completos
    print(f"Acurácia de CAPTCHAs completamente corretos: {recognition_rates[-1]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Processo de reconhecimento de CAPTCHA")
    parser.add_argument("--model_type", type=str, choices=["simple", "branches"], required=True, help="Tipo de modelo (simple ou branches).")
    parser.add_argument("--train", action="store_true", help="Treina o modelo com os dados disponíveis.")
    parser.add_argument("--evaluate", action="store_true", help="Avalia o modelo treinado no conjunto de teste.")
    parser.add_argument("--model_path", type=str, default="captcha_model.h5", help="Caminho para salvar ou carregar o modelo.")
    args = parser.parse_args()

    if args.model_type == "simple":
        input_shape = INPUT_SHAPE_SIMPLE
        load_data = load_data_simple
        create_model = create_cnn_simple
    else:
        input_shape = INPUT_SHAPE_BRANCHES
        load_data = load_data_with_branches
        create_model = create_cnn_branches

    if args.train:
        X_train, y_train = load_data(TRAIN_DIR, LABELS_DIR)
        X_val, y_val = load_data(VALIDATION_DIR, LABELS_DIR)
        model = create_model(input_shape, NUM_CLASSES, NUM_CHARS)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, callbacks=[early_stopping])
        model.save(args.model_path)
        print(f"Modelo salvo em {args.model_path}.")

    if args.evaluate:
        model = tf.keras.models.load_model(args.model_path)
        if args.model_type == "simple":
            X_test, y_test = load_data(TEST_DIR, LABELS_DIR)
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
            print(f"Acurácia no conjunto de teste: {test_acc:.2f}")
        else:  # branches
            evaluate_with_graph(model, TEST_DIR, LABELS_DIR, NUM_CHARS)


if __name__ == "__main__":
    main()