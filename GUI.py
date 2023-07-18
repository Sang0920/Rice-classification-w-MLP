import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
from sklearn.model_selection import train_test_split
import pickle
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image

from neural_net_mlp import NeuralNetMLP

data_dir = '../Rice_Image_Dataset'
data_dir = pathlib.Path(data_dir)

arborio = list(data_dir.glob('Arborio/*'))[:600]
basmati = list(data_dir.glob('Basmati/*'))[:600]
ipsala = list(data_dir.glob('Ipsala/*'))[:600]
jasmine = list(data_dir.glob('Jasmine/*'))[:600]
karacadag = list(data_dir.glob('Karacadag/*'))[:600]

df_images = {
    'arborio': arborio,
    'basmati': basmati,
    'ipsala': ipsala,
    'jasmine': jasmine,
    'karacadag': karacadag
}

# Contains numerical labels for the categories
df_labels = {
    'arborio': 0,
    'basmati': 1,
    'ipsala': 2,
    'jasmine': 3,
    'karacadag': 4
}

root = Tk()
root.title("Rice Image Classification")
root.geometry("750x750")

def show_samples():
    fig, ax = plt.subplots(ncols=5, figsize=(20,5))
    fig.suptitle('Rice Category')
    arborio_image = img.imread(arborio[0])
    basmati_image = img.imread(basmati[0])
    ipsala_image = img.imread(ipsala[0])
    jasmine_image = img.imread(jasmine[0])
    karacadag_image = img.imread(karacadag[0])

    ax[0].set_title('arborio')
    ax[1].set_title('basmati')
    ax[2].set_title('ipsala')
    ax[3].set_title('jasmine')
    ax[4].set_title('karacadag')

    ax[0].imshow(arborio_image)
    ax[1].imshow(basmati_image)
    ax[2].imshow(ipsala_image)
    ax[3].imshow(jasmine_image)
    ax[4].imshow(karacadag_image)

    plt.show()

def standardize_images():
    X = []
    y = []
    for category, images in df_images.items():
        for image in images:
            img = cv2.imread(str(image))
            img = img[0:224, 0:224]
            img = cv2.resize(img, (64, 64))
            X.append(img)
            y.append(df_labels[category])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], 64*64*3)
    X = X.astype('float32')
    X /= 255.0
    
    return X, y

def train_model(num_epochs, num_hidden_layers, learning_rate):
    print('Standardizing images...')
    X, y = standardize_images()
    print('Done!')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)
    model = NeuralNetMLP(n_hidden=num_hidden_layers,
                            l2=0.01,
                            epochs=num_epochs,
                            eta=learning_rate,
                            minibatch_size=100,
                            shuffle=True,
                            seed=1)
    model.fit(X_train=X_train,
                y_train=y_train,
                X_valid=X_test,
                y_valid=y_test)

    plt.plot(range(model.epochs), model.eval_['cost'])
    plt.ylabel('Cost')
    plt.xlabel('Epochs')
    plt.show()

    plt.plot(range(model.epochs), model.eval_['train_acc'],
            label='training')
    plt.plot(range(model.epochs), model.eval_['valid_acc'],
            label='validation', linestyle='--')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='lower right')
    plt.show()

    y_train_pred = model.predict(X_train)
    acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
    print('Training accuracy: %.2f%%' % (acc * 100))
    y_test_pred = model.predict(X_test)
    acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
    print('Test accuracy: %.2f%%' % (acc * 100))

    # save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    img = Image.open(file_path)
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    img_label.configure(image=img)
    img_label.image = img
    return file_path

def predict_image():
    file_path = load_image()
    img = cv2.imread(file_path)
    # img = cv2.resize(img, (224, 224))
    img = img[0:224, 0:224]
    img = cv2.resize(img, (64, 64))
    img = img.reshape(1, 64*64*3).astype('float32') / 255.0

    # Load the saved model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Predict the image
    prediction = model.predict(img)
    predicted_class = list(df_labels.keys())[list(df_labels.values()).index(prediction)]
    prediction_label.configure(text=f"Predicted Class: {predicted_class}")

epoch_label = Label(root, text="Number of Epochs:", font=("Arial", 14))
epoch_label.pack()

hidden_layer_label = Label(root, text="Number of Hidden Layers:", font=("Arial", 14))
hidden_layer_label.pack()

learning_rate_label = Label(root, text="Learning Rate:", font=("Arial", 14))
learning_rate_label.pack()

# Increase the font size for entry fields
epoch_entry = Entry(root, font=("Arial", 14))
epoch_entry.pack()

hidden_layer_entry = Entry(root, font=("Arial", 14))
hidden_layer_entry.pack()

learning_rate_entry = Entry(root, font=("Arial", 14))
learning_rate_entry.pack()

# Increase the font size for buttons
train_button = Button(root, text="Train Model", font=("Arial", 14), command=lambda: train_model(int(epoch_entry.get()), int(hidden_layer_entry.get()), float(learning_rate_entry.get())))
train_button.pack()

# Uncomment and modify the font size for other buttons if needed
# cost_button = Button(root, text="Show Cost", font=("Arial", 14), command=lambda: plt.show())
# cost_button.pack()

# validation_button = Button(root, text="Show Training Validation", font=("Arial", 14), command=lambda: plt.show())
# validation_button.pack()

# test_button = Button(root, text="Test MLP Model", font=("Arial", 14), command=lambda: print("Testing MLP Model..."))
# test_button.pack()

# image_button = Button(root, text="Load Image", font=("Arial", 14), command=load_image)
# image_button.pack()

img_label = Label(root)
img_label.pack()

predict_button = Button(root, text="Predict Image", font=("Arial", 14), command=predict_image)
predict_button.pack()

prediction_label = Label(root, text="Predicted Class:", font=("Arial", 14))
prediction_label.pack()

root.mainloop()