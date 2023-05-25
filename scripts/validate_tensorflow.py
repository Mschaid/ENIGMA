import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def print_tf_specs():
        print("TensorFlow version:", tf.__version__)
        print('********************************')

        # Check if GPUs are detectable
        print('GPUs available:')
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            print(gpu)
        print('********************************')


def test_tf():
    np.random.seed(42)

    # Number of data points
    num_data_points = 1000

    # Generate random values for features
    feature1 = np.random.rand(num_data_points)
    feature2 = 2 * feature1 + np.random.randn(num_data_points)
    feature3 = -3 * feature1 + np.random.randn(num_data_points)
    feature4 = 0.5 * feature1 + np.random.randn(num_data_points)
    feature5 = feature1 + np.random.randn(num_data_points)

    # Generate target values (linear combination of features)
    target = 2 * feature1 + 3 * feature2 - 4 * feature3 + feature4 + 0.5 * feature5 + np.random.randn(num_data_points)

    # Create a pandas DataFrame
    data = pd.DataFrame({'feature1': feature1,
                        'feature2': feature2,
                        'feature3': feature3,
                        'feature4': feature4,
                        'feature5': feature5,
                        'target': target})

    # Split the dataset into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2, random_state=42)

    # Scale the data using StandardScaler
    # scaler = StandardScaler()
    # train_data = scaler.fit_transform(train_data)
    # test_data = scaler.transform(test_data)

    # Define the DNN architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model on GPUs

    model.fit(train_data, train_labels, epochs=100, batch_size=32, validation_data=(test_data, test_labels))
    model.evaluate(test_data, test_labels)
    predictions = model.predict(test_data)
    print(predictions)
    
if __name__ == '__main__':
    # Print TensorFlow version
    print_tf_specs()
    test_tf()
