# Python packages
import mlcm
import train_model
import timeit
import tensorflow as tf
import utils
from datetime import datetime


# Select GPU
utils.set_gpu(0)

# Load data
X_train, y_train, X_val, y_val, X_test, y_test = utils.load_data()

# Sequence of classes names
target_names = ['NORM', 'STTC', 'CD', 'MI', 'HYP']

# target_names = ['NORM', 'AMI', 'IMI', 'STTC', 'LAFB/LPFB', 'LVH', 'ISC_', 'IRBBB', 'ISCA',\
                # '_AVB', 'IVCD', 'NST_', 'CRBBB', 'CLBBB', 'LAO/LAE', 'ISCI', 'LMI', 'RVH',\
                # 'RAO/RAE', 'WPW', 'ILBBB', 'SEHYP', 'PMI']

# Get the model
# model_name = 'rajpurkar'
model_name = 'ribeiro'
# model_name = 'ahmed'
# model_name = 'baloglu'

if model_name == 'rajpurkar':
    batch_size = 128
    optimizer = 'adam'
    learning_rate = 0.001

elif model_name == 'ribeiro':
    batch_size = 16
    optimizer = 'adam'
    learning_rate = 0.01

elif model_name == 'ahmed':
    batch_size = 16
    optimizer = 'adam'
    learning_rate = 0.001

else:
    batch_size = 256
    optimizer = 'adam'
    learning_rate = 0.001


# Get the model
input_shape = X_train.shape[1:]
model = utils.get_model(input_shape, model_name)
# model.summary()
# dot_img_file = f'{model_name}.png'
# tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

# Get the time
time = datetime.now().isoformat()

# Paths
model_name_path = f'{model_name}_{time}'

params = {'X_train': X_train, 'y_train': y_train,
          'X_val': X_val, 'y_val': y_val, 'model_name': model_name_path,
          'save_parameters': True, 'learning_rate': learning_rate, 'epochs': 100, 'factor': 0.5,
          'patience_RLR': 10, 'patience_ES': 15, 'min_lr': 1e-6, 'loss': 'binary_crossentropy',
          'batch_size': batch_size, 'monitor': 'val_loss', 'optimizer': optimizer}


# Get and print the memory usage of the model
gbytes, mbytes = utils.get_model_memory_usage(params['batch_size'], model)
print(f'Model: {model_name} - (GPU) Memory requirements: {gbytes} GB and {mbytes} MB')

# Train the model
# tic = timeit.default_timer()
history = train_model.training(model, **params)
# toc = timeit.default_timer()

# Evaluate the model
score = model.evaluate(X_test, y_test)
print(f'#############################################################################')
# print(f'Tempo de treinamento: {toc - tic:.2f} segundos')
print(f'Custo de teste: {score[0]:.2f}')
print(f'Acurácia de teste: {score[1]:.2%}')
print(f'#############################################################################')

# Prediction of the model
y_pred = model.predict(X_test)
# Test
y_pred_val = model.predict(X_val)

# Convert the predictions to binary values
y_pred = (y_pred > 0.5).astype('int')
# test
y_pred_val = (y_pred_val > 0.5).astype('int')


# Save results
# Calculate the mlcm
cm, _ = mlcm.cm(y_test, y_pred, print_note=False)
# Get the metrics from mlcm
d = utils.get_mlcm_metrics(cm)
# Save the reports from mlcm
utils.get_mlcm_report(cm, target_names, model_name_path, dataset='test')
# Plot the confusiona matrix
utils.plot_confusion_matrix(cm, model_name_path, target_names, plot_path = 'results', dataset = 'test')
# Plot the loss function
utils.plot_results(history, model_name_path, metric='loss')

