#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'a6dff912ab48f7a273f5704ad9ab1311'))
	print(os.getcwd())
except:
	pass

#%%
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#%% [markdown]
# ## Overview
# All the code we'll look at is in the next cell. We will step through each step after.

#%%
import tensorflow as tf
import numpy as np

print(tf.__version__)

from tensorflow.contrib.learn.python.learn.datasets import base

# Data files
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# Load datasets.
training_set = base.load_csv_with_header(filename=IRIS_TRAINING,
                                         features_dtype=np.float32,
                                         target_dtype=np.int)
test_set = base.load_csv_with_header(filename=IRIS_TEST,
                                     features_dtype=np.float32,
                                     target_dtype=np.int)

# Specify that all features have real-value data
feature_name = "flower_features"
feature_columns = [tf.feature_column.numeric_column(feature_name, 
                                                    shape=[4])]
classifier = tf.estimator.LinearClassifier(
    feature_columns=feature_columns,
    n_classes=3,
    model_dir="/tmp/iris_model")

def input_fn(dataset):
    def _fn():
        features = {feature_name: tf.constant(dataset.data)}
        label = tf.constant(dataset.target)
        return features, label
    return _fn

# Fit model.
classifier.train(input_fn=input_fn(training_set),
               steps=1000)
print('fit done')

# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=input_fn(test_set), 
                                     steps=100)["accuracy"]
print('\nAccuracy: {0:f}'.format(accuracy_score))

# Export the model for serving
feature_spec = {'flower_features': tf.FixedLenFeature(shape=[4], dtype=np.float32)}

serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

classifier.export_savedmodel(export_dir_base='/tmp/iris_model' + '/export', 
                            serving_input_receiver_fn=serving_fn)

#%% [markdown]
# ## Imports

#%%
import tensorflow as tf
import numpy as np

print(tf.__version__)

#%% [markdown]
# ## Data set
# From https://en.wikipedia.org/wiki/Iris_flower_data_set
# 
# 3 types of Iris Flowers: 
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/450px-Kosaciec_szczecinkowaty_Iris_setosa.jpg" style="width: 100px; display:inline"/>
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/800px-Iris_versicolor_3.jpg" style="width: 150px;display:inline"/>
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/736px-Iris_virginica.jpg" style="width: 150px;display:inline"/>
# * Iris Setosa
# * Iris Versicolour
# * Iris Virginica
# 
# 
# 
#%% [markdown]
# ## Data Columns:
#    1. sepal length in cm 
#    2. sepal width in cm 
#    3. petal length in cm 
#    4. petal width in cm
# 
# <img src="petal_sepal.png" style="width: 200px;"/>
# <img src="https://storage.googleapis.com/image-uploader/AIA_images/data_table.png" style="width: 450px"/>
#%% [markdown]
# ## Load data in

#%%
from tensorflow.contrib.learn.python.learn.datasets import base

# Data files
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# Load datasets.
training_set = base.load_csv_with_header(filename=IRIS_TRAINING,
                                         features_dtype=np.float32,
                                         target_dtype=np.int)
test_set = base.load_csv_with_header(filename=IRIS_TEST,
                                     features_dtype=np.float32,
                                     target_dtype=np.int)

print(training_set.data)

print(training_set.target)

#%% [markdown]
# ## Feature columns and model creation

#%%
# Specify that all features have real-value data
feature_name = "flower_features"
feature_columns = [tf.feature_column.numeric_column(feature_name, 
                                                    shape=[4])]

classifier = tf.estimator.LinearClassifier(
    feature_columns=feature_columns,
    n_classes=3,
    model_dir="/tmp/iris_model")

#%% [markdown]
# ## Input function

#%%
def input_fn(dataset):
    def _fn():
        features = {feature_name: tf.constant(dataset.data)}
        label = tf.constant(dataset.target)
        return features, label
    return _fn

print(input_fn(training_set)())

# raw data -> input function -> feature columns -> model

#%% [markdown]
# ## Training

#%%
# Fit model.
classifier.train(input_fn=input_fn(training_set),
               steps=1000)
print('fit done')


#%%
## Evaluation


#%%
# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=input_fn(test_set), 
                                     steps=100)["accuracy"]
print('\nAccuracy: {0:f}'.format(accuracy_score))

#%% [markdown]
# # Estimators review
# 
# ### Load datasets.
# 
#     training_data = load_csv_with_header()
# 
# ### define input functions
# 
#     def input_fn(dataset)
#    
# ### Define feature columns
# 
#     feature_columns = [tf.feature_column.numeric_column(feature_name, shape=[4])]
# 
# ### Create model
# 
#     classifier = tf.estimator.LinearClassifier()
# 
# ### Train
# 
#     classifier.train()
# 
# ### Evaluate
# 
#     classifier.evaluate()
#%% [markdown]
# ## Exporting a model for serving predictions
# 

#%%
feature_spec = {'flower_features': tf.FixedLenFeature(shape=[4], dtype=np.float32)}

serving_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

classifier.export_savedmodel(export_dir_base='/tmp/iris_model' + '/export', 
                            serving_input_receiver_fn=serving_fn)




