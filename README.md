#This project address the classification of the flower of iris a linear regression on the TensorFlow framework.


## The steps are as follows:

### Load datasets.

    training_data = load_csv_with_header()

### define input functions

    def input_fn(dataset)

### Define feature columns

    feature_columns = [tf.feature_column.numeric_column(feature_name, shape=[4])]

### Create model

   classifier = tf.estimator.LinearClassifier()


### Train

   classifier.train()


### Evaluate

    classifier.evaluate()

### Exporting a model for serving predictions

