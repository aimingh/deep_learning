import os
import numpy as np
import pandas as pd
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import pandas as pd

path_test = './datasets/dogs_vs_cats/test'
path_test_gt = './datasets/dogs_vs_cats/sampleSubmission.csv'
# test_num = len(os.listdir(path_test))
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        path_test,
        target_size=(224, 224),
        batch_size=1,
        shuffle=False,
        class_mode=None)

# Model 불러오기
model = models.load_model('AlexNet.h5')
model.summary()

# Test
test_generator.reset()
pred = model.predict_generator(
            test_generator,
            steps=test_generator.n,
            verbose=1)

pred = pred.reshape((test_generator.n))
predicted_class_indices = (pred>0.5)+0

labels = {'cats': 0, 'dogs': 1}
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)