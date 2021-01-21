from tensorflow.keras import layers, models

def down_conv(x, n_filters, kernel_size, dropout_rate):
    x = layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), padding = 'same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def unet(input_size, 
         n_filters = 64, 
         kernel_size=3, 
         dropout_rate=0.1,
         output_channerl = 1):
    input = layers.Input(input_size)

    # conv
    conv1_1 = down_conv(input, n_filters, kernel_size, dropout_rate)  
    conv1_2 = down_conv(conv1_1, n_filters, kernel_size, dropout_rate)  
    pool1 = layers.MaxPooling2D(pool_size = (2, 2))(conv1_2)
    pool1 = layers.Dropout(rate = dropout_rate)(pool1)

    conv2_1 = down_conv(pool1, 2*n_filters, kernel_size, dropout_rate)  
    conv2_2 = down_conv(conv2_1, 2*n_filters, kernel_size, dropout_rate)  
    pool2 = layers.MaxPooling2D(pool_size = (2, 2))(conv2_2)
    pool2 = layers.Dropout(rate = dropout_rate)(pool2)

    conv3_1 = down_conv(pool2, 4*n_filters, kernel_size, dropout_rate)  
    conv3_2 = down_conv(conv3_1, 4*n_filters, kernel_size, dropout_rate)  
    pool3 = layers.MaxPooling2D(pool_size = (2, 2))(conv3_2)
    pool3 = layers.Dropout(rate = dropout_rate)(pool3)

    conv4_1 = down_conv(pool3, 8*n_filters, kernel_size, dropout_rate)  
    conv4_2 = down_conv(conv4_1, 8*n_filters, kernel_size, dropout_rate)  
    pool4 = layers.MaxPooling2D(pool_size = (2, 2))(conv4_2)
    pool4 = layers.Dropout(rate = dropout_rate)(pool4)

    conv5_1 = down_conv(pool4, 16*n_filters, kernel_size, dropout_rate)  
    conv5_2 = down_conv(conv5_1, 16*n_filters, kernel_size, dropout_rate)  

    # deconv
    upconv6 = layers.Conv2DTranspose(filters = 8*n_filters, kernel_size = (kernel_size, kernel_size), strides = (2, 2), padding = 'same')(conv5_2)
    upconv6 = layers.concatenate([conv4_2, upconv6])
    conv6_1 = down_conv(upconv6, 8*n_filters, kernel_size, dropout_rate) 
    conv6_2 = down_conv(conv6_1, 8*n_filters, kernel_size, dropout_rate)  

    upconv7 = layers.Conv2DTranspose(filters = 4*n_filters, kernel_size = (kernel_size, kernel_size), strides = (2, 2), padding = 'same')(conv6_2)
    upconv7 = layers.concatenate([conv3_2, upconv7])
    conv7_1 = down_conv(upconv7, 4*n_filters, kernel_size, dropout_rate)  
    conv7_2 = down_conv(conv7_1, 4*n_filters, kernel_size, dropout_rate)  

    upconv8 = layers.Conv2DTranspose(filters = 2*n_filters, kernel_size = (kernel_size, kernel_size), strides = (2, 2), padding = 'same')(conv7_2)
    upconv8 = layers.concatenate([conv2_2, upconv8])
    conv8_1 = down_conv(upconv8, 2*n_filters, kernel_size, dropout_rate)  
    conv8_2 = down_conv(conv8_1, 2*n_filters, kernel_size, dropout_rate)  

    upconv9 = layers.Conv2DTranspose(filters = n_filters, kernel_size = (kernel_size, kernel_size), strides = (2, 2), padding = 'same')(conv8_2)
    upconv9 = layers.concatenate([conv1_2, upconv9])
    conv9_1 = down_conv(upconv9, n_filters, kernel_size, dropout_rate)  
    conv9_2 = down_conv(conv9_1, n_filters, kernel_size, dropout_rate)  

    # last layer
    output = layers.Conv2D(filters = output_channerl, kernel_size = (1, 1), activation = 'sigmoid')(conv9_2)

    model = models.Model(input, output)

    return model