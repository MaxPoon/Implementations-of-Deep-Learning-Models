conv1_params = {
    'filters': 256,
    'kernel_size': 9,
    'strides': 1,
    'padding': 'valid',
    'activation': 'relu'
}

conv2_params = {
    'filters': 256,
    'kernel_size': 9,
    'strides': 2,
    'padding': 'valid',
    'activation': 'relu',
}

batch_size = 256
input_shape = [28, 28, 1]
primary_capsules_shape = [1152, 8]

digits_capsules_params = {
    'num_capsule': 10,
    'dim_capsule': 16,
    'routing_iterations': 3
}
dense1, dense2 = 512, 1024
margin_loss_lambda = 0.5
reconstruction_loss_coefficient = 0.0005