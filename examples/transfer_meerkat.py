








w_radio = calc_w(uv_radio)


gunet = GUNet_var(
    Nd, 
    uv=uv_radio,
    op=NUFFT2D_TF, 
    depth=4, 
    conv_layers=2,
    input_type="measurements", 
    measurement_weights=np.ones(len(uv_radio)),
    batch_size=batch_size,
    residual=False,
    metrics=[PSNRMetric()]
    )

operator = ""
network = "GUNet_var"
postfix = "_specific_new2"

data = "TNG"

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    monitor='val_PSNR',
    mode='max',
    save_best_only='True',
    save_weights_only=True,
    save_freq='epoch'#save_freq* (train_size//batch_size)
)

checkpoint_folder = project_folder+ f"models/{data}/NUFFT_Random_var/{network}_{ISNR}dB{postfix}"
latest = tf.train.latest_checkpoint(checkpoint_folder)
gunet.load_weights(latest)


z_train = np.ones_like(y_train)
gunet.fit((y_train, z_train), x_train, epochs = 100, batch_size=batch_size)