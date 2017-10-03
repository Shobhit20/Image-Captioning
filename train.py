import SceneDesc

sd = SceneDesc.scenedesc()
model = sd.create_model()
batch_size = 512
model.fit_generator(sd.data_process(batch_size=batch_size), steps_per_epoch=sd.no_samples/batch_size, epochs=30, verbose=2, callbacks=None)
model.save('output/Model.h5', overwrite=True)
model.save_weights('output/Weights.h5',overwrite=True)
