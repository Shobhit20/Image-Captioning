import SceneDesc

import sys

def train(epoch):
    sd = SceneDesc.scenedesc()
    model = sd.create_model()
    batch_size = 512
    model.fit_generator(sd.data_process(batch_size=batch_size), steps_per_epoch=sd.no_samples/batch_size, epochs=epoch, verbose=2, callbacks=None)
    model.save('Output/Model.h5', overwrite=True)
    model.save_weights('Output/Weights.h5',overwrite=True)
 
if __name__=="__main__":
    train(int(sys.argv[1]))
