
#Ensembling

def fit_model():
    model = get_model_bn_do()
    model.fit_generator(batches, batches.N, nb_epoch=1, verbose=0,
                        validation_data=test_batches, nb_val_samples=test_batches.N)
    model.optimizer.lr=0.1
    model.fit_generator(batches, batches.N, nb_epoch=4, verbose=0,
                        validation_data=test_batches, nb_val_samples=test_batches.N)
    model.optimizer.lr=0.01
    model.fit_generator(batches, batches.N, nb_epoch=12, verbose=0,
                        validation_data=test_batches, nb_val_samples=test_batches.N)
    model.optimizer.lr=0.001
    model.fit_generator(batches, batches.N, nb_epoch=18, verbose=0,
                        validation_data=test_batches, nb_val_samples=test_batches.N)
    return model

models = [fit_model() for i in range(6)]

path = "data/mnist/"
model_path = path + 'models/'

for i,m in enumerate(models):
    m.save_weights(model_path+'cnn-mnist23-'+str(i)+'.pkl')

evals = np.array([m.evaluate(X_test, y_test, batch_size=256) for m in models])

evals.mean(axis=0)

array([ 0.016,  0.995])

all_preds = np.stack([m.predict(X_test, batch_size=256) for m in models])

all_preds.shape


avg_preds = all_preds.mean(axis=0)

keras.metrics.categorical_accuracy(y_test, avg_preds).eval()



