const { loadModel, readInput } = require('./utils');

const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

const Straights = 'straights-aug';
const Uppercuts = 'uppercuts-aug';
const Negative = 'no-hits-aug';
const Epochs = 500;
const BatchSize = 0.1;
const InputShape = 1024;

const train = async () => {
    const mobileNet = await loadModel();
    const model = tf.sequential();
    model.add(tf.layers.inputLayer({ inputShape: [InputShape] }));
    model.add(tf.layers.dense({ units: 1024, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
    await model.compile({
        optimizer: tf.train.adam(1e-6),
        loss: tf.losses.sigmoidCrossEntropy,
        metrics: ['accuracy']
    });

  const straights = require('fs')
    .readdirSync(Straights)
    .filter(f => f.endsWith('.jpg'))
    .map(f => `${Straights}/${f}`);

  const uppercuts = require('fs')
    .readdirSync(Uppercuts)
    .filter(f => f.endsWith('.jpg'))
    .map(f => `${Uppercuts}/${f}`);

  const negatives = require('fs')
    .readdirSync(Negative)
    .filter(f => f.endsWith('.jpg'))
    .map(f => `${Negative}/${f}`);

  console.log('Building the training set');

  const ys = tf.tensor2d(
    new Array(straights.length)
      .fill([1, 0, 0])
      .concat(new Array(uppercuts.length).fill([0, 1, 0]))
      .concat(new Array(negatives.length).fill([0, 0, 1])),
    [straights.length + uppercuts.length + negatives.length, 3]
  );

  console.log('Getting the punches');
  const arr = straights
    .map((path) => mobileNet(readInput(path)))
    .concat(uppercuts.map((path) => mobileNet(readInput(path))))
    .concat(negatives.map((path) => mobileNet(readInput(path))));

  const xs = tf.stack(arr);
  console.log('Fitting the model');
  await model.fit(xs, ys, {
    epochs: Epochs,
    batchSize: parseInt(((straights.length + uppercuts.length + negatives.length) * BatchSize).toFixed(0)),
    callbacks: {
      onBatchEnd: async (_, logs) => {
        console.log('Cost: %s, accuracy: %s', logs.loss.toFixed(5), logs.acc.toFixed(5));
        await tf.nextFrame();
      }
    }
  });

  console.log('Saving the model');
  await model.save('file://straights_uppercuts');
};

train().catch(e => e.message);