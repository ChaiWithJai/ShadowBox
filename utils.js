const tf = require('@tensorflow/tfjs');
const fs = require('fs');
const jpeg = require('jpeg-js')
const { image, valueAndGrads } = require('@tensorflow/tfjs');
const mobilenet = require('@tensorflow-models/mobilenet');

const TotalChannels = 3;

const readImage = path => {
    const buf = fs.readFileSync(path);
    const pixels = jpeg.decode(buf, true);
    return pixels;
}

const imageByteArray = (image, numChannels) => {
    const pixels = image.data;
    const numPixels = image.width * image.height;
    const values = new Int32Array(numPixels * numChannels);

    for (let i = 0; i < numPixels; i++) {
        for (let channel = 0; channel < numChannels; ++channel) {
            values[i * numChannels + channel] = pixels[i * 4 + channel];
        }
    }
    return values;
};

const imageToInput = (image, numChannels) => {
    const values = imageByteArray(image, numChannels);
    const outShape = [image.height, image.width, numChannels];
    const input = tf.tensor3d(values, outShape, 'int32');

    return input;
};

const Layer = 'global_average_pooling2d_1';
const ModelPath = './mobile-net/model.json';


module.exports.readInput = img => imageToInput(readImage(img), TotalChannels);


module.exports.loadModel = async () => {
    const mn = new mobilenet.MobileNet(1, 1);
    mn.path = `file://${ModelPath}`;
    await mn.load();
    return (input) => mn.infer(input, Layer).reshape([1024]);
};