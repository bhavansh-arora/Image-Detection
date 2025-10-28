package com.ai.imagedetection;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class LogoClassifier {
    private static final String TAG = "LogoClassifier";
    private final Interpreter interpreter;
    private final List<String> labels;
    private final int imageSize = 224; // must match training size

    private final boolean isQuantized;

    public LogoClassifier(AssetManager assetManager) throws IOException {
        interpreter = new Interpreter(loadModelFile(assetManager, "logo_model.tflite"));
        labels = loadLabels(assetManager, "labels.txt");
        isQuantized = interpreter.getInputTensor(0).dataType().toString().equals("UINT8");
        Log.d(TAG, "Model type: " + (isQuantized ? "Quantized (UINT8)" : "Float32"));
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private List<String> loadLabels(AssetManager assetManager, String labelPath) throws IOException {
        List<String> labels = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            labels.add(line);
        }
        reader.close();
        return labels;
    }

private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
    Bitmap resized = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true);

    int bytesPerChannel = isQuantized ? 1 : 4;
    ByteBuffer buffer = ByteBuffer.allocateDirect(bytesPerChannel * imageSize * imageSize * 3);
    buffer.order(ByteOrder.nativeOrder());

    int[] intValues = new int[imageSize * imageSize];
    resized.getPixels(intValues, 0, imageSize, 0, 0, imageSize, imageSize);
    int pixelIndex = 0;

    for (int i = 0; i < imageSize; i++) {
        for (int j = 0; j < imageSize; j++) {
            int pixelValue = intValues[pixelIndex++];
            int r = (pixelValue >> 16) & 0xFF;
            int g = (pixelValue >> 8) & 0xFF;
            int b = pixelValue & 0xFF;

            if (isQuantized) {
                // For quantized models → raw bytes
                buffer.put((byte) r);
                buffer.put((byte) g);
                buffer.put((byte) b);
            } else {
                // For float models → normalized floats [-1,1]
                buffer.putFloat((r / 127.5f) - 1.0f);
                buffer.putFloat((g / 127.5f) - 1.0f);
                buffer.putFloat((b / 127.5f) - 1.0f);
            }
        }
    }

    return buffer;
}

    public String classify(Bitmap bitmap) {
        ByteBuffer input = convertBitmapToByteBuffer(bitmap);

        // Choose output type based on quantization
        Object output;
        if (isQuantized) {
            output = new byte[1][labels.size()];
        } else {
            output = new float[1][labels.size()];
        }

        interpreter.run(input, output);

        float[] probabilities = new float[labels.size()];

        if (isQuantized) {
            // Convert uint8 → float probabilities (0–255 → 0.0–1.0)
            byte[][] quantizedOutput = (byte[][]) output;
            for (int i = 0; i < labels.size(); i++) {
                probabilities[i] = (quantizedOutput[0][i] & 0xFF) / 255.0f;
            }
        } else {
            probabilities = ((float[][]) output)[0];
        }

        // 🔍 Debug print
        StringBuilder probs = new StringBuilder();
        for (int i = 0; i < probabilities.length; i++) {
            probs.append(labels.get(i))
                    .append(": ")
                    .append(String.format("%.3f ", probabilities[i]));
        }
        Log.d(TAG, "Probabilities → " + probs);

        // Find top class
        int maxIndex = 0;
        float maxProb = 0f;
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                maxIndex = i;
            }
        }

        // 🧠 Threshold for unknowns / NoLogo
        if (maxProb < 0.4f || labels.get(maxIndex).equalsIgnoreCase("NoLogo")) {
            return "⚠️ No known logo detected (" + String.format("%.1f", maxProb * 100) + "%)";
        }

        String result = labels.get(maxIndex) + " (" + String.format("%.1f", maxProb * 100) + "%)";
        Log.d(TAG, "Prediction: " + result);
        return result;
    }


    public void close() {
        interpreter.close();
    }
}
