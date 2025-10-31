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
import java.util.HashMap;
import java.util.List;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Map;

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
        Object output = isQuantized ? new byte[1][labels.size()] : new float[1][labels.size()];
        interpreter.run(input, output);

        float[] probabilities = new float[labels.size()];
        if (isQuantized) {
            byte[][] q = (byte[][]) output;
            for (int i = 0; i < labels.size(); i++) {
                probabilities[i] = (q[0][i] & 0xFF) / 255f;
            }
        } else {
            probabilities = ((float[][]) output)[0];
        }

        // --- basic stats ---
        int top1 = -1, top2 = -1;
        float maxProb = -1f, secondProb = -1f;
        for (int i = 0; i < probabilities.length; i++) {
            float p = probabilities[i];
            if (p > maxProb) {
                secondProb = maxProb;
                top2 = top1;
                maxProb = p;
                top1 = i;
            } else if (p > secondProb) {
                secondProb = p;
                top2 = i;
            }
        }
        float gap = maxProb - secondProb;
        int noLogoIndex = labels.indexOf("NoLogo");
        float noLogoProb = (noLogoIndex >= 0) ? probabilities[noLogoIndex] : 0f;

        // --- entropy + stats ---
        float entropy = 0f;
        for (float p : probabilities) if (p > 0) entropy -= p * Math.log(p);
        float mean = 0f;
        for (float p : probabilities) mean += p;
        mean /= probabilities.length;
        float variance = 0f;
        for (float p : probabilities) variance += (p - mean) * (p - mean);
        variance /= probabilities.length;
        float stddev = (float) Math.sqrt(variance);

        // --- log everything for tuning ---
        Log.d(TAG, "==== LOGO DEBUG ====");
        Log.d(TAG, "Top1: " + labels.get(top1) + "  (" + maxProb + ")");
        Log.d(TAG, "Top2: " + labels.get(top2) + "  (" + secondProb + ")");
        Log.d(TAG, "Gap: " + gap);
        Log.d(TAG, "NoLogoProb: " + noLogoProb);
        Log.d(TAG, "Entropy: " + entropy);
        Log.d(TAG, "StdDev: " + stddev);
        StringBuilder dist = new StringBuilder("Distribution: ");
        for (int i = 0; i < labels.size(); i++) {
            dist.append(labels.get(i))
                    .append(":")
                    .append(String.format("%.3f ", probabilities[i]));
        }
        Log.d(TAG, dist.toString());
        Log.d(TAG, "=====================");

        // --- new sanity layer ---
        Log.d(TAG, "EdgeDensity value: " + getEdgeDensity(bitmap));
        boolean looksLogoLike = !isFlatImage(bitmap) && getEdgeDensity(bitmap) > 0.015f;
        if (!looksLogoLike) {
            Log.d(TAG, "Reject → Visual cues do not match a printed logo");
            return "⚠️ No logo detected (visual mismatch)";
        }

// dynamic bias for NoLogo
        float boostedNoLogo = noLogoProb + 0.25f * (1f - maxProb);
        if (boostedNoLogo > 0.3f) {
            Log.d(TAG, "Reject → Boosted NoLogo triggers (" + boostedNoLogo + ")");
            return "⚠️ No known logo detected";
        }


        // --- final decision logic ---
        boolean reject =
                (maxProb < 0.8f) ||          // low confidence
                        (gap < 0.15f) ||             // ambiguous top2
                        (entropy > 2.2f) ||          // spread out probabilities
                        (noLogoProb > 0.4f && noLogoProb > 0.5f * maxProb);

        if (reject) {
            Log.d(TAG, "Decision → REJECT (No known logo)");
            return "⚠️ No known logo detected";
        }

        String result = labels.get(top1) + " (" + String.format("%.1f", maxProb * 100) + "%)";
        Log.d(TAG, "Decision → ACCEPT → " + result);
        return result;
    }

    private boolean isFlatImage(Bitmap bmp) {
        int[] pixels = new int[bmp.getWidth() * bmp.getHeight()];
        bmp.getPixels(pixels, 0, bmp.getWidth(), 0, 0, bmp.getWidth(), bmp.getHeight());
        long sum = 0, sumSq = 0;
        for (int c : pixels) {
            int gray = (int)(0.3 * ((c >> 16) & 0xFF) + 0.59 * ((c >> 8) & 0xFF) + 0.11 * (c & 0xFF));
            sum += gray;
            sumSq += gray * gray;
        }
        double mean = sum / (double)pixels.length;
        double var = (sumSq / (double)pixels.length) - mean * mean;
        return var < 20; // tweak
    }

    private float getEdgeDensity(Bitmap bmp) {
        int w = bmp.getWidth(), h = bmp.getHeight();
        int step = Math.max(1, Math.min(w, h) / 50);
        int edgeCount = 0, total = 0;
        for (int y = step; y < h - step; y += step) {
            for (int x = step; x < w - step; x += step) {
                int c1 = bmp.getPixel(x, y);
                int c2 = bmp.getPixel(x + step, y);
                int diff = Math.abs(((c1 >> 16) & 0xFF) - ((c2 >> 16) & 0xFF));
                if (diff > 20) edgeCount++;
                total++;
            }
        }
        return (float) edgeCount / total;
    }


    // --- Cosine Similarity Embedding Check ---
// Suppose you have precomputed centroids for each known brand
// You can later load them from assets or hardcode temporary ones

    private final HashMap<String, float[]> brandCentroids = new HashMap<String, float[]>() {{
        put("Adidas", new float[]{ /* example placeholder vector */ });
        put("RalphLauren", new float[]{ /* example placeholder vector */ });
    }};

    // Get 128D or feature vector (assuming penultimate layer output)
    public float[] getEmbedding(Bitmap bitmap) {
        ByteBuffer input = convertBitmapToByteBuffer(bitmap);

        // Let's assume your model output can be treated as an embedding
        // If your model has a softmax layer at the end, you can still
        // temporarily use those probabilities as a coarse embedding
        Object output = isQuantized ? new byte[1][labels.size()] : new float[1][labels.size()];
        interpreter.run(input, output);

        float[] embedding = new float[labels.size()];
        if (isQuantized) {
            byte[][] q = (byte[][]) output;
            for (int i = 0; i < labels.size(); i++) embedding[i] = (q[0][i] & 0xFF) / 255f;
        } else {
            embedding = ((float[][]) output)[0];
        }
        return embedding;
    }

    // Compute cosine similarity between two vectors
    private float cosineSimilarity(float[] a, float[] b) {
        if (a == null || b == null || a.length == 0 || b.length == 0) {
            Log.e(TAG, "❌ Invalid vectors for cosine similarity");
            return 0f;
        }

        float dot = 0f, magA = 0f, magB = 0f;
        int len = Math.min(a.length, b.length); // ensure no out-of-range access

        for (int i = 0; i < len; i++) {
            dot += a[i] * b[i];
            magA += a[i] * a[i];
            magB += b[i] * b[i];
        }

        return (float) (dot / (Math.sqrt(magA) * Math.sqrt(magB) + 1e-6));
    }

    // Compare current image embedding against known brands
    public String compareWithCentroids(Bitmap bitmap, float threshold) {
        float[] embedding = getEmbedding(bitmap);

        if (embedding == null || embedding.length == 0) {
            Log.e(TAG, "❌ Embedding extraction failed or empty!");
            return "⚠️ Embedding error — cannot verify logo";
        }

        String bestBrand = "Unknown";
        float bestSim = -1f;

        for (Map.Entry<String, float[]> e : brandCentroids.entrySet()) {
            float[] centroid = e.getValue();
            if (centroid == null || centroid.length == 0) {
                Log.w(TAG, "⚠️ Centroid for " + e.getKey() + " is empty — skipping");
                continue;
            }

            float sim = cosineSimilarity(embedding, centroid);
            Log.d(TAG, "Cosine with " + e.getKey() + ": " + sim);
            if (sim > bestSim) {
                bestSim = sim;
                bestBrand = e.getKey();
            }
        }

        if (bestSim < 0) {
            return "⚠️ No centroids available for comparison";
        }

        if (bestSim < threshold) {
            return "⚠️ No known logo detected (cosine=" + String.format("%.2f", bestSim) + ")";
        } else {
            return "✅ Verified: " + bestBrand + " (" + String.format("%.2f", bestSim) + ")";
        }
    }



    public void close() {
        interpreter.close();
    }
}
