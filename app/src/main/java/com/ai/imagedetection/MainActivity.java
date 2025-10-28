package com.ai.imagedetection;

//import android.os.Bundle;
//import android.util.Log;
//
//import androidx.activity.EdgeToEdge;
//import androidx.appcompat.app.AppCompatActivity;
//import androidx.core.graphics.Insets;
//import androidx.core.view.ViewCompat;
//import androidx.core.view.WindowInsetsCompat;
//
//import org.opencv.android.OpenCVLoader;
//
//public class MainActivity extends AppCompatActivity {
//
//    @Override
//    protected void onCreate(Bundle savedInstanceState) {
//        super.onCreate(savedInstanceState);
//        EdgeToEdge.enable(this);
//        setContentView(R.layout.activity_main);
//        if (!OpenCVLoader.initDebug()) {
//            Log.e("OpenCV", "Unable to load OpenCV!");
//        } else {
//            Log.d("OpenCV", "OpenCV loaded successfully!");
//        }
//    }
//}

//package com.example.myapplication;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;

import org.json.JSONArray;
import org.json.JSONObject;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    private ImageView scannedImage;
    private TextRecognizer recognizer;
    private TextView statusText;
    private Bitmap lastCapturedBitmap;
    private static final int CAMERA_PERMISSION_CODE = 101;
    private LogoClassifier logoClassifier;

    // For ORB image-based matching
    private ActivityResultLauncher<Void> cameraLauncher;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        File libFile = new File(getApplicationInfo().nativeLibraryDir, "libopencv_java4.so");
        Log.d("OpenCV", "Library present at runtime: " + libFile.exists());
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        scannedImage = findViewById(R.id.scanned_image);
        statusText = findViewById(R.id.status);
        Button btnCapture = findViewById(R.id.scan);

        recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);

        try {
            logoClassifier = new LogoClassifier(getAssets());
            Log.d("TFLite", "✅ Model loaded");
        } catch (IOException e) {
            Log.e("TFLite", "❌ Failed to load model: " + e.getMessage());
        }

        cameraLauncher = registerForActivityResult(
                new ActivityResultContracts.TakePicturePreview(),
                bitmap -> {
                    if (bitmap != null) {
                        scannedImage.setImageBitmap(bitmap);
                        lastCapturedBitmap = bitmap;
                        processImage(bitmap);
                    }
                });

        btnCapture.setOnClickListener(v -> {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);
            } else {
                cameraLauncher.launch(null);
            }
        });
    }
    private void processImage(Bitmap bitmap) {
        InputImage image = InputImage.fromBitmap(bitmap, 0);

        recognizer.process(image)
                .addOnSuccessListener(visionText -> {
                    String detectedText = visionText.getText();
                    checkWordsInJson(detectedText);
                })
                .addOnFailureListener(e -> {
                    e.printStackTrace();
                    Toast.makeText(this, "Failed to read text", Toast.LENGTH_SHORT).show();
                });
    }

    private void checkWordsInJson(String detectedText) {
        List<String> words = loadJson();
        List<String> found = new ArrayList<>();

//        for (String word : words) {
//            if (detectedText.toLowerCase().contains(word.toLowerCase())) {
//                found.add(word);
//            }
//        }

        String cleanText = detectedText.toLowerCase()
                .replaceAll("[^a-z0-9 ]", " ") // remove punctuation
                .replaceAll("\\s+", " ");       // normalize spaces

        for (String word : words) {
            String cleanWord = word.toLowerCase().replaceAll("[^a-z0-9 ]", " ");
            if (cleanText.contains(cleanWord)) {
                found.add(word);
            }
        }

        if (!found.isEmpty()) {
            String result = "✅ Match found: " + String.join(", ", found);
            statusText.setText(result);
            Toast.makeText(this, "Found: " + String.join(", ", found), Toast.LENGTH_LONG).show();
        } else {
//            statusText.setText("❌ Match not found.");
            Toast.makeText(this, "🔍 No text match found. Checking logo...", Toast.LENGTH_SHORT).show();
            // ❌ If text didn’t match, try logo match via ORB
            statusText.setText("🔍 No text match found. Checking logo...");
            if (logoClassifier != null && lastCapturedBitmap != null) {
                String prediction = logoClassifier.classify(lastCapturedBitmap);
                statusText.setText("🧠 Predicted: " + prediction);
                Toast.makeText(this, prediction, Toast.LENGTH_LONG).show();
            } else {
                statusText.setText("⚠️ Model not ready or no image");
            }

        }
    }

    private List<String> loadJson() {
        List<String> words = new ArrayList<>();
        try {
            InputStream is = getAssets().open("keywords.json");
            int size = is.available();
            byte[] buffer = new byte[size];
            is.read(buffer);
            is.close();
            String json = new String(buffer, "UTF-8");

            JSONObject jsonObject = new JSONObject(json);
            JSONArray jsonArray = jsonObject.getJSONArray("brands");

            for (int i = 0; i < jsonArray.length(); i++) {
                words.add(jsonArray.getString(i));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return words;
    }
    private String readStream(InputStream is) throws Exception {
        StringBuilder sb = new StringBuilder();
        byte[] buffer = new byte[1024];
        int len;
        while ((len = is.read(buffer)) != -1) {
            sb.append(new String(buffer, 0, len, StandardCharsets.UTF_8));
        }
        is.close();
        return sb.toString();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                cameraLauncher.launch(null);
            } else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show();
            }
        }
    }
}
