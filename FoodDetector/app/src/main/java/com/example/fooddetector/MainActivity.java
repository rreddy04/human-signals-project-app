package com.example.fooddetector;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
//import android.os.Handler;
//import android.os.Looper;
import android.util.Log;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.pytorch.Device;
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
//import java.util.ArrayList;
import java.util.Arrays;
//import java.util.List;
//import java.util.concurrent.AbstractExecutorService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

//import be.tarsos.dsp.AudioDispatcher;
//import be.tarsos.dsp.AudioEvent;
//import be.tarsos.dsp.AudioProcessor;
//import be.tarsos.dsp.SpectralPeakProcessor;
//import be.tarsos.dsp.io.TarsosDSPAudioFormat;
//import be.tarsos.dsp.io.UniversalAudioInputStream;
//import be.tarsos.dsp.mfcc.MFCC;
//import be.tarsos.dsp.util.fft.FFT;
//
//
//import java.io.ByteArrayInputStream;
//import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final int RECORD_AUDIO_PERMISSION_REQUEST_CODE = 101;

    private static final String MODEL_PATH = "modelcnn.ptl";
    private AudioRecord audioRecord;
    private byte[] audioData;
    private byte[] finalAudioData;
    private boolean isRecording = false;

    private Module model;
    private TextView label;

    // Audio recording parameters
    private static final int SAMPLE_RATE = 48000; // Sample rate in Hz
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    private static final int AUDIO_BUFFER_SIZE = 1024; // Example buffer size, adjust as needed


    // Declare the ActivityResultLauncher for requesting permission
    private final ActivityResultLauncher<String> requestPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) {
                    // Permission is granted. Continue the action or workflow in your app.
                    startRecording();
                } else {
                    // Explain to the user that the feature is unavailable because the
                    // features requires a permission that the user has denied.
                    Log.e(TAG, "Audio recording permission denied.");
                }
            });


    // Executor for feature extraction and prediction
    private final ExecutorService predictionExecutor = Executors.newSingleThreadExecutor();

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName);
             FileOutputStream os = new FileOutputStream(file)) {

            byte[] buffer = new byte[1024];
            int read;
            while ((read = is.read(buffer)) != -1) {
                os.write(buffer, 0, read);
            }
            os.flush();
        }

        return file.getAbsolutePath();
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        String modelPath = null;
        try {
            modelPath = assetFilePath(this, MODEL_PATH);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        Log.i("modelload", modelPath);

        try {
            if (model == null) {
                model = LiteModuleLoader.load(modelPath);
            }
        } catch (Exception e) {
            Log.e(TAG, "Error loading model!", e);
        }

        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);


        label = findViewById(R.id.label);
        label.setText("Initialized, prediction goes here");

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        checkAudioPermission();
    }

    private void checkAudioPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            // Permission already granted
            startRecording();
        } else {
            // Request permission
            requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO);
        }
    }

    private void startRecording() {
        // Calculate buffer size
        int bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
        if (bufferSize == AudioRecord.ERROR_BAD_VALUE || bufferSize == AudioRecord.ERROR) {
            Log.e(TAG, "Invalid buffer size");
            return;
        }

        // Create AudioRecord
        audioRecord = new AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize
        );

        // Check if AudioRecord was successfully created
        if (audioRecord.getState() == AudioRecord.STATE_UNINITIALIZED) {
            Log.e(TAG, "Failed to initialize AudioRecord");
            return;
        }

        // Initialize audioData
        // 5 seconds
        int recordingDurationMillis = 5000;
        int maxBufferSize = SAMPLE_RATE * 2 * (recordingDurationMillis / 1000); // 2 bytes per sample (16-bit),
        audioData = new byte[maxBufferSize];
        // Start recording
        audioRecord.startRecording();
        isRecording = true;
        Log.d(TAG, "Recording started");

        // Process 5 seconds of audio
        Thread recordingThread = new Thread(() -> {
            int bytesRead = 0;
            byte[] fiveSecondsAudioData = new byte[maxBufferSize];
            int currentPosition = 0;
            while (isRecording) {
                int result = audioRecord.read(audioData, 0, audioData.length);
                if (result > 0) {
                    System.arraycopy(audioData, 0, fiveSecondsAudioData, currentPosition, result);
                    currentPosition += result;
                    if (currentPosition >= maxBufferSize) {
                        // Process 5 seconds of audio
                        finalAudioData = Arrays.copyOf(fiveSecondsAudioData, maxBufferSize);
                        currentPosition = 0;
                        Log.d(TAG, "Finished reading audio data.");
                        predictionExecutor.submit(() -> processAudioChunk(finalAudioData));
                    }

                } else {
                    Log.e(TAG, "Error reading audio data, code: " + result);
                }
            }
        });
        recordingThread.start();
    }

    private void processAudioChunk(byte[] audioChunk) {
        Log.d("prediction", "Processing new audio chunk.");
        makePredictionWithModel(convertByteArrayToFloatArray(audioChunk));
    }

    private float[] convertByteArrayToFloatArray(byte[] byteArray) {
        // Assuming 16-bit PCM encoding
        if (byteArray.length % 2 != 0) {
            throw new IllegalArgumentException("ByteArray length must be a multiple of 2 for 16-bit PCM");
        }

        int numSamples = byteArray.length / 2; // 2 bytes per sample for 16-bit
        float[] floatArray = new float[numSamples];

        ByteBuffer buffer = ByteBuffer.wrap(byteArray);
        buffer.order(ByteOrder.LITTLE_ENDIAN); // Assuming little-endian encoding

        for (int i = 0; i < numSamples; i++) {
            short sample = buffer.getShort();
            floatArray[i] = sample / 32768.0f; // Normalize to [-1.0, 1.0]
        }

        return floatArray;
    }

    private void makePredictionWithModel(float[] featureVector) {
        // Convert float[] to Tensor
        Log.i("prediction", "thinking");

        if (featureVector.length != 240000) {
            Log.e(TAG, "Unexpected feature vector size: " + featureVector.length);
            return;
        }

        long[] shape = new long[]{1, 1, 240000};
        Tensor inputTensor = Tensor.fromBlob(featureVector, shape);
        Tensor outputTensor = null;

        // Run the model
        Log.i("prediction", "ambatu predict");
        try {
            Log.i("prediction", "Input tensor shape: " + Arrays.toString(inputTensor.shape()));
            outputTensor = model.forward(IValue.from(inputTensor)).toTensor();
            Log.i("prediction", "got output");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        float[] scores = outputTensor.getDataAsFloatArray();
        Log.i("prediction", Arrays.toString(scores));


        // Find the predicted class (the one with the highest score)
        int predictedClass = 0;
        float maxScore = -Float.MAX_VALUE;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                predictedClass = i;
            }
        }

        // Define class labels (replace with your actual labels)
        String[] classLabels = {"Carrot", "PB", "Sandwich", "Rice", "Fries", "Grapes", "Banana", "Apple", "Chips", "Popcorn"};

        // Return the predicted class
        label.setText("Prediction: " + classLabels[predictedClass] + " (Score: " + maxScore + ")");
        Log.i("prediction", "Prediction: " + classLabels[predictedClass] + " (Score: " + maxScore + ")");
    }
}