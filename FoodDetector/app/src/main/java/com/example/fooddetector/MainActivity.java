package com.example.fooddetector;

import android.Manifest;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
//import android.os.Handler;
//import android.os.Looper;
import android.util.Log;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

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
    private boolean isRecording = false;

    private Module model;

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



//    public class ExtractedFeatures {
//        private final double[][] mfccs;
//        private final double[][] contrast;
//        private final double[][] chroma;
//        private final double[] zcr;
//        private final double[] rms;
//        private final double[] rolloff;
//        private final double[] bandwidth;
//        private final double[] flatness;
//
//        public ExtractedFeatures(double[][] mfccs, double[][] contrast, double[][] chroma, double[] zcr, double[] rms, double[] rolloff, double[] bandwidth, double[] flatness) {
//            this.mfccs = mfccs;
//            this.contrast = contrast;
//            this.chroma = chroma;
//            this.zcr = zcr;
//            this.rms = rms;
//            this.rolloff = rolloff;
//            this.bandwidth = bandwidth;
//            this.flatness = flatness;
//        }
//
//        public double[][] getMfccs() {
//            return mfccs;
//        }
//
//        public double[][] getContrast() {
//            return contrast;
//        }
//
//        public double[][] getChroma() {
//            return chroma;
//        }
//
//        public double[] getZcr() {
//            return zcr;
//        }
//
//        public double[] getRms() {
//            return rms;
//        }
//
//        public double[] getRolloff() {
//            return rolloff;
//        }
//
//        public double[] getBandwidth() {
//            return bandwidth;
//        }
//
//        public double[] getFlatness() {
//            return flatness;
//        }
//
//        @Override
//        public String toString() {
//            return "ExtractedFeatures{" +
//                    "mfccs=" + Arrays.deepToString(mfccs) +
//                    ", contrast=" + Arrays.deepToString(contrast) +
//                    ", chroma=" + Arrays.deepToString(chroma) +
//                    ", zcr=" + Arrays.toString(zcr) +
//                    ", rms=" + Arrays.toString(rms) +
//                    ", rolloff=" + Arrays.toString(rolloff) +
//                    ", bandwidth=" + Arrays.toString(bandwidth) +
//                    ", flatness=" + Arrays.toString(flatness) +
//                    '}';
//        }
//    }


    // Executor for feature extraction and prediction
    private final ExecutorService predictionExecutor = Executors.newSingleThreadExecutor();
//    private ExecutorService featureExtractionExecutor = Executors.newSingleThreadExecutor();
//    private Handler mainHandler = new Handler(Looper.getMainLooper());

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        try {
            if (model == null) {
                model = LiteModuleLoader.load(MODEL_PATH);
            }
        } catch (Exception e) {
            Log.e(TAG, "Error loading model!", e);
        }

        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

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
                        byte[] finalAudioData = Arrays.copyOf(fiveSecondsAudioData, maxBufferSize);
                        currentPosition = 0;
                        predictionExecutor.submit(() -> processAudioChunk(finalAudioData));
                    }

                } else {
                    Log.e(TAG, "Error reading audio data, code: " + result);
                }
            }
            Log.d(TAG, "Finished reading audio data.");
        });
        recordingThread.start();
    }

    private void processAudioChunk(byte[] audioChunk) {
        Log.d(TAG, "Processing new audio chunk.");
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

    private String makePredictionWithModel(float[] featureVector) {
        // Convert float[] to Tensor
        long[] shape = new long[]{1, 240000};
        Tensor inputTensor = Tensor.fromBlob(featureVector, shape);

        // Run the model
        Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor();
        float[] scores = outputTensor.getDataAsFloatArray();

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
        String[] classLabels = {"Food", "Not Food"};

        // Return the predicted class
        return "Prediction: " + classLabels[predictedClass] + " (Score: " + maxScore + ")";
    }
}