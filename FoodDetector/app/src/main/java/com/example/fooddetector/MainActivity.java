package com.example.fooddetector;

import android.Manifest;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import be.tarsos.dsp.AudioDispatcher;
import be.tarsos.dsp.AudioEvent;
import be.tarsos.dsp.AudioProcessor;
import be.tarsos.dsp.SpectralPeakProcessor;
import be.tarsos.dsp.io.TarsosDSPAudioFormat;
import be.tarsos.dsp.io.UniversalAudioInputStream;
import be.tarsos.dsp.util.fft.FFT;

import java.io.ByteArrayInputStream;
import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final int RECORD_AUDIO_PERMISSION_REQUEST_CODE = 101;

    private AudioRecord audioRecord;
    private byte[] audioData;
    private final int recordingDurationMillis = 5000; // 5 seconds
    private boolean isRecording = false;
    private Thread recordingThread;

    // Audio recording parameters
    private static final int SAMPLE_RATE = 44100; // Sample rate in Hz
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    private static final int AUDIO_BUFFER_SIZE = 1024; // Example buffer size, adjust as needed
    private static final int OVERLAP = 512; //Example overlap

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
    // Array to store extracted features
    private List<double[]> extractedFeatures = new ArrayList<>();

    // Executor for feature extraction and prediction
    private ExecutorService predictionExecutor = Executors.newSingleThreadExecutor();

    private Handler mainHandler = new Handler(Looper.getMainLooper());

    @Override
    protected void onCreate(Bundle savedInstanceState) {
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
        int maxBufferSize = SAMPLE_RATE * 2 * (recordingDurationMillis / 1000); // 2 bytes per sample (16-bit),
        audioData = new byte[maxBufferSize];

        // Start recording
        audioRecord.startRecording();
        isRecording = true;
        Log.d(TAG, "Recording started");

        recordingThread = new Thread(() -> {
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
        // You can add your code to extract features here
        extractAudioFeatures(audioChunk);
        // ... Run prediction ...
        runPrediction(audioChunk);

    }

    private void extractAudioFeatures(byte[] audioChunk) {
        try {
            ByteArrayInputStream bais = new ByteArrayInputStream(audioChunk);
            TarsosDSPAudioFormat audioFormat = new TarsosDSPAudioFormat(SAMPLE_RATE, 16, 1, true, false);
            UniversalAudioInputStream uais = new UniversalAudioInputStream(bais, audioFormat);

            AudioDispatcher dispatcher = new AudioDispatcher(uais, AUDIO_BUFFER_SIZE, OVERLAP);
            FFT fft = new FFT(AUDIO_BUFFER_SIZE);

            dispatcher.addAudioProcessor(new AudioProcessor() {
                @Override
                public void processingFinished() {
                    // TODO: control what happens when processing is finished
                }

                @Override
                public boolean process(AudioEvent audioEvent) {
                    float[] audioFloatBuffer = audioEvent.getFloatBuffer();
                    float[] amplitudes = new float[audioFloatBuffer.length];
                    System.arraycopy(audioFloatBuffer, 0, amplitudes, 0, audioFloatBuffer.length);

                    fft.forwardTransform(amplitudes);

                    double[] featureVector = new double[amplitudes.length];

                    for (int i = 0; i < amplitudes.length; i++) {
                        featureVector[i] = amplitudes[i];
                    }

                    extractedFeatures.add(featureVector);
                    return true;
                }
            });
            dispatcher.run();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private void runPrediction(byte[] audioChunk) {
        // TODO: Implement your prediction logic here
        // 1. Get the extracted features: You can access them from the 'extractedFeatures' list.
        //    The extractAudioFeatures method add these to the extractedFeatures array.
        // 2. Prepare the features: You might need to convert the feature array to a format compatible with your prediction model.
        // 3. Make a prediction: Use your prediction model (e.g., a pre-trained machine learning model) to make a prediction based on the audio features.
        // 4. Handle the prediction result: Display the result to the user, log it, or take some other action based on the prediction.
        Log.d(TAG, "Running prediction on audio chunk.");
    }
}