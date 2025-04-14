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

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.AbstractExecutorService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import be.tarsos.dsp.AudioDispatcher;
import be.tarsos.dsp.AudioEvent;
import be.tarsos.dsp.AudioProcessor;
import be.tarsos.dsp.SpectralPeakProcessor;
import be.tarsos.dsp.io.TarsosDSPAudioFormat;
import be.tarsos.dsp.io.UniversalAudioInputStream;
import be.tarsos.dsp.mfcc.MFCC;
import be.tarsos.dsp.util.fft.FFT;

import be.tarsos.dsp.util.ChromaProcessor;
import be.tarsos.dsp.util.SpectralContrast;
import be.tarsos.dsp.util.SpectralFlatness;
import be.tarsos.dsp.util.SpectralRolloff;
import be.tarsos.dsp.util.SpectralSpread;
import be.tarsos.dsp.ZeroCrossingRate;
import be.tarsos.dsp.util.RMS;

import java.io.ByteArrayInputStream;
import java.io.IOException;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    private static final int RECORD_AUDIO_PERMISSION_REQUEST_CODE = 101;
    private static final int N_MFCC = 40;
    private static final int N_CHROMA = 12;
    private static final int N_BANDS = 6;

    private static final int WINDOW = 2048;
    private static final int OVERLAP = WINDOW - 512;

    private AudioRecord audioRecord;
    private byte[] audioData;
    private final int recordingDurationMillis = 5000; // 5 seconds
    private boolean isRecording = false;
    private Thread recordingThread;

    private Module mModule;

    // Audio recording parameters
    private static final int SAMPLE_RATE = 48000; // Sample rate in Hz
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



    public class ExtractedFeatures {
        private final double[][] mfccs;
        private final double[][] contrast;
        private final double[][] chroma;
        private final double[] zcr;
        private final double[] rms;
        private final double[] rolloff;
        private final double[] bandwidth;
        private final double[] flatness;

        public ExtractedFeatures(double[][] mfccs, double[][] contrast, double[][] chroma, double[] zcr, double[] rms, double[] rolloff, double[] bandwidth, double[] flatness) {
            this.mfccs = mfccs;
            this.contrast = contrast;
            this.chroma = chroma;
            this.zcr = zcr;
            this.rms = rms;
            this.rolloff = rolloff;
            this.bandwidth = bandwidth;
            this.flatness = flatness;
        }

        public double[][] getMfccs() {
            return mfccs;
        }

        public double[][] getContrast() {
            return contrast;
        }

        public double[][] getChroma() {
            return chroma;
        }

        public double[] getZcr() {
            return zcr;
        }

        public double[] getRms() {
            return rms;
        }

        public double[] getRolloff() {
            return rolloff;
        }

        public double[] getBandwidth() {
            return bandwidth;
        }

        public double[] getFlatness() {
            return flatness;
        }

        @Override
        public String toString() {
            return "ExtractedFeatures{" +
                    "mfccs=" + Arrays.deepToString(mfccs) +
                    ", contrast=" + Arrays.deepToString(contrast) +
                    ", chroma=" + Arrays.deepToString(chroma) +
                    ", zcr=" + Arrays.toString(zcr) +
                    ", rms=" + Arrays.toString(rms) +
                    ", rolloff=" + Arrays.toString(rolloff) +
                    ", bandwidth=" + Arrays.toString(bandwidth) +
                    ", flatness=" + Arrays.toString(flatness) +
                    '}';
        }
    }


    // Executor for feature extraction and prediction
    private ExecutorService predictionExecutor = Executors.newSingleThreadExecutor();
    private ExecutorService featureExtractionExecutor = Executors.newSingleThreadExecutor();
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

    private double[] floatArrayToDoubleArray(float[] floatArray) {
        double[] doubleArray = new double[floatArray.length];
        for (int i = 0; i < floatArray.length; i++) {
            doubleArray[i] = (double) floatArray[i];
        }
        return doubleArray;
    }
    private void processAudioChunk(byte[] audioChunk) {
        Log.d(TAG, "Processing new audio chunk.");
        // You can add your code to extract features here
        extractAudioFeatures(audioChunk);
    }


    private void extractAudioFeatures(byte[] finalAudioData) {
        featureExtractionExecutor.submit(() -> {
            try {
                // Create a ByteArrayInputStream from the finalAudioData
                ByteArrayInputStream bais = new ByteArrayInputStream(finalAudioData);

                // Original Audio Format
                TarsosDSPAudioFormat originalAudioFormat = new TarsosDSPAudioFormat(SAMPLE_RATE, 16, 1, true, false);
                float targetSampleRate = 22050.0f;

                // Create a UniversalAudioInputStream from the ByteArrayInputStream and AudioFormat
                UniversalAudioInputStream uais = new UniversalAudioInputStream(bais, originalAudioFormat);

                // MFCC parameters for the resampled audio
                int amountOfMelFilters = 40;
                float lowerFilterFreq = 20.0f;
                float upperFilterFreq = targetSampleRate / 2.0f;
                MFCC mfcc = new MFCC(AUDIO_BUFFER_SIZE, targetSampleRate, N_MFCC, amountOfMelFilters, lowerFilterFreq, upperFilterFreq);
                final int MAX_FRAMES = SAMPLE_RATE * 5 / OVERLAP; // Calculated based on audio settings
                double[][] mfccs = new double[MAX_FRAMES][N_MFCC];
                final Integer[] mfccFrameCount = {0};

                //Spectral contrast parameters
                final int N_BANDS = 6;
                double[][] contrast = new double[MAX_FRAMES][N_BANDS + 1];
                final Integer[] contrastFrameCount = {0};
                SpectralContrast spectralContrast = new SpectralContrast(AUDIO_BUFFER_SIZE, targetSampleRate);

                //Chroma parameters
                final int N_CHROMA = 12;
                double[][] chroma = new double[MAX_FRAMES][N_CHROMA];
                final Integer[] chromaFrameCount = {0};
                ChromaProcessor chromaProcessor = new ChromaProcessor(AUDIO_BUFFER_SIZE, targetSampleRate);

                //ZCR parameters
                double[] zcr = new double[MAX_FRAMES];
                final Integer[] zcrFrameCount = {0};
                ZeroCrossingRate zeroCrossingRate = new ZeroCrossingRate(AUDIO_BUFFER_SIZE, targetSampleRate);

                //RMS parameters
                double[] rms = new double[MAX_FRAMES];
                final Integer[] rmsFrameCount = {0};
                RMS rmsProcessor = new RMS(AUDIO_BUFFER_SIZE, targetSampleRate);

                //rolloff parameters
                double[] rolloff = new double[MAX_FRAMES];
                final Integer[] rolloffFrameCount = {0};
                SpectralRolloff spectralRolloff = new SpectralRolloff(AUDIO_BUFFER_SIZE, targetSampleRate);

                //bandwidth parameters
                double[] bandwidth = new double[MAX_FRAMES];
                final Integer[] bandwidthFrameCount = {0};
                SpectralSpread spectralSpread = new SpectralSpread(AUDIO_BUFFER_SIZE, targetSampleRate);

                //flatness parameters
                double[] flatness = new double[MAX_FRAMES];
                final Integer[] flatnessFrameCount = {0};
                SpectralFlatness spectralFlatness = new SpectralFlatness(AUDIO_BUFFER_SIZE, targetSampleRate);


                    // Audio dispatcher for the resampled audio
                AudioDispatcher resampledDispatcher = new AudioDispatcher(uais, AUDIO_BUFFER_SIZE, OVERLAP);
                resampledDispatcher.addAudioProcessor(mfcc);
                resampledDispatcher.addAudioProcessor(spectralContrast);
                resampledDispatcher.addAudioProcessor(chromaProcessor);
                resampledDispatcher.addAudioProcessor(zeroCrossingRate);
                resampledDispatcher.addAudioProcessor(rmsProcessor);
                resampledDispatcher.addAudioProcessor(spectralRolloff);
                resampledDispatcher.addAudioProcessor(spectralSpread);
                resampledDispatcher.addAudioProcessor(spectralFlatness);
                resampledDispatcher.addAudioProcessor(new AudioProcessor()){
                    @Override
                        public boolean process(AudioEvent audioEvent) {
                            float[] currentMfccs = mfcc.getMFCC();
                            if (currentMfccs != null) {
                                if (mfccFrameCount[0] < MAX_FRAMES) {
                                    mfccs[mfccFrameCount[0]++] = floatArrayToDoubleArray(currentMfccs);
                                } else {
                                    Log.e(TAG, "Exceeded maximum number of MFCC frames. Increase MAX_FRAMES");
                                }
                            }
                            double[] currentContrast = spectralContrast.getSpectralContrast();
                            if (currentContrast != null) {
                                if (contrastFrameCount[0] < MAX_FRAMES) {
                                    contrast[contrastFrameCount[0]++] = currentContrast;
                                } else {
                                    Log.e(TAG, "Exceeded maximum number of contrast frames. Increase MAX_FRAMES");
                                }
                            }
                            float[] currentChroma = chromaProcessor.getCHROMA();
                            if (currentChroma != null) {
                                if (chromaFrameCount[0] < MAX_FRAMES) {
                                    chroma[chromaFrameCount[0]++] = floatArrayToDoubleArray(currentChroma);
                                } else {
                                    Log.e(TAG, "Exceeded maximum number of chroma frames. Increase MAX_FRAMES");
                                }
                            }
                            float currentZcr = zeroCrossingRate.getZcr();
                            if (zcrFrameCount[0] < MAX_FRAMES) {
                                zcr[zcrFrameCount[0]++] = currentZcr;
                            } else {
                                Log.e(TAG, "Exceeded maximum number of zcr frames. Increase MAX_FRAMES");
                            }
                            float currentRms = rmsProcessor.getRms();
                            if (rmsFrameCount[0] < MAX_FRAMES) {
                                rms[rmsFrameCount[0]++] = currentRms;
                            } else {
                                Log.e(TAG, "Exceeded maximum number of rms frames. Increase MAX_FRAMES");
                            }
                            float currentRolloff = spectralRolloff.getSpectralRolloff();
                            if (rolloffFrameCount[0] < MAX_FRAMES) {
                                rolloff[rolloffFrameCount[0]++] = currentRolloff;
                            } else {
                                Log.e(TAG, "Exceeded maximum number of rolloff frames. Increase MAX_FRAMES");
                            }
                            float currentBandwidth = spectralSpread.getSpectralSpread();
                            if (bandwidthFrameCount[0] < MAX_FRAMES) {
                                bandwidth[bandwidthFrameCount[0]++] = currentBandwidth;
                            } else {
                                Log.e(TAG, "Exceeded maximum number of bandwidth frames. Increase MAX_FRAMES");
                            }
                            float currentFlatness = spectralFlatness.getSpectralFlatness();
                            if (flatnessFrameCount[0] < MAX_FRAMES) {
                                flatness[flatnessFrameCount[0]++] = currentFlatness;
                            } else {
                                Log.e(TAG, "Exceeded maximum number of flatness frames. Increase MAX_FRAMES");
                            }
                            return true;
                    }

                    @Override
                    public void processingFinished() {
                        Log.d(TAG, "Audio processing finished.");
                    }
                });

                resampledDispatcher.run();


                double[][] mfccsFinal = new double[mfccFrameCount[0]][];
                for(int i = 0; i < mfccFrameCount[0]; i++){
                    mfccsFinal[i] = mfccs[i];
                }

                mainHandler.post(() -> {
                    runPrediction(new ExtractedFeatures(mfccsFinal));
                });

            } catch (Exception e) {
                // Handle any exceptions that occur during processing
                Log.e(TAG, "Error extracting audio features", e);
                throw new RuntimeException(e);
            }
        });

    private String makePredictionWithModel(ExtractedFeatures extractedFeatures) {

        double[][] featureVector = extractedFeatures.getMfccs();

        // Load the model from assets
        try {
            if (mModule == null) {
                mModule = LiteModuleLoader.load("model.ptl");
            }
        } catch (Exception e) {
            Log.e(TAG, "Error loading model!", e);
            return "Error: Model not loaded!"; // Or handle the error appropriately
        }

        // Convert double[][] to Tensor
        float[] floatInput = new float[featureVector.length * featureVector[0].length];
        int k = 0;
        for (double[] doubles : featureVector) {
            for (double aDouble : doubles) {
                floatInput[k++] = (float) aDouble;
            }
        }
        long[] shape = new long[]{1, featureVector.length, featureVector[0].length};
        Tensor inputTensor = Tensor.fromBlob(floatInput, shape);


        // Run the model
        Tensor outputTensor = mModule.forward(IValue.from(inputTensor)).toTensor();
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

    private void runPrediction(ExtractedFeatures extractedFeatures) {
        // Check if there are features to process
        if (extractedFeatures == null || extractedFeatures.getMfccs().length == 0) {
            Log.e(TAG, "No features to run prediction on.");
            return;
        }

        predictionExecutor.submit(() -> {
            Log.d(TAG, "Running prediction on extracted features.");

            // TODO: Implement your prediction logic here

            // 1. Get the extracted features:
            //    The extractedFeatures list now contains one double[] (the mean MFCC vector) per audio clip.
            //    You can iterate through them, or just take the first/last element depending on your design.
            //    For example, let's assume you're only using the first (and only) mean MFCC vector in this example:
            double[][] featureVector = extractedFeatures.getMfccs();

            // 2. Prepare the features:
            //    You might need to convert the feature array to a format compatible with your prediction model.
            //    This may involve reshaping, scaling, or other preprocessing.
            //    This depends entirely on your prediction model's input requirements.

            // 3. Make a prediction:
            //    Use your prediction model (e.g., a pre-trained machine learning model) to make a prediction
            //    based on the audio features.
            //    This is where you would interface with your model's API.
            //    Here's a placeholder for that:

            String predictionResult = makePredictionWithModel(extractedFeatures);

            // 4. Handle the prediction result:
            //    Display the result to the user, log it, or take some other action based on the prediction.
            //    Since this is running in a background thread, you should use the mainHandler to update
            //    the UI or perform other tasks on the main thread.
            mainHandler.post(() -> {
                Log.d(TAG, "Prediction Result: " + predictionResult);
                // Update UI or perform other main-thread tasks here
            });
        });
    }
}