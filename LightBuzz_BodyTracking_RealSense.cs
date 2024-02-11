using UnityEngine;
using LightBuzz.BodyTracking; // Import LightBuzz.BodyTracking namespace
using System.Collections.Generic;
using System.IO;
using System;
using System.Globalization;
using UnityEngine.UIElements;
using System.Linq;
using Unity.VisualScripting;
using Newtonsoft.Json;
using LightBuzz.BodyTracking.Video;
using UnityEngine.Profiling;
using static TMPro.SpriteAssetUtilities.TexturePacker_JsonArray;
using System.IO.Ports;
using System.Threading;

public class LightBuzz_BodyTracking_RealSense : MonoBehaviour
{
    [SerializeField] private DeviceConfiguration _configuration;
    [SerializeField] private LightBuzzViewer _colorViewer;
    [SerializeField] private LightBuzzViewer _depthViewer;

    private Sensor _sensor;
    private readonly VideoRecorder _recorder = new VideoRecorder();

    // Editable session number
    private string file = "lightbuzz_table_6"; // You can change this number as needed

    private string basePath = @"C:\Users\tejwa\adaptive-language-controller\sessions\";
    private string posesPath;
    private string audioPath;
    private string videoMetadataPath;
    private AudioClip audioClip;
    private bool isRecordingAudio = false;
    private SerialPort serialPort = new SerialPort("COM5", 9600);
    private Thread serialThread;
    private bool isReadingSerial = false;

    private string latestForceSensorReading = "N/A"; // To store the latest reading

    private void Start()
    {
        // Construct the file paths using the session number
        posesPath = Path.Combine(basePath + file, "poses.jsonl");
        videoMetadataPath = Path.Combine(basePath + file, "video metadata");
        audioPath = Path.Combine(basePath + file, "audio.wav");

        // Ensure the directories exist
        Directory.CreateDirectory(Path.GetDirectoryName(posesPath));
        Directory.CreateDirectory(videoMetadataPath);

        Debug.Log("starting sensor");
        _sensor = Sensor.Create(_configuration);
        _sensor?.Open();

        if (_sensor == null || !_sensor.IsOpen)
        {
            Debug.LogError("Sensor is not open. Check the configuration settings.");
        }

        // Initialize the recorder with the new path
        _recorder.Settings = new VideoRecordingSettings
        {
            Path = videoMetadataPath, // Use the new path for video metadata
            RecordColor = true,
            RecordDepth = true,
            RecordBody = true,
            Smoothing = _sensor.Configuration.Smoothing,
            FrameRate = _sensor.FPS,
            ColorResolution = new Size(_sensor.Width, _sensor.Height),
            DepthResolution = new Size(_sensor.Width, _sensor.Height),
            ColorFormat = _sensor.ColorFormat
        };

        // Subscribe to the recorder events.
        _recorder.OnRecordingStarted += Recorder_OnRecordingStarted;
        _recorder.OnRecordingStopped += Recorder_OnRecordingStopped;
        _recorder.OnRecordingCanceled += Recorder_OnRecordingCanceled;
        _recorder.OnRecordingCompleted += Recorder_OnRecordingCompleted;

        _recorder.Start();

        StartAudioRecording();
        StartSerialReading();
    }

    private void OnDestroy()
    {
        if (_recorder != null)
        {
            // Unsubscribe from the events and dispose the recorder.
            _recorder.OnRecordingStarted -= Recorder_OnRecordingStarted;
            _recorder.OnRecordingStopped -= Recorder_OnRecordingStopped;
            _recorder.OnRecordingCanceled -= Recorder_OnRecordingCanceled;
            _recorder.OnRecordingCompleted -= Recorder_OnRecordingCompleted;
            _recorder.Stop();
            _recorder.Dispose();
            // Close and dispose the camera.
            _sensor.Close();
            _sensor.Dispose();

            StopAudioRecording();
            StopSerialReading();

            _sensor = null;
        }
    }


    private void Update()
    {
        if (_sensor == null || !_sensor.IsOpen) return;

        FrameData frame = _sensor.Update();

        if (frame == null) return;
        var bodies = frame.BodyData;

        if (_recorder.IsRecording)
        {
            _recorder.Update(frame);
        }

        // Check for Enter key press to trigger OnDestroy and exit application
        if (Input.GetKeyDown(KeyCode.Return)) // KeyCode.Return represents the Enter key
        {
            OnDestroy(); // Manually call OnDestroy to ensure cleanup
            Application.Quit(); // Quit the application
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false; // If running in the Unity Editor, stop the play mode
#endif
        }

        List<Dictionary<string, object>> poses = new List<Dictionary<string, object>>();

        //Convert timestamps to Unix time
        DateTime dateTime = DateTime.UtcNow;
        double unixTimeSeconds = (dateTime - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).TotalSeconds;

        var skeleton = new Dictionary<string, object>
    {
        {"timestamp", unixTimeSeconds},
        {"skeletons", new List<Dictionary<string, object>>() }
    };

        foreach (var body in bodies)
        {
            var jointsData = new Dictionary<string, object>
        {
            {"user_id", body.ID}
        };

            foreach (JointType jointType in Enum.GetValues(typeof(JointType)))
            {
                if (body.Joints.ContainsKey(jointType))
                {
                    var joint = body.Joints[jointType];
                    jointsData[jointType.ToString()] = new Dictionary<string, object>
                {
                    {"confidence", joint.Confidence},
                    { "pos2D", new List<float> { joint.Position2D.X, joint.Position2D.Y} },
                    { "pos3D", new List<float> { joint.Position3D.X, joint.Position3D.Y, joint.Position3D.Z } },
                    { "orientation", new List<float> { joint.Orientation.X, joint.Orientation.Y, joint.Orientation.Z, joint.Orientation.W } },
                    {"tracking_state", joint.TrackingState.ToString() },
                    {"type", joint.Type.ToString() }
                };
                }
            }

            ((List<Dictionary<string, object>>)skeleton["skeletons"]).Add(jointsData);
        }

        string json = JsonConvert.SerializeObject(skeleton);
        File.AppendAllText(posesPath, json + Environment.NewLine);

        //Debug.Log($"Latest Force Sensor Reading: {latestForceSensorReading}");

        if (frame != null)
        {
            _colorViewer.Load(frame);
            _depthViewer.Load(frame);
        }
    }


    private void Recorder_OnRecordingStarted()
    {
        Debug.Log("Recording started");
    }
    private void Recorder_OnRecordingStopped()
    {
        Debug.Log("Recording stopped");
    }
    private void Recorder_OnRecordingCanceled()
    {
        Debug.Log("Recording canceled");
    }
    private void Recorder_OnRecordingCompleted()
    {
        Debug.Log("Recording completed");
    }
    // This is a custom action (e.g. a button click)
    // that starts and stops the recording.
    public void OnRecord_Click()
    {
        if (!_recorder.IsRecording)
        {
            _recorder.Start();
            Debug.Log("Recording toggled (start)");
        }
        else
        {
            _recorder.Stop();
            Debug.Log("Recording toggled (stop)");
        }
    }
    private void StartAudioRecording()
    {
        var maxRecordingTime = 3599; //3599 seconds
        var samplingRate = 44100; //default sampling rate
        audioClip = Microphone.Start(null, false, maxRecordingTime, samplingRate); // Start with no fixed length
        isRecordingAudio = true;
    }

    private void StopAudioRecording()
    {
        if (isRecordingAudio)
        {
            var position = Microphone.GetPosition(null);
            var soundData = new float[audioClip.samples * audioClip.channels];
            audioClip.GetData(soundData, 0);
            var newData = new float[position * audioClip.channels];
            Array.Copy(soundData, newData, newData.Length);
            var newClip = AudioClip.Create(audioClip.name, position, audioClip.channels, audioClip.frequency, false);
            newClip.SetData(newData, 0);

            Microphone.End(null); // Stop the microphone
            isRecordingAudio = false;
            WAVUtility.Save(newClip, audioPath); // Save the trimmed audio clip
            Debug.Log($"Audio saved to {audioPath}");
        }
    }


    private void StartSerialReading()
    {
        try
        {
            serialPort.Open(); // Attempt to open the serial port
            if (serialPort.IsOpen)
            {
                isReadingSerial = true;
                serialThread = new Thread(ReadSerial);
                serialThread.Start();
                Debug.Log("Successfully connected to the serial port.");
            }
            else
            {
                Debug.LogError("Failed to open the serial port.");
            }
        }
        catch (Exception ex)
        {
            // Log any exceptions that occur during the attempt to open the serial port
            Debug.LogError($"Failed to open the serial port: {ex.Message}");
        }
    }

    private void StopSerialReading()
    {
        if (isReadingSerial && serialPort != null)
        {
            isReadingSerial = false;
            serialPort.Close();
            serialThread.Join();
        }
    }

    private void ReadSerial()
    {
        string forceDataPath = Path.Combine(basePath, file, "force_data.csv");
        using (StreamWriter writer = new StreamWriter(forceDataPath, true)) // Ensure the file is appended to, not overwritten
        {
            writer.WriteLine("timestamp,reading"); // Write the header if needed
            while (isReadingSerial)
            {
                try
                {
                    string line = serialPort.ReadLine();

                    //Convert timestamps to Unix time
                    DateTime dateTime = DateTime.UtcNow; // Capture the current UTC time
                    double unixTimeSeconds = (dateTime - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc)).TotalSeconds; // Convert to Unix time

                    latestForceSensorReading = line; // Update the latest reading

                    // Write the Unix timestamp and the reading to the file
                    writer.WriteLine($"{unixTimeSeconds},{line}");
                }
                catch (TimeoutException) { }
            }
        }
    }

}
public static class WAVUtility
{
    public static void Save(AudioClip audioClip, string filePath)
    {
        var wavFile = AudioClipToWAV(audioClip);
        File.WriteAllBytes(filePath, wavFile);
    }

    public static byte[] AudioClipToWAV(AudioClip audioClip)
    {
        using (var memoryStream = new MemoryStream())
        {
            using (var writer = new BinaryWriter(memoryStream))
            {
                ushort blockAlign = (ushort)(audioClip.channels * 2);
                ushort bitsPerSample = 16;
                string riff = "RIFF";
                string wave = "WAVE";
                string fmt = "fmt ";
                string data = "data";
                int subChunk1Size = 16;
                int sampleRate = audioClip.frequency;
                int byteRate = sampleRate * blockAlign;
                int subChunk2Size = audioClip.samples * blockAlign;
                int chunkSize = 4 + (8 + subChunk1Size) + (8 + subChunk2Size);

                // Write header
                writer.Write(riff.ToCharArray());
                writer.Write(chunkSize);
                writer.Write(wave.ToCharArray());
                writer.Write(fmt.ToCharArray());
                writer.Write(subChunk1Size);
                writer.Write((ushort)1); // PCM format
                writer.Write((ushort)audioClip.channels);
                writer.Write(sampleRate);
                writer.Write(byteRate);
                writer.Write(blockAlign);
                writer.Write(bitsPerSample);
                writer.Write(data.ToCharArray());
                writer.Write(subChunk2Size);

                // Convert audio clip data to WAV data
                float[] samples = new float[audioClip.samples * audioClip.channels];
                audioClip.GetData(samples, 0);
                foreach (var sample in samples)
                {
                    writer.Write((short)(sample * 32767));
                }
            }

            return memoryStream.ToArray();
        }
    }
}