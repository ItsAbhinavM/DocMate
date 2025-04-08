const { app, BrowserWindow, session } = require("electron/main");
const path = require("path");

process.env["ELECTRON_DISABLE_SECURITY_WARNINGS"] = "true";

const fs = require("fs");
const ffmpeg = require("fluent-ffmpeg");
const ffmpegPath = require("ffmpeg-static");
const sdk = require("microsoft-cognitiveservices-speech-sdk");

ffmpeg.setFfmpegPath(ffmpegPath);

const createWindow = () => {
    const win = new BrowserWindow({
        width: 800,
        height: 600,
        autoHideMenuBar: true,
        transparent: true,
        vibrancy: {
            theme: "light",
            effect: "acrylic",
            disableOnBlur: true,
        },
        //frame: false,
        webPreferences: {
            enableRemoteModule: true,
            nodeIntegration: true,
            preload: path.join(__dirname, "./preload.js"),
        },
        // webPreferences:{
        // 	preload: 'preload.js'
        // }
    });

    win.loadFile("index.html");

    session.defaultSession.setPermissionRequestHandler(
        (webContents, permission, callback) => {
            if (permission === "media") {
                callback(true); // Grant media permissions
            } else {
                callback(false);
            }
        },
    );
};

app.whenReady().then(() => {
    createWindow();

    app.on("activate", () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on("window-all-closed", () => {
    if (process.platform !== "darwin") {
        app.quit();
    }
});

("use strict");

//require("./App");

const { ipcMain } = require("electron");
const os = require("os");
const { env } = require("process");
// const getMyFriends = () => {
// 	return [
// 		{id: 6030, name: "Twilley"},
// 		{id: 967, name: "Wickkiser"},
// 		{id: 5073, name: "Essick"},
// 		{id: 8886, name: "Marotta"},
// 		{id: 7416, name: "Banh"}
// 	];
// }

// const sendMessage = (message) => {
// 	console.log(message);
// }

// ipcMain
// 	.on("main-window-ready", (e) => {
// 		console.log("Main window is ready");
// 	})
// 	.on("send-message", (e, message) => {
// 		sendMessage(message);
// 	})

// ipcMain.handle("get-friends", (e) => {
// 	return getMyFriends();
// })

ipcMain.on("activate-app", () => {
    console.log("Voice activation succesfull");
    if (win) {
        win.show();
    }
});
ipcMain.on("send-audio", async(event, buffer) => {
    const webmPath = path.join(__dirname, "temp.webm");
    const wavPath = path.join(__dirname, "temp.wav");

    fs.writeFileSync(webmPath, buffer);

    // Convert WebM to WAV
    convertToWav(webmPath, wavPath)
        .then(() => recognizeSpeech(wavPath))
        .then((text) => {
            console.log("Transcription complete:", text);
            event.sender.send("transcription-complete", text);
        })
        .catch((err) => {
            console.error("Error:", err);
        });
});

function convertToWav(inputPath, outputPath) {
    return new Promise((resolve, reject) => {
        ffmpeg(inputPath)
            .audioChannels(1)
            .audioFrequency(16000)
            .audioCodec("pcm_s16le")
            .format("wav")
            .on("end", () => resolve())
            .on("error", reject)
            .save(outputPath);
    });
}

function recognizeSpeech(filePath) {
    return new Promise((resolve, reject) => {
        const speechConfig = sdk.SpeechConfig.fromSubscription(
            env.AZURE_API_KEY,
            "centralindia",
        );
        const audioConfig = sdk.AudioConfig.fromWavFileInput(
            fs.readFileSync(filePath),
        );
        const recognizer = new sdk.SpeechRecognizer(speechConfig, audioConfig);

        recognizer.recognizeOnceAsync((result) => {
            if (result.reason === sdk.ResultReason.RecognizedSpeech) {
                resolve(result.text);
            } else {
                reject(result.errorDetails || "Speech not recognized.");
            }
        });
    });
}

// ipcMain.on("upload-file", (event, file) => {
//   const docsDir = path.join(os.homedir(), "docs");
//   const docsPath = path.join(docsDir, file.name);
//   if (!fs.existsSync(docsDir)) {
//     fs.mkdirSync(docsDir, { recursive: true });
//     console.log("Created docs folder at:", docsDir);
//   }
//   fs.writeFile(docsPath, Buffer.from(file.buffer), (err) => {
//     if (err) {
//       console.error("Failed to save file:", err);
//     } else {
//       console.log("File saved to:", docsPath);
//     }
//   });
// });