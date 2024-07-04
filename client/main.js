import { Tensor, InferenceSession } from "onnxruntime-web";
let intervalId = null;
let cameraOn = false;
function formatMemoryUsage(bytes){
    const mb = bytes /(1024*1024);
    return `${mb.toFixed(2)} MB`;
}
function isMemoryAvailable(){
    return performance && performance.memory && performance.memory.usedJSHeapSize;
}
if (isMemoryAvailable()) {
    console.log(`Memory used before: ${formatMemoryUsage(performance.memory.usedJSHeapSize)}`);
}
const session = await InferenceSession.create(
    "SCI_new.onnx"
);
document.getElementById('startBtn').addEventListener('click', () => {
    if (cameraOn) {
        stopCamera();
    } else {
        startCamera();
    }
});
function stopCamera() {
    const video = document.getElementById('video');
  
    if (intervalId !== null) {
        clearInterval(intervalId);
    }

    video.pause();
    video.srcObject.getTracks().forEach(track => track.stop());
    
    cameraOn = false;
    document.getElementById('startBtn').innerText = 'Continue';
  
}



async function startCamera() {
    cameraOn = true;
    document.getElementById('startBtn').innerText = 'Stop';
    const video = document.getElementById('video');

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.play();

        video.onloadedmetadata = () => {

            if (intervalId !== null) {
                clearInterval(intervalId);
            }

            intervalId = setInterval(() => {
                processFrame(video);
            }, 1000); // Call API every 1 seconds
        };
    } catch (err) {
        console.error('Error accessing the camera: ', err);
    }
}



async function processFrame(video) {
    const canvasInput = document.createElement('canvas');
    const ctxInput = canvasInput.getContext('2d');
    canvasInput.width = video.videoWidth;
    canvasInput.height = video.videoHeight;
    ctxInput.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    const src = cv.imread(canvasInput);
    const processedFrame = new cv.Mat();
    cv.cvtColor(src, processedFrame, cv.COLOR_RGBA2RGB, 0);

    const threshold = getThreshold(processedFrame);
    if (threshold < 8) {
        clearResponse();
        displayWarning("Face recognition failed due to insufficient lighting. Please move to a brighter area.");
    } else {
        clearWarning();
        if (threshold < 100) {
            await enhancedImage(ctxInput);
            
        } else{
            sendFrameToServer(canvasInput, 'no_enhance');
        }
    }
    if (isMemoryAvailable()) {
        console.log(`Memory used: ${formatMemoryUsage(performance.memory.usedJSHeapSize)}`);
    }
    src.delete();
    processedFrame.delete();
}

// LLIE
async function enhancedImage(ctxInput) {
    const imageData = ctxInput.getImageData(0, 0, video.videoWidth, video.videoHeight);
    const data = imageData.data;
    const imgArray = new Float32Array(video.videoWidth * video.videoHeight * 3);

    for (let i = 0; i < data.length; i += 4) {
        imgArray[i / 4 * 3] = data[i] / 255.0;     // R
        imgArray[i / 4 * 3 + 1] = data[i + 1] / 255.0; // G
        imgArray[i / 4 * 3 + 2] = data[i + 2] / 255.0; // B
    }

    const inputTensor = new Tensor('float32', imgArray, [1, 3, video.videoWidth, video.videoHeight]);

    const feeds = { 'modelInput': inputTensor };

    try {
        const inferenceOutput = await session.run(feeds);

        const outputData = inferenceOutput['modelOutput'].data;

        const reshapedOutput = new Float32Array(video.videoWidth * video.videoHeight * 3);
        for (let i = 0; i < outputData.length; i++) {
            reshapedOutput[i] = outputData[i];
        }

        const canvasData = new Uint8ClampedArray(video.videoWidth * video.videoHeight * 4);

        // Denormalize the values and arrange them in RGBA format
        let j = 0;
        for (let i = 0; i < reshapedOutput.length; i += 3) {
            canvasData[j] = Math.round(reshapedOutput[i] * 255);     // R
            canvasData[j + 1] = Math.round(reshapedOutput[i + 1] * 255); // G
            canvasData[j + 2] = Math.round(reshapedOutput[i + 2] * 255); // B
            canvasData[j + 3] = 255;                                  // A (fully opaque)
            j += 4;
        }
        const imageData = new ImageData(canvasData, video.videoWidth, video.videoHeight);
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.putImageData(imageData, 0, 0);
        sendFrameToServer(canvas, "enhance");
        

} catch (error) {
        console.error("Error: ", error)
    }

}


function getThreshold(image) {
    const grayImage = new cv.Mat();
    cv.cvtColor(image, grayImage, cv.COLOR_RGBA2GRAY, 0);

    const meanScalar = cv.mean(grayImage);
    const averageIntensity = meanScalar[0];

    grayImage.delete();

    return averageIntensity;
}

function sendFrameToServer(canvas,message) {
    canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('frame', blob, 'frame.png');
        formData.append('message',message ); 
        fetch('https://aiserver.daotao.ai/linknk/upload_frame', {
            method: 'POST',
            body: formData
        }).then(response => response.json())
            .then(data => {
                displayServerResponse(data.message);
            }).catch(error => {
                console.error('Error sending frame to server:', error);
            });
    }, 'image/png');
}

function displayServerResponse(message) {
    const responseDiv = document.getElementById('response');
    if (message === "No Face") {
        responseDiv.innerText = "No face detected. Please position yourself in front of the camera.";
        responseDiv.style.color = 'red';
    } else if (message === "Real") {
        responseDiv.innerText = "REAL";
        responseDiv.style.color = 'green';
    } else if (message === "Fake") {
        responseDiv.innerText = "FAKE";
        responseDiv.style.color = 'red';
    }
}

// Display a warning message
function displayWarning(message) {
    let warningDiv = document.getElementById('warning');
    if (!warningDiv) {
        warningDiv = document.createElement('div');
        warningDiv.id = 'warning';
        warningDiv.style.color = 'red';
        document.body.appendChild(warningDiv);
    }
    warningDiv.innerText = message;
}

function clearWarning() {
    const warningDiv = document.getElementById('warning');
    if (warningDiv) {
        warningDiv.innerText = '';
    }
    
}
function clearResponse() {
    const responseDiv = document.getElementById('response');
    if (responseDiv) {
        responseDiv.innerText = '';
    }
}